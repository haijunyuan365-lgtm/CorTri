import os
import sys
import json
import csv
import math
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ====== 路径修正：确保能导入你项目里的 CorrelationModel ======
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from correlation_models import CorrelationModel


# =========================================================
# 1. 与你 stage1 一致的配置
# =========================================================
DATASET_CONFIGS = {
    "mosei_senti": {
        "text_in_dim": 768,
        "audio_in_dim": 74,
        "vision_in_dim": 35,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "out_dim": 64,
    },
    "ch_sims": {
        "text_in_dim": 768,
        "audio_in_dim": 25,
        "vision_in_dim": 177,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "out_dim": 64,
    }
}


# =========================================================
# 2. 读 pkl
# =========================================================
def load_split_from_pkl(pkl_path, split="valid"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    split_data = data[split]
    return split_data


# =========================================================
# 3. 长度推断
# =========================================================
def infer_text_len(text_emb, text_bert=None):
    """
    优先尝试从 text_bert 推断 attention mask；
    如果不可靠，则退回到 text embedding 非零行数。
    """
    # text_bert: shape may be [3, T]
    if text_bert is not None:
        tb = np.asarray(text_bert)
        if tb.ndim == 2:
            candidate_lens = []
            for r in range(tb.shape[0]):
                vals = np.unique(tb[r])
                if set(vals.tolist()).issubset({0, 1}):
                    s = int(tb[r].sum())
                    if 1 <= s <= tb.shape[1]:
                        candidate_lens.append(s)
            if len(candidate_lens) > 0:
                return max(candidate_lens)

    # fallback：按 embedding 非零行判断
    text_emb = np.asarray(text_emb)
    row_norm = np.linalg.norm(text_emb, axis=-1)
    nz = int((row_norm > 1e-8).sum())
    return max(1, nz)


# =========================================================
# 4. 一些统计函数
# =========================================================
def compute_pairwise_corr_matrix(x, y):
    """
    x: [Tx, D] torch.Tensor
    y: [Ty, D] torch.Tensor
    return: [Tx, Ty] numpy array
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    corr = torch.matmul(x, y.transpose(0, 1))
    return corr.detach().cpu().numpy()


def summarize_corr_matrix(corr, topk_ratio=0.10):
    """
    返回：
    - mean_corr
    - topk_concentration
    """
    mean_corr = float(np.mean(corr))

    score = (corr + 1.0) / 2.0
    score = np.clip(score, 0.0, 1.0)

    flat = score.reshape(-1)
    k = max(1, int(len(flat) * topk_ratio))
    topk_sum = np.sort(flat)[-k:].sum()
    total_sum = flat.sum() + 1e-12
    topk_concentration = float(topk_sum / total_sum)

    return mean_corr, topk_concentration


def short_text(s, max_len=140):
    s = str(s).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


# =========================================================
# 5. 模型前向：单样本
# =========================================================
@torch.no_grad()
def forward_one_sample(
    model,
    text_np,
    audio_np,
    vision_np,
    text_len,
    audio_len,
    vision_len,
    device,
    max_audio_vision_len=400,
    vision_noise_std=0.0,
    seed=1234,
):
    """
    对单个样本跑 stage1。
    这里专门支持：只对 vision 加噪声。
    """
    text_len = int(text_len)
    audio_len = int(audio_len)
    vision_len = int(vision_len)

    # 截断到真实长度
    text_np = np.asarray(text_np)[:text_len]
    audio_np = np.asarray(audio_np)[:min(audio_len, max_audio_vision_len)]
    vision_np = np.asarray(vision_np)[:min(vision_len, max_audio_vision_len)]

    # 转 tensor
    text = torch.tensor(text_np, dtype=torch.float32).unsqueeze(0).to(device)    # [1, Tt, D]
    audio = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, Ta, D]
    vision = torch.tensor(vision_np, dtype=torch.float32).unsqueeze(0).to(device)# [1, Tv, D]

    # 只扰动 vision
    if vision_noise_std > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        vision = vision + vision_noise_std * torch.randn_like(vision)

    # 因为我们已经按真实长度裁掉了，所以 pad mask 全 False
    text_pad = torch.zeros((1, text.size(1)), dtype=torch.bool, device=device)
    audio_pad = torch.zeros((1, audio.size(1)), dtype=torch.bool, device=device)
    vision_pad = torch.zeros((1, vision.size(1)), dtype=torch.bool, device=device)

    t_out, a_out, v_out = model(
        text, audio, vision,
        text_pad_mask=text_pad,
        audio_pad_mask=audio_pad,
        vision_pad_mask=vision_pad
    )

    t_out = t_out[0]
    a_out = a_out[0]
    v_out = v_out[0]

    C_ta = compute_pairwise_corr_matrix(t_out, a_out)
    C_tv = compute_pairwise_corr_matrix(t_out, v_out)
    C_av = compute_pairwise_corr_matrix(a_out, v_out)

    mats = {"TA": C_ta, "TV": C_tv, "AV": C_av}
    stats = {}

    for k, mat in mats.items():
        mean_corr, topk_conc = summarize_corr_matrix(mat, topk_ratio=0.10)
        stats[k] = {
            "mean": mean_corr,
            "topk": topk_conc
        }

    overall_mean = (stats["TA"]["mean"] + stats["TV"]["mean"] + stats["AV"]["mean"]) / 3.0
    overall_topk = (stats["TA"]["topk"] + stats["TV"]["topk"] + stats["AV"]["topk"]) / 3.0

    return mats, stats, overall_mean, overall_topk


# =========================================================
# 6. 主逻辑：扫描 valid split，选最适合做图的样本
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mosei_senti", choices=["mosei_senti", "ch_sims"])
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--pkl_path", type=str, required=True, help="例如 /path/to/unaligned_50.pkl")
    parser.add_argument("--ckpt", type=str, required=True, help="stage1 模型权重 .pt")
    parser.add_argument("--out_dir", type=str, default="vis_sample_selection")
    parser.add_argument("--vision_noise_std", type=float, default=0.10)
    parser.add_argument("--max_audio_vision_len", type=int, default=400)

    # 长度过滤
    parser.add_argument("--min_text_len", type=int, default=8)
    parser.add_argument("--min_audio_len", type=int, default=80)
    parser.add_argument("--min_vision_len", type=int, default=60)

    # 情感强度过滤（不是必须，但一般更好看）
    parser.add_argument("--min_abs_label", type=float, default=0.5)

    # 输出 topk
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # ---------- load model ----------
    config = DATASET_CONFIGS[args.dataset_name]
    model = CorrelationModel(**config).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[Info] Loaded ckpt: {args.ckpt}")

    # ---------- load data ----------
    split_data = load_split_from_pkl(args.pkl_path, split=args.split)

    text_all = split_data["text"]
    audio_all = split_data["audio"]
    vision_all = split_data["vision"]
    ids_all = split_data["id"]
    raw_text_all = split_data["raw_text"]
    labels_all = split_data["regression_labels"]

    audio_lengths = split_data.get("audio_lengths", None)
    vision_lengths = split_data.get("vision_lengths", None)
    text_bert_all = split_data.get("text_bert", None)

    n = len(ids_all)
    print(f"[Info] Loaded split={args.split}, #samples={n}")

    results = []

    # ---------- scan ----------
    for idx in range(n):
        try:
            text_np = text_all[idx]
            audio_np = audio_all[idx]
            vision_np = vision_all[idx]

            text_bert = text_bert_all[idx] if text_bert_all is not None else None

            text_len = infer_text_len(text_np, text_bert)
            audio_len = int(audio_lengths[idx]) if audio_lengths is not None else int(audio_np.shape[0])
            vision_len = int(vision_lengths[idx]) if vision_lengths is not None else int(vision_np.shape[0])

            label = float(labels_all[idx])

            # ---------- 基础过滤 ----------
            if text_len < args.min_text_len:
                continue
            if audio_len < args.min_audio_len:
                continue
            if vision_len < args.min_vision_len:
                continue
            if abs(label) < args.min_abs_label:
                continue

            # ---------- original ----------
            orig_mats, orig_stats, orig_overall_mean, orig_overall_topk = forward_one_sample(
                model=model,
                text_np=text_np,
                audio_np=audio_np,
                vision_np=vision_np,
                text_len=text_len,
                audio_len=audio_len,
                vision_len=vision_len,
                device=device,
                max_audio_vision_len=args.max_audio_vision_len,
                vision_noise_std=0.0,
                seed=1234,
            )

            # ---------- vision-perturbed ----------
            pert_mats, pert_stats, pert_overall_mean, pert_overall_topk = forward_one_sample(
                model=model,
                text_np=text_np,
                audio_np=audio_np,
                vision_np=vision_np,
                text_len=text_len,
                audio_len=audio_len,
                vision_len=vision_len,
                device=device,
                max_audio_vision_len=args.max_audio_vision_len,
                vision_noise_std=args.vision_noise_std,
                seed=1234,
            )

            # ---------- 排序指标 ----------
            delta_tv = orig_stats["TV"]["mean"] - pert_stats["TV"]["mean"]
            delta_av = orig_stats["AV"]["mean"] - pert_stats["AV"]["mean"]
            delta_ta_abs = abs(orig_stats["TA"]["mean"] - pert_stats["TA"]["mean"])

            # 主排序分数：
            # 希望 TV / AV 明显下降，同时 TA 保持稳定
            q_score = 0.45 * delta_tv + 0.45 * delta_av - 0.10 * delta_ta_abs

            results.append({
                "idx": idx,
                "id": ids_all[idx],
                "label": label,
                "text_len": int(text_len),
                "audio_len": int(audio_len),
                "vision_len": int(vision_len),
                "raw_text": short_text(raw_text_all[idx], max_len=160),

                "orig_overall_mean": float(orig_overall_mean),
                "orig_overall_topk": float(orig_overall_topk),
                "pert_overall_mean": float(pert_overall_mean),
                "pert_overall_topk": float(pert_overall_topk),

                "orig_ta_mean": float(orig_stats["TA"]["mean"]),
                "orig_tv_mean": float(orig_stats["TV"]["mean"]),
                "orig_av_mean": float(orig_stats["AV"]["mean"]),
                "orig_ta_topk": float(orig_stats["TA"]["topk"]),
                "orig_tv_topk": float(orig_stats["TV"]["topk"]),
                "orig_av_topk": float(orig_stats["AV"]["topk"]),

                "pert_ta_mean": float(pert_stats["TA"]["mean"]),
                "pert_tv_mean": float(pert_stats["TV"]["mean"]),
                "pert_av_mean": float(pert_stats["AV"]["mean"]),
                "pert_ta_topk": float(pert_stats["TA"]["topk"]),
                "pert_tv_topk": float(pert_stats["TV"]["topk"]),
                "pert_av_topk": float(pert_stats["AV"]["topk"]),

                "delta_tv": float(delta_tv),
                "delta_av": float(delta_av),
                "delta_ta_abs": float(delta_ta_abs),
                "q_score": float(q_score),
            })

        except Exception as e:
            print(f"[Warn] idx={idx} failed: {e}")
            continue

    if len(results) == 0:
        print("[Error] No valid candidates found. Please relax the thresholds.")
        return

    # ---------- 先按 original overall mean 做分位数过滤 ----------
    orig_scores = np.array([r["orig_overall_mean"] for r in results], dtype=np.float32)
    thresh = float(np.percentile(orig_scores, 70))

    filtered = [r for r in results if r["orig_overall_mean"] >= thresh]
    if len(filtered) == 0:
        filtered = results

    # ---------- 最终排序 ----------
    filtered.sort(key=lambda x: (x["q_score"], x["orig_overall_mean"], x["orig_overall_topk"]), reverse=True)

    topk = filtered[:args.topk]

    # ---------- print ----------
    print("\n" + "=" * 120)
    print("Top candidates for 4.4.1 visualization")
    print(f"(split={args.split}, vision_noise_std={args.vision_noise_std}, orig_mean_percentile_threshold={thresh:.4f})")
    print("=" * 120)

    for rank, r in enumerate(topk, 1):
        print(
            f"[Rank {rank}] idx={r['idx']} | id={r['id']} | label={r['label']:.3f} | "
            f"lens=(T:{r['text_len']}, A:{r['audio_len']}, V:{r['vision_len']}) | "
            f"orig_mean={r['orig_overall_mean']:.4f} | q_score={r['q_score']:.4f} | "
            f"ΔTV={r['delta_tv']:.4f} | ΔAV={r['delta_av']:.4f} | |ΔTA|={r['delta_ta_abs']:.4f}\n"
            f"          text: {r['raw_text']}"
        )

    # ---------- save csv ----------
    csv_path = os.path.join(args.out_dir, "top_candidates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(topk[0].keys()))
        writer.writeheader()
        writer.writerows(topk)

    # ---------- save best json ----------
    best = topk[0]
    json_path = os.path.join(args.out_dir, "best_candidate.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 120)
    print(f"Saved top candidates to: {csv_path}")
    print(f"Saved best candidate to: {json_path}")
    print("-" * 120)


if __name__ == "__main__":
    main()