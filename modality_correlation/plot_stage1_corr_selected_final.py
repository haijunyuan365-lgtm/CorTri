import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec


# =========================================================
# 路径修正：确保能导入你项目里的 CorrelationModel
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from correlation_models import CorrelationModel


# =========================================================
# 1. 与 stage1 一致的配置
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
# 2. 数据读取与长度推断
# =========================================================
def load_split_from_pkl(pkl_path, split="valid"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data[split]


def infer_text_len(text_emb, text_bert=None):
    """
    优先尝试从 text_bert 推断 mask；
    若不可靠，则退回到 text embedding 非零行数。
    """
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

    text_emb = np.asarray(text_emb)
    row_norm = np.linalg.norm(text_emb, axis=-1)
    nz = int((row_norm > 1e-8).sum())
    return max(1, nz)


# =========================================================
# 3. 相关矩阵与指标
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


def mean_corr(corr):
    return float(np.mean(corr))


def symmetric_max_align_score(corr):
    """
    corr: [Tx, Ty]
    s_xy = 0.5 * (mean_i max_j C_ij + mean_j max_i C_ij)
    """
    row_best = np.max(corr, axis=1).mean()
    col_best = np.max(corr, axis=0).mean()
    return float(0.5 * (row_best + col_best))


def compute_stats(mats_orig, mats_pert):
    keys = ["TA", "TV", "AV"]

    mean_orig = {k: mean_corr(mats_orig[k]) for k in keys}
    mean_pert = {k: mean_corr(mats_pert[k]) for k in keys}

    align_orig = {k: symmetric_max_align_score(mats_orig[k]) for k in keys}
    align_pert = {k: symmetric_max_align_score(mats_pert[k]) for k in keys}
    align_drop = {k: align_orig[k] - align_pert[k] for k in keys}

    overall = {
        "orig_mean_overall": float(np.mean([mean_orig[k] for k in keys])),
        "pert_mean_overall": float(np.mean([mean_pert[k] for k in keys])),
        "orig_align_overall": float(np.mean([align_orig[k] for k in keys])),
        "pert_align_overall": float(np.mean([align_pert[k] for k in keys])),
    }

    return mean_orig, mean_pert, align_orig, align_pert, align_drop, overall


# =========================================================
# 4. 单样本前向
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
    text_len = int(text_len)
    audio_len = int(audio_len)
    vision_len = int(vision_len)

    text_np = np.asarray(text_np)[:text_len]
    audio_np = np.asarray(audio_np)[:min(audio_len, max_audio_vision_len)]
    vision_np = np.asarray(vision_np)[:min(vision_len, max_audio_vision_len)]

    text = torch.tensor(text_np, dtype=torch.float32).unsqueeze(0).to(device)
    audio = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)
    vision = torch.tensor(vision_np, dtype=torch.float32).unsqueeze(0).to(device)

    # 仅对 vision 加噪声
    if vision_noise_std > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        noise = torch.randn(vision.shape, device=vision.device, dtype=vision.dtype)
        vision = vision + vision_noise_std * noise

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

    return {
        "TA": C_ta,
        "TV": C_tv,
        "AV": C_av,
    }


# =========================================================
# 5. 画图辅助
# =========================================================
def set_compact_ticks(ax, n_x, n_y):
    """
    为热力图设置更紧凑的刻度，避免过密。
    """
    def build_ticks(n):
        if n <= 6:
            return list(range(n))
        ticks = np.linspace(0, n - 1, 5)
        ticks = np.unique(np.round(ticks).astype(int))
        return ticks.tolist()

    xticks = build_ticks(n_x)
    yticks = build_ticks(n_y)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=8)


# =========================================================
# 6. 绘图
# =========================================================
def plot_selected_sample_figure(
    mats_orig,
    mats_pert,
    align_orig,
    align_pert,
    align_drop,
    save_png,
    save_pdf=None,
    cmap="RdBu_r",
    vmin_pct=1.0,
    vmax_pct=99.0,
):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10

    # 统一色条范围
    all_vals = np.concatenate([
        mats_orig["TA"].ravel(),
        mats_orig["TV"].ravel(),
        mats_orig["AV"].ravel(),
        mats_pert["TA"].ravel(),
        mats_pert["TV"].ravel(),
        mats_pert["AV"].ravel(),
    ])

    vmin = float(np.percentile(all_vals, vmin_pct))
    vmax = float(np.percentile(all_vals, vmax_pct))
    if vmax - vmin < 1e-8:
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())

    fig = plt.figure(figsize=(15.5, 6.2)) 
    
    # 【修复重点1】: 调大 wspace 到 0.38，彻底拉开矩阵图，防止 Y 轴文字重叠。
    # 稍微缩小了用来缓冲的第 5 列 (0.25)，让 Colorbar 和右侧图表靠得更协调。
    gs = gridspec.GridSpec(
        2, 6,
        width_ratios=[1, 1, 1, 0.05, 0.25, 1.35], 
        wspace=0.38, 
        hspace=0.30  
    )

    keys = ["TA", "TV", "AV"]
    titles = ["Text–Audio", "Text–Vision", "Audio–Vision"]
    axis_names = {
        "TA": ("Audio index", "Text index"),
        "TV": ("Vision index", "Text index"),
        "AV": ("Vision index", "Audio index"),
    }

    # ---------- 第一行：Original ----------
    last_im = None
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        last_im = ax.imshow(
            mats_orig[key],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )
        ax.set_title(titles[i], fontsize=12, pad=8)
        ax.set_xlabel(axis_names[key][0], fontsize=10)
        ax.set_ylabel(axis_names[key][1], fontsize=10)
        set_compact_ticks(ax, mats_orig[key].shape[1], mats_orig[key].shape[0])

    # ---------- 第二行：Vision-perturbed ----------
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(
            mats_pert[key],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )
        ax.set_xlabel(axis_names[key][0], fontsize=10)
        ax.set_ylabel(axis_names[key][1], fontsize=10)
        set_compact_ticks(ax, mats_pert[key].shape[1], mats_pert[key].shape[0])

    # ---------- Colorbar ----------
    cax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Correlation value", fontsize=11, labelpad=8)
    cb.ax.tick_params(labelsize=9)

    # ---------- 右侧 summary ----------
    subgs = gs[:, 5].subgridspec(2, 1, hspace=0.55)

    labels = ["T-A", "T-V", "A-V"]
    x = np.arange(len(labels))
    width = 0.32

    # (a) Symmetric max-alignment score
    ax1 = fig.add_subplot(subgs[0, 0])

    vals_orig = [align_orig[k] for k in keys]
    vals_pert = [align_pert[k] for k in keys]

    bars1 = ax1.bar(x - width/2, vals_orig, width, label="Original")
    bars2 = ax1.bar(x + width/2, vals_pert, width, label="Vision-perturbed")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("Symmetric max-alignment score", fontsize=12, pad=8)
    ax1.tick_params(labelsize=9)
    
    ax1.set_ylim(0.0, 1.25) 
    ax1.set_yticks(np.linspace(0.0, 1.0, 6))
    
    # 【修复重点2】: 将图例规规矩矩地收进坐标系内部，贴着上边缘，不再压线。
    ax1.legend(fontsize=9, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))

    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width()/2,
                h + 0.03, 
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0
            )

    # (b) Alignment-score drop
    ax2 = fig.add_subplot(subgs[1, 0])
    vals_drop = [align_drop[k] for k in keys]
    bars3 = ax2.bar(x, vals_drop, width=0.50)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Drop", fontsize=10)
    ax2.set_title("Alignment-score drop after visual perturbation", fontsize=12, pad=8)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.tick_params(labelsize=9)

    max_drop = max(vals_drop) if max(vals_drop) > 0 else 0.01
    upper = max(0.015, max_drop * 1.4) 
    ax2.set_ylim(0.0, upper)

    for b in bars3:
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width()/2,
            h + (upper * 0.03), 
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    # ---------- 行标签 ----------
    fig.text(0.02, 0.72, "Original sample", rotation=90,
             va="center", ha="center", fontsize=12)
    fig.text(0.02, 0.28, "Vision-perturbed sample", rotation=90,
             va="center", ha="center", fontsize=12)

    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.93, bottom=0.10,
    )

    plt.savefig(save_png, dpi=700, bbox_inches="tight")
    if save_pdf is not None:
        plt.savefig(save_pdf, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 7. 主逻辑
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mosei_senti", choices=["mosei_senti", "ch_sims"])
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--idx", type=int, default=1840)
    parser.add_argument("--sample_id", type=str, default="y8HXGm1-Ecw$_$5")
    parser.add_argument("--vision_noise_std", type=float, default=0.10)
    parser.add_argument("--max_audio_vision_len", type=int, default=400)
    parser.add_argument("--out_dir", type=str, default="vis_stage1_selected_final")
    parser.add_argument("--vmin_pct", type=float, default=1.0)
    parser.add_argument("--vmax_pct", type=float, default=99.0)
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
    print(f"[Info] Loaded split={args.split}")

    idx = args.idx
    sample_id = split_data["id"][idx]
    if args.sample_id is not None and sample_id != args.sample_id:
        print(f"[Warn] sample_id mismatch: expected={args.sample_id}, actual={sample_id}")

    text_np = split_data["text"][idx]
    audio_np = split_data["audio"][idx]
    vision_np = split_data["vision"][idx]
    raw_text = split_data["raw_text"][idx]
    label = float(split_data["regression_labels"][idx])

    text_bert = split_data["text_bert"][idx] if "text_bert" in split_data else None
    audio_len = int(split_data["audio_lengths"][idx]) if "audio_lengths" in split_data else int(audio_np.shape[0])
    vision_len = int(split_data["vision_lengths"][idx]) if "vision_lengths" in split_data else int(vision_np.shape[0])
    text_len = infer_text_len(text_np, text_bert)

    print(f"[Info] Using sample idx={idx}, id={sample_id}")
    print(f"[Info] label={label:.3f}, lengths: T={text_len}, A={audio_len}, V={vision_len}")
    print(f"[Info] raw_text: {raw_text[:200]}{'...' if len(str(raw_text)) > 200 else ''}")

    # ---------- forward ----------
    mats_orig = forward_one_sample(
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

    mats_pert = forward_one_sample(
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

    mean_orig, mean_pert, align_orig, align_pert, align_drop, overall = compute_stats(mats_orig, mats_pert)

    print("\n[Mean correlation]")
    print(f"  TA: {mean_orig['TA']:.6f} -> {mean_pert['TA']:.6f}")
    print(f"  TV: {mean_orig['TV']:.6f} -> {mean_pert['TV']:.6f}")
    print(f"  AV: {mean_orig['AV']:.6f} -> {mean_pert['AV']:.6f}")

    print("\n[Symmetric max-alignment score]")
    print(f"  TA: {align_orig['TA']:.6f} -> {align_pert['TA']:.6f} | drop={align_drop['TA']:.6f}")
    print(f"  TV: {align_orig['TV']:.6f} -> {align_pert['TV']:.6f} | drop={align_drop['TV']:.6f}")
    print(f"  AV: {align_orig['AV']:.6f} -> {align_pert['AV']:.6f} | drop={align_drop['AV']:.6f}")

    # ---------- save arrays ----------
    np.save(os.path.join(args.out_dir, "C_ta_original.npy"), mats_orig["TA"])
    np.save(os.path.join(args.out_dir, "C_tv_original.npy"), mats_orig["TV"])
    np.save(os.path.join(args.out_dir, "C_av_original.npy"), mats_orig["AV"])
    np.save(os.path.join(args.out_dir, "C_ta_vision_perturbed.npy"), mats_pert["TA"])
    np.save(os.path.join(args.out_dir, "C_tv_vision_perturbed.npy"), mats_pert["TV"])
    np.save(os.path.join(args.out_dir, "C_av_vision_perturbed.npy"), mats_pert["AV"])

    # ---------- save meta ----------
    meta = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "idx": idx,
        "sample_id": sample_id,
        "label": label,
        "text_len": int(text_len),
        "audio_len": int(min(audio_len, args.max_audio_vision_len)),
        "vision_len": int(min(vision_len, args.max_audio_vision_len)),
        "vision_noise_std": float(args.vision_noise_std),
        "raw_text": str(raw_text),
        "mean_orig": mean_orig,
        "mean_pert": mean_pert,
        "align_orig": align_orig,
        "align_pert": align_pert,
        "align_drop": align_drop,
        "overall": overall,
        "heatmap_percentile_range": {
            "vmin_pct": args.vmin_pct,
            "vmax_pct": args.vmax_pct,
        }
    }

    with open(os.path.join(args.out_dir, "selected_sample_stats.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------- plot ----------
    save_png = os.path.join(args.out_dir, "stage1_corr_selected_final.png")
    save_pdf = os.path.join(args.out_dir, "stage1_corr_selected_final.pdf")

    plot_selected_sample_figure(
        mats_orig=mats_orig,
        mats_pert=mats_pert,
        align_orig=align_orig,
        align_pert=align_pert,
        align_drop=align_drop,
        save_png=save_png,
        save_pdf=save_pdf,
        cmap="RdBu_r",
        vmin_pct=args.vmin_pct,
        vmax_pct=args.vmax_pct,
    )

    print("\n[Saved]")
    print(f"  Figure PNG : {save_png}")
    print(f"  Figure PDF : {save_pdf}")
    print(f"  Stats JSON : {os.path.join(args.out_dir, 'selected_sample_stats.json')}")


if __name__ == "__main__":
    main()