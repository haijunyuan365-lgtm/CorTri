import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from correlation_models import CorrelationModel
from correlation_dataset import UnifiedMultimodalDataset

# 与你训练脚本一致
dataset_specific_configs = {
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

def build_pad_mask(lengths, max_len):
    # lengths: [B]
    B = lengths.size(0)
    ar = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, max_len)
    return ar >= lengths.unsqueeze(1)  # True is padding

def get_collate_fn(max_len=400):
    def _safe_trunc(x):
        if x.size(0) == 0:
            return torch.zeros(1, x.size(-1), dtype=x.dtype)
        return x[:max_len]

    def _pad_and_mask(seqs):
        lens = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True)  # [B,T,D]
        pad_mask = build_pad_mask(lens, padded.size(1))
        return padded, pad_mask

    def collate_fn(batch):
        # UnifiedMultimodalDataset(for_correlation=True) returns:
        # ((meta, text, audio, vision), (text_neg, audio_neg, vision_neg), label, META)
        texts     = [_safe_trunc(item[0][1]) for item in batch]
        audios    = [_safe_trunc(item[0][2]) for item in batch]
        visions   = [_safe_trunc(item[0][3]) for item in batch]

        texts, text_pad = _pad_and_mask(texts)
        audios, audio_pad = _pad_and_mask(audios)
        visions, vision_pad = _pad_and_mask(visions)

        return texts, audios, visions, text_pad, audio_pad, vision_pad

    return collate_fn

@torch.no_grad()
def max_align_score(anchor, other, other_pad_mask=None, anchor_pad_mask=None, neg_inf=-1e4):
    """
    anchor: [B, Ta, D]
    other:  [B, To, D]
    *_pad_mask: [B, T] True means padding
    returns: score [B]   (masked max-align then masked mean over anchor tokens)
    """
    a = F.normalize(anchor, p=2, dim=-1)
    o = F.normalize(other,  p=2, dim=-1)
    sim = torch.bmm(a, o.transpose(1,2))  # [B, Ta, To]

    if other_pad_mask is not None:
        sim = sim.masked_fill(other_pad_mask.unsqueeze(1), neg_inf)

    max_sim = sim.max(dim=-1).values  # [B, Ta]

    if anchor_pad_mask is not None:
        valid = (~anchor_pad_mask).float()
        max_sim = max_sim * valid
        denom = valid.sum(dim=-1).clamp(min=1.0)
        score = max_sim.sum(dim=-1) / denom
    else:
        score = max_sim.mean(dim=-1)

    return score

def aggregate(stats):
    # stats: list of dicts containing sums and counts
    out = {}
    for k in stats[0].keys():
        if k.endswith("_count"):
            continue
        count_key = k + "_count"
        s = sum(x[k] for x in stats)
        c = sum(x[count_key] for x in stats)
        out[k] = (s / max(c, 1.0))
    return out

@torch.no_grad()
def evaluate(model, loader, device, margin=0.2, use_shuffle_neg=True):
    model.eval()
    batch_stats = []

    for (text, audio, vision, text_pad, audio_pad, vision_pad) in loader:
        text = text.to(device); audio = audio.to(device); vision = vision.to(device)
        text_pad = text_pad.to(device); audio_pad = audio_pad.to(device); vision_pad = vision_pad.to(device)

        f_t, f_a, f_v = model(
            text, audio, vision,
            text_pad_mask=text_pad, audio_pad_mask=audio_pad, vision_pad_mask=vision_pad
        )

        # 负样本：推荐用 batch 内 shuffle（更干净），用于评估判别力
        if use_shuffle_neg:
            perm = torch.randperm(f_t.size(0), device=device)
            f_t_n = f_t[perm]
            tpad_n = text_pad[perm]

            perm = torch.randperm(f_a.size(0), device=device)
            f_a_n = f_a[perm]
            apad_n = audio_pad[perm]

            perm = torch.randperm(f_v.size(0), device=device)
            f_v_n = f_v[perm]
            vpad_n = vision_pad[perm]
        else:
            # 不 shuffle 的话，就用自身当 neg（无意义），这里只是兜底
            f_t_n, tpad_n = f_t, text_pad
            f_a_n, apad_n = f_a, audio_pad
            f_v_n, vpad_n = f_v, vision_pad

        # 三对模态：计算 pos/neg 分数、gap、margin_acc
        # T-A
        ta_pos = max_align_score(f_t, f_a, other_pad_mask=audio_pad, anchor_pad_mask=text_pad)
        ta_neg = max_align_score(f_t, f_a_n, other_pad_mask=apad_n, anchor_pad_mask=text_pad)
        ta_gap = (ta_pos - ta_neg)
        ta_acc = (ta_pos >= ta_neg + margin).float()

        # T-V
        tv_pos = max_align_score(f_t, f_v, other_pad_mask=vision_pad, anchor_pad_mask=text_pad)
        tv_neg = max_align_score(f_t, f_v_n, other_pad_mask=vpad_n, anchor_pad_mask=text_pad)
        tv_gap = (tv_pos - tv_neg)
        tv_acc = (tv_pos >= tv_neg + margin).float()

        # A-V
        av_pos = max_align_score(f_a, f_v, other_pad_mask=vision_pad, anchor_pad_mask=audio_pad)
        av_neg = max_align_score(f_a, f_v_n, other_pad_mask=vpad_n, anchor_pad_mask=audio_pad)
        av_gap = (av_pos - av_neg)
        av_acc = (av_pos >= av_neg + margin).float()

        B = f_t.size(0)
        st = {
            "ta_pos_sum": ta_pos.sum().item(), "ta_pos_sum_count": B,
            "ta_neg_sum": ta_neg.sum().item(), "ta_neg_sum_count": B,
            "ta_gap_sum": ta_gap.sum().item(), "ta_gap_sum_count": B,
            "ta_acc_sum": ta_acc.sum().item(), "ta_acc_sum_count": B,

            "tv_pos_sum": tv_pos.sum().item(), "tv_pos_sum_count": B,
            "tv_neg_sum": tv_neg.sum().item(), "tv_neg_sum_count": B,
            "tv_gap_sum": tv_gap.sum().item(), "tv_gap_sum_count": B,
            "tv_acc_sum": tv_acc.sum().item(), "tv_acc_sum_count": B,

            "av_pos_sum": av_pos.sum().item(), "av_pos_sum_count": B,
            "av_neg_sum": av_neg.sum().item(), "av_neg_sum_count": B,
            "av_gap_sum": av_gap.sum().item(), "av_gap_sum_count": B,
            "av_acc_sum": av_acc.sum().item(), "av_acc_sum_count": B,
        }
        batch_stats.append(st)

    agg = aggregate(batch_stats)

    # 转成更好读的格式
    def pick(prefix):
        return {
            "pos": agg[f"{prefix}_pos_sum"],
            "neg": agg[f"{prefix}_neg_sum"],
            "gap": agg[f"{prefix}_gap_sum"],
            "margin_acc": agg[f"{prefix}_acc_sum"],
        }

    report = {
        "T-A": pick("ta"),
        "T-V": pick("tv"),
        "A-V": pick("av"),
    }

    # overall
    report["OVERALL"] = {
        "pos": (report["T-A"]["pos"] + report["T-V"]["pos"] + report["A-V"]["pos"]) / 3.0,
        "neg": (report["T-A"]["neg"] + report["T-V"]["neg"] + report["A-V"]["neg"]) / 3.0,
        "gap": (report["T-A"]["gap"] + report["T-V"]["gap"] + report["A-V"]["gap"]) / 3.0,
        "margin_acc": (report["T-A"]["margin_acc"] + report["T-V"]["margin_acc"] + report["A-V"]["margin_acc"]) / 3.0,
    }
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/autodl-fs')
    parser.add_argument('--dataset_name', type=str, default='mosei_senti')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--ckpt', type=str, default='pre_trained_models/correlation_model.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=400)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--no_shuffle_neg', action='store_true', help='disable batch-shuffle negatives (not recommended)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    ds = UnifiedMultimodalDataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type=args.split,
        if_align=False,
        for_correlation=True,
        perturbation_ratio=0.0,
        noise_std=0.0
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=get_collate_fn(max_len=args.max_len),
        drop_last=False
    )

    # model
    config = dataset_specific_configs[args.dataset_name]
    model = CorrelationModel(**config).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    report = evaluate(
        model, loader, device,
        margin=args.margin,
        use_shuffle_neg=(not args.no_shuffle_neg)
    )

    print("\n========== Stage-1 Correlation Model Eval ==========")
    print(f"split={args.split} | ckpt={args.ckpt} | margin={args.margin} | shuffle_neg={not args.no_shuffle_neg}")
    for k in ["T-A", "T-V", "A-V", "OVERALL"]:
        r = report[k]
        print(f"[{k}] pos={r['pos']:.4f}  neg={r['neg']:.4f}  gap={r['gap']:.4f}  margin_acc={r['margin_acc']:.4f}")
    print("===================================================\n")

if __name__ == "__main__":
    main()
