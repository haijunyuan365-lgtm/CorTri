import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleLoss(nn.Module):
    def __init__(self, margin=0.2, neg_inf=-1e4):
        super(TripleLoss, self).__init__()
        self.margin = margin
        self.neg_inf = neg_inf  # 用于 masked max

    def forward(
        self,
        F_anchor, F_pos, F_neg,
        anchor_pad_mask=None, pos_pad_mask=None, neg_pad_mask=None
    ):
        """
        F_anchor: [B, Ta, D]
        F_pos:    [B, Tp, D]
        F_neg:    [B, Tn, D]
        *_pad_mask: [B, T]  bool, True 表示 padding

        修复2：
        - 对 pos/neg 的 padding 列在 max 之前 masked_fill(-inf)
        - anchor 的 padding 行在 mean 时不计入
        """

        anchor_norm = F.normalize(F_anchor, p=2, dim=-1)
        pos_norm    = F.normalize(F_pos,    p=2, dim=-1)
        neg_norm    = F.normalize(F_neg,    p=2, dim=-1)

        sim_pos = torch.bmm(anchor_norm, pos_norm.transpose(1, 2))  # [B, Ta, Tp]
        sim_neg = torch.bmm(anchor_norm, neg_norm.transpose(1, 2))  # [B, Ta, Tn]

        # mask 掉 pos/neg 的 padding 列，防止 max 匹配到 padding
        if pos_pad_mask is not None:
            sim_pos = sim_pos.masked_fill(pos_pad_mask.unsqueeze(1), self.neg_inf)
        if neg_pad_mask is not None:
            sim_neg = sim_neg.masked_fill(neg_pad_mask.unsqueeze(1), self.neg_inf)

        max_pos = sim_pos.max(dim=-1).values  # [B, Ta]
        max_neg = sim_neg.max(dim=-1).values  # [B, Ta]

        # anchor 的 padding 行不参与平均
        if anchor_pad_mask is not None:
            valid = (~anchor_pad_mask).float()  # [B, Ta]
            max_pos = max_pos * valid
            max_neg = max_neg * valid
            denom = valid.sum(dim=-1).clamp(min=1.0)  # [B]
            score_pos = max_pos.sum(dim=-1) / denom
            score_neg = max_neg.sum(dim=-1) / denom
        else:
            score_pos = max_pos.mean(dim=-1)
            score_neg = max_neg.mean(dim=-1)

        dist_pos = 1.0 - score_pos
        dist_neg = 1.0 - score_neg

        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()
