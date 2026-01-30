# correlation_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripleLoss, self).__init__()
        self.margin = margin

    def forward(self, F_anchor, F_pos, F_neg, mask_anchor=None, mask_pos=None, mask_neg=None):
        """
        F_anchor: [B, T1, D]
        F_pos:    [B, T2, D]
        F_neg:    [B, T3, D]
        mask_*:   [B, T*]  True=valid
        """

        # normalize
        F_anchor = F.normalize(F_anchor, p=2, dim=-1, eps=1e-8)
        F_pos    = F.normalize(F_pos,    p=2, dim=-1, eps=1e-8)
        F_neg    = F.normalize(F_neg,    p=2, dim=-1, eps=1e-8)

        # Corr matrices
        Corr_pos = torch.bmm(F_anchor, F_pos.transpose(1, 2))  # [B,T1,T2]
        Corr_neg = torch.bmm(F_anchor, F_neg.transpose(1, 2))  # [B,T1,T3]

        def masked_max_mean(Corr, m1, m2):
            # m1: [B,T1], m2:[B,T2]
            if (m1 is None) or (m2 is None):
                max_corr = Corr.max(dim=2)[0].max(dim=1)[0]      # [B]
                mean_corr = Corr.mean(dim=(1,2))                 # [B]
                return max_corr, mean_corr

            pair_mask = (m1.unsqueeze(2) & m2.unsqueeze(1))      # [B,T1,T2]

            # max：无效位置设 -inf
            Corr_for_max = Corr.masked_fill(~pair_mask, -1e9)
            max_corr = Corr_for_max.max(dim=2)[0].max(dim=1)[0]  # [B]

            # mean：只对有效位置平均
            Corr_for_sum = Corr.masked_fill(~pair_mask, 0.0)
            denom = pair_mask.sum(dim=(1,2)).float().clamp_min(1.0)  # [B]
            mean_corr = Corr_for_sum.sum(dim=(1,2)) / denom          # [B]
            return max_corr, mean_corr

        max_pos, mean_pos = masked_max_mean(Corr_pos, mask_anchor, mask_pos)
        max_neg, mean_neg = masked_max_mean(Corr_neg, mask_anchor, mask_neg)

        corr_pos = 0.5 * (max_pos + mean_pos)
        corr_neg = 0.5 * (max_neg + mean_neg)

        dist_pos = 1.0 - corr_pos
        dist_neg = 1.0 - corr_neg

        loss = F.relu(dist_pos - dist_neg + self.margin).mean()
        return loss
