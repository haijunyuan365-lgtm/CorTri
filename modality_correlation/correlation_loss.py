import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleLoss(nn.Module):
    def __init__(self, margin=0.2, neg_inf=-1e4):
        super(TripleLoss, self).__init__()
        self.margin = margin

    def forward(self, F_anchor, F_pos, F_neg, **kwargs):
        """
        注意这里的 **kwargs 是为了吸收 main_correlation.py 传进来的 mask 参数，
        这样你就完全不需要去改 main_correlation.py 的代码了！
        F_anchor, F_pos, F_neg 现在的形状都是 [B, D]
        """
        
        # Average pooling to [B, D]
        F_anchor_mean = F_anchor.mean(dim=1)
        F_pos_mean = F_pos.mean(dim=1)
        F_neg_mean = F_neg.mean(dim=1)

        # Calculate distances
        # The higher the cosine similarity, the lower the distance. Here, we define the distance as 1 - cos_sim
        dist_pos = 1 - F.cosine_similarity(F_anchor_mean, F_pos_mean, dim=-1)
        dist_neg = 1 - F.cosine_similarity(F_anchor_mean, F_neg_mean, dim=-1)

        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()