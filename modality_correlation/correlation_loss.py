# correlation_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripleLoss, self).__init__()
        self.margin = margin

    def forward(self, F_anchor, F_pos, F_neg):
        """
        F_anchor: [B, T1, D] (例如 Text)
        F_pos:    [B, T2, D] (例如 Audio Positive)
        F_neg:    [B, T3, D] (例如 Audio Negative)
        
        Modification for Idea:
        不再使用 .mean(dim=1) 进行全局池化。
        而是计算细粒度相关性矩阵，并基于矩阵的最大匹配度(Max-Alignment)计算距离。
        这迫使模型学习 Token/Frame 级别的细粒度相关性，而非全局统计量的相关性。
        """
        
        # 1. 归一化特征 (为了计算 Cosine Similarity 矩阵)
        # norm: [B, T, D]
        anchor_norm = F.normalize(F_anchor, p=2, dim=-1)
        pos_norm = F.normalize(F_pos, p=2, dim=-1)
        neg_norm = F.normalize(F_neg, p=2, dim=-1)

        # 2. 计算细粒度相关性矩阵 [B, T_anchor, T_other]
        # matrix[b, i, j] 代表 anchor 第 i 个时刻与 pos/neg 第 j 个时刻的相似度
        sim_matrix_pos = torch.bmm(anchor_norm, pos_norm.transpose(1, 2))
        sim_matrix_neg = torch.bmm(anchor_norm, neg_norm.transpose(1, 2))

        # 3. 计算相似度分数 (Similarity Score)
        # 策略: Max-over-time Pooling (Chamfer-like score)
        # 对于 Anchor 中的每一个时间步，找到 Pos/Neg 中最相似的时间步作为匹配分，然后平均。
        # 这样既保留了细粒度信息，又允许局部错位 (Local Misalignment)。
        
        # values shape: [B, T_anchor] -> mean -> [B]
        score_pos = sim_matrix_pos.max(dim=-1).values.mean(dim=-1)
        score_neg = sim_matrix_neg.max(dim=-1).values.mean(dim=-1)

        # 4. 计算距离与 Loss
        # Cosine Similarity 越大，距离越小
        dist_pos = 1.0 - score_pos
        dist_neg = 1.0 - score_neg

        # Triplet Loss: max(0, dist_pos - dist_neg + margin)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        
        return loss.mean()