import torch
from torch import nn
import sys
import os
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.eval_metrics import *
# 引入 sklearn 指标库
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from modality_correlation.correlation_loss import TripleLoss

class ARLoss(nn.Module):
    """
    AR loss (for mosi/mosei regression-style labels).
    Encourages prediction to stay closer to the rounded target integer
    than to the adjacent integer on the wrong side.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # input/target: [B, D]
        eps = 1e-6
        zero = torch.tensor(0., device=target.device)

        # rounded "class" target
        classes = torch.round(target)

        B, D = input.shape
        loss = []

        for i in range(B):
            delta = input[i] - classes[i]   # [D]
            row = []
            for j in range(D):
                if delta[j] >= 0:
                    z = torch.ceil(input[i][j]).detach()
                    temp = torch.max(
                        zero, torch.abs(delta[j]) - torch.abs(input[i][j] - z) + eps
                    )
                else:
                    z = torch.floor(input[i][j]).detach()
                    temp = torch.max(
                        zero, torch.abs(delta[j]) - torch.abs(input[i][j] - z)
                    )
                row.append(temp)
            loss.append(torch.stack(row))

        loss = torch.stack(loss)  # [B, D]
        if self.reduction == 'sum':
            return loss.sum()
        return loss.mean()
