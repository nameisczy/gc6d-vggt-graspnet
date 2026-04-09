# -*- coding: utf-8 -*-
"""
占位 Encoder：仅用于从单数据点跑通 pipeline。
输入: point_cloud (B, N, 3)
输出: feature (B, feat_dim)
"""

import torch
import torch.nn as nn


class PlaceholderEncoder(nn.Module):
    """简单点云 encoder：mean pool + MLP，无正则、易过拟合单样本。"""

    def __init__(self, point_dim: int = 3, feat_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: (B, N, 3)
        return: (B, feat_dim)
        """
        x = point_cloud.mean(dim=1)  # (B, 3)
        return self.mlp(x)  # (B, feat_dim)
