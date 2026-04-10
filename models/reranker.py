# -*- coding: utf-8 -*-
"""
Residual reranker：对 baseline 解码后的抓取候选预测标量 residual，与 baseline_score 融合后用于排序训练。

默认 9 维：[score, width, tolerance, center(3), approach_dir(3)]；
兼容 in_dim=3 的旧权重（仅 score/width/tolerance）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualReranker(nn.Module):
    """
    输入每 grasp 特征 (B, K, C)：
      - C=3：[baseline_score, width, tolerance]
      - C=9：上述 + grasp_center_xyz + approach_dir_xyz（approach 为与可微解码一致的 viewing / approach 方向，单位向量）
    输出 residual (B, K, 1)，由外部与 baseline 按配置融合为 final_score。
    """

    def __init__(self, in_dim: int = 9, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        h1 = int(hidden1)
        h2 = int(hidden2)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, K, in_dim) 或 (B, in_dim)，须与构造时 in_dim 一致。
        返回: (B, K, 1) 或 (B, 1)
        """
        if x.dim() == 2:
            if x.shape[-1] != self.in_dim:
                raise ValueError(f"expected in_dim={self.in_dim}, got {x.shape[-1]}")
            return self.mlp(x).unsqueeze(-1)
        b, k, c = x.shape
        if c != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {c}")
        y = self.mlp(x.reshape(b * k, c)).view(b, k, 1)
        return y
