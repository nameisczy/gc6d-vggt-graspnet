# -*- coding: utf-8 -*-
"""
LIFT3D action head 风格：MLP 特征提取 + LayerNorm + 分离的 t/rotation/width 头，
与 LIFT3D 官方 action head 结构一致，输出 10D 或 17D。保留用于与 GraspNet 等 proposal head 对比。
"""

from typing import List

import torch
import torch.nn as nn

EPS = 1e-8


def _r6_to_R_gram_schmidt(r6: torch.Tensor) -> torch.Tensor:
    """r6 (..., 6) -> R (..., 3, 3) 可微 Gram-Schmidt。"""
    c1 = r6[..., 0:3]
    c2 = r6[..., 3:6]
    c1 = c1 / (c1.norm(dim=-1, keepdim=True) + EPS)
    c2 = c2 - (c1 * (c1 * c2).sum(dim=-1, keepdim=True))
    c2 = c2 / (c2.norm(dim=-1, keepdim=True) + EPS)
    c3 = torch.linalg.cross(c1, c2, dim=-1)
    c3 = c3 / (c3.norm(dim=-1, keepdim=True) + EPS)
    return torch.stack([c1, c2, c3], dim=-1)


class MatureGraspHead(nn.Module):
    """
    LIFT3D action head 风格：共享 MLP + 分离的 translation / rotation / width 头。
    输出 10D（output_17d=False）或 17D（output_17d=True）。
    （mature / lift3d_action 均使用此类，便于与 graspnet proposal head 对比。）
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dims: List[int] = (256, 256, 128),
        width_min: float = 0.01,
        width_max: float = 0.12,
        use_layer_norm: bool = True,
        dropout_p: float = 0.0,
        output_17d: bool = False,
        height: float = 0.02,
        depth: float = 0.04,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.width_min = width_min
        self.width_max = width_max
        self.dropout_p = dropout_p
        self.output_17d = output_17d
        self.height = height
        self.depth = depth
        self._dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        layers = []
        prev = feat_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.feature_extractor = nn.Sequential(*layers)

        self.translation_head = nn.Linear(prev, 3)
        self.rotation_head = nn.Linear(prev, 6)
        self.width_head = nn.Linear(prev, 1)
        if output_17d:
            self.score_head = nn.Linear(prev, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            mid = (self.width_min + self.width_max) / 2.0
            self.width_head.bias.fill_(
                float(torch.logit(torch.tensor((mid - self.width_min) / (self.width_max - self.width_min + 1e-8))))
            )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, feat_dim) -> (B, 10) 或 (B, 17)
        """
        feat = self._dropout(feat)
        features = self.feature_extractor(feat)
        t = self.translation_head(features)
        r6 = self.rotation_head(features)
        w_raw = self.width_head(features)
        w = self.width_min + (self.width_max - self.width_min) * torch.sigmoid(w_raw)
        if not self.output_17d:
            return torch.cat([t, r6, w], dim=1)
        score = self.score_head(features)
        R = _r6_to_R_gram_schmidt(r6)
        R_flat = R.reshape(feat.shape[0], 9)
        B = feat.shape[0]
        out = torch.cat(
            [
                score,
                w,
                t.new_full((B, 1), self.height),
                t.new_full((B, 1), self.depth),
                R_flat,
                t,
                t.new_full((B, 1), 0.0),
            ],
            dim=1,
        )
        return out

    def forward_proposals(self, feat: torch.Tensor) -> torch.Tensor:
        """return (B, 1, 10) 或 (B, 1, 17)。"""
        out = self.forward(feat)
        return out.unsqueeze(1)
