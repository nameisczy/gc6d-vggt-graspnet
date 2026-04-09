# -*- coding: utf-8 -*-
"""
GraspNet 风格 proposal head：输出多组 (R, t, width, score) 抓取候选，
再聚合为单条 10D 与现有 pipeline 兼容。风格参考 Contact-GraspNet / 6DoF-GraspNet。
"""

from typing import List

import torch
import torch.nn as nn


class GraspNetProposalHead(nn.Module):
    """
    输出 K 个 grasp proposals，每个 (t, R6d, width, score) = 11 维；
    通过 score 的 softmax 加权聚合为 (B, 10)，与 GC6DGraspHead 输出一致。
    """

    def __init__(
        self,
        feat_dim: int,
        num_proposals: int = 4,
        hidden_dims: List[int] = (256, 256),
        width_min: float = 0.01,
        width_max: float = 0.12,
        use_layer_norm: bool = True,
        dropout_p: float = 0.0,
        score_temperature: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_proposals = num_proposals
        self.width_min = width_min
        self.width_max = width_max
        self.score_temperature = score_temperature
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

        # 每个 proposal: 3 + 6 + 1 + 1 = 11 (t, R6d, width_raw, score_logit)
        self.proposal_head = nn.Linear(prev, num_proposals * 11)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, feat_dim)
        return: (B, 10) 与 GC6DGraspHead 一致，由 K 个 proposal 按 score softmax 加权得到。
        """
        feat = self._dropout(feat)
        features = self.feature_extractor(feat)  # (B, H)
        out = self.proposal_head(features)  # (B, K*11)
        B = out.shape[0]
        K = self.num_proposals
        out = out.view(B, K, 11)

        t = out[:, :, :3]  # (B, K, 3)
        r6 = out[:, :, 3:9]  # (B, K, 6)
        w_raw = out[:, :, 9:10]  # (B, K, 1)
        score_logit = out[:, :, 10:11]  # (B, K, 1)

        w = self.width_min + (self.width_max - self.width_min) * torch.sigmoid(w_raw)
        weights = torch.softmax(score_logit / self.score_temperature, dim=1)  # (B, K, 1)

        t_out = (weights * t).sum(dim=1)  # (B, 3)
        r6_out = (weights * r6).sum(dim=1)  # (B, 6)
        w_out = (weights * w).sum(dim=1)  # (B, 1)
        return torch.cat([t_out, r6_out, w_out], dim=1)  # (B, 10)

    def forward_proposals(self, feat: torch.Tensor) -> torch.Tensor:
        """返回 (B, K, 11)：t(3), R6d(6), width(1), score_logit(1)，便于分析或 NMS。"""
        feat = self._dropout(feat)
        features = self.feature_extractor(feat)
        out = self.proposal_head(features)
        B = out.shape[0]
        out = out.view(B, self.num_proposals, 11)
        w = self.width_min + (self.width_max - self.width_min) * torch.sigmoid(out[:, :, 9:10])
        out = torch.cat([out[:, :, :9], w, out[:, :, 10:11]], dim=-1)
        return out  # (B, K, 11), last dim: t(3), r6(6), w(1), score(1)
