# -*- coding: utf-8 -*-
"""
针对 GC6D 的 6-DOF grasp 预测头。
输出 10D: [translation(3), rotation_matrix 前两列(6), width(1)]，与 offline 数据集 action 一致。
output_17d=True 时输出 GC6D 17D: [score, w, height, depth, R(9), t(3), object_id]。
"""

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


class GC6DGraspHead(nn.Module):
    """
    输入: feature (B, feat_dim)
    输出: action (B, 10) 或 (B, 17)。
    output_17d=False（默认，兼容旧 ckpt）: (B, 10)。output_17d=True: (B, 17) GC6D 格式。
    width 用 sigmoid 映射到 [width_min, width_max]，默认 [0.01, 0.12]。
    dropout_p: 在 fc 前对 feat 做 dropout，小样本时设 0.2~0.3 抑制过拟合。
    """

    def __init__(
        self,
        feat_dim: int,
        width_min: float = 0.01,
        width_max: float = 0.12,
        dropout_p: float = 0.0,
        output_17d: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.width_min = width_min
        self.width_max = width_max
        self.dropout_p = dropout_p
        self.output_17d = output_17d
        self._dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        # output_17d: 11 = t(3)+r6(6)+w(1)+score(1)；否则 10 维向后兼容
        self.fc = nn.Linear(feat_dim, 11 if output_17d else 10)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, feat_dim)
        return: (B, 10) 或 (B, 17)
        """
        feat = self._dropout(feat)
        out = self.fc(feat)
        t = out[:, :3]
        r6 = out[:, 3:9]
        w_raw = out[:, 9:10]
        w = self.width_min + (self.width_max - self.width_min) * torch.sigmoid(w_raw)
        if not self.output_17d:
            return torch.cat([t, r6, w], dim=1)
        score = out[:, 10:11]
        R = _r6_to_R_gram_schmidt(r6)
        R_flat = R.reshape(out.shape[0], 9)
        height = 0.02
        depth = 0.04
        object_id = 0.0
        return torch.cat(
            [score, w, out.new_full((out.shape[0], 1), height), out.new_full((out.shape[0], 1), depth), R_flat, t, out.new_full((out.shape[0], 1), object_id)],
            dim=1,
        )

    def forward_proposals(self, feat: torch.Tensor) -> torch.Tensor:
        """return (B, 1, 10) 或 (B, 1, 17)。"""
        out = self.forward(feat)
        return out.unsqueeze(1)
