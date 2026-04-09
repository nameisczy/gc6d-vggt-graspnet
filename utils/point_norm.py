# -*- coding: utf-8 -*-
"""点云与 query 点坐标归一化（与 LIFT3DEncoder.normalize_seed_xyz 一致）。"""

from __future__ import annotations

import torch


def normalize_xyz_with_pc(point_cloud: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """
    point_cloud: (B, N, 3)
    xyz: (B, *, 3) 任意点数
    return: 与 point_cloud 同一中心化 + 尺度下的坐标
    """
    center = point_cloud.mean(dim=1, keepdim=True)
    pc0 = point_cloud - center
    scale = pc0.abs().reshape(point_cloud.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    return (xyz - center) / scale
