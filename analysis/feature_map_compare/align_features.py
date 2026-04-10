# -*- coding: utf-8 -*-
"""将各类参考点上的特征通过 3D 最近邻对齐到 pc_common（与训练时 normalize 一致）。"""

from __future__ import annotations

from typing import Optional

import torch

from models.lift3d_local_fusion import nearest_neighbor_gather_features
from utils.point_norm import normalize_xyz_with_pc


def nn_assign_features_world(
    point_cloud: torch.Tensor,
    pc_common: torch.Tensor,
    ref_xyz: torch.Tensor,
    ref_feat: torch.Tensor,
) -> torch.Tensor:
    """
    point_cloud: (1, N_pc, 3) 场景点云（与模型输入一致）
    pc_common: (N, 3) 目标点（世界坐标）
    ref_xyz: (1, N_ref, 3)
    ref_feat: (1, C, N_ref)
    返回 features (N, C) float32
    """
    if point_cloud.dim() != 3 or pc_common.dim() != 2:
        raise ValueError(f"unexpected shapes pc={point_cloud.shape} pc_common={pc_common.shape}")
    q = pc_common.unsqueeze(0)
    qn = normalize_xyz_with_pc(point_cloud, q)
    rn = normalize_xyz_with_pc(point_cloud, ref_xyz)
    if ref_feat.dim() != 3:
        raise ValueError(ref_feat.shape)
    out = nearest_neighbor_gather_features(qn, rn, ref_feat)
    return out[0].permute(1, 0).contiguous()


def denormalize_lift3d_centers(pc: torch.Tensor, centers_norm: torch.Tensor) -> torch.Tensor:
    """与 lift3d_clip_replacement 中 patch center 反变换到世界坐标。"""
    center = pc.mean(dim=1, keepdim=True)
    pc0 = pc - center
    scale = pc0.abs().reshape(pc.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    return centers_norm * scale + center
