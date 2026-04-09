# -*- coding: utf-8 -*-
"""
LIFT3D-CLIP（lift3d_clip_base）replacement：预训练 Lift3dCLIP 的 patch 特征 → 对齐 seed → 投影 → GraspNet head。
不使用 GraspNet backbone 的 seed_features。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

from .alignment_projectors import make_conv1d_projector
from .graspnet_adapter import load_graspnet_pretrained
from .lift3d_clip_patch_features import lift3d_clip_forward_patch_tokens
from .lift3d_local_fusion import nearest_neighbor_gather_features


def _ensure_lift3d_path(lift3d_root: Optional[str]) -> str:
    root = lift3d_root or os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"LIFT3D root not found: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _load_lift3d_clip_backbone_pretrained(lift3d_root: str):
    from lift3d.models.lift3d.model_loader import lift3d_clip_base

    return lift3d_clip_base()


def _normalize_pc_lift3d(pc: torch.Tensor) -> torch.Tensor:
    """与 ``LIFT3DEncoder._normalize`` 一致：每 batch 一个全局 scale，避免 (B,N,1) 与 seed 维冲突。"""
    center = pc.mean(dim=1, keepdim=True)
    pc0 = pc - center
    scale = pc0.abs().reshape(pc.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    return pc0 / scale


def _normalize_seed_lift3d(pc: torch.Tensor, seed_xyz: torch.Tensor) -> torch.Tensor:
    """与 ``LIFT3DEncoder.normalize_seed_xyz`` 一致。"""
    center = pc.mean(dim=1, keepdim=True)
    pc0 = pc - center
    scale = pc0.abs().reshape(pc.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    return (seed_xyz - center) / scale


def _normalize_centers_lift3d(pc: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return _normalize_seed_lift3d(pc, centers)


class Lift3DClipReplacementGraspNet(nn.Module):
    """encoder: Lift3dCLIP；冻结；仅训 replacement_projector + vpmodule + grasp_generator。"""

    def __init__(self, clip_backbone: nn.Module, grasp_net: nn.Module, feat_dim: int = 768):
        super().__init__()
        self.encoder = clip_backbone
        self.grasp_net = grasp_net
        self.feat_dim = feat_dim
        self.replacement_projector = make_conv1d_projector(feat_dim, 256)
        self.use_adapter = False
        self.encoder_type = "lift3d_clip_replacement"
        self.requires_images = False
        self.model_mode = "lift3d_replacement_clip"

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is not None:
            raise ValueError("Lift3DClipReplacementGraspNet 仅点云")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        _sf, seed_xyz, end_points = backbone(point_cloud, end_points)

        pc_n = _normalize_pc_lift3d(point_cloud)
        patch_tok, centers = lift3d_clip_forward_patch_tokens(self.encoder, pc_n)
        feat_b = patch_tok.transpose(1, 2).contiguous()

        seed_n = _normalize_seed_lift3d(point_cloud, seed_xyz)
        ctr_n = _normalize_centers_lift3d(point_cloud, centers)
        lift_raw = nearest_neighbor_gather_features(seed_n, ctr_n, feat_b.float())

        seed_features = self.replacement_projector(lift_raw)
        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        return end_points


def build_lift3d_clip_replacement_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    lift3d_root: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Lift3DClipReplacementGraspNet:
    import torch as _t

    if device is None:
        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    _ensure_lift3d_path(lift3d_root)
    clip_backbone = _load_lift3d_clip_backbone_pretrained(
        lift3d_root or os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
    )
    clip_backbone = clip_backbone.to(device)
    for p in clip_backbone.parameters():
        p.requires_grad = False
    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    fd = int(getattr(clip_backbone, "feature_dim", 768))
    model = Lift3DClipReplacementGraspNet(clip_backbone=clip_backbone, grasp_net=grasp_net, feat_dim=fd)
    return model.to(device)
