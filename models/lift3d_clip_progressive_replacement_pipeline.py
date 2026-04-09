# -*- coding: utf-8 -*-
"""
LIFT3D-CLIP 渐进式替代：f_in = (1-alpha)*f_graspnet + alpha*f_lift_adapted
（与 vggt_progressive_replacement 同构，输入为点云 + Lift3dCLIP patch 特征）。
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
from .lift3d_clip_replacement_pipeline import (
    _ensure_lift3d_path,
    _load_lift3d_clip_backbone_pretrained,
    _normalize_centers_lift3d,
    _normalize_pc_lift3d,
    _normalize_seed_lift3d,
)
from .lift3d_local_fusion import nearest_neighbor_gather_features


def _make_progressive_adapter() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(256, 256, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(256, 256, kernel_size=1),
    )


class Lift3DClipProgressiveReplacementGraspNet(nn.Module):
    def __init__(
        self,
        clip_backbone: nn.Module,
        grasp_net: nn.Module,
        *,
        feat_dim: int = 768,
        progressive_alpha: float = 0.5,
    ):
        super().__init__()
        self.encoder = clip_backbone
        self.grasp_net = grasp_net
        self.feat_dim = feat_dim
        self.progressive_alpha = float(progressive_alpha)
        self.replacement_projector = make_conv1d_projector(feat_dim, 256)
        self.progressive_ln = nn.LayerNorm(256, elementwise_affine=False)
        self.progressive_adapter = _make_progressive_adapter()
        self.encoder_type = "lift3d_clip_progressive_replacement"
        self.model_mode = "lift3d_clip_progressive_replacement"
        self.requires_images = False

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is not None:
            raise ValueError("Lift3DClipProgressiveReplacementGraspNet 仅点云")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        f_graspnet, seed_xyz, end_points = backbone(point_cloud, end_points)

        pc_n = _normalize_pc_lift3d(point_cloud)
        patch_tok, centers = lift3d_clip_forward_patch_tokens(self.encoder, pc_n)
        feat_b = patch_tok.transpose(1, 2).contiguous()
        seed_n = _normalize_seed_lift3d(point_cloud, seed_xyz)
        ctr_n = _normalize_centers_lift3d(point_cloud, centers)
        lift_raw = nearest_neighbor_gather_features(seed_n, ctr_n, feat_b.float())

        x = self.replacement_projector(lift_raw)
        x_ln = self.progressive_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        f_lift = self.progressive_adapter(x_ln)

        a = self.progressive_alpha
        seed_features = (1.0 - a) * f_graspnet + a * f_lift

        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        return end_points


def build_lift3d_clip_progressive_replacement_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    lift3d_root: Optional[str] = None,
    train_encoder: bool = False,
    progressive_alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Lift3DClipProgressiveReplacementGraspNet:
    import torch as _t

    if device is None:
        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    _ensure_lift3d_path(lift3d_root)
    lr = lift3d_root or os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
    clip_backbone = _load_lift3d_clip_backbone_pretrained(lr)
    clip_backbone = clip_backbone.to(device)
    for p in clip_backbone.parameters():
        p.requires_grad = train_encoder
    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    fd = int(getattr(clip_backbone, "feature_dim", 768))
    model = Lift3DClipProgressiveReplacementGraspNet(
        clip_backbone=clip_backbone,
        grasp_net=grasp_net,
        feat_dim=fd,
        progressive_alpha=progressive_alpha,
    )
    return model.to(device)
