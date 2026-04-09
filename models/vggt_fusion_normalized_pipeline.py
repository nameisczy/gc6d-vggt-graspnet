# -*- coding: utf-8 -*-
"""
VGGT + GraspNet backbone：seed_features 保留；VGGT 局部特征 LN → 投影 → concat → fusion MLP（可选残差）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .alignment_projectors import make_conv1d_projector, make_fusion_mlp_concat512
from .graspnet_adapter import load_graspnet_pretrained
from .lift3d_local_fusion import nearest_neighbor_gather_features
from utils.point_norm import normalize_xyz_with_pc
from .vggt_encoder import VGGTEncoder
from .vggt_replacement_pipeline import _vggt_local_features_b768k


class VGGTFusionNormalizedGraspNet(nn.Module):
    def __init__(
        self,
        encoder: VGGTEncoder,
        grasp_net: nn.Module,
        vggt_dim: int = 768,
        fuse_residual: bool = False,
        fuse_alpha: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.grasp_net = grasp_net
        self.vggt_dim = vggt_dim
        self.vggt_ln = nn.LayerNorm(vggt_dim)
        self.vggt_proj = nn.Sequential(
            nn.Conv1d(vggt_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1),
        )
        self.fuse_residual = bool(fuse_residual)
        self.fuse_alpha = float(fuse_alpha)
        self.use_adapter = False
        self.encoder_type = "vggt_fusion_normalized"
        self.requires_images = True
        self.model_mode = "vggt_fusion_normalized"

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is None:
            raise ValueError("VGGTFusionNormalizedGraspNet 需要 images")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        seed_features, seed_xyz, end_points = backbone(point_cloud, end_points)

        pts, feat_b768k = _vggt_local_features_b768k(self.encoder, images)
        seed_n = normalize_xyz_with_pc(point_cloud, seed_xyz)
        pts_n = normalize_xyz_with_pc(point_cloud, pts)
        vggt_raw = nearest_neighbor_gather_features(seed_n, pts_n, feat_b768k).float()
        x_ln = self.vggt_ln(vggt_raw.permute(0, 2, 1)).permute(0, 2, 1)
        vggt_proj = self.vggt_proj(x_ln)
        fused = torch.cat([seed_features, vggt_proj], dim=1)
        mixed = self.fusion_mlp(fused)
        if self.fuse_residual:
            seed_features_out = seed_features + self.fuse_alpha * mixed
        else:
            seed_features_out = mixed

        end_points = view_estimator.vpmodule(seed_xyz, seed_features_out, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        return end_points


def build_vggt_fusion_normalized_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    vggt_ckpt: Optional[str] = None,
    feat_dim: int = 256,
    sample_k: int = 1024,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    fuse_residual: bool = False,
    fuse_alpha: float = 0.1,
    device: Optional[torch.device] = None,
) -> VGGTFusionNormalizedGraspNet:
    import torch as _t

    if device is None:
        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    encoder = VGGTEncoder(
        feat_dim=feat_dim,
        freeze_backbone=True,
        ckpt_path=vggt_ckpt,
        sample_k=sample_k,
        lora_r=lora_r,
        lora_scale=lora_scale,
        lora_last_n_blocks=lora_last_n_blocks,
    )
    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    model = VGGTFusionNormalizedGraspNet(
        encoder=encoder,
        grasp_net=grasp_net,
        vggt_dim=768,
        fuse_residual=fuse_residual,
        fuse_alpha=fuse_alpha,
    )
    return model.to(device)
