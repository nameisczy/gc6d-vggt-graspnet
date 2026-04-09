# -*- coding: utf-8 -*-
"""
DINOv2 replacement：BEV + DINOv2 patch 特征按 seed 采样 → Conv1d 投影 → GraspNet head。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .alignment_projectors import make_conv1d_projector
from .dinov2_seed_encoder import DINOv2SeedEncoder
from .graspnet_adapter import load_graspnet_pretrained


class Dinov2ReplacementGraspNet(nn.Module):
    def __init__(self, dinov2_encoder: DINOv2SeedEncoder, grasp_net: nn.Module):
        super().__init__()
        self.encoder = dinov2_encoder
        self.grasp_net = grasp_net
        self.replacement_projector = make_conv1d_projector(768, 256)
        self.use_adapter = False
        self.encoder_type = "dinov2_replacement"
        self.requires_images = False
        self.model_mode = "lift3d_replacement_dinov2"

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is not None:
            raise ValueError("Dinov2ReplacementGraspNet 仅点云")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        _sf, seed_xyz, end_points = backbone(point_cloud, end_points)

        enc_b768s = self.encoder(point_cloud, seed_xyz)
        seed_features = self.replacement_projector(enc_b768s.float())
        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        return end_points


def build_dinov2_replacement_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dinov2ReplacementGraspNet:
    import torch as _t

    if device is None:
        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    enc = DINOv2SeedEncoder(device=device)
    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    model = Dinov2ReplacementGraspNet(enc, grasp_net)
    return model.to(device)
