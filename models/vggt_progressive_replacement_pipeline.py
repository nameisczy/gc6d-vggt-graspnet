# -*- coding: utf-8 -*-
"""
VGGT 渐进式替代：f_in = (1-alpha)*f_graspnet + alpha*f_vggt_adapted，
其中 f_vggt_adapted = LayerNorm -> adapter(VGGT projector 输出)。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .alignment_projectors import make_conv1d_projector
from .graspnet_adapter import load_graspnet_pretrained
from .lift3d_local_fusion import nearest_neighbor_gather_features
from .vggt_encoder import VGGTEncoder
from utils.point_norm import normalize_xyz_with_pc

from .vggt_replacement_pipeline import _vggt_local_features_b768k


def _make_progressive_adapter() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(256, 256, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(256, 256, kernel_size=1),
    )


class VGGProgressiveReplacementGraspNet(nn.Module):
    def __init__(
        self,
        encoder: VGGTEncoder,
        grasp_net: nn.Module,
        *,
        progressive_alpha: float = 0.5,
        vggt_dim: int = 768,
        score_calibration_mode: str = "none",
        score_delta_scale: float = 0.1,
        score_calibration_hidden: int = 12,
    ):
        super().__init__()
        self.encoder = encoder
        self.grasp_net = grasp_net
        self.vggt_dim = vggt_dim
        self.progressive_alpha = float(progressive_alpha)
        self.replacement_projector = make_conv1d_projector(vggt_dim, 256)
        self.progressive_ln = nn.LayerNorm(256, elementwise_affine=False)
        self.progressive_adapter = _make_progressive_adapter()
        self.encoder_type = "vggt_progressive_replacement"
        self.model_mode = "vggt_progressive_replacement"
        self.requires_images = True
        self.score_calibration_mode = str(score_calibration_mode)
        self.score_delta_scale = float(score_delta_scale)
        self.score_calibration_hidden = int(score_calibration_hidden)
        if self.score_calibration_mode == "residual":
            h = int(score_calibration_hidden)
            self.score_calibration_head = nn.Sequential(
                nn.Conv1d(12, h, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(h, 12, kernel_size=1),
            )
        else:
            self.score_calibration_head = None

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is None:
            raise ValueError("VGGProgressiveReplacementGraspNet 需要 images")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        f_graspnet, seed_xyz, end_points = backbone(point_cloud, end_points)

        pts, feat_b768k = _vggt_local_features_b768k(self.encoder, images)
        seed_n = normalize_xyz_with_pc(point_cloud, seed_xyz)
        pts_n = normalize_xyz_with_pc(point_cloud, pts)
        vggt_raw = nearest_neighbor_gather_features(seed_n, pts_n, feat_b768k).float()
        x = self.replacement_projector(vggt_raw)
        x_ln = self.progressive_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        f_vggt_adapted = self.progressive_adapter(x_ln)

        a = self.progressive_alpha
        seed_features = (1.0 - a) * f_graspnet + a * f_vggt_adapted

        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)

        if self.score_calibration_head is not None:
            score = end_points["grasp_score_pred"]
            B, A, Ns, D = score.shape
            x2 = score.reshape(B, A, Ns * D)
            delta = self.score_calibration_head(x2).reshape(B, A, Ns, D)
            end_points["grasp_score_pred_raw_pretrained"] = score
            end_points["grasp_score_delta"] = torch.tanh(delta) * self.score_delta_scale
            end_points["grasp_score_pred"] = score + end_points["grasp_score_delta"]
        return end_points


def build_vggt_progressive_replacement_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    vggt_ckpt: Optional[str] = None,
    feat_dim: int = 256,
    sample_k: int = 1024,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    progressive_alpha: float = 0.5,
    score_calibration_mode: str = "none",
    score_delta_scale: float = 0.1,
    score_calibration_hidden: int = 12,
    device: Optional[torch.device] = None,
) -> VGGProgressiveReplacementGraspNet:
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
    model = VGGProgressiveReplacementGraspNet(
        encoder=encoder,
        grasp_net=grasp_net,
        progressive_alpha=progressive_alpha,
        vggt_dim=768,
        score_calibration_mode=score_calibration_mode,
        score_delta_scale=score_delta_scale,
        score_calibration_hidden=score_calibration_hidden,
    )
    return model.to(device)
