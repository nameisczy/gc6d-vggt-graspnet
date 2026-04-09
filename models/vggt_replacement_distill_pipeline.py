# -*- coding: utf-8 -*-
"""
VGGT feature distillation：teacher = GraspNet backbone seed feature；
student = VGGT -> projector -> LayerNorm -> adapter；student 送入 vpmodule。
训练时在 end_points 中附带 distill_loss（L2 或 cosine）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment_projectors import make_conv1d_projector
from .graspnet_adapter import load_graspnet_pretrained
from .lift3d_local_fusion import nearest_neighbor_gather_features
from .vggt_encoder import VGGTEncoder
from utils.point_norm import normalize_xyz_with_pc

from .vggt_replacement_pipeline import _vggt_local_features_b768k


def _make_distill_adapter() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(256, 256, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(256, 256, kernel_size=1),
    )


class VGGReplacementDistillGraspNet(nn.Module):
    def __init__(
        self,
        encoder: VGGTEncoder,
        grasp_net: nn.Module,
        *,
        vggt_dim: int = 768,
        distill_loss_type: str = "l2",
    ):
        super().__init__()
        self.encoder = encoder
        self.grasp_net = grasp_net
        self.vggt_dim = vggt_dim
        self.distill_loss_type = str(distill_loss_type)
        self.distill_projector = make_conv1d_projector(vggt_dim, 256)
        self.distill_ln = nn.LayerNorm(256, elementwise_affine=False)
        self.distill_adapter = _make_distill_adapter()
        self.encoder_type = "vggt_replacement_distill"
        self.model_mode = "vggt_replacement_distill"
        self.requires_images = True

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is None:
            raise ValueError("VGGReplacementDistillGraspNet 需要 images")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        f_teacher, seed_xyz, end_points = backbone(point_cloud, end_points)
        f_teacher_det = f_teacher.detach()

        pts, feat_b768k = _vggt_local_features_b768k(self.encoder, images)
        seed_n = normalize_xyz_with_pc(point_cloud, seed_xyz)
        pts_n = normalize_xyz_with_pc(point_cloud, pts)
        vggt_raw = nearest_neighbor_gather_features(seed_n, pts_n, feat_b768k).float()
        x = self.distill_projector(vggt_raw)
        x_ln = self.distill_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        f_student = self.distill_adapter(x_ln)

        if self.distill_loss_type == "l2":
            l_dist = F.mse_loss(f_student, f_teacher_det)
        elif self.distill_loss_type == "cosine":
            s = f_student.reshape(-1, 256)
            t = f_teacher_det.reshape(-1, 256)
            cos = F.cosine_similarity(s, t, dim=1, eps=1e-8).mean()
            l_dist = 1.0 - cos
        else:
            raise ValueError("distill_loss_type 应为 l2 或 cosine")

        end_points = view_estimator.vpmodule(seed_xyz, f_student, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        end_points["distill_loss"] = l_dist
        return end_points


def build_vggt_replacement_distill_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    vggt_ckpt: Optional[str] = None,
    feat_dim: int = 256,
    sample_k: int = 1024,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    distill_loss_type: str = "l2",
    device: Optional[torch.device] = None,
) -> VGGReplacementDistillGraspNet:
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
    model = VGGReplacementDistillGraspNet(
        encoder=encoder,
        grasp_net=grasp_net,
        distill_loss_type=distill_loss_type,
    )
    return model.to(device)
