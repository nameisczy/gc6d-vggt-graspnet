# -*- coding: utf-8 -*-
"""
LIFT3D-CLIP distillation：teacher = GraspNet backbone seed；student = Lift3D→projector→LN→adapter。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from .vggt_replacement_distill_pipeline import _distill_feature_debug_print


def _make_distill_adapter() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(256, 256, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(256, 256, kernel_size=1),
    )


class Lift3DClipReplacementDistillGraspNet(nn.Module):
    def __init__(self, clip_backbone: nn.Module, grasp_net: nn.Module, *, feat_dim: int = 768, distill_loss_type: str = "l2"):
        super().__init__()
        self.encoder = clip_backbone
        self.grasp_net = grasp_net
        self.feat_dim = feat_dim
        self.distill_loss_type = str(distill_loss_type)
        self.distill_projector = make_conv1d_projector(feat_dim, 256)
        self.distill_ln = nn.LayerNorm(256, elementwise_affine=False)
        self.distill_adapter = _make_distill_adapter()
        self.encoder_type = "lift3d_clip_replacement_distill"
        self.model_mode = "lift3d_clip_replacement_distill"
        self.requires_images = False

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is not None:
            raise ValueError("Lift3DClipReplacementDistillGraspNet 仅点云")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        f_teacher, seed_xyz, end_points = backbone(point_cloud, end_points)
        f_teacher_det = f_teacher.detach()

        pc_n = _normalize_pc_lift3d(point_cloud)
        patch_tok, centers = lift3d_clip_forward_patch_tokens(self.encoder, pc_n)
        feat_b = patch_tok.transpose(1, 2).contiguous()
        seed_n = _normalize_seed_lift3d(point_cloud, seed_xyz)
        ctr_n = _normalize_centers_lift3d(point_cloud, centers)
        lift_raw = nearest_neighbor_gather_features(seed_n, ctr_n, feat_b.float())

        x = self.distill_projector(lift_raw)
        x_ln = self.distill_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        f_student = self.distill_adapter(x_ln)
        _distill_feature_debug_print("lift3d_clip_replacement_distill", f_student)

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


def build_lift3d_clip_replacement_distill_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    lift3d_root: Optional[str] = None,
    train_encoder: bool = False,
    distill_loss_type: str = "l2",
    device: Optional[torch.device] = None,
) -> Lift3DClipReplacementDistillGraspNet:
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
    model = Lift3DClipReplacementDistillGraspNet(
        clip_backbone=clip_backbone,
        grasp_net=grasp_net,
        feat_dim=fd,
        distill_loss_type=distill_loss_type,
    )
    return model.to(device)
