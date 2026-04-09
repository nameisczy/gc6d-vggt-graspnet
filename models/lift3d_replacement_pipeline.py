# -*- coding: utf-8 -*-
"""
LIFT3D replacement：仅用 GraspNet backbone 的 seed_xyz，不用其 seed_features。
LIFT3D 局部特征 → 最近邻到 seed → Conv1d 投影到 256 → vpmodule + grasp_generator。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .alignment_projectors import make_conv1d_projector
from .graspnet_adapter import load_graspnet_pretrained
from .lift3d_encoder import LIFT3DEncoder
from .lift3d_local_fusion import nearest_neighbor_gather_features


class Lift3DReplacementGraspNet(nn.Module):
    """encoder 冻结；GraspNet backbone 冻结；仅训 replacement_projector + vpmodule + grasp_generator。"""

    def __init__(
        self,
        encoder: LIFT3DEncoder,
        grasp_net: nn.Module,
        lift3d_backbone_channels: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.grasp_net = grasp_net
        self.lift3d_backbone_channels = lift3d_backbone_channels
        self.replacement_projector = make_conv1d_projector(lift3d_backbone_channels, 256)
        self.use_adapter = False
        self.encoder_type = "lift3d_replacement"
        self.requires_images = False
        self.model_mode = "lift3d_replacement"

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is not None:
            raise ValueError("Lift3DReplacementGraspNet 仅支持点云输入")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        _seed_unused, seed_xyz, end_points = backbone(point_cloud, end_points)

        seed_xyz_norm = self.encoder.normalize_seed_xyz(point_cloud, seed_xyz)
        p_list, f_list = self.encoder.forward_seg_feat(point_cloud)
        p_last = p_list[-1]
        f_last = f_list[-1]
        if f_last.dim() == 4:
            f_last = f_last.squeeze(-1)
        lift3d_raw = nearest_neighbor_gather_features(seed_xyz_norm, p_last, f_last).float()
        seed_features = self.replacement_projector(lift3d_raw)

        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        return end_points


def build_lift3d_replacement_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    lift3d_root: Optional[str] = None,
    lift3d_ckpt: Optional[str] = None,
    encoder_feat_dim: int = 256,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Lift3DReplacementGraspNet:
    import torch as _t

    if device is None:
        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    encoder = LIFT3DEncoder(
        lift3d_root=lift3d_root,
        feat_dim=encoder_feat_dim,
        use_lora=True,
        lora_r=lora_r,
        lora_scale=lora_scale,
        lora_last_n_blocks=lora_last_n_blocks,
        ckpt_path=lift3d_ckpt,
    )
    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    enc = encoder.backbone.model
    lift3d_c = int(getattr(enc, "out_channels", 512))
    model = Lift3DReplacementGraspNet(
        encoder=encoder,
        grasp_net=grasp_net,
        lift3d_backbone_channels=lift3d_c,
    )
    return model.to(device)
