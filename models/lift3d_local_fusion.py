# -*- coding: utf-8 -*-
"""
LIFT3D 局部点特征与 GraspNet seed_features 融合（仅 LIFT3D 线，不涉及 VGGT）。

- PointNext ``forward_seg_feat`` 得到多尺度点坐标与特征；取最后一层 (语义最丰富、点数最少)，
  对 GraspNet 的 ``seed_xyz`` 在**与 LIFT3D 输入相同归一化空间**下做最近邻，gather 特征。
- ``lift3d_seed_proj``: Conv1d(C,256,1) 对齐到 256 维后再与 ``seed_features`` 融合。

fusion_mode:
- ``concat_proj``: cat 后 Conv1d(512,256,1)
- ``residual_proj``: Conv1d(256,256,1) 残差，``seed + alpha * proj(lift3d_seed)``
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .graspnet_adapter import load_graspnet_pretrained
from .lift3d_encoder import LIFT3DEncoder


def nearest_neighbor_gather_features(
    query_xyz: torch.Tensor,
    ref_xyz: torch.Tensor,
    ref_feat: torch.Tensor,
) -> torch.Tensor:
    """
    query_xyz: (B, S, 3)
    ref_xyz: (B, Nc, 3)
    ref_feat: (B, C, Nc)
    return: (B, C, S)
    """
    if ref_feat.dim() != 3:
        raise ValueError(f"ref_feat 期望 3 维 (B,C,Nc)，实际 {tuple(ref_feat.shape)}")
    if ref_xyz.dim() != 3 or ref_xyz.shape[-1] != 3:
        raise ValueError(f"ref_xyz 期望 (B,Nc,3)，实际 {tuple(ref_xyz.shape)}")
    if query_xyz.dim() != 3 or query_xyz.shape[-1] != 3:
        raise ValueError(f"query_xyz 期望 (B,S,3)，实际 {tuple(query_xyz.shape)}")

    # 某些实现下特征可能是 (B, Nc, C)，这里自动转为 (B, C, Nc)
    if ref_feat.shape[-1] != ref_xyz.shape[1] and ref_feat.shape[1] == ref_xyz.shape[1]:
        ref_feat = ref_feat.transpose(1, 2).contiguous()

    # 若 ref_xyz / ref_feat 点数不一致，裁到共同最小长度，避免 gather 越界
    nc_xyz = int(ref_xyz.shape[1])
    nc_feat = int(ref_feat.shape[-1])
    nc = min(nc_xyz, nc_feat)
    if nc <= 0:
        raise RuntimeError(f"无有效参考点：nc_xyz={nc_xyz}, nc_feat={nc_feat}")
    if nc_xyz != nc:
        ref_xyz = ref_xyz[:, :nc, :].contiguous()
    if nc_feat != nc:
        ref_feat = ref_feat[:, :, :nc].contiguous()

    # (B, S, Nc)
    dist = torch.cdist(query_xyz.contiguous(), ref_xyz, p=2.0)
    idx = dist.argmin(dim=-1).long()  # (B, S)
    idx = idx.clamp_(min=0, max=nc - 1)

    B, C, _ = ref_feat.shape
    idx_exp = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, S)
    return torch.gather(ref_feat, 2, idx_exp)


class Lift3DLocalFusionGraspNet(nn.Module):
    """
    encoder: LIFT3DEncoder（仅用于局部特征；forward 中不用 global adapter）
    grasp_net: 预训练 GraspNet
    """

    def __init__(
        self,
        encoder: LIFT3DEncoder,
        grasp_net: nn.Module,
        fusion_mode: str,
        lift3d_backbone_channels: int,
        residual_alpha: float = 1.0,
    ):
        super().__init__()
        if fusion_mode not in ("concat_proj", "residual_proj"):
            raise ValueError(f"fusion_mode must be concat_proj or residual_proj, got {fusion_mode}")
        self.encoder = encoder
        self.grasp_net = grasp_net
        self.fusion_mode = fusion_mode
        self.residual_alpha = residual_alpha
        self.lift3d_backbone_channels = lift3d_backbone_channels

        self.lift3d_seed_proj = nn.Conv1d(lift3d_backbone_channels, 256, kernel_size=1, bias=True)
        if fusion_mode == "concat_proj":
            self.fusion_concat_proj = nn.Conv1d(512, 256, kernel_size=1, bias=True)
        else:
            self.fusion_residual_proj = nn.Conv1d(256, 256, kernel_size=1, bias=True)

        # 与 EncoderAdapterGraspNet / eval 兼容字段
        self.use_adapter = False
        self.adapter = None
        self.adapter_cond_coeff = 1.0
        self.adapter_cond_mode = fusion_mode
        self.encoder_type = "lift3d"
        self.cond_gate = None
        self.film_proj = None
        self.concat_proj = None  # 旧 additive/concat global 用名；本模型用 fusion_concat_proj

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is not None:
            raise ValueError("Lift3DLocalFusionGraspNet 仅支持点云输入")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        seed_features, seed_xyz, end_points = backbone(point_cloud, end_points)

        seed_xyz_norm = self.encoder.normalize_seed_xyz(point_cloud, seed_xyz)

        p_list, f_list = self.encoder.forward_seg_feat(point_cloud)
        p_last = p_list[-1]
        f_last = f_list[-1]
        if f_last.dim() == 4:
            f_last = f_last.squeeze(-1)
        # f_last: (B, C, Nc)
        C = f_last.shape[1]
        if C != self.lift3d_seed_proj.in_channels:
            raise RuntimeError(
                f"lift3d backbone C={C} != proj.in_ch={self.lift3d_seed_proj.in_channels}，请检查 checkpoint 与配置"
            )

        lift3d_raw = nearest_neighbor_gather_features(seed_xyz_norm, p_last, f_last)
        lift3d_raw = lift3d_raw.float()
        lift3d_seed = self.lift3d_seed_proj(lift3d_raw)

        if self.fusion_mode == "concat_proj":
            fused = torch.cat([seed_features, lift3d_seed], dim=1)
            seed_features = self.fusion_concat_proj(fused)
        else:
            delta = self.fusion_residual_proj(lift3d_seed)
            seed_features = seed_features + self.residual_alpha * delta

        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        end_points["_cond"] = lift3d_seed.mean(dim=-1)
        return end_points


def build_lift3d_local_fusion_graspnet(
    *,
    fusion_mode: str,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    lift3d_root: Optional[str] = None,
    lift3d_ckpt: Optional[str] = None,
    residual_alpha: float = 1.0,
    encoder_feat_dim: int = 256,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Lift3DLocalFusionGraspNet:
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
    model = Lift3DLocalFusionGraspNet(
        encoder=encoder,
        grasp_net=grasp_net,
        fusion_mode=fusion_mode,
        lift3d_backbone_channels=lift3d_c,
        residual_alpha=residual_alpha,
    )
    return model.to(device)
