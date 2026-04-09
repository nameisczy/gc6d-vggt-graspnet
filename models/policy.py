# -*- coding: utf-8 -*-
"""
Policy: Placeholder 或 LIFT3D Encoder + grasp head，输出 10D action。
head 类型：simple（单层）| lift3d_action（LIFT3D 官方 action head 风格）| graspnet（GraspNet 风格 proposal）。
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .placeholder_encoder import PlaceholderEncoder
from .gc6d_grasp_head import GC6DGraspHead
from .mature_grasp_head import MatureGraspHead
from .graspnet_proposal_head import GraspNetProposalHead


def _forward_proposals_from_feat(feat: torch.Tensor, grasp_head: nn.Module) -> torch.Tensor:
    """feat (B, feat_dim) -> (B, K, 10) 或 (B, 1, 17)。17D head 时返回 as-is；否则取前 10 维。"""
    if hasattr(grasp_head, "forward_proposals"):
        out = grasp_head.forward_proposals(feat)
        if out.shape[-1] == 17:
            return out
        return out[:, :, :10]
    out = grasp_head(feat)
    return out.unsqueeze(1)


def _forward_proposals_raw_from_feat(feat: torch.Tensor, grasp_head: nn.Module) -> torch.Tensor:
    """feat (B, feat_dim) -> (B, K, 11)、(B, K, 10) 或 (B, 1, 17)。17D 时返回 as-is。"""
    if hasattr(grasp_head, "forward_proposals"):
        out = grasp_head.forward_proposals(feat)
        return out
    out = grasp_head(feat)
    return out.unsqueeze(1)


def _make_grasp_head(
    head_type: str,
    feat_dim: int,
    width_min: float = 0.01,
    width_max: float = 0.12,
    head_dropout: float = 0.0,
    mature_hidden_dims: Optional[List[int]] = None,
    num_proposals: int = 4,
    graspnet_hidden_dims: Optional[List[int]] = None,
) -> nn.Module:
    """head_type: 'simple' | 'simple_17d' | 'lift3d_action' | 'mature' | 'mature_17d' | 'graspnet'。simple_17d/mature_17d 输出 (B, 17)。"""
    if head_type == "simple_17d":
        return GC6DGraspHead(
            feat_dim=feat_dim,
            width_min=width_min,
            width_max=width_max,
            dropout_p=head_dropout,
            output_17d=True,
        )
    if head_type in ("mature", "lift3d_action"):
        return MatureGraspHead(
            feat_dim=feat_dim,
            hidden_dims=mature_hidden_dims or [256, 256, 128],
            width_min=width_min,
            width_max=width_max,
            dropout_p=head_dropout,
        )
    if head_type == "mature_17d":
        return MatureGraspHead(
            feat_dim=feat_dim,
            hidden_dims=mature_hidden_dims or [256, 256, 128],
            width_min=width_min,
            width_max=width_max,
            dropout_p=head_dropout,
            output_17d=True,
            height=0.02,
            depth=0.04,
        )
    if head_type == "graspnet":
        return GraspNetProposalHead(
            feat_dim=feat_dim,
            num_proposals=num_proposals,
            hidden_dims=list(graspnet_hidden_dims or [256, 256]),
            width_min=width_min,
            width_max=width_max,
            dropout_p=head_dropout,
        )
    return GC6DGraspHead(
        feat_dim=feat_dim,
        width_min=width_min,
        width_max=width_max,
        dropout_p=head_dropout,
    )


class GC6DGraspPolicy(nn.Module):
    """占位 encoder + GC6D grasp head，单数据点可训至 loss=0。"""

    def __init__(
        self,
        encoder_feat_dim: int = 256,
        width_min: float = 0.01,
        width_max: float = 0.12,
    ):
        super().__init__()
        self.encoder_type = "placeholder"
        self.encoder = PlaceholderEncoder(point_dim=3, feat_dim=encoder_feat_dim)
        self.grasp_head = GC6DGraspHead(
            feat_dim=encoder_feat_dim,
            width_min=width_min,
            width_max=width_max,
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: (B, N, 3)
        return: (B, 10) action
        """
        feat = self.encoder(point_cloud)
        return self.grasp_head(feat)


def build_lift3d_policy(
    encoder_feat_dim: int = 256,
    width_min: float = 0.01,
    width_max: float = 0.12,
    lift3d_root: Optional[str] = None,
    point_next_config: str = "point_next.yaml",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    normalize_pc: bool = True,
    grasp_head_type: str = "simple",
    mature_head_hidden_dims: Optional[List[int]] = None,
    num_proposals: int = 4,
    graspnet_hidden_dims: Optional[List[int]] = None,
) -> "GC6DGraspPolicyLIFT3D":
    """构建 LIFT3D PointNext encoder + grasp head 的 policy。"""
    from .lift3d_encoder import LIFT3DEncoder

    encoder = LIFT3DEncoder(
        lift3d_root=lift3d_root,
        point_next_config=point_next_config,
        feat_dim=encoder_feat_dim,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_scale=lora_scale,
        normalize_pc=normalize_pc,
    )
    return GC6DGraspPolicyLIFT3D(
        encoder=encoder,
        encoder_feat_dim=encoder_feat_dim,
        width_min=width_min,
        width_max=width_max,
        grasp_head_type=grasp_head_type,
        mature_head_hidden_dims=mature_head_hidden_dims,
        num_proposals=num_proposals,
        graspnet_hidden_dims=graspnet_hidden_dims,
    )


def build_lift3d_clip_policy(
    encoder_feat_dim: int = 256,
    width_min: float = 0.01,
    width_max: float = 0.12,
    lift3d_root: Optional[str] = None,
    freeze_backbone: bool = True,
    normalize_pc: bool = True,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    head_dropout: float = 0.0,
    adapter_dropout: float = 0.0,
    grasp_head_type: str = "simple",
    mature_head_hidden_dims: Optional[List[int]] = None,
    num_proposals: int = 4,
    graspnet_hidden_dims: Optional[List[int]] = None,
) -> "GC6DGraspPolicyLIFT3D":
    """构建 LIFT3D lift3d_clip encoder + LoRA + grasp head。"""
    from .lift3d_clip_encoder import LIFT3DClipEncoder

    encoder = LIFT3DClipEncoder(
        lift3d_root=lift3d_root,
        feat_dim=encoder_feat_dim,
        freeze_backbone=freeze_backbone,
        normalize_pc=normalize_pc,
        lora_r=lora_r,
        lora_scale=lora_scale,
        adapter_dropout=adapter_dropout,
    )
    return GC6DGraspPolicyLIFT3D(
        encoder=encoder,
        encoder_feat_dim=encoder_feat_dim,
        width_min=width_min,
        width_max=width_max,
        head_dropout=head_dropout,
        grasp_head_type=grasp_head_type,
        mature_head_hidden_dims=mature_head_hidden_dims,
        num_proposals=num_proposals,
        graspnet_hidden_dims=graspnet_hidden_dims,
    )


class GC6DGraspPolicyLIFT3D(nn.Module):
    """LIFT3D PointNext（可选 LoRA）+ 适配器 + grasp head。"""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_feat_dim: int = 256,
        width_min: float = 0.01,
        width_max: float = 0.12,
        head_dropout: float = 0.0,
        grasp_head_type: str = "simple",
        mature_head_hidden_dims: Optional[List[int]] = None,
        num_proposals: int = 4,
        graspnet_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.encoder_type = "lift3d"
        self.grasp_head_type = grasp_head_type
        self.grasp_head_num_proposals = num_proposals if grasp_head_type == "graspnet" else None
        self.encoder = encoder
        self.grasp_head = _make_grasp_head(
            head_type=grasp_head_type,
            feat_dim=encoder_feat_dim,
            width_min=width_min,
            width_max=width_max,
            head_dropout=head_dropout,
            mature_hidden_dims=mature_head_hidden_dims,
            num_proposals=num_proposals,
            graspnet_hidden_dims=graspnet_hidden_dims,
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: (B, N, 3)
        return: (B, 10) action
        """
        feat = self.encoder(point_cloud)
        return self.grasp_head(feat)

    def forward_proposals(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """return (B, K, 10) 或 (B, 1, 10)。"""
        feat = self.encoder(point_cloud)
        return _forward_proposals_from_feat(feat, self.grasp_head)

    def forward_proposals_raw(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """return (B, K, 11) GraspNet 原始输出，或 (B, K, 10)。供 11D 直接转 17D 评估。"""
        feat = self.encoder(point_cloud)
        return _forward_proposals_raw_from_feat(feat, self.grasp_head)


def build_lift3d_clip_policy_multimodal(
    encoder_feat_dim: int = 256,
    width_min: float = 0.01,
    width_max: float = 0.12,
    lift3d_root: Optional[str] = None,
    freeze_backbone: bool = True,
    normalize_pc: bool = True,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    head_dropout: float = 0.0,
    adapter_dropout: float = 0.0,
    grasp_head_type: str = "simple",
    mature_head_hidden_dims: Optional[List[int]] = None,
    num_proposals: int = 4,
    graspnet_hidden_dims: Optional[List[int]] = None,
    freeze_image_encoder: bool = True,
) -> "GC6DGraspPolicyLIFT3DMultimodal":
    """LIFT3D 点云 + 图像双模态：pc encoder + image encoder(VGGT) + fusion(concat→256) + grasp head。"""
    from .lift3d_clip_encoder import LIFT3DClipEncoder
    from .vggt_encoder import VGGTEncoder

    encoder_pc = LIFT3DClipEncoder(
        lift3d_root=lift3d_root,
        feat_dim=encoder_feat_dim,
        freeze_backbone=freeze_backbone,
        normalize_pc=normalize_pc,
        lora_r=lora_r,
        lora_scale=lora_scale,
        adapter_dropout=adapter_dropout,
    )
    encoder_img = VGGTEncoder(
        feat_dim=encoder_feat_dim,
        freeze_backbone=freeze_image_encoder,
        ckpt_path=None,
        lora_r=lora_r,
        lora_scale=lora_scale,
        adapter_dropout=0.0,
    )
    fusion = nn.Sequential(
        nn.Linear(encoder_feat_dim * 2, encoder_feat_dim),
        nn.ReLU(inplace=True),
    )
    grasp_head = _make_grasp_head(
        head_type=grasp_head_type,
        feat_dim=encoder_feat_dim,
        width_min=width_min,
        width_max=width_max,
        head_dropout=head_dropout,
        mature_hidden_dims=mature_head_hidden_dims,
        num_proposals=num_proposals,
        graspnet_hidden_dims=graspnet_hidden_dims,
    )
    return GC6DGraspPolicyLIFT3DMultimodal(
        encoder=encoder_pc,
        encoder_img=encoder_img,
        fusion=fusion,
        grasp_head=grasp_head,
        encoder_feat_dim=encoder_feat_dim,
        grasp_head_type=grasp_head_type,
        num_proposals=num_proposals,
    )


class GC6DGraspPolicyLIFT3DMultimodal(nn.Module):
    """LIFT3D 点云 + 图像：encoder(pc) + encoder_img(images) + fusion + grasp head。"""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_img: nn.Module,
        fusion: nn.Module,
        grasp_head: nn.Module,
        encoder_feat_dim: int = 256,
        grasp_head_type: str = "simple",
        num_proposals: int = 4,
    ):
        super().__init__()
        self.encoder_type = "lift3d_multimodal"
        self.grasp_head_type = grasp_head_type
        self.grasp_head_num_proposals = num_proposals if grasp_head_type == "graspnet" else None
        self.encoder = encoder
        self.encoder_img = encoder_img
        self.fusion = fusion
        self.grasp_head = grasp_head
        self.encoder_feat_dim = encoder_feat_dim

    def forward(
        self,
        point_cloud: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        point_cloud: (B, N, 3), images: (B, 3, 224, 224)
        return: (B, 10) action
        """
        feat_pc = self.encoder(point_cloud)
        feat_img = self.encoder_img(images)
        feat = self.fusion(torch.cat([feat_pc, feat_img], dim=1))
        return self.grasp_head(feat)

    def forward_proposals(self, point_cloud: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """return (B, K, 10) 或 (B, 1, 10)。"""
        feat_pc = self.encoder(point_cloud)
        feat_img = self.encoder_img(images)
        feat = self.fusion(torch.cat([feat_pc, feat_img], dim=1))
        return _forward_proposals_from_feat(feat, self.grasp_head)

    def forward_proposals_raw(self, point_cloud: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """return (B, K, 11) 或 (B, K, 10)。供 11D 直接转 17D 评估。"""
        feat_pc = self.encoder(point_cloud)
        feat_img = self.encoder_img(images)
        feat = self.fusion(torch.cat([feat_pc, feat_img], dim=1))
        return _forward_proposals_raw_from_feat(feat, self.grasp_head)


def build_vggt_base_policy(
    encoder_feat_dim: int = 256,
    width_min: float = 0.01,
    width_max: float = 0.12,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    head_dropout: float = 0.0,
    adapter_dropout: float = 0.0,
    grasp_head_type: str = "simple",
    mature_head_hidden_dims: Optional[List[int]] = None,
    num_proposals: int = 4,
    graspnet_hidden_dims: Optional[List[int]] = None,
) -> "GC6DGraspPolicyVGGT":
    """VGGT base：backbone 注入 LoRA，训 LoRA + adapter + head。"""
    from .vggt_encoder import VGGTEncoder
    encoder = VGGTEncoder(
        feat_dim=encoder_feat_dim,
        freeze_backbone=True,
        ckpt_path=None,
        lora_r=lora_r,
        lora_scale=lora_scale,
        adapter_dropout=adapter_dropout,
    )
    model = GC6DGraspPolicyVGGT(
        encoder=encoder,
        encoder_feat_dim=encoder_feat_dim,
        width_min=width_min,
        width_max=width_max,
        head_dropout=head_dropout,
        grasp_head_type=grasp_head_type,
        mature_head_hidden_dims=mature_head_hidden_dims,
        num_proposals=num_proposals,
        graspnet_hidden_dims=graspnet_hidden_dims,
    )
    model.encoder_type = "vggt_base"
    return model


def build_vggt_ft_policy(
    encoder_feat_dim: int = 256,
    width_min: float = 0.01,
    width_max: float = 0.12,
    ckpt_path: Optional[str] = None,
    freeze_backbone: bool = True,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    head_dropout: float = 0.0,
    adapter_dropout: float = 0.0,
    grasp_head_type: str = "simple",
    mature_head_hidden_dims: Optional[List[int]] = None,
    num_proposals: int = 4,
    graspnet_hidden_dims: Optional[List[int]] = None,
) -> "GC6DGraspPolicyVGGT":
    """VGGT 微调：backbone 注入 LoRA；Stage1 只训 adapter+head，Stage2 只训 LoRA，Stage3 联合。"""
    from .vggt_encoder import VGGTEncoder
    encoder = VGGTEncoder(
        feat_dim=encoder_feat_dim,
        freeze_backbone=freeze_backbone,
        ckpt_path=ckpt_path,
        lora_r=lora_r,
        lora_scale=lora_scale,
        adapter_dropout=adapter_dropout,
    )
    model = GC6DGraspPolicyVGGT(
        encoder=encoder,
        encoder_feat_dim=encoder_feat_dim,
        width_min=width_min,
        width_max=width_max,
        head_dropout=head_dropout,
        grasp_head_type=grasp_head_type,
        mature_head_hidden_dims=mature_head_hidden_dims,
        num_proposals=num_proposals,
        graspnet_hidden_dims=graspnet_hidden_dims,
    )
    model.encoder_type = "vggt_ft"
    return model


class GC6DGraspPolicyVGGT(nn.Module):
    """VGGT 图像 encoder + adapter + grasp head；输入 RGB 图像。"""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_feat_dim: int = 256,
        width_min: float = 0.01,
        width_max: float = 0.12,
        head_dropout: float = 0.0,
        grasp_head_type: str = "simple",
        mature_head_hidden_dims: Optional[List[int]] = None,
        num_proposals: int = 4,
        graspnet_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.encoder_type = "vggt_ft"
        self.grasp_head_type = grasp_head_type
        self.grasp_head_num_proposals = num_proposals if grasp_head_type == "graspnet" else None
        self.encoder = encoder
        self.grasp_head = _make_grasp_head(
            head_type=grasp_head_type,
            feat_dim=encoder_feat_dim,
            width_min=width_min,
            width_max=width_max,
            head_dropout=head_dropout,
            mature_hidden_dims=mature_head_hidden_dims,
            num_proposals=num_proposals,
            graspnet_hidden_dims=graspnet_hidden_dims,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224)
        return: (B, 10) action
        """
        feat = self.encoder(images)
        return self.grasp_head(feat)

    def forward_proposals(self, images: torch.Tensor) -> torch.Tensor:
        """return (B, K, 10) 或 (B, 1, 10)。"""
        feat = self.encoder(images)
        return _forward_proposals_from_feat(feat, self.grasp_head)

    def forward_proposals_raw(self, images: torch.Tensor) -> torch.Tensor:
        """return (B, K, 11) 或 (B, K, 10)。供 11D 直接转 17D 评估。"""
        feat = self.encoder(images)
        return _forward_proposals_raw_from_feat(feat, self.grasp_head)
