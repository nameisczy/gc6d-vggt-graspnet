# -*- coding: utf-8 -*-
"""
LIFT3D 官方点云 encoder（lift3d_clip_base）：768 维 → 适配器 → feat_dim。
支持：冻结 backbone 仅训 head（Stage1）、仅训 backbone 内 LoRA（Stage2）、联合训（Stage3）。
"""

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn


def _ensure_lift3d_path(lift3d_root: Optional[str] = None) -> str:
    root = lift3d_root or os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"LIFT3D root not found: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _load_lift3d_clip_backbone(lift3d_root: str, lora_r: int = 8, lora_scale: float = 1.0):
    """加载 LIFT3D 官方 lift3d_clip_base()，注入 LoRA（与 VGGT 配置对齐），返回 backbone。"""
    from lift3d.models.lift3d.model_loader import lift3d_clip_base
    from .lora import inject_lora
    model = lift3d_clip_base()
    if not hasattr(model, "feature_dim"):
        model.feature_dim = getattr(model, "embed_dim", 768)
    inject_lora(model, r=lora_r, scale=lora_scale)
    return model


def get_backbone_lora_params(backbone: nn.Module, last_n_blocks: Optional[int] = None) -> List[torch.nn.Parameter]:
    """返回 backbone 中 LoRA 参数。last_n_blocks 非空时只返回最后 n 个 block（Stage3 降强度）。"""
    from .lora import get_lora_params, get_lora_params_from_last_n_blocks
    if last_n_blocks is not None and last_n_blocks > 0:
        return get_lora_params_from_last_n_blocks(backbone, last_n_blocks)
    return get_lora_params(backbone)


class LIFT3DClipEncoder(nn.Module):
    """
    LIFT3D 官方 encoder (lift3d_clip_base) + 768→feat_dim 适配器。
    forward(point_cloud) -> (B, feat_dim)。
    - Stage1: freeze_backbone=True，只训 adapter + 下游 head。
    - Stage2: 仅解冻 backbone 内 LoRA 参数，训 LoRA。
    - Stage3: 解冻 head（及可选 adapter/LoRA）联合训。
    """

    def __init__(
        self,
        lift3d_root: Optional[str] = None,
        feat_dim: int = 256,
        freeze_backbone: bool = True,
        normalize_pc: bool = True,
        lora_r: int = 8,
        lora_scale: float = 1.0,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()
        self._lift3d_root = _ensure_lift3d_path(lift3d_root)
        self.feat_dim = feat_dim
        self.normalize_pc = normalize_pc
        self.adapter_dropout = float(adapter_dropout)

        self.backbone = _load_lift3d_clip_backbone(self._lift3d_root, lora_r=lora_r, lora_scale=lora_scale)
        backbone_dim = getattr(self.backbone, "feature_dim", 768)
        if backbone_dim != 768:
            raise ValueError(f"Expected backbone feature_dim 768, got {backbone_dim}")

        self.adapter = nn.Sequential(
            nn.Linear(768, feat_dim),
            nn.ReLU(inplace=True),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _normalize(self, pc: torch.Tensor) -> torch.Tensor:
        """与 LIFT3D 风格一致的点云归一化。"""
        center = pc.mean(dim=1, keepdim=True)
        pc = pc - center
        scale = pc.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        return pc / scale

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: (B, N, 3)
        return: (B, feat_dim)
        """
        if self.normalize_pc:
            point_cloud = self._normalize(point_cloud)
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)
        pts = point_cloud[:, :, :3].contiguous().float()
        with torch.amp.autocast(device_type="cuda", enabled=pts.is_cuda):
            emb = self.backbone(pts)
        h = self.adapter(emb)
        if self.adapter_dropout > 0 and self.training:
            h = torch.nn.functional.dropout(h, p=self.adapter_dropout, training=True)
        return h

    def get_backbone_lora_params(self, last_n_blocks: Optional[int] = None) -> List[torch.nn.Parameter]:
        return get_backbone_lora_params(self.backbone, last_n_blocks=last_n_blocks)

    def set_backbone_lora_trainable(self, trainable: bool, last_n_blocks: Optional[int] = None):
        from .lora import get_lora_params
        all_lora = get_lora_params(self.backbone)
        to_train = self.get_backbone_lora_params(last_n_blocks=last_n_blocks) if last_n_blocks else all_lora
        to_train_ids = {id(p) for p in to_train}
        for p in all_lora:
            p.requires_grad = trainable and (id(p) in to_train_ids)

    def set_adapter_trainable(self, trainable: bool):
        for p in self.adapter.parameters():
            p.requires_grad = trainable
