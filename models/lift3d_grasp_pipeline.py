# -*- coding: utf-8 -*-
"""
LIFT3D 线：点云 → LIFT3D(PointNext) encoder → adapter → cond → 注入 GraspNet seed_features → head。

仅 ``encoder_type=lift3d``，与 ``eval_benchmark`` / ``load_policy_from_checkpoint`` 的 state_dict 键一致
（``encoder`` / ``adapter`` / ``grasp_net``），无额外包装层。

VGGT 不在此文件处理。
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .graspnet_adapter import EncoderAdapterGraspNet

from .graspnet_adapter import build_encoder_adapter_graspnet


def build_lift3d_grasp_pipeline(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    lift3d_root: Optional[str] = None,
    lift3d_ckpt: Optional[str] = None,
    encoder_feat_dim: int = 256,
    adapter_cond_coeff: float = 1.0,
    adapter_cond_mode: str = "additive",
    use_adapter: bool = True,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    device: "torch.device" = None,
) -> "EncoderAdapterGraspNet":
    """
    构建 **LIFT3D + adapter + 预训练 GraspNet**（Stage2 equivalent 的默认结构）。

    - ``adapter_cond_coeff`` 默认 **1.0**（新 baseline）
    - ``adapter_cond_mode`` 默认 **additive**
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_encoder_adapter_graspnet(
        encoder_type="lift3d",
        graspnet_ckpt=graspnet_ckpt,
        encoder_feat_dim=encoder_feat_dim,
        graspnet_root=graspnet_root,
        lift3d_root=lift3d_root or os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D")),
        lift3d_ckpt=lift3d_ckpt,
        vggt_ckpt=None,
        lora_r=lora_r,
        lora_scale=lora_scale,
        lora_last_n_blocks=lora_last_n_blocks,
        device=device,
        use_adapter=use_adapter,
        adapter_cond_coeff=adapter_cond_coeff,
        adapter_cond_mode=adapter_cond_mode,
    )
