# -*- coding: utf-8 -*-
"""
LIFT3D PointNext encoder 封装：可选 LoRA、固定 512->feat_dim 适配器。
输入 point_cloud (B, N, 3)，输出 (B, feat_dim)。

说明：与 ``lift3d.models.lift3d.model_loader.lift3d_clip_base()``（CLIP 点云编码）不是同一路；
本管线需要 PointNext 的 ``forward_seg_feat`` 做局部特征，故用 ``PointNextModel`` + 可选 ``ckpt_path``。
不传 ckpt 时 backbone 为配置初始化（权重需自行提供 checkpoint 路径）。
"""

import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

from .lora import inject_lora, get_lora_params, get_non_lora_params


def _ensure_lift3d_path(lift3d_root: Optional[str] = None):
    if lift3d_root is None:
        lift3d_root = os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
    lift3d_root = os.path.abspath(os.path.expanduser(lift3d_root))
    if not os.path.isdir(lift3d_root):
        raise FileNotFoundError(f"LIFT3D root not found: {lift3d_root}")
    if lift3d_root not in sys.path:
        sys.path.insert(0, lift3d_root)
    # openpoints/cpp/pointnet2_batch/__init__.py 里 import pointnet2_batch_cuda，.so 在该目录，需加入 path
    pointnet2_batch_dir = os.path.join(lift3d_root, "lift3d", "models", "point_next", "openpoints", "cpp", "pointnet2_batch")
    if os.path.isdir(pointnet2_batch_dir) and pointnet2_batch_dir not in sys.path:
        sys.path.insert(0, pointnet2_batch_dir)
    return lift3d_root


def _load_point_next(lift3d_root: str, config_name: str = "point_next.yaml", ckpt_path: Optional[str] = None):
    """加载 LIFT3D PointNextModel。ckpt_path 非空时加载预训练权重到 backbone。"""
    from lift3d.models.point_next.point_next import PointNextModel

    # config 相对 point_next.py 所在目录
    point_next_dir = os.path.join(lift3d_root, "lift3d", "models", "point_next")
    config_path = os.path.join(point_next_dir, config_name)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"PointNext config not found: {config_path}")
    model = PointNextModel(config_path)
    if ckpt_path and ckpt_path != "__NONE__":
        ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
        if os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # PointNextModel 含 .model (PointNextEncoder)，state_dict 的 key 为 "model.xxx"
            target_keys = set(model.state_dict().keys())
            # 若 ckpt 的 key 无 "model." 前缀（仅 Encoder），则加上前缀以匹配
            sd_use = {}
            for k, v in sd.items():
                if k in target_keys:
                    sd_use[k] = v
                elif not k.startswith("model.") and ("model." + k) in target_keys:
                    sd_use["model." + k] = v
                else:
                    sd_use[k] = v
            miss, unexp = model.load_state_dict(sd_use, strict=False)
            logging.getLogger(__name__).info(
                "[LIFT3D] loaded pretrained %s missing=%d unexpected=%d", ckpt_path, len(miss), len(unexp)
            )
        else:
            logging.getLogger(__name__).warning("[LIFT3D] ckpt_path not found, backbone random init: %s", ckpt_path)
    return model


class LIFT3DEncoder(nn.Module):
    """
    LIFT3D PointNext + 可选 LoRA + 可选 512->feat_dim 适配器。
    forward(point_cloud) -> (B, feat_dim)
    """

    def __init__(
        self,
        lift3d_root: Optional[str] = None,
        point_next_config: str = "point_next.yaml",
        feat_dim: int = 256,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_scale: float = 1.0,
        lora_last_n_blocks: Optional[int] = None,
        normalize_pc: bool = True,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self._lift3d_root = _ensure_lift3d_path(lift3d_root)
        self.feat_dim = feat_dim
        self.normalize_pc = normalize_pc
        self.use_lora = use_lora

        self.backbone = _load_point_next(self._lift3d_root, point_next_config, ckpt_path=ckpt_path)
        # PointNext 输出 512
        if self.backbone.feature_dim != 512:
            raise ValueError(f"Expected backbone feature_dim 512, got {self.backbone.feature_dim}")

        if use_lora:
            inject_lora(self.backbone, r=lora_r, scale=lora_scale, last_n_blocks=lora_last_n_blocks)

        self.adapter = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.ReLU(inplace=True),
        )

    def _normalize(self, pc: torch.Tensor) -> torch.Tensor:
        """样本级中心化 + 样本级缩放，保持整帧点云几何比例。"""
        center = pc.mean(dim=1, keepdim=True)
        pc = pc - center
        # 每样本一个 scale：该样本所有点的最大绝对值
        scale = pc.abs().reshape(pc.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
        return pc / scale

    def normalize_seed_xyz(self, point_cloud: torch.Tensor, seed_xyz: torch.Tensor) -> torch.Tensor:
        """
        将 GraspNet 的 seed_xyz 变到与 ``_normalize(point_cloud)`` 后点云同一坐标系，
        以便与 ``forward_seg_feat`` 输出的 ref 坐标做最近邻。
        point_cloud / seed_xyz: 与模型 forward 相同的原始空间 (B,N,3) / (B,S,3)。
        """
        if not self.normalize_pc:
            return seed_xyz
        center = point_cloud.mean(dim=1, keepdim=True)
        pc0 = point_cloud - center
        scale = pc0.abs().reshape(point_cloud.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(
            min=1e-6
        )
        return (seed_xyz - center) / scale

    def forward_seg_feat(self, point_cloud: torch.Tensor):
        """
        与 ``forward`` 使用同一套 ``_normalize``，返回 PointNext ``forward_seg_feat`` 的多尺度 p, f。
        p[i]: (B, Ni, 3)，f[i]: (B, Ci, Ni)。
        """
        if self.normalize_pc:
            point_cloud = self._normalize(point_cloud)
        x = point_cloud[:, :, :3].contiguous().float()
        self.backbone.model.to(x.device)
        if self.training:
            p, f = self.backbone.model.forward_seg_feat(x)
        else:
            use_amp = x.is_cuda
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                p, f = self.backbone.model.forward_seg_feat(x)
        # eval 下 autocast 可能输出 FP16，下游 Conv1d / 与 GraspNet FP32 特征拼接需统一为 float32
        def _to_f32(t):
            return t.float() if isinstance(t, torch.Tensor) else t

        return [_to_f32(t) for t in p], [_to_f32(t) for t in f]

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: (B, N, 3)
        return: (B, feat_dim)
        """
        if self.normalize_pc:
            point_cloud = self._normalize(point_cloud)
        # PointNextModel.forward 只取 xyz
        x = point_cloud[:, :, :3].contiguous().float()
        if self.training:
            # 训练时绕过 PointNextModel.forward 内的 autocast，直接调 forward_cls_feat，避免 Half/float 混用报错
            # 确保 backbone.model 与输入同设备（避免 weight 在 CPU、input 在 CUDA）
            self.backbone.model.to(x.device)
            emb = self.backbone.model.forward_cls_feat(x)
        else:
            use_amp = x.is_cuda
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                emb = self.backbone(x)
        return self.adapter(emb.float())

    def get_lora_params(self):
        return get_lora_params(self.backbone) if self.use_lora else []

    def get_non_lora_params(self):
        backbone_non_lora = get_non_lora_params(self.backbone)
        return list(self.adapter.parameters()) + backbone_non_lora
