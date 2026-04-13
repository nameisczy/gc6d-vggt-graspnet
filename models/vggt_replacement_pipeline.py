# -*- coding: utf-8 -*-
"""
VGGT replacement：仅用 GraspNet backbone 的 seed_xyz；VGGT 的 3D 点特征经最近邻对齐到 seed，
再投影到 256 维送入 head。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .alignment_projectors import make_conv1d_projector
from .graspnet_adapter import load_graspnet_pretrained
from .lift3d_local_fusion import nearest_neighbor_gather_features
from utils.point_norm import normalize_xyz_with_pc

from .vggt_encoder import VGGTEncoder


def apply_vggt_replacement_align_and_scale(model: nn.Module, seed_features: torch.Tensor) -> torch.Tensor:
    """
    ``replacement_projector`` 输出之后、``vpmodule`` 之前：与 ``VGGTReplacementGraspNet.forward`` 一致。
    非 ``VGGTReplacementGraspNet``（无 ``replacement_align_mode``）时原样返回，便于特征提取与 forward 共用。
    """
    if not hasattr(model, "replacement_align_mode"):
        return seed_features
    mode = str(model.replacement_align_mode)
    if mode == "layernorm":
        x = seed_features.permute(0, 2, 1).contiguous()
        seed_features = model.replacement_align_ln(x).permute(0, 2, 1).contiguous()
    elif mode == "layernorm_affine":
        x = seed_features.permute(0, 2, 1).contiguous()
        x = model.replacement_align_ln(x)
        x = x * model.replacement_affine_gamma.view(1, 1, 256) + model.replacement_affine_beta.view(1, 1, 256)
        seed_features = x.permute(0, 2, 1).contiguous()
    elif mode == "adapter":
        seed_features = model.replacement_adapter(seed_features)
    elif mode in ("ln_adapter", "layernorm_adapter"):
        x = seed_features.permute(0, 2, 1).contiguous()
        x = model.replacement_align_ln(x).permute(0, 2, 1).contiguous()
        seed_features = model.replacement_adapter(x)
    elif mode in ("layernorm_affine_adapter", "layernorm_affine_deep_adapter"):
        x = seed_features.permute(0, 2, 1).contiguous()
        x = model.replacement_align_ln(x)
        x = x * model.replacement_affine_gamma.view(1, 1, 256) + model.replacement_affine_beta.view(1, 1, 256)
        x = x.permute(0, 2, 1).contiguous()
        seed_features = model.replacement_adapter(x)

    if hasattr(model, "replacement_scale_mode"):
        if model.replacement_scale_mode == "fixed":
            seed_features = model.replacement_fixed_alpha * seed_features
        elif model.replacement_scale_mode == "ln_learnable":
            x = seed_features.permute(0, 2, 1).contiguous()
            x = model.replacement_scale_ln(x)
            seed_features = x.permute(0, 2, 1).contiguous() * model.replacement_learnable_scale
    return seed_features


def _make_replacement_adapter(hidden: int, depth: int) -> nn.Sequential:
    layers = [nn.Conv1d(256, hidden, kernel_size=1), nn.ReLU(inplace=True)]
    for _ in range(max(depth - 2, 0)):
        layers.extend([nn.Conv1d(hidden, hidden, kernel_size=1), nn.ReLU(inplace=True)])
    layers.append(nn.Conv1d(hidden, 256, kernel_size=1))
    return nn.Sequential(*layers)


def _vggt_local_features_b768k(
    encoder: VGGTEncoder,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    images: (B,3,224,224)
    return: pts (B,k,3), feat (B,768,k) 与 VGGT 高置信度采样点一致。

    **中间量**：``encoder.backbone``（含注入的 LoRA）预测 ``world_points``，再经 ``pt_mlp``(xyz)→768
    供最近邻 gather 到 GraspNet seed。**可视化/对比请用** ``extract_vggt_variant_pre_vpmodule`` 的 **256 维**
    ``replacement_projector`` + 混合/对齐 之后、与 ``forward`` 中 ``vpmodule`` 输入一致的特征，而非单独使用本函数的 768。
    """
    if images.dim() == 4:
        images_v = images.unsqueeze(1)
    else:
        images_v = images
    out = encoder.backbone(images_v)
    wp = out["world_points"]
    conf = out["world_points_conf"]
    if wp.ndim != 5:
        raise ValueError("VGGT world_points 维数异常: %s" % (tuple(wp.shape),))
    B, V, H, W, _ = wp.shape
    wp_flat = wp.reshape(B, V * H * W, 3)
    cf_flat = conf.reshape(B, V * H * W)
    k = min(encoder.sample_k, wp_flat.shape[1])
    idx = torch.topk(cf_flat, k=k, dim=1, largest=True).indices
    pts = torch.gather(wp_flat, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    feat = encoder.pt_mlp(pts).float()
    feat = feat.transpose(1, 2)
    return pts, feat


class VGGTReplacementGraspNet(nn.Module):
    def __init__(
        self,
        encoder: VGGTEncoder,
        grasp_net: nn.Module,
        vggt_dim: int = 768,
        replacement_align_mode: str = "none",
        replacement_affine_init_scale: float = 1.0,
        replacement_adapter_hidden: int = 256,
        replacement_adapter_depth: int = 2,
        replacement_scale_mode: str = "none",
        replacement_fixed_alpha: float = 1.0,
        replacement_learnable_scale_init: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.grasp_net = grasp_net
        self.vggt_dim = vggt_dim
        self.replacement_projector = make_conv1d_projector(vggt_dim, 256)
        self.replacement_align_mode = str(replacement_align_mode)
        self.replacement_affine_init_scale = float(replacement_affine_init_scale)
        self.replacement_adapter_hidden = int(replacement_adapter_hidden)
        self.replacement_adapter_depth = int(replacement_adapter_depth)
        if self.replacement_align_mode in (
            "layernorm",
            "layernorm_affine",
            "ln_adapter",
            "layernorm_adapter",
            "layernorm_affine_adapter",
            "layernorm_affine_deep_adapter",
        ):
            self.replacement_align_ln = nn.LayerNorm(256, elementwise_affine=False)
        else:
            self.replacement_align_ln = None
        if self.replacement_align_mode in ("layernorm_affine", "layernorm_affine_adapter", "layernorm_affine_deep_adapter"):
            self.replacement_affine_gamma = nn.Parameter(
                torch.full((256,), float(replacement_affine_init_scale), dtype=torch.float32)
            )
            self.replacement_affine_beta = nn.Parameter(torch.zeros(256, dtype=torch.float32))
        else:
            self.register_parameter("replacement_affine_gamma", None)
            self.register_parameter("replacement_affine_beta", None)
        if self.replacement_align_mode in (
            "adapter",
            "ln_adapter",
            "layernorm_adapter",
            "layernorm_affine_adapter",
            "layernorm_affine_deep_adapter",
        ):
            h = int(replacement_adapter_hidden)
            depth = 3 if self.replacement_align_mode == "layernorm_affine_deep_adapter" else int(replacement_adapter_depth)
            self.replacement_adapter = _make_replacement_adapter(h, depth)
        else:
            self.replacement_adapter = None
        self.replacement_scale_mode = str(replacement_scale_mode)
        self.replacement_fixed_alpha = float(replacement_fixed_alpha)
        self.replacement_learnable_scale_init = float(replacement_learnable_scale_init)
        if self.replacement_scale_mode == "ln_learnable":
            self.replacement_scale_ln = nn.LayerNorm(256)
            self.replacement_learnable_scale = nn.Parameter(
                torch.tensor(float(replacement_learnable_scale_init), dtype=torch.float32)
            )
        else:
            self.replacement_scale_ln = None
            self.register_parameter("replacement_learnable_scale", None)
        self.use_adapter = False
        self.encoder_type = "vggt_replacement"
        self.requires_images = True
        self.model_mode = "vggt_replacement"

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is None:
            raise ValueError("VGGTReplacementGraspNet 需要 images (B,3,224,224)")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        _seed_unused, seed_xyz, end_points = backbone(point_cloud, end_points)

        pts, feat_b768k = _vggt_local_features_b768k(self.encoder, images)
        seed_n = normalize_xyz_with_pc(point_cloud, seed_xyz)
        pts_n = normalize_xyz_with_pc(point_cloud, pts)
        vggt_raw = nearest_neighbor_gather_features(seed_n, pts_n, feat_b768k).float()
        seed_features = self.replacement_projector(vggt_raw)
        seed_features = apply_vggt_replacement_align_and_scale(self, seed_features)

        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        return end_points


def build_vggt_replacement_graspnet(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    vggt_ckpt: Optional[str] = None,
    feat_dim: int = 256,
    sample_k: int = 1024,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    replacement_align_mode: str = "none",
    replacement_affine_init_scale: float = 1.0,
    replacement_adapter_hidden: int = 256,
    replacement_adapter_depth: int = 2,
    replacement_scale_mode: str = "none",
    replacement_fixed_alpha: float = 1.0,
    replacement_learnable_scale_init: float = 1.0,
    device: Optional[torch.device] = None,
) -> VGGTReplacementGraspNet:
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
    model = VGGTReplacementGraspNet(
        encoder=encoder,
        grasp_net=grasp_net,
        vggt_dim=768,
        replacement_align_mode=replacement_align_mode,
        replacement_affine_init_scale=replacement_affine_init_scale,
        replacement_adapter_hidden=replacement_adapter_hidden,
        replacement_adapter_depth=replacement_adapter_depth,
        replacement_scale_mode=replacement_scale_mode,
        replacement_fixed_alpha=replacement_fixed_alpha,
        replacement_learnable_scale_init=replacement_learnable_scale_init,
    )
    return model.to(device)
