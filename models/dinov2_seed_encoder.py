# -*- coding: utf-8 -*-
"""
DINOv2（facebook/dinov2 ViT-B/14）+ BEV 点云栅格 → patch 特征 → 按 seed 的 xy 双线性采样，得到 (B,768,S)。
用于 ``lift3d_replacement_dinov2``：不依赖 LIFT3D 仓库内的独立 DINO 权重（官方仅提供 lift3d_clip_base）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_pc(pc: torch.Tensor) -> torch.Tensor:
    c = pc.mean(dim=1, keepdim=True)
    p0 = pc - c
    s = p0.abs().reshape(pc.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    return p0 / s


def _normalize_seed(pc: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    c = pc.mean(dim=1, keepdim=True)
    p0 = pc - c
    s = p0.abs().reshape(pc.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    return (seed - c) / s


def _bev_image_and_bounds(
    pc_n: torch.Tensor, size: int = 224
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pc_n: (B, N, 3) 已归一化
    return: img (B,3,H,W), xmin,xmax,ymin,ymax 为 (B,1) 便于与 (B,S) 广播
    """
    B, N, _ = pc_n.shape
    x, y, z = pc_n[..., 0], pc_n[..., 1], pc_n[..., 2]
    xmin = x.min(dim=1, keepdim=True)[0]
    xmax = x.max(dim=1, keepdim=True)[0]
    ymin = y.min(dim=1, keepdim=True)[0]
    ymax = y.max(dim=1, keepdim=True)[0]
    u = (x - xmin) / (xmax - xmin + 1e-6) * (size - 1)
    v = (y - ymin) / (ymax - ymin + 1e-6) * (size - 1)
    gi = u.long().clamp(0, size - 1)
    gj = v.long().clamp(0, size - 1)
    flat_idx = (gj * size + gi).long()
    img_z = torch.zeros(B, size * size, device=pc_n.device, dtype=pc_n.dtype)
    img_z.scatter_reduce_(1, flat_idx, z, reduce="amax", include_self=False)
    img_hit = torch.zeros(B, size * size, device=pc_n.device, dtype=pc_n.dtype)
    one = torch.ones_like(z)
    img_hit.scatter_reduce_(1, flat_idx, one, reduce="amax", include_self=False)
    img0 = img_z.view(B, 1, size, size)
    img1 = img_hit.view(B, 1, size, size)
    img2 = img0 / (img1 + 1e-6)
    img = torch.cat([img0, img1, img2], dim=1)
    return img, xmin, xmax, ymin, ymax


def _load_dinov2_vitb14(device: torch.device) -> nn.Module:
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device)


class DINOv2SeedEncoder(nn.Module):
    """
    冻结 DINOv2；forward(point_cloud, seed_xyz) -> (B, 768, S)
    """

    def __init__(self, device: Optional[torch.device] = None, img_size: int = 224):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dinov2 = _load_dinov2_vitb14(device)
        self.embed_dim = 768
        self.img_size = img_size
        self.patch_size = 14
        self._grid = img_size // self.patch_size  # 16
        for p in self.dinov2.parameters():
            p.requires_grad = False
        self.eval()

    def forward_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B,3,H,W)，已 ImageNet normalize。兼容 hub 版 ``forward_features`` 返回 Tensor 或 dict。"""
        out = self.dinov2.forward_features(images)
        if isinstance(out, dict):
            x = out.get("x_norm", out.get("x_prenorm", None))
            if x is None:
                for v in out.values():
                    if torch.is_tensor(v) and v.dim() == 3:
                        x = v
                        break
            if x is None:
                raise RuntimeError("dinov2 forward_features 未找到 3D token 张量: keys=%s" % (list(out.keys()),))
        else:
            x = out
        return x[:, 1:, :]

    def forward(self, point_cloud: torch.Tensor, seed_xyz: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: (B, N, 3)
        seed_xyz: (B, S, 3)
        return: (B, 768, S)
        """
        pc_n = _normalize_pc(point_cloud)
        seed_n = _normalize_seed(point_cloud, seed_xyz)
        img, xmin, xmax, ymin, ymax = _bev_image_and_bounds(pc_n, self.img_size)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        img = (img - mean) / std

        tok = self.forward_patch_tokens(img)
        B, P, C = tok.shape
        assert P == self._grid * self._grid
        fh, fw = self._grid, self._grid
        feat_map = tok.reshape(B, fh, fw, C).permute(0, 3, 1, 2)

        sx, sy = seed_n[..., 0], seed_n[..., 1]
        ux = (sx - xmin) / (xmax - xmin + 1e-6) * (self.img_size - 1)
        uy = (sy - ymin) / (ymax - ymin + 1e-6) * (self.img_size - 1)
        gx = ((ux + 0.5) / float(self.img_size)) * 2 - 1
        gy = ((uy + 0.5) / float(self.img_size)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(2)
