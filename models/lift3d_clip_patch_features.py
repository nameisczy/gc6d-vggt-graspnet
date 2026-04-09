# -*- coding: utf-8 -*-
"""
从 LIFT3D 官方 ``Lift3dCLIP``（lift3d_clip_base）提取 patch token，供 replacement 使用。
不修改 ~/LIFT3D 源码：此处复现 ``Lift3dCLIP.forward`` 中 transformer 前向，并返回非 CLS token。
"""

from __future__ import annotations

import torch


def lift3d_clip_forward_patch_tokens(model: torch.nn.Module, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    model: Lift3dCLIP（lift3d_clip_base 返回值）
    pts: (B, N, 3) 已与 LIFT3D 训练一致的归一化点云
    return:
      patch_tokens: (B, G, 768)  G=num_group（通常 64）
      centers: (B, G, 3)  每组中心，用于与 seed_xyz 最近邻对齐
    """
    tokens, pos = [], []
    if pts.dim() == 2:
        pts = pts.unsqueeze(0).float()
    pts = pts[:, :, :3].float()
    batch_size = pts.shape[0]
    pts_trans = pts.clone().transpose(1, 2).contiguous()
    center, group_input_tokens = model.patch_embed(pts_trans, pts)
    group_input_tokens = group_input_tokens.transpose(1, 2)

    pos_x, pos_y, _ = model.get_pos_2d(center)
    model.patch_pos_embed_2D = model.pos_embed_2d[:, 1:]

    interpolated_pos_embed = model.bilinear_interpolation_3d_to_2d(
        pos_x, pos_y, model.patch_pos_embed_2D
    )
    interpolated_pos_embed = interpolated_pos_embed.reshape(
        center.shape[0], -1, center.shape[1], model.trans_dim
    )
    interpolated_pos_embed = interpolated_pos_embed.mean(dim=1)

    tokens.append(group_input_tokens)
    pos.append(interpolated_pos_embed)
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    cls_pos = model.cls_pos.expand(batch_size, -1, -1)
    tokens.insert(0, cls_tokens)
    pos.insert(0, cls_pos)

    tokens = torch.cat(tokens, dim=1)
    pos = torch.cat(pos, dim=1)

    x = (tokens + pos).permute(1, 0, 2)
    for _, block in enumerate(model.resblocks):
        x = block(x)

    x = x.permute(1, 0, 2)
    x = model.norm(x)
    patch_tokens = x[:, 1:, :]
    return patch_tokens, center
