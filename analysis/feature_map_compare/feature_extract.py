# -*- coding: utf-8 -*-
"""
各模型在 pc_common 上的可比特征：统一记录 layer 名称、对齐方式与张量形状。

默认与训练/推理一致的可视化目标为 **任务适配后、进入 vpmodule 之前** 的 seed 特征（通常 256 维），
**不是** 纯几何的 ``world_points + pt_mlp`` 768：

- **VGGT（``vggt_mode=seed256``）**：``encoder.backbone``（含 LoRA）→ 中间 768 仅用于 NN gather 到 seed →
  ``replacement_projector`` →（``vggt_raw`` 含 replacement 对齐/缩放；progressive 含 α 混合与 adapter；
  distill / fusion 等同 ``forward``）→ 与 ``vpmodule`` 输入一致。
- **Lift3D-CLIP / DINOv2**：``replacement_projector`` 后及 progressive 分支后的 seed 特征。
- **dense768** 模式仅作可选对照（全图 ``pt_mlp`` 768，**无** projector/混合），旧式几何对照。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from analysis.feature_map_compare.align_features import (
    denormalize_lift3d_centers,
    nn_assign_features_world,
)
from models.lift3d_clip_patch_features import lift3d_clip_forward_patch_tokens
from models.lift3d_clip_replacement_pipeline import (
    _normalize_pc_lift3d,
)
from models.vggt_replacement_pipeline import (
    _vggt_local_features_b768k,
    apply_vggt_replacement_align_and_scale,
)
from utils.point_norm import normalize_xyz_with_pc

# 与训练/推理一致：VGGT 预训练权重标识（见 ``VGGTEncoder._load_vggt_backbone``）
PRETRAINED_VGGT_SOURCE = "facebook/VGGT-1B"
PRETRAINED_LIFT3D_CLIP_SOURCE = "LIFT3D lift3d_clip_base()"
PRETRAINED_DINOV2_SOURCE = "torch.hub facebookresearch/dinov2 dinov2_vitb14 pretrained=True"


def normalize_checkpoint_manifest_entry(value: Any) -> Dict[str, Any]:
    """
    支持两种 manifest 写法：

    - 字符串（ckpt 路径）→ ``{"type": "checkpoint", "path": ...}``
    - 字典：``{"type": "pretrained"}`` 或 ``{"type": "checkpoint", "path": "..."}``
    """
    if isinstance(value, str):
        p = os.path.expanduser(value.strip())
        return {"type": "checkpoint", "path": p}
    if isinstance(value, dict):
        t = value.get("type", "checkpoint")
        if t == "checkpoint":
            p = value.get("path")
            if not p or not isinstance(p, str):
                raise ValueError(f"checkpoint 类型需要非空字符串 path，当前: {value!r}")
            return {"type": "checkpoint", "path": os.path.expanduser(p.strip())}
        if t == "pretrained":
            return {"type": "pretrained"}
        raise ValueError(f"不支持的 manifest type: {t!r}（仅支持 checkpoint / pretrained）")
    raise TypeError(f"manifest 项必须是 str 或 dict，当前: {type(value).__name__}")


def load_model_from_manifest(
    entry: Dict[str, Any],
    model_name: str,
    *,
    device: torch.device,
    graspnet_ckpt: Optional[str],
    graspnet_root: Optional[str],
    lift3d_root: Optional[str],
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    统一加载：checkpoint 走 ``load_policy_from_checkpoint``；pretrained 仅支持
    ``vggt_raw`` / ``lift3d_clip_raw`` / ``lift3d_dinov2_raw``（与各自 alignment 构建函数一致，不重写 forward）。
    返回 ``(model, log_meta)``，``log_meta`` 含 ``load_type``、``source``（及可选 ``checkpoint_path``）。
    """
    from utils.load_model import load_policy_from_checkpoint

    et = entry.get("type", "checkpoint")
    log_meta: Dict[str, Any] = {"load_type": et, "source": None, "checkpoint_path": None}

    if et == "checkpoint":
        ckpt = entry["path"]
        log_meta["source"] = ckpt
        log_meta["checkpoint_path"] = ckpt
        model = load_policy_from_checkpoint(
            ckpt,
            device=str(device),
            graspnet_ckpt=graspnet_ckpt,
            graspnet_root=graspnet_root,
            lift3d_root=lift3d_root,
        )
        return model, log_meta

    if et != "pretrained":
        raise ValueError(f"未知 entry type: {et!r}")

    if model_name == "vggt_raw":
        from models.vggt_replacement_pipeline import build_vggt_replacement_graspnet

        if not graspnet_ckpt or not os.path.isfile(os.path.expanduser(graspnet_ckpt)):
            raise RuntimeError(
                f"预训练 vggt_raw 需要有效的 --graspnet_ckpt（GraspNet 骨干），当前: {graspnet_ckpt!r}"
            )
        try:
            model = build_vggt_replacement_graspnet(
                graspnet_ckpt=os.path.expanduser(graspnet_ckpt),
                graspnet_root=graspnet_root,
                vggt_ckpt=None,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(
                f"预训练 VGGT（{PRETRAINED_VGGT_SOURCE}）加载失败；请确认已安装 vggt 且可访问 Hugging Face 权重。"
            ) from e
        log_meta["source"] = PRETRAINED_VGGT_SOURCE
        return model, log_meta

    if model_name == "lift3d_clip_raw":
        from models.lift3d_clip_replacement_pipeline import build_lift3d_clip_replacement_graspnet

        if not graspnet_ckpt or not os.path.isfile(os.path.expanduser(graspnet_ckpt)):
            raise RuntimeError(
                f"预训练 lift3d_clip_raw 需要有效的 --graspnet_ckpt，当前: {graspnet_ckpt!r}"
            )
        try:
            model = build_lift3d_clip_replacement_graspnet(
                graspnet_ckpt=os.path.expanduser(graspnet_ckpt),
                graspnet_root=graspnet_root,
                lift3d_root=lift3d_root,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(
                f"预训练 Lift3D-CLIP（{PRETRAINED_LIFT3D_CLIP_SOURCE}）加载失败；请确认 LIFT3D_ROOT 有效且依赖齐全。"
            ) from e
        log_meta["source"] = PRETRAINED_LIFT3D_CLIP_SOURCE
        return model, log_meta

    if model_name == "lift3d_dinov2_raw":
        from models.dinov2_replacement_pipeline import build_dinov2_replacement_graspnet

        if not graspnet_ckpt or not os.path.isfile(os.path.expanduser(graspnet_ckpt)):
            raise RuntimeError(
                f"预训练 lift3d_dinov2_raw 需要有效的 --graspnet_ckpt，当前: {graspnet_ckpt!r}"
            )
        try:
            model = build_dinov2_replacement_graspnet(
                graspnet_ckpt=os.path.expanduser(graspnet_ckpt),
                graspnet_root=graspnet_root,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(
                f"预训练 DINOv2（{PRETRAINED_DINOV2_SOURCE}）加载失败；请确认 torch.hub 可拉取 facebookresearch/dinov2。"
            ) from e
        log_meta["source"] = PRETRAINED_DINOV2_SOURCE
        return model, log_meta

    raise ValueError(
        f"model_name={model_name!r} 不支持 type=pretrained（仅 vggt_raw / lift3d_clip_raw / lift3d_dinov2_raw）"
    )


def _freeze(m: nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


@torch.no_grad()
def vggt_world_grid_pt_mlp(
    encoder: nn.Module,
    images: torch.Tensor,
    *,
    conf_thresh: float = 0.05,
    max_points: int = 80000,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    全图 world_points 打平为 M，每点 pt_mlp -> 768 维（与 ``VGGTEncoder`` / replacement 一致）。
    按置信度过滤后，若仍过多则取 top-conf 子集。
    """
    meta: Dict[str, Any] = {
        "layer": "encoder.backbone['world_points'] + encoder.pt_mlp",
        "conf_thresh": conf_thresh,
        "max_points": max_points,
    }
    if images.dim() == 4:
        images_v = images.unsqueeze(1)
    else:
        images_v = images
    out = encoder.backbone(images_v)
    wp = out["world_points"]
    conf = out["world_points_conf"]
    B, V, H, W, _ = wp.shape
    M = V * H * W
    wp_flat = wp.reshape(B, M, 3)
    cf = conf.reshape(B, M)
    scores = cf[0].clone()
    scores[scores <= conf_thresh] = -1.0
    k = min(max_points, M)
    top_idx = torch.topk(scores, k=k, largest=True).indices
    wp_used = wp_flat[:, top_idx, :]
    feat = encoder.pt_mlp(wp_used).float()
    meta["M_used"] = int(k)
    meta["original_M"] = int(M)
    meta["feature_dim"] = int(feat.shape[-1])
    meta["subsample"] = "topk_confidence"
    ref_feat = feat.transpose(1, 2).contiguous()
    return wp_used, ref_feat, meta


@torch.no_grad()
def extract_vggt_family_dense_world(
    model: nn.Module,
    point_cloud: torch.Tensor,
    images: torch.Tensor,
    pc_common: torch.Tensor,
    *,
    model_name: str,
    max_points: int = 80000,
) -> Dict[str, Any]:
    """
    VGGT：全图 ``world_points`` 经 ``pt_mlp`` 得到 768 维，再 NN 对齐到 ``pc_common``。
    使用当前 checkpoint 内的 ``encoder``（含 LoRA 等），故不同实验权重会得到不同图。

    注意：progressive / distill / fusion 的 α 混合发生在 **GraspNet seed** 上，无法用纯 world 768 表达；
    若要与 vpmodule 输入 256 维一致，请使用 ``vggt_mode=seed256``。
    """
    enc = model.encoder
    wp_used, ref_feat, mmeta = vggt_world_grid_pt_mlp(enc, images, max_points=max_points)
    feat_on_pc = nn_assign_features_world(point_cloud, pc_common, wp_used, ref_feat)
    meta = dict(mmeta)
    meta["alignment"] = "nearest_neighbor_world_pt_mlp_768_to_pc_common"
    meta["feature_stage"] = "dense_pt_mlp_only_legacy"
    meta["warning_if_interpreting_progressive"] = (
        "World 768-D 不包含 seed 空间中的 α 混合；对比 progressive/distill 请用 seed256 模式。"
    )
    meta["model_name"] = model_name
    return {"points": pc_common, "features": feat_on_pc, "meta": meta}


@torch.no_grad()
def extract_graspnet_backbone_features(
    model: nn.Module,
    point_cloud: torch.Tensor,
    pc_common: torch.Tensor,
) -> Dict[str, Any]:
    """GraspNet view_estimator.backbone 的 seed_features，(B,256,S) -> pc_common。"""
    view_estimator = model.grasp_net.view_estimator
    end_points: Dict[str, Any] = {"point_clouds": point_cloud}
    seed_features, seed_xyz, _end = view_estimator.backbone(point_cloud, end_points)
    feat_on_pc = nn_assign_features_world(point_cloud, pc_common, seed_xyz, seed_features)
    return {
        "points": pc_common,
        "features": feat_on_pc,
        "meta": {
            "checkpoint_tensor": "view_estimator.backbone -> seed_features",
            "original_shape": list(seed_features.shape),
            "alignment": "nearest_neighbor_in_normalize_xyz_with_pc_space",
            "feature_dim": int(feat_on_pc.shape[-1]),
            "feature_stage": "graspnet_backbone_seed256_pre_vpmodule",
        },
    }


@torch.no_grad()
def extract_vggt_variant_pre_vpmodule(
    model: nn.Module,
    point_cloud: torch.Tensor,
    images: torch.Tensor,
    pc_common: torch.Tensor,
    *,
    variant: str,
) -> Dict[str, Any]:
    """
    variant: vggt_raw | vggt_progressive | vggt_distill | vggt_fusion_progressive | vggt_prog_enc_lora
    均复现 ``forward`` 中进入 ``vpmodule`` 之前的 **任务适配** ``seed_features``（256），
    含 ``replacement_projector``、progressive α / distill / fusion、以及 ``vggt_raw`` 的 replacement 对齐与缩放。
    中间 ``_vggt_local_features_b768k`` 仅为 VGGT→seed 对齐用，最终着色特征与此 768 几何路径不同。
    """
    view_estimator = model.grasp_net.view_estimator
    backbone = view_estimator.backbone
    f_graspnet, seed_xyz, end_points = backbone(point_cloud, end_points={"point_clouds": point_cloud})
    enc = model.encoder
    pts, feat_b768k = _vggt_local_features_b768k(enc, images)
    seed_n = normalize_xyz_with_pc(point_cloud, seed_xyz)
    pts_n = normalize_xyz_with_pc(point_cloud, pts)
    from models.lift3d_local_fusion import nearest_neighbor_gather_features

    vggt_raw = nearest_neighbor_gather_features(seed_n, pts_n, feat_b768k).float()

    if variant == "vggt_raw":
        proj = model.replacement_projector(vggt_raw)
        seed_features = apply_vggt_replacement_align_and_scale(model, proj)
        tensor_note = "replacement_projector + align/scale (same as VGGTReplacementGraspNet.forward pre-vpmodule)"
    elif variant == "vggt_progressive" or variant == "vggt_prog_enc_lora":
        x = model.replacement_projector(vggt_raw)
        x_ln = model.progressive_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = model.progressive_adapter(x_ln)
        a = float(model.progressive_alpha)
        seed_features = (1.0 - a) * f_graspnet + a * x
        tensor_note = f"(1-a)*f_graspnet+a*adapter(LN(proj(vggt_raw))), a={a}"
    elif variant == "vggt_distill":
        x = model.distill_projector(vggt_raw)
        x_ln = model.distill_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        seed_features = model.distill_adapter(x_ln)
        tensor_note = "distill_adapter(LN(distill_projector(vggt_raw))) -> vpmodule (student path)"
    elif variant == "vggt_fusion_progressive":
        x = model.fusion_to_seed_projector(vggt_raw)
        x_ln = model.fusion_seed_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = model.fusion_seed_adapter(x_ln)
        a = float(model.progressive_alpha)
        seed_features = (1.0 - a) * f_graspnet + a * x
        tensor_note = f"fusion progressive (1-a)*f_graspnet+a*fusion_adapter(LN(fusion_proj(vggt_raw)))"
    else:
        raise ValueError(variant)

    feat_on_pc = nn_assign_features_world(point_cloud, pc_common, seed_xyz, seed_features)
    return {
        "points": pc_common,
        "features": feat_on_pc,
        "meta": {
            "checkpoint_tensor": tensor_note,
            "original_shape": list(seed_features.shape),
            "alignment": "nearest_neighbor_in_normalize_xyz_with_pc_space (seed_xyz refs)",
            "feature_dim": 256,
            "variant": variant,
            "feature_stage": "pre_vpmodule_seed256_post_projector_and_mixing",
        },
    }


@torch.no_grad()
def extract_lift3d_clip_variant(
    model: nn.Module,
    point_cloud: torch.Tensor,
    pc_common: torch.Tensor,
    *,
    progressive: bool,
) -> Dict[str, Any]:
    from models.lift3d_local_fusion import nearest_neighbor_gather_features

    clip = model.encoder
    view_estimator = model.grasp_net.view_estimator
    backbone = view_estimator.backbone
    f_graspnet, seed_xyz, end_points = backbone(point_cloud, end_points={"point_clouds": point_cloud})
    pc_n = _normalize_pc_lift3d(point_cloud)
    patch_tok, centers = lift3d_clip_forward_patch_tokens(clip, pc_n)
    feat_b = patch_tok.transpose(1, 2).contiguous()
    ctr_world = denormalize_lift3d_centers(point_cloud, centers)
    seed_n = normalize_xyz_with_pc(point_cloud, seed_xyz)
    ctr_n = normalize_xyz_with_pc(point_cloud, ctr_world)
    lift_raw = nearest_neighbor_gather_features(seed_n, ctr_n, feat_b.float())
    if not progressive:
        seed_features = model.replacement_projector(lift_raw)
        tensor_note = "replacement_projector(NN gather Lift3D-CLIP 768 patch tokens to seeds)"
    else:
        x = model.replacement_projector(lift_raw)
        x_ln = model.progressive_ln(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = model.progressive_adapter(x_ln)
        a = float(model.progressive_alpha)
        seed_features = (1.0 - a) * f_graspnet + a * x
        tensor_note = f"(1-a)*f_graspnet+a*adapter(LN(proj(lift_raw))), a={a}"

    feat_on_pc = nn_assign_features_world(point_cloud, pc_common, seed_xyz, seed_features)
    return {
        "points": pc_common,
        "features": feat_on_pc,
        "meta": {
            "checkpoint_tensor": tensor_note,
            "original_shape": list(seed_features.shape),
            "alignment": "NN on seeds from lift3d patch centers (768->256)",
            "feature_dim": 256,
            "progressive": progressive,
            "feature_stage": "pre_vpmodule_seed256_post_projector_and_mixing",
        },
    }


@torch.no_grad()
def extract_dinov2_variant(
    model: nn.Module,
    point_cloud: torch.Tensor,
    pc_common: torch.Tensor,
    *,
    progressive: bool,
) -> Dict[str, Any]:
    """DINOv2：与 ``Dinov2ReplacementGraspNet`` / progressive 的 forward 一致，取 vpmodule 前 256 维 seed 特征。"""
    enc = model.encoder
    view_estimator = model.grasp_net.view_estimator
    backbone = view_estimator.backbone
    f_graspnet, seed_xyz, _ = backbone(point_cloud, end_points={"point_clouds": point_cloud})
    enc_b768s = enc(point_cloud, seed_xyz)
    if not progressive:
        seed_features = model.replacement_projector(enc_b768s.float())
        tensor_note = "replacement_projector(DINOv2SeedEncoder(pc, seed_xyz) BEV->768 per seed)"
    else:
        x_proj = model.replacement_projector(enc_b768s.float())
        x_ln = model.progressive_ln(x_proj.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        f_lift = model.progressive_adapter(x_ln)
        a = float(model.progressive_alpha)
        seed_features = (1.0 - a) * f_graspnet + a * f_lift
        tensor_note = f"(1-a)*f_graspnet+a*adapter(LN(proj(DINOv2(pc,seed)))), a={a}"
    feat_on_pc = nn_assign_features_world(point_cloud, pc_common, seed_xyz, seed_features)

    return {
        "points": pc_common,
        "features": feat_on_pc,
        "meta": {
            "checkpoint_tensor": tensor_note,
            "original_shape": list(seed_features.shape),
            "alignment": "nearest_neighbor seeds -> pc_common",
            "feature_dim": 256,
            "progressive": progressive,
            "feature_stage": "pre_vpmodule_seed256_post_projector_and_mixing",
        },
    }


def extract_by_model_name(
    model: nn.Module,
    model_name: str,
    point_cloud: torch.Tensor,
    images: Optional[torch.Tensor],
    pc_common: torch.Tensor,
    *,
    vggt_mode: str = "seed256",
    vggt_dense_max_points: int = 80000,
) -> Dict[str, Any]:
    """model_name 与实验命名一致。``vggt_mode``: ``seed256``（默认，适配后 seed）| ``dense768``（仅 pt_mlp 稠密对照）。"""
    _freeze(model)

    if model_name == "graspnet_backbone":
        return extract_graspnet_backbone_features(model, point_cloud, pc_common)

    vggt_names = (
        "vggt_raw",
        "vggt_progressive_alpha05",
        "vggt_prog_enc_lora",
        "vggt_distill",
        "vggt_fusion_progressive_alpha05",
    )
    if model_name in vggt_names:
        if images is None:
            return {"points": pc_common, "features": None, "meta": {"error": "VGGT 需要 images"}}
        if vggt_mode == "dense768":
            return extract_vggt_family_dense_world(
                model,
                point_cloud,
                images,
                pc_common,
                model_name=model_name,
                max_points=vggt_dense_max_points,
            )
        if model_name == "vggt_raw":
            return extract_vggt_variant_pre_vpmodule(
                model, point_cloud, images, pc_common, variant="vggt_raw"
            )
        if model_name == "vggt_progressive_alpha05":
            return extract_vggt_variant_pre_vpmodule(
                model, point_cloud, images, pc_common, variant="vggt_progressive"
            )
        if model_name == "vggt_prog_enc_lora":
            return extract_vggt_variant_pre_vpmodule(
                model, point_cloud, images, pc_common, variant="vggt_prog_enc_lora"
            )
        if model_name == "vggt_distill":
            return extract_vggt_variant_pre_vpmodule(
                model, point_cloud, images, pc_common, variant="vggt_distill"
            )
        if model_name == "vggt_fusion_progressive_alpha05":
            return extract_vggt_variant_pre_vpmodule(
                model, point_cloud, images, pc_common, variant="vggt_fusion_progressive"
            )

    if model_name == "lift3d_clip_raw":
        return extract_lift3d_clip_variant(model, point_cloud, pc_common, progressive=False)
    if model_name == "lift3d_clip_progressive_alpha05":
        return extract_lift3d_clip_variant(model, point_cloud, pc_common, progressive=True)

    if model_name == "lift3d_dinov2_raw":
        return extract_dinov2_variant(model, point_cloud, pc_common, progressive=False)
    if model_name == "lift3d_dinov2_progressive_alpha05":
        return extract_dinov2_variant(model, point_cloud, pc_common, progressive=True)

    return {
        "points": pc_common,
        "features": None,
        "meta": {
            "error": f"unknown model_name={model_name}, model_mode={getattr(model, 'model_mode', None)}",
        },
    }
