# -*- coding: utf-8 -*-
"""
训练损失：主任务（17D matching）+ 可选 collision / quality 辅助项。

quality_aux：默认关闭；此处提供接口，返回 0 或与 pred 相关的占位（便于后续接 DexNet 等）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# 主损失仍复用 utils.loss（与旧 benchmark 一致），避免重复实现 matching 逻辑
from utils.loss import action_loss_topk_matched_17d, pad_gt_grasp_group_17d
from models.graspnet_adapter import pred_decode_17d_differentiable, pred_decode_17d

_DISTILL_MODEL_MODES = (
    "vggt_replacement_distill",
    "vggt_fusion_distill",
    "lift3d_replacement_distill_clip",
    "lift3d_replacement_distill_dinov2",
)


def _is_distill_model(model: torch.nn.Module) -> bool:
    return getattr(model, "model_mode", None) in _DISTILL_MODEL_MODES


def model_requires_images(model: torch.nn.Module) -> bool:
    enc = getattr(model, "encoder", None)
    if enc is not None and type(enc).__name__ == "VGGTEncoder":
        return True
    return bool(getattr(model, "requires_images", False))


def collision_clearance_aux(
    pred_17d: torch.Tensor,
    scene_pc: torch.Tensor,
    *,
    margin: float = 0.01,
    max_scene_points: int = 4096,
) -> torch.Tensor:
    """
    可微「清障」代理：惩罚 grasp 中心到场景点过近（非 GraspNet 碰撞检测的逐像素复现，
    但作为辅助项可鼓励与点云保持 margin）。

    pred_17d: (B, K, 17)，scene_pc: (B, N, 3)
    """
    B, N, _ = scene_pc.shape
    pc = scene_pc
    if N > max_scene_points:
        # 随机子采样（每 batch 不同子集需固定 seed 时可在训练循环外做）
        idx = torch.randperm(N, device=scene_pc.device)[:max_scene_points]
        pc = scene_pc[:, idx]
    t = pred_17d[:, :, 13:16]
    d = torch.cdist(t, pc)
    d_min = d.min(dim=-1).values
    return F.relu(margin - d_min).mean()


def quality_aux_stub(pred_17d: torch.Tensor) -> torch.Tensor:
    """quality_aux 关闭或占位：返回与 pred 同设备的标量 0。"""
    return pred_17d.new_zeros(())


def _nearest_gt_quality(
    pred_17d: torch.Tensor,
    gt_17d: torch.Tensor,
    *,
    center_sigma: float = 0.04,
    width_sigma: float = 0.02,
) -> torch.Tensor:
    """
    用最近 GT 的中心/宽度接近程度构造一个最小 quality proxy，范围约在 (0, 1]。
    pred_17d: (B, K, 17), gt_17d: (B, G, 17)
    """
    pred_center = pred_17d[:, :, 13:16]
    gt_center = gt_17d[:, :, 13:16]
    pred_width = pred_17d[:, :, 1:2]
    gt_width = gt_17d[:, :, 1:2]
    center_dist = torch.cdist(pred_center, gt_center)
    width_dist = torch.cdist(pred_width, gt_width)
    joint = center_dist / center_sigma + width_dist / width_sigma
    nn_joint, _ = joint.min(dim=-1)
    return torch.exp(-nn_joint)


def quality_score_aux(
    pred_17d: torch.Tensor,
    gt_17d: torch.Tensor,
    *,
    mode: str = "quality_mse",
    margin: float = 0.05,
    max_pairs: int = 256,
) -> torch.Tensor:
    """
    最小 quality-aware / ranking-aware 辅助：
    - quality_mse: 让 sigmoid(score) 拟合最近 GT quality proxy
    - margin_ranking: 对明显更好的 grasp 施加 pairwise 排序约束
    """
    pred_score = pred_17d[:, :, 0]
    quality = _nearest_gt_quality(pred_17d, gt_17d)
    if mode == "quality_mse":
        return F.mse_loss(torch.sigmoid(pred_score), quality)
    if mode != "margin_ranking":
        raise ValueError(f"未知 quality/ranking aux mode: {mode}")

    score_i = pred_score.unsqueeze(2)
    score_j = pred_score.unsqueeze(1)
    qual_i = quality.unsqueeze(2)
    qual_j = quality.unsqueeze(1)
    better = (qual_i - qual_j) > margin
    if not better.any():
        return pred_17d.new_zeros(())
    pair_loss = F.relu(margin - (score_i - score_j))
    chosen = pair_loss[better]
    if chosen.numel() > max_pairs:
        idx = torch.randperm(chosen.numel(), device=chosen.device)[:max_pairs]
        chosen = chosen[idx]
    return chosen.mean()


def compute_train_loss(
    *,
    model: torch.nn.Module,
    point_cloud: torch.Tensor,
    metas: Any,
    device: torch.device,
    loss_mode: str = "bidir",
    loss_alpha: float = 0.7,
    best_gt_weight: float = 0.3,
    pred2gt_agg: str = "min",
    max_grasps_decode: int = 128,
    sort_and_truncate_decode: bool = True,
    collision_aux: bool = False,
    collision_aux_weight: float = 0.1,
    collision_margin: float = 0.01,
    quality_aux: bool = False,
    quality_aux_weight: float = 0.0,
    quality_aux_mode: str = "quality_mse",
    ranking_aux: bool = False,
    ranking_aux_weight: float = 0.0,
    ranking_margin: float = 0.05,
    data_dir: Optional[str] = None,
    distill_weight: float = 1.0,
    distill_task_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    返回 (total_loss, log_dict)。
    VGGT / vggt_replacement 需传 data_dir 以便从 meta 加载 RGB。
    vggt_replacement_distill / lift3d_*_distill：distill_weight * distill_loss + distill_task_weight * main(+aux)。
    """
    images = None
    if model_requires_images(model):
        if not data_dir:
            raise ValueError("该模型需要 RGB，请传入 data_dir（offline_unified 根目录）")
        from utils.batch_images import load_images_batch

        images = load_images_batch(metas, data_dir, device)
    end_points = model(point_cloud=point_cloud, images=images)

    if _is_distill_model(model):
        l_d = end_points.get("distill_loss")
        if l_d is None:
            raise ValueError("distill 模式前向应设置 end_points['distill_loss']")
        total = distill_weight * l_d
        log: Dict[str, float] = {
            "loss_distill": float(l_d.detach().item()),
            "loss_main": 0.0,
            "loss_collision_aux": 0.0,
            "loss_quality_aux": 0.0,
            "loss_ranking_aux": 0.0,
        }
        if distill_task_weight > 0:
            pred_17d = pred_decode_17d_differentiable(
                end_points,
                device,
                max_grasps=max_grasps_decode,
                sort_and_truncate=sort_and_truncate_decode,
            )
            gt_17d = pad_gt_grasp_group_17d(metas, device)
            main = action_loss_topk_matched_17d(
                pred_17d,
                gt_17d,
                mode=loss_mode,
                alpha=loss_alpha,
                best_gt_weight=best_gt_weight,
                pred2gt_agg=pred2gt_agg,
                rank_weights=None,
            )
            total = total + distill_task_weight * main
            log["loss_main"] = float(main.detach().item())
            if collision_aux:
                l_col = collision_clearance_aux(pred_17d, point_cloud, margin=collision_margin)
                total = total + collision_aux_weight * l_col
                log["loss_collision_aux"] = float(l_col.detach().item())
            if quality_aux:
                l_q = quality_score_aux(pred_17d, gt_17d, mode=quality_aux_mode, margin=ranking_margin)
                total = total + quality_aux_weight * l_q
                log["loss_quality_aux"] = float(l_q.detach().item())
            if ranking_aux:
                l_rank = quality_score_aux(pred_17d, gt_17d, mode="margin_ranking", margin=ranking_margin)
                total = total + ranking_aux_weight * l_rank
                log["loss_ranking_aux"] = float(l_rank.detach().item())
        log["loss_total"] = float(total.detach().item())
        return total, log

    pred_17d = pred_decode_17d_differentiable(
        end_points,
        device,
        max_grasps=max_grasps_decode,
        sort_and_truncate=sort_and_truncate_decode,
    )
    gt_17d = pad_gt_grasp_group_17d(metas, device)
    main = action_loss_topk_matched_17d(
        pred_17d,
        gt_17d,
        mode=loss_mode,
        alpha=loss_alpha,
        best_gt_weight=best_gt_weight,
        pred2gt_agg=pred2gt_agg,
        rank_weights=None,
    )
    total = main
    log: Dict[str, float] = {"loss_main": float(main.detach().item())}

    if collision_aux:
        l_col = collision_clearance_aux(pred_17d, point_cloud, margin=collision_margin)
        total = total + collision_aux_weight * l_col
        log["loss_collision_aux"] = float(l_col.detach().item())
    else:
        log["loss_collision_aux"] = 0.0

    if quality_aux:
        l_q = quality_score_aux(pred_17d, gt_17d, mode=quality_aux_mode, margin=ranking_margin)
        total = total + quality_aux_weight * l_q
        log["loss_quality_aux"] = float(l_q.detach().item())
    else:
        log["loss_quality_aux"] = 0.0

    if ranking_aux:
        l_rank = quality_score_aux(pred_17d, gt_17d, mode="margin_ranking", margin=ranking_margin)
        total = total + ranking_aux_weight * l_rank
        log["loss_ranking_aux"] = float(l_rank.detach().item())
    else:
        log["loss_ranking_aux"] = 0.0

    log["loss_total"] = float(total.detach().item())
    return total, log


@torch.no_grad()
def evaluate_mean_val_loss(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    loss_mode: str = "bidir",
    loss_alpha: float = 0.7,
    best_gt_weight: float = 0.3,
    pred2gt_agg: str = "min",
    max_grasps: int = 128,
    data_dir: Optional[str] = None,
    distill_weight: float = 1.0,
    distill_task_weight: float = 0.0,
) -> float:
    """验证集 loss：默认 17D matching；distill 模式时为 distill 加权均值（与 task 组合时同 compute_train_loss）。"""
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        pcs, _, _, metas = batch
        pcs = pcs.to(device)
        gt_17d = pad_gt_grasp_group_17d(metas, device)
        images = None
        if model_requires_images(model):
            if not data_dir:
                raise ValueError("验证需要 data_dir 以加载 RGB")
            from utils.batch_images import load_images_batch

            images = load_images_batch(metas, data_dir, device)
        end_points = model(point_cloud=pcs, images=images)
        if _is_distill_model(model):
            l_d = end_points.get("distill_loss")
            if l_d is None:
                raise ValueError("distill 验证需 end_points['distill_loss']")
            loss_val = distill_weight * float(l_d.detach().item())
            if distill_task_weight > 0:
                pred_17d = pred_decode_17d(end_points, device, max_grasps=max_grasps)
                main = action_loss_topk_matched_17d(
                    pred_17d,
                    gt_17d,
                    mode=loss_mode,
                    alpha=loss_alpha,
                    best_gt_weight=best_gt_weight,
                    pred2gt_agg=pred2gt_agg,
                    rank_weights=None,
                )
                loss_val = loss_val + distill_task_weight * float(main.detach().item())
            total += loss_val
            n += 1
            continue
        pred_17d = pred_decode_17d(end_points, device, max_grasps=max_grasps)
        loss = action_loss_topk_matched_17d(
            pred_17d,
            gt_17d,
            mode=loss_mode,
            alpha=loss_alpha,
            best_gt_weight=best_gt_weight,
            pred2gt_agg=pred2gt_agg,
            rank_weights=None,
        )
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(n, 1)
