# -*- coding: utf-8 -*-
"""
Reranker 排序损失：在 top-K 可微候选上，基于与 GT 的近邻关系定义正负样本，pairwise hinge。
不修改 pred_decode / 碰撞 / dump；仅用于训练循环。

GT：先用 quality 阈值过滤，再按该标量降序取 top-K（默认 dim=3，与 17D 中 depth 列一致，可用 ranking_gt_quality_index 覆盖）。
正样本：最近邻距离 < pos_dist_thresh 且 最近 GT 的 quality > quality_thresh；hinge 按该 GT 的 quality 加权。
负样本：在 non-positive 候选中优先取 baseline_score 高者（难例）。
quality reranker：final = baseline * sigmoid(reranker 输出)。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from models.graspnet_adapter import per_seed_soft_scalars_from_end_points

logger = logging.getLogger(__name__)

# 17D: [0]=score [1]=w [2]=height [3]=depth — 默认用 depth 作 GT quality（与 pred 的 tolerance 槽位语义接近，可改）
DEFAULT_GT_QUALITY_INDEX = 3


def gather_topk_sorted_features(
    end_points: dict,
    top_k: int,
    *,
    detach_scalars: bool = False,
    extended_features: bool = True,
    normalize_center_by_scene: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    与 pred_decode_17d_differentiable(sort=True) 一致：按 score 降序取前 top_k。

    - extended_features=False：feat (B, K, 3)，与旧版一致。
    - extended_features=True：feat (B, K, 9) =
        [score, width, tol, cx, cy, cz, ax, ay, az]
      其中 center 可选按场景包围盒归一化；approach 为 -grasp_top_view_xyz 按 seed gather 后 L2 归一
     （与 graspnet_adapter 中可微解码的 approach 定义一致）。

    返回：
      feat：供 ResidualReranker；
      pred_center_raw：(B, K, 3) 未归一化中心，**仅用于**与 GT 算距离 / 正样本 mask；
      idx：(B, K) 选用的 seed 下标。
    """
    score_val, width_val, tol_val = per_seed_soft_scalars_from_end_points(end_points)
    if detach_scalars:
        score_val = score_val.detach()
        width_val = width_val.detach()
        tol_val = tol_val.detach()

    sort_idx = torch.argsort(score_val, dim=1, descending=True)
    k = min(top_k, score_val.shape[1])
    idx = sort_idx[:, :k]
    gs = lambda t: torch.gather(t, 1, idx)

    center = end_points["fp2_xyz"].float()
    pred_center_raw = torch.gather(center, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

    if not extended_features:
        feat = torch.stack([gs(score_val), gs(width_val), gs(tol_val)], dim=-1)
        if detach_scalars:
            feat = feat.detach()
        return feat, pred_center_raw, idx

    # ---- 9 维：几何特征缺失时回退为 3 维 + 零填充（仍要求 reranker.in_dim==9）----
    B = score_val.shape[0]
    device = score_val.device
    base3 = torch.stack([gs(score_val), gs(width_val), gs(tol_val)], dim=-1)

    has_approach = "grasp_top_view_xyz" in end_points
    if has_approach:
        approach_raw = -end_points["grasp_top_view_xyz"].float()
        pred_approach = torch.gather(approach_raw, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        pred_approach = F.normalize(pred_approach, dim=-1, eps=1e-8)
    else:
        logger.warning("[reranker] end_points 无 grasp_top_view_xyz，approach 置零")
        pred_approach = torch.zeros(B, k, 3, device=device, dtype=base3.dtype)

    center_feat = pred_center_raw
    if normalize_center_by_scene:
        pc = end_points.get("point_clouds")
        if pc is not None and pc.dim() == 3 and pc.shape[0] == B:
            bb_min = pc.min(dim=1)[0]
            bb_max = pc.max(dim=1)[0]
            c0 = (bb_min + bb_max) * 0.5
            scale = (bb_max - bb_min).max(dim=-1)[0].clamp(min=1e-4)
            center_feat = (pred_center_raw - c0.unsqueeze(1)) / scale.view(B, 1, 1)
        else:
            logger.debug("[reranker] 无 point_clouds 或形状不符，center 特征不归一化")

    if detach_scalars:
        center_feat = center_feat.detach()
        pred_approach = pred_approach.detach()

    feat = torch.cat([base3, center_feat, pred_approach], dim=-1)
    return feat, pred_center_raw, idx


def compute_fused_scores(
    baseline: torch.Tensor,
    residual: torch.Tensor,
    *,
    reranker_type: str,
    fusion: str,
    lam: float,
    bounded: bool = True,
) -> torch.Tensor:
    """
    baseline, residual: (B, K)，融合为最终排序分数 (B, K)。

    reranker_type == "quality"：
      final = baseline * sigmoid(residual)（与 baseline 尺度一致的可学习重加权）

    bounded=True（默认）：
      br = tanh(residual)，减轻对 baseline 分布的破坏。
      - ranking + add: baseline + lam * br
      - ranking + mul: baseline * (1 + lam * br)

    bounded=False（旧行为）：
      - ranking + add: baseline + lam * residual
      - ranking + mul: baseline * sigmoid(residual)
    """
    r = residual
    if r.dim() == 3:
        r = r.squeeze(-1)
    lam_f = float(lam)

    if reranker_type == "quality":
        return baseline * torch.sigmoid(r)

    if bounded:
        br = torch.tanh(r)
        if fusion == "add":
            return baseline + lam_f * br
        return baseline * (1.0 + lam_f * br)

    if fusion == "add":
        return baseline + lam_f * r
    return baseline * torch.sigmoid(r)


def filter_gt_for_ranking(
    gt_17d: torch.Tensor,
    *,
    quality_index: int,
    quality_thresh: float,
    gt_top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    先按 gt_quality > quality_thresh 过滤，再按 quality 降序取前 gt_top_k。
    返回 (gt_out, valid_gt_mask)，形状 (B, G', 17)、(B, G')。
    """
    B, G, _ = gt_17d.shape
    device = gt_17d.device
    valid_center = gt_17d[:, :, 13:16].abs().sum(dim=-1) > 1e-5
    q_all = gt_17d[:, :, quality_index]
    G_cap = min(G, int(gt_top_k))
    out = gt_17d.new_zeros(B, G_cap, 17)
    mask = torch.zeros(B, G_cap, dtype=torch.bool, device=device)
    for b in range(B):
        keep = valid_center[b] & (q_all[b] > float(quality_thresh))
        if not keep.any():
            continue
        idx = torch.where(keep)[0]
        q_sub = q_all[b, idx]
        order = torch.argsort(q_sub, descending=True)
        take_n = min(int(gt_top_k), int(idx.numel()))
        idx_sorted = idx[order[:take_n]]
        L = int(idx_sorted.numel())
        out[b, :L] = gt_17d[b, idx_sorted]
        mask[b, :L] = True
    return out, mask


def _positive_mask_and_weight_from_gt(
    pred_center: torch.Tensor,
    gt_17d: torch.Tensor,
    valid_gt_mask: torch.Tensor,
    pos_dist_thresh: float,
    quality_thresh: float,
    quality_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pred_center: (B, K, 3)
    gt_17d: (B, G, 17)
    valid_gt_mask: (B, G)
    正样本：最近邻距离 <= pos_dist_thresh 且 最近 GT 的 quality > quality_thresh。
    返回 pos_mask (B,K)、pos_weight (B,K)，权重为最近 GT 的 quality（非正为 0）。
    """
    gt_c = gt_17d[:, :, 13:16]
    d = torch.cdist(pred_center, gt_c)
    d = d.masked_fill(~valid_gt_mask.unsqueeze(1), float("inf"))
    d_min, g_star = d.min(dim=-1)
    g_star = g_star.clamp(0, gt_17d.size(1) - 1)
    qt = torch.gather(gt_17d[:, :, quality_index], 1, g_star)
    finite = torch.isfinite(d_min)
    pos = finite & (d_min <= float(pos_dist_thresh)) & (qt > float(quality_thresh))
    pos_w = torch.where(pos, qt, torch.zeros_like(qt))
    return pos, pos_w


def _zero_ranking_loss_with_grad(final_score: torch.Tensor, reranker: Optional[torch.nn.Module]) -> torch.Tensor:
    """
    无可采样 pair 时仍需返回与 reranker 相连的标量，否则 backward 报
    'element 0 of tensors does not require grad and does not have a grad_fn'。
    """
    z = final_score.sum() * 0.0
    if z.requires_grad or z.grad_fn is not None:
        return z
    if reranker is not None:
        for p in reranker.parameters():
            if p.requires_grad and p.numel() > 0:
                return p.flatten()[0] * 0.0
    return z


def pairwise_ranking_hinge_loss(
    final_score: torch.Tensor,
    pos_mask: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
    baseline_score: Optional[torch.Tensor] = None,
    neg_sample_strategy: str = "high_baseline",
    margin: float = 0.1,
    neg_samples_per_pos: int = 3,
    max_pairs: int = 2048,
    reranker: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    final_score: (B, K)
    pos_mask: (B, K) bool
    pos_weight: (B, K) 可选，与 hinge 相乘（通常取最近 GT 的 quality）
    baseline_score: (B, K) 可选，neg_sample_strategy=high_baseline 时用于挑选高分负样本
    对每个 (batch, pos, neg) 采样：L = w_pos * relu(margin - (s_pos - s_neg))
    """
    device = final_score.device
    B, K = final_score.shape
    # 与 final_score 同图上的零标量，避免「无 pair 时 new_zeros 断图」
    total_loss = final_score.sum() * 0.0
    pair_count = 0
    max_pairs = int(max_pairs)

    for b in range(B):
        pos_idx = torch.where(pos_mask[b])[0]
        neg_idx = torch.where(~pos_mask[b])[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        n_neg = min(neg_samples_per_pos, int(neg_idx.numel()))
        for pi in pos_idx:
            sp = final_score[b, pi]
            w_pi = pos_weight[b, pi] if pos_weight is not None else final_score.new_tensor(1.0)
            if n_neg <= 0:
                continue
            if (
                neg_sample_strategy == "high_baseline"
                and baseline_score is not None
            ):
                scores = baseline_score[b, neg_idx]
                order = torch.argsort(scores, descending=True)
                choice = neg_idx[order[:n_neg]]
            else:
                choice = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:n_neg]]
            for nj in choice:
                sn = final_score[b, nj]
                total_loss = total_loss + w_pi * F.relu(float(margin) - (sp - sn))
                pair_count += 1
                if pair_count >= max_pairs:
                    return total_loss / max(pair_count, 1)
    if pair_count == 0:
        return _zero_ranking_loss_with_grad(final_score, reranker)
    return total_loss / float(pair_count)


def reranker_ranking_loss_from_endpoints(
    end_points: dict,
    gt_17d: torch.Tensor,
    reranker: torch.nn.Module,
    *,
    top_k: int = 50,
    margin: float = 0.1,
    reranker_type: str = "ranking",
    fusion: str = "add",
    lam: float = 0.1,
    bounded: bool = True,
    extended_features: Optional[bool] = None,
    normalize_center_by_scene: bool = True,
    pos_dist_thresh: float = 0.05,
    neg_samples_per_pos: int = 3,
    max_pairs: int = 2048,
    detach_scalars_for_reranker: bool = False,
    valid_gt_mask: Optional[torch.Tensor] = None,
    quality_thresh: float = 0.02,
    gt_top_k: int = 100,
    gt_quality_index: int = DEFAULT_GT_QUALITY_INDEX,
    neg_sample_strategy: str = "high_baseline",
) -> Tuple[torch.Tensor, dict]:
    """
    完整 reranker 排序损失。返回 (loss, log_dict)。
    extended_features=None：按 reranker.in_dim 自动选择（3 或 9）。
    detach_scalars_for_reranker=True：标量与几何特征 detach（reranker_only 推荐）。
    """
    in_dim = int(getattr(reranker, "in_dim", 3))
    if extended_features is None:
        extended_features = in_dim >= 9
    if in_dim == 3:
        extended_features = False
    if extended_features and in_dim < 9:
        raise ValueError("extended_features=True 需要 ResidualReranker(in_dim>=9)")

    gt_work, valid_gt = filter_gt_for_ranking(
        gt_17d,
        quality_index=int(gt_quality_index),
        quality_thresh=float(quality_thresh),
        gt_top_k=int(gt_top_k),
    )
    _ = valid_gt_mask  # 保留旧参数；正样本/GT 掩码由 filter_gt_for_ranking 与 valid_gt 决定

    feat, pred_center_raw, _ = gather_topk_sorted_features(
        end_points,
        top_k,
        detach_scalars=detach_scalars_for_reranker,
        extended_features=extended_features,
        normalize_center_by_scene=normalize_center_by_scene,
    )

    if (not extended_features) and in_dim >= 9:
        z = feat.new_zeros(feat.shape[0], feat.shape[1], in_dim - feat.shape[-1])
        feat = torch.cat([feat, z], dim=-1)

    baseline = feat[:, :, 0]
    residual = reranker(feat)
    if residual.dim() == 2:
        residual = residual.unsqueeze(-1)
    r = residual.squeeze(-1)

    final_score = compute_fused_scores(
        baseline,
        r,
        reranker_type=reranker_type,
        fusion=fusion,
        lam=lam,
        bounded=bounded,
    )
    pos_mask, pos_w = _positive_mask_and_weight_from_gt(
        pred_center_raw,
        gt_work,
        valid_gt,
        pos_dist_thresh,
        quality_thresh,
        int(gt_quality_index),
    )
    loss = pairwise_ranking_hinge_loss(
        final_score,
        pos_mask,
        pos_weight=pos_w,
        baseline_score=baseline,
        neg_sample_strategy=neg_sample_strategy,
        margin=margin,
        neg_samples_per_pos=neg_samples_per_pos,
        max_pairs=max_pairs,
        reranker=reranker,
    )
    n_pos = int(pos_mask.sum().item())
    log = {
        "loss_ranking": float(loss.detach().item()),
        "ranking_pos_count": float(n_pos),
    }
    return loss, log
