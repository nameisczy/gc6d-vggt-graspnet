# -*- coding: utf-8 -*-
"""
Reranker 排序损失（baseline-aware、保守）：
- Top-K **仅**由 baseline（seed score）决定：``torch.topk(baseline, K)``，候选集不因 reranker 改变。
- 融合（ranking + add）：``final = baseline + lambda * residual``（小 lambda，避免整表重排）。
- 正样本：在 Top-K 内 **仅** 按 **高 baseline**（前 top_frac）定义，**不**用 GT 距离约束正样本。
- 可选软约束：若某候选与 GT 平移距离 < 阈值，则对该正样本 hinge 加权略增（仍不强制贴 GT）。
- 负样本：**难负例** — baseline 排名在 (top 20%, top 60%] 内；忽略最差 40%。
- 稳定项：``L_total = L_ranking + stability_weight * MSE(final, baseline)``。
- quality reranker：``final = baseline * sigmoid(residual)``（与 ranking 路径独立）。

GT 平移与 eval 一致：``[:, 13:16]``。
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from models.graspnet_adapter import per_seed_soft_scalars_from_end_points

logger = logging.getLogger(__name__)

# 17D: [0]=score [1]=w [2]=height [3]=depth — 保留旧参数名兼容（当前正/负定义不再用 quality 维）
DEFAULT_GT_QUALITY_INDEX = 3

# 防止分数漂移：MSE(final, baseline) 权重
RERANK_STABILITY_WEIGHT = 0.01

# 正样本在「近 GT」时的 hinge 额外权重（仅当 near_gt 时生效；0 表示关闭）
RANKING_NEAR_GT_SOFT_BONUS = 0.25

# 负样本：baseline 排名在 (pos_top_frac, neg_top_frac] 内（忽略最差 40% 作负例）
RANKING_NEG_TOP_FRAC_DEFAULT = 0.6


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

    # 仅用 baseline（seed score）选 Top-K；与 eval 一致，reranker 不改变候选集合
    k = min(int(top_k), int(score_val.shape[1]))
    _, idx = torch.topk(score_val, k, dim=-1, largest=True, sorted=True)
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

    reranker_type == "ranking" + fusion == "add"（训练默认，保守）：
      final = baseline + lam * residual（小 lam，与 eval 对齐、避免整表重排）

    reranker_type == "quality"：
      final = baseline * sigmoid(residual)

    ranking + mul：仍可用 tanh 有界残差（bounded）或旧式 sigmoid（unbounded）。
    """
    r = residual
    if r.dim() == 3:
        r = r.squeeze(-1)
    lam_f = float(lam)

    if reranker_type == "quality":
        return baseline * torch.sigmoid(r)

    # ranking：add 路径固定为 baseline + lambda * residual（与「小 lambda」保守策略一致）
    if reranker_type == "ranking" and fusion == "add":
        return baseline + lam_f * r

    if bounded:
        br = torch.tanh(r)
        if fusion == "add":
            return baseline + lam_f * br
        return baseline * (1.0 + lam_f * br)

    if fusion == "add":
        return baseline + lam_f * r
    return baseline * torch.sigmoid(r)


def _baseline_aware_pos_neg_masks(
    pred_center: torch.Tensor,
    gt_17d: torch.Tensor,
    baseline_topk: torch.Tensor,
    pos_dist_thresh: float,
    *,
    baseline_top_frac: float = 0.2,
    neg_top_frac: float = RANKING_NEG_TOP_FRAC_DEFAULT,
    near_gt_soft_bonus: float = RANKING_NEAR_GT_SOFT_BONUS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred_center: (B, K, 3)，已与 Top-K baseline 候选对齐。
    baseline_topk: (B, K)，对应位置的 baseline 分数。
    按 **baseline 排名**（每行按分数降序，rank 0 为最高）：
    - 正样本：rank < ceil(K * pos_top_frac)（默认 top 20%）
    - 负样本：pos_top_frac < rank <= neg_top_frac（默认 rank 严格大于 20% 且 <= top 60%），
      即 **忽略** 排名最差 (1 - neg_top_frac)（默认 40%）作为负例，专注难负例。

    平移 GT [:, 13:16] 仅用于可选 soft 权重（近 GT 略增权），**不**参与正负定义。

    返回 pos_mask (B,K)、neg_mask (B,K)、pos_pair_weight (B,K)：
    pos_pair_weight 在正样本位置上为 ``1 + bonus``（若 near GT）否则 ``1``。
    """
    valid_g = gt_17d[:, :, 13:16].abs().sum(dim=-1) > 1e-5
    gt_c = gt_17d[:, :, 13:16]
    d = torch.cdist(pred_center, gt_c)
    d = d.masked_fill(~valid_g.unsqueeze(1), float("inf"))
    d_min, _ = d.min(dim=-1)
    near_gt = torch.isfinite(d_min) & (d_min <= float(pos_dist_thresh))

    B, K = baseline_topk.shape
    device = baseline_topk.device
    sorted_idx = torch.argsort(baseline_topk, dim=-1, descending=True)
    ranks = torch.arange(K, device=device, dtype=torch.long).view(1, K).expand(B, -1)
    rank = torch.zeros(B, K, dtype=torch.long, device=device)
    rank.scatter_(1, sorted_idx, ranks)

    n_pos = max(1, int(math.ceil(float(K) * float(baseline_top_frac))))
    n_neg_cap = max(n_pos, int(math.ceil(float(K) * float(neg_top_frac))))

    pos_mask = rank < n_pos
    neg_mask = (rank >= n_pos) & (rank < n_neg_cap)

    bonus = float(near_gt_soft_bonus)
    pos_pair_weight = torch.ones_like(baseline_topk, dtype=baseline_topk.dtype)
    if bonus > 0.0:
        pos_pair_weight = pos_pair_weight + bonus * (pos_mask & near_gt).to(dtype=baseline_topk.dtype)
    return pos_mask, neg_mask, pos_pair_weight


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
    neg_mask: torch.Tensor,
    *,
    pos_pair_weight: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    neg_samples_per_pos: int = 5,
    max_pairs: int = 2048,
    reranker: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    final_score: (B, K)
    pos_mask / neg_mask: (B, K) bool，负样本仅在 neg_mask 为 True 的下标上采样。
    pos_pair_weight: (B, K) 可选，对正样本侧 hinge 加权（默认全 1）。
    L = w_pos * relu(margin - (s_pos - s_neg))
    """
    device = final_score.device
    B, K = final_score.shape
    # 与 final_score 同图上的零标量，避免「无 pair 时 new_zeros 断图」
    total_loss = final_score.sum() * 0.0
    pair_count = 0
    max_pairs = int(max_pairs)

    for b in range(B):
        pos_idx = torch.where(pos_mask[b])[0]
        neg_idx = torch.where(neg_mask[b])[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        n_neg = min(neg_samples_per_pos, int(neg_idx.numel()))
        for pi in pos_idx:
            sp = final_score[b, pi]
            w_pi = (
                pos_pair_weight[b, pi]
                if pos_pair_weight is not None
                else final_score.new_tensor(1.0)
            )
            if n_neg <= 0:
                continue
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
    margin: float = 0.05,
    reranker_type: str = "ranking",
    fusion: str = "add",
    lam: float = 0.01,
    bounded: bool = True,
    extended_features: Optional[bool] = None,
    normalize_center_by_scene: bool = True,
    pos_dist_thresh: float = 0.05,
    neg_samples_per_pos: int = 5,
    max_pairs: int = 2048,
    detach_scalars_for_reranker: bool = False,
    valid_gt_mask: Optional[torch.Tensor] = None,
    quality_thresh: float = 0.02,
    gt_top_k: int = 100,
    gt_quality_index: int = DEFAULT_GT_QUALITY_INDEX,
    neg_sample_strategy: str = "high_baseline",
    ranking_baseline_top_frac: float = 0.2,
    ranking_neg_top_frac: float = RANKING_NEG_TOP_FRAC_DEFAULT,
    ranking_near_gt_soft_bonus: float = RANKING_NEAR_GT_SOFT_BONUS,
    stability_weight: float = RERANK_STABILITY_WEIGHT,
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
    pos_mask, neg_mask, pos_pair_weight = _baseline_aware_pos_neg_masks(
        pred_center_raw,
        gt_17d,
        baseline,
        pos_dist_thresh,
        baseline_top_frac=float(ranking_baseline_top_frac),
        neg_top_frac=float(ranking_neg_top_frac),
        near_gt_soft_bonus=float(ranking_near_gt_soft_bonus),
    )
    loss_rank = pairwise_ranking_hinge_loss(
        final_score,
        pos_mask,
        neg_mask,
        pos_pair_weight=pos_pair_weight,
        margin=margin,
        neg_samples_per_pos=neg_samples_per_pos,
        max_pairs=max_pairs,
        reranker=reranker,
    )
    loss_stab = F.mse_loss(final_score, baseline)
    loss = loss_rank + float(stability_weight) * loss_stab
    n_pos = int(pos_mask.sum().item())
    log = {
        "loss_ranking": float(loss_rank.detach().item()),
        "loss_ranking_stability": float(loss_stab.detach().item()),
        "ranking_pos_count": float(n_pos),
    }
    return loss, log
