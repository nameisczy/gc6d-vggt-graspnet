# -*- coding: utf-8 -*-
"""
Action 回归 loss：支持单 GT、多 GT、top-K 预测 + matching（Hungarian / 最近邻）。

当前 loss 计算（action_loss_topk_matched）：
- cost = _mse_per_pair(pred_k, gt_multi) → (B,K,M)，cost[b,k,m] = MSE(pred[b,k], gt[b,m])。
- mode="bidir"（默认）：双向最近邻。
  (A) 预测→GT：每个 pred 找最近 GT，min_m cost → (B,K)，对 K 求平均 → loss_pred2gt (B,)；
  (B) GT→预测：每个 GT 找最近 pred，min_k cost → (B,M)，对 M 求平均 → loss_gt2pred (B,)；
  每样本 loss = alpha * loss_pred2gt + (1-alpha) * loss_gt2pred，推荐 alpha=0.7。

支持：
- return_components=True：返回 mse_t / mse_rot6d / mse_w（对 matched GT 的平均），便于排查哪一维卡住。
- action_weights=(w_t, w_rot, w_w)：对 t(3)/rot6d(6)/width(1) 加权，避免 rotation 主导。
- use_smooth_l1：用 SmoothL1 替代 MSE，对噪声/多解更稳。
- pred2gt_top_frac：只对 cost 最小的前 frac 的 pred 反传（如 0.25）。
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# action 10 维: t(0:3), rot6d(3:9), width(9:10)
T_SLICE = slice(0, 3)
R6_SLICE = slice(3, 9)
W_SLICE = slice(9, 10)

# 17D: [0]=score, [1]=w, [2]=height, [3]=depth, [4:13]=R(9), [13:16]=t, [16]=object_id
D17_W, D17_R, D17_T = 1, slice(4, 13), slice(13, 16)
EPS = 1e-8


def _r6_to_R_torch(r6: torch.Tensor) -> torch.Tensor:
    """R6d (..., 6) -> 正交阵 R (..., 3, 3)。与 action2grasp.r6_to_R 一致，可微。"""
    c1 = r6[..., 0:3]
    c2 = r6[..., 3:6]
    c1 = c1 / (c1.norm(dim=-1, keepdim=True) + EPS)
    c2 = c2 - (c1 * (c1 * c2).sum(dim=-1, keepdim=True))
    c2 = c2 / (c2.norm(dim=-1, keepdim=True) + EPS)
    c3 = torch.linalg.cross(c1, c2, dim=-1)
    c3 = c3 / (c3.norm(dim=-1, keepdim=True) + EPS)
    return torch.stack([c1, c2, c3], dim=-1)


def proposals_11d_to_17d_torch(
    pred_11: torch.Tensor,
    height: float = 0.02,
    depth: float = 0.04,
    width_min: float = 0.01,
    width_max: float = 0.12,
) -> torch.Tensor:
    """
    pred_11: (B, K, 11) [t(3), R6d(6), w(1), score(1)]，可微 11D→17D 与 eval 一致。
    返回 (B, K, 17)，R 行优先，height/depth 固定，object_id=0。
    """
    B, K, _ = pred_11.shape
    t_raw = pred_11[:, :, 0:3]
    t = torch.stack([
        t_raw[..., 0].clamp(-0.5, 0.5),
        t_raw[..., 1].clamp(-0.5, 0.5),
        t_raw[..., 2].clamp(0.0, 1.5),
    ], dim=-1)
    r6 = pred_11[:, :, 3:9]
    w = pred_11[:, :, 9:10].clamp(width_min, width_max)
    score = pred_11[:, :, 10:11]
    R = _r6_to_R_torch(r6)
    R_flat = R.reshape(B, K, 9)
    out = pred_11.new_zeros(B, K, 17)
    out[:, :, 0:1] = score
    out[:, :, 1:2] = w
    out[:, :, 2] = height
    out[:, :, 3] = depth
    out[:, :, 4:13] = R_flat
    out[:, :, 13:16] = t
    out[:, :, 16] = 0
    return out


def proposals_10d_to_17d_torch(
    pred_10: torch.Tensor,
    height: float = 0.02,
    depth: float = 0.04,
    width_min: float = 0.01,
    width_max: float = 0.12,
) -> torch.Tensor:
    """
    MLP head 等输出 (B, K, 10) [t(3), R6d(6), w(1)] 可微转 17D，与 11D 路径一致，score 置 0。
    """
    B, K, _ = pred_10.shape
    t_raw = pred_10[:, :, 0:3]
    t = torch.stack([
        t_raw[:, :, 0].clamp(-0.5, 0.5),
        t_raw[:, :, 1].clamp(-0.5, 0.5),
        t_raw[:, :, 2].clamp(0.0, 1.5),
    ], dim=-1)
    r6 = pred_10[:, :, 3:9]
    w = pred_10[:, :, 9:10].clamp(width_min, width_max)
    R = _r6_to_R_torch(r6)
    R_flat = R.reshape(B, K, 9)
    out = pred_10.new_zeros(B, K, 17)
    out[:, :, 0:1] = 0.0
    out[:, :, 1:2] = w
    out[:, :, 2] = height
    out[:, :, 3] = depth
    out[:, :, 4:13] = R_flat
    out[:, :, 13:16] = t
    out[:, :, 16] = 0
    return out


def _proposals_to_17d_torch(
    pred: torch.Tensor,
    height: float = 0.02,
    depth: float = 0.04,
    width_min: float = 0.01,
    width_max: float = 0.12,
) -> torch.Tensor:
    """(B, K, 10)、(B, K, 11) 或 (B, K, 17) 统一为 (B, K, 17)。17D 时直接返回。"""
    if pred.shape[-1] == 17:
        return pred
    if pred.shape[-1] == 11:
        return proposals_11d_to_17d_torch(pred, height=height, depth=depth, width_min=width_min, width_max=width_max)
    return proposals_10d_to_17d_torch(pred, height=height, depth=depth, width_min=width_min, width_max=width_max)


def pad_gt_grasp_group_17d(metas: list, device: torch.device) -> torch.Tensor:
    """
    从 batch 的 metas 中收集 gt_grasp_group，pad 成 (B, max_M, 17)。
    无 gt_grasp_group 的样本用首行 0 填（loss 时需配合 gt_primary 或仅用有 GT 的样本）。
    """
    list_gg = []
    for m in metas:
        gg = m.get("gt_grasp_group")
        if gg is not None:
            if not isinstance(gg, torch.Tensor):
                gg = torch.from_numpy(np.asarray(gg, dtype=np.float32))
            list_gg.append(gg)
        else:
            list_gg.append(None)
    max_m = 1
    for gg in list_gg:
        if gg is not None and gg.dim() >= 2 and gg.shape[0] > 0:
            max_m = max(max_m, gg.shape[0])
    B = len(metas)
    out = torch.zeros(B, max_m, 17, dtype=torch.float32, device=device)
    for i in range(B):
        gg = list_gg[i] if i < len(list_gg) else None
        if gg is not None and gg.numel() >= 17:
            if gg.dim() == 1:
                gg = gg.unsqueeze(0)
            gg = gg.to(device)
            m = min(gg.shape[0], max_m)
            out[i, :m] = gg[:m]
    return out


def _cost_components_per_pair(
    pred: torch.Tensor,
    gt: torch.Tensor,
    use_smooth_l1: bool = False,
) -> tuple:
    """pred (B,K,10), gt (B,M,10) -> cost_t, cost_rot, cost_w 各 (B,K,M)，每块内先 MSE/SmoothL1 再 mean。"""
    pred_exp = pred.unsqueeze(2).expand(-1, -1, gt.shape[1], -1)
    gt_exp = gt.unsqueeze(1).expand(-1, pred.shape[1], -1, -1)
    if use_smooth_l1:
        raw = F.smooth_l1_loss(pred_exp, gt_exp, reduction="none")
    else:
        raw = (pred_exp - gt_exp) ** 2
    cost_t = raw[:, :, :, T_SLICE].mean(dim=3)
    cost_rot = raw[:, :, :, R6_SLICE].mean(dim=3)
    cost_w = raw[:, :, :, W_SLICE].mean(dim=3)
    return cost_t, cost_rot, cost_w


def _mse_per_pair(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred (B,K,10), gt (B,M,10) -> (B,K,M) per-element MSE then mean over dim 10."""
    pred_exp = pred.unsqueeze(2).expand(-1, -1, gt.shape[1], -1)
    gt_exp = gt.unsqueeze(1).expand(-1, pred.shape[1], -1, -1)
    return F.mse_loss(pred_exp, gt_exp, reduction="none").mean(dim=3)


def action_loss_topk_matched(
    pred_k: torch.Tensor,
    gt_primary: torch.Tensor,
    gt_multi: torch.Tensor,
    mode: str = "bidir",
    reduction: str = "mean",
    alpha: float = 0.7,
    action_weights: tuple = None,
    return_components: bool = False,
    use_smooth_l1: bool = False,
    pred2gt_top_frac: float = 1.0,
    best_gt_weight: float = 0.0,
):
    """
    pred_k: (B, K, 10) 模型预测的 top-K grasps
    gt_primary: (B, 10) 主 GT
    gt_multi: (B, M, 10) 多 GT（由 pad_actions_multi 得到）
    mode: "bidir" = 双向最近邻（默认）；"min" = 仅 GT→预测；"hungarian" = 一对一匹配
    alpha: mode="bidir" 时，loss = alpha * (预测→GT) + (1-alpha) * (GT→预测)，推荐 0.7
    action_weights: (w_t, w_rot, w_w) 对 t/rot6d/width 加权，如 (1.0, 0.2, 0.5)；None=等权
    return_components: 若 True，返回 (loss, {"mse_t", "mse_rot6d", "mse_w"})，便于排查哪一维卡住
    use_smooth_l1: 用 SmoothL1 替代 MSE
    pred2gt_top_frac: 预测→GT 分支只对 cost 最小的前 frac 的 pred 反传，如 0.25；1.0=全部
    best_gt_weight: 若>0，加一项「至少一个 pred 逼近主 GT」min_k MSE(pred_k, gt_primary)，缓解多 GT 时 loss 停滞；推荐 0.2~0.4。
    """
    B, K, _ = pred_k.shape
    M = gt_multi.shape[1]
    need_components = return_components or action_weights is not None
    if need_components or use_smooth_l1:
        cost_t, cost_rot, cost_w = _cost_components_per_pair(pred_k, gt_multi, use_smooth_l1=use_smooth_l1)
        if action_weights is not None:
            w_t, w_rot, w_w = action_weights
            cost = w_t * cost_t + w_rot * cost_rot + w_w * cost_w
        else:
            cost = (cost_t * 3 + cost_rot * 6 + cost_w * 1) / 10.0
    else:
        cost = _mse_per_pair(pred_k, gt_multi)
        cost_t = cost_rot = cost_w = None

    def _pred2gt_loss(c: torch.Tensor) -> torch.Tensor:
        # (B, K) 每个 pred 到最近 GT 的 cost
        cost_min = c.min(dim=2).values
        if pred2gt_top_frac >= 1.0:
            return cost_min.mean(dim=1)
        k_keep = max(1, int(K * pred2gt_top_frac))
        # 每个样本取 cost 最小的 k_keep 个 pred 的 mean（可微）
        idx = cost_min.argsort(dim=1)[:, :k_keep]
        selected = cost_min.gather(1, idx)
        return selected.mean(dim=1)

    if mode == "bidir":
        loss_pred2gt = _pred2gt_loss(cost)
        loss_gt2pred = cost.min(dim=1).values.mean(dim=1)
        loss_per_sample = alpha * loss_pred2gt + (1.0 - alpha) * loss_gt2pred
    elif mode == "min":
        loss_per_gt = cost.min(dim=1).values
        loss_per_sample = loss_per_gt.mean(dim=1)
    else:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("mode='hungarian' requires scipy. Install with: pip install scipy")
        device = pred_k.device
        loss_list = []
        for b in range(B):
            C = cost[b].detach().cpu().numpy()
            k_idx, m_idx = linear_sum_assignment(C)
            n_pairs = max(1, len(k_idx))
            assign = torch.zeros(K, M, dtype=pred_k.dtype, device=device)
            for i in range(len(k_idx)):
                assign[k_idx[i], m_idx[i]] = 1.0
            loss_list.append((cost[b] * assign).sum() / n_pairs)
        loss_per_sample = torch.stack(loss_list, dim=0)

    if reduction == "mean":
        loss_val = loss_per_sample.mean()
    else:
        loss_val = loss_per_sample.sum()

    if best_gt_weight > 0 and gt_primary is not None:
        diff = (pred_k - gt_primary.unsqueeze(1)) ** 2
        mse_per_k = diff.mean(dim=2)
        loss_best = mse_per_k.min(dim=1).values
        if reduction == "mean":
            loss_best = loss_best.mean()
        else:
            loss_best = loss_best.sum()
        loss_val = (1.0 - best_gt_weight) * loss_val + best_gt_weight * loss_best

    if not return_components:
        return loss_val
    # 对 matched 的 pred→gt 打印分量：每个 (b,k) 取 m_star = argmin_m cost[b,k,m]，再对 cost_t/rot/w 取平均
    if cost_t is None:
        cost_t, cost_rot, cost_w = _cost_components_per_pair(pred_k, gt_multi, use_smooth_l1=False)
    m_star = cost.min(dim=2).indices
    ct = cost_t.gather(2, m_star.unsqueeze(2)).squeeze(2).mean().item()
    cr = cost_rot.gather(2, m_star.unsqueeze(2)).squeeze(2).mean().item()
    cw = cost_w.gather(2, m_star.unsqueeze(2)).squeeze(2).mean().item()
    return loss_val, {"mse_t": ct, "mse_rot6d": cr, "mse_w": cw}


def action_loss_multi_gt(
    pred: torch.Tensor,
    gt_primary: torch.Tensor,
    gt_multi: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    pred: (B, 10)
    gt_primary: (B, 10)，主 GT（每个样本至少有一个）
    gt_multi: (B, M, 10) 或 None。若提供，每个样本取 min over M 的 MSE，再 mean over B。
    reduction: "mean" | "sum"
    """
    if gt_multi is None:
        return F.mse_loss(pred, gt_primary, reduction=reduction)
    # (B, M, 10) -> per-sample per-gt MSE -> (B, M)
    B, M, _ = gt_multi.shape
    pred_exp = pred.unsqueeze(1).expand(-1, M, -1)  # (B, M, 10)
    mse_per_gt = F.mse_loss(pred_exp, gt_multi, reduction="none").mean(dim=2)  # (B, M)
    loss_per_sample = mse_per_gt.min(dim=1).values  # (B,) min over gt
    if reduction == "mean":
        return loss_per_sample.mean()
    return loss_per_sample.sum()


def _cost_17d_per_pair(pred_17: torch.Tensor, gt_17: torch.Tensor) -> torch.Tensor:
    """pred_17 (B,K,17), gt_17 (B,M,17) -> (B,K,M) MSE 仅在 w,R,t 上（dims 1,4:13,13:16）。"""
    pred_exp = pred_17.unsqueeze(2).expand(-1, -1, gt_17.shape[1], -1)
    gt_exp = gt_17.unsqueeze(1).expand(-1, pred_17.shape[1], -1, -1)
    idx = [1] + list(range(4, 13)) + list(range(13, 16))
    diff = (pred_exp - gt_exp)[..., idx]
    return (diff ** 2).mean(dim=-1)


def action_loss_topk_matched_17d(
    pred_k: torch.Tensor,
    gt_multi_17d: torch.Tensor,
    mode: str = "bidir",
    reduction: str = "mean",
    alpha: float = 0.7,
    pred2gt_top_frac: float = 1.0,
    best_gt_weight: float = 0.0,
    height: float = 0.02,
    depth: float = 0.04,
    pred2gt_agg: str = "min",
    rank_weights: Optional[torch.Tensor] = None,
):
    """
    在 17D 空间算 loss：pred (B,K,10)、(B,K,11) 或 (B,K,17)。17D 时直接用 pred 作为 pred_17。
    与 GT (B,M,17) 在 w,R,t 上做 matching。best_gt 用 gt_multi_17d[:, 0] 作为主 GT。

    pred2gt_agg: 预测→GT 分支对 K 个预测的聚合。"mean"=对 K 个 pred 的 cost 取平均（需全部预测都好
    才可接近 0，过拟合单样本时难以下降）；"min"=取最优一个 pred 的 cost（有一个预测匹配即可接近 0）。
    """
    if pred_k.shape[-1] == 17:
        pred_17 = pred_k
    else:
        pred_17 = _proposals_to_17d_torch(pred_k, height=height, depth=depth)
    B, K, _ = pred_k.shape
    M = gt_multi_17d.shape[1]
    cost = _cost_17d_per_pair(pred_17, gt_multi_17d)

    def _pred2gt_loss(c: torch.Tensor) -> torch.Tensor:
        cost_min = c.min(dim=2).values  # (B, K)
        if rank_weights is not None:
            # rank_weights: (K,) 或 (1,K)，前排权重大，加权聚合
            w = rank_weights.to(cost_min.device).reshape(1, -1)
            w = w[:, : cost_min.shape[1]]
            return (cost_min * w).sum(dim=1) / (w.sum(dim=1) + EPS)
        if pred2gt_top_frac >= 1.0:
            if pred2gt_agg == "min":
                return cost_min.min(dim=1).values
            return cost_min.mean(dim=1)
        k_keep = max(1, int(K * pred2gt_top_frac))
        idx = cost_min.argsort(dim=1)[:, :k_keep]
        selected = cost_min.gather(1, idx)
        return selected.mean(dim=1)

    if mode == "bidir":
        loss_pred2gt = _pred2gt_loss(cost)
        loss_gt2pred = cost.min(dim=1).values.mean(dim=1)
        loss_per_sample = alpha * loss_pred2gt + (1.0 - alpha) * loss_gt2pred
    elif mode == "min":
        loss_per_gt = cost.min(dim=1).values
        loss_per_sample = loss_per_gt.mean(dim=1)
    else:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("mode='hungarian' requires scipy")
        device = pred_k.device
        loss_list = []
        for b in range(B):
            C = cost[b].detach().cpu().numpy()
            k_idx, m_idx = linear_sum_assignment(C)
            assign = torch.zeros_like(cost[b], device=device)
            for ki, mi in zip(k_idx, m_idx):
                assign[ki, mi] = 1.0
            n_pairs = max(1, len(k_idx))
            loss_list.append((cost[b] * assign).sum() / n_pairs)
        loss_per_sample = torch.stack(loss_list, dim=0)

    if reduction == "mean":
        loss_val = loss_per_sample.mean()
    else:
        loss_val = loss_per_sample.sum()

    if best_gt_weight > 0 and gt_multi_17d.shape[1] > 0:
        gt_primary_17d = gt_multi_17d[:, 0]
        pred_17_flat = pred_17.reshape(B * K, 17).to(gt_primary_17d.dtype)
        gt_rep = gt_primary_17d.unsqueeze(1).expand(-1, K, -1).reshape(B * K, 17)
        idx = [1] + list(range(4, 13)) + list(range(13, 16))
        diff = (pred_17_flat - gt_rep)[:, idx]
        mse_per_k = (diff ** 2).mean(dim=1).reshape(B, K)
        loss_best = mse_per_k.min(dim=1).values.mean()
        loss_val = (1.0 - best_gt_weight) * loss_val + best_gt_weight * loss_best

    return loss_val


def pad_actions_multi(
    metas: list,
    gt_primary: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    从 batch 的 metas 中收集 actions_multi，pad 成 (B, max_M, 10)。
    无 actions_multi 的样本用 gt_primary 填一行，min over M 即等价于单 GT。
    gt_primary: (B, 10)
    """
    list_multi = [m.get("actions_multi") for m in metas]
    max_m = 1
    for am in list_multi:
        if am is not None and am.dim() >= 2 and am.shape[0] > 0:
            max_m = max(max_m, am.shape[0])
    B = gt_primary.shape[0]
    out = torch.zeros(B, max_m, 10, dtype=torch.float32, device=device)
    gt_primary = gt_primary.to(device)
    for i in range(B):
        am = list_multi[i] if i < len(list_multi) else None
        if am is not None and am.numel() >= 10:
            if am.dim() == 1:
                am = am.unsqueeze(0)
            am = am.to(device)
            m = min(am.shape[0], max_m)
            out[i, :m] = am[:m]
        else:
            out[i, 0] = gt_primary[i]
    return out


def ranking_align_loss(score_train: torch.Tensor, score_bench: torch.Tensor) -> torch.Tensor:
    """
    可微排序对齐 loss：让 score_train 的排序逼近 score_bench（用 1 - Pearson 相关）。
    score_train, score_bench: (B, K)，训练 decode 与 benchmark decode 的 score。
    返回标量，越小表示两路 score 排序越一致。
    """
    st = score_train.reshape(-1)
    sb = score_bench.reshape(-1).detach()
    st_c = st - st.mean()
    sb_c = sb - sb.mean()
    cov = (st_c * sb_c).sum()
    std_t = (st_c.pow(2).sum() + EPS).sqrt()
    std_b = (sb_c.pow(2).sum() + EPS).sqrt()
    corr = cov / (std_t * std_b + EPS)
    return (1.0 - corr).clamp(0.0, 2.0)
