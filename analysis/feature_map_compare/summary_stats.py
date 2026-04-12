# -*- coding: utf-8 -*-
"""标量摘要：范数统计、top-k 重叠、可选 GT 接触距离。"""

from __future__ import annotations

import csv
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def l2_norm_per_point(feat: np.ndarray) -> np.ndarray:
    """(N, D) -> (N,)"""
    return np.linalg.norm(feat, axis=1)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    order = np.argsort(-scores[idx])
    return idx[order]


def overlap_ratio_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """a,b: 点索引集合 overlap |A∩B|/k"""
    k = min(k, len(a), len(b))
    if k == 0:
        return float("nan")
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / float(k)


def pca_first_component(feat: np.ndarray) -> np.ndarray:
    """(N,D) 中心化后第一主成分投影值 (N,)"""
    x = np.asarray(feat, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    pc1 = u[:, 0]
    return pc1


def summarize_models(
    model_feats: Dict[str, np.ndarray],
    *,
    topk: int = 64,
) -> Dict[str, Any]:
    """model_feats: name -> (N, D) numpy"""
    stats: Dict[str, Any] = {}
    topk_map: Dict[str, np.ndarray] = {}
    for name, f in model_feats.items():
        fn = l2_norm_per_point(f)
        stats[name] = {
            "l2_norm_mean": float(np.mean(fn)),
            "l2_norm_std": float(np.std(fn)),
        }
        topk_map[name] = topk_indices(fn, topk)
    pairs: List[Tuple[str, str, float]] = []
    names = list(model_feats.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            o = overlap_ratio_topk(topk_map[ni], topk_map[nj], topk)
            pairs.append((ni, nj, o))
    return {
        "per_model": stats,
        "top_k": topk,
        "topk_indices": {k: v.tolist() for k, v in topk_map.items()},
        "pairwise_topk_overlap": [{"a": a, "b": b, "overlap_ratio": float(o)} for a, b, o in pairs],
    }


def gt_nearest_pc_common_mapping(
    pc_common: np.ndarray,
    gt_grasps_17: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    对每个 GT 抓取平移 t（17D 行中 13:16），在 pc_common 上找最近点。
    返回每行 GT 的最近距离与索引。
    """
    if gt_grasps_17 is None or gt_grasps_17.size == 0:
        return None
    g = np.asarray(gt_grasps_17, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] < 16:
        return None
    T = g[:, 13:16]
    pc = np.asarray(pc_common, dtype=np.float64)
    # (N,G)
    d = np.linalg.norm(pc[:, None, :] - T[None, :, :], axis=2)
    nn_idx = np.argmin(d, axis=0)
    nn_dist = d[nn_idx, np.arange(T.shape[0])]
    return {
        "num_gt_grasps": int(T.shape[0]),
        "per_gt_nearest_pc_index": nn_idx.tolist(),
        "per_gt_nearest_distance": nn_dist.astype(np.float64).tolist(),
        "mean_gt_to_pc_nearest_dist": float(np.mean(nn_dist)),
    }


def per_point_min_dist_to_gt_translations(
    pc_common: np.ndarray,
    gt_grasps_17: np.ndarray,
) -> Optional[np.ndarray]:
    """
    每个 ``pc_common`` 点到**任意** GT 抓取平移（17D 行中 13:16）的最小欧氏距离，形状 ``(N,)``。
    用于 grasp 对齐可视化：与 ``gt_nearest_pc_common_mapping`` 使用同一平移定义。
    """
    if gt_grasps_17 is None or gt_grasps_17.size == 0:
        return None
    g = np.asarray(gt_grasps_17, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] < 16:
        return None
    T = g[:, 13:16]
    pc = np.asarray(pc_common, dtype=np.float64)
    d = np.linalg.norm(pc[:, None, :] - T[None, :, :], axis=2)
    return np.min(d, axis=1).astype(np.float64)


def topk_distance_to_nearest_gt_translation(
    pc_common: np.ndarray,
    gt_grasps_17: np.ndarray,
    top_idx: np.ndarray,
    *,
    dist_threshold: float,
) -> Optional[Dict[str, float]]:
    """
    top-k 特征点到**任意** GT 抓取平移的最小距离：均值与阈值内占比。
    """
    if gt_grasps_17 is None or gt_grasps_17.size == 0:
        return None
    g = np.asarray(gt_grasps_17, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] < 16:
        return None
    T = g[:, 13:16]
    idx = np.asarray(top_idx, dtype=np.int64).ravel()
    if idx.size == 0:
        return None
    pts = np.asarray(pc_common, dtype=np.float64)[idx]
    # (K, G)
    d = np.linalg.norm(pts[:, None, :] - T[None, :, :], axis=2)
    dmin = np.min(d, axis=1)
    return {
        "mean_dist_topk_to_nearest_gt_translation": float(np.mean(dmin)),
        "frac_topk_within_dist_threshold": float(np.mean(dmin < dist_threshold)),
        "dist_threshold": float(dist_threshold),
    }


def pairwise_pearson_feature_norms(feats_l2: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    在 pc_common 上，各模型逐点 L2 范数向量之间的 Pearson 相关（长度均为 N）。
    返回对称矩阵（字典列表 + names 顺序）。
    """
    names = sorted(feats_l2.keys())
    if len(names) < 2:
        return {
            "model_names": names,
            "matrix": [],
            "note": "need_at_least_two_models",
        }

    def pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.shape != b.shape or a.size < 2:
            return float("nan")
        a = a - np.mean(a)
        b = b - np.mean(b)
        den = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if den < 1e-20:
            return float("nan")
        return float(np.sum(a * b) / den)

    n = len(names)
    mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            r = pearson(feats_l2[names[i]], feats_l2[names[j]])
            mat[i, j] = r
            mat[j, i] = r
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({"model_a": names[i], "model_b": names[j], "pearson_r": float(mat[i, j])})
    return {
        "model_names": names,
        "matrix": mat.tolist(),
        "pairwise_list": rows,
    }


def score_entropy_normalized(scores: np.ndarray) -> float:
    """非负分数归一化为概率分布后的 Shannon 熵（自然底）。"""
    s = np.asarray(scores, dtype=np.float64).ravel()
    s = np.clip(s, 0.0, None)
    z = float(np.sum(s))
    if z < 1e-20:
        return float("nan")
    p = s / z
    p = np.clip(p, 1e-20, 1.0)
    return float(-np.sum(p * np.log(p)))


def topk_energy_ratio(scores: np.ndarray, k: int) -> float:
    """sum(top-k scores) / sum(all)，scores 非负。"""
    s = np.asarray(scores, dtype=np.float64).ravel()
    s = np.clip(s, 0.0, None)
    tot = float(np.sum(s))
    if tot < 1e-20:
        return float("nan")
    k = min(k, s.size)
    if k <= 0:
        return float("nan")
    part = np.partition(s, -k)[-k:]
    return float(np.sum(part) / tot)


def concentration_metrics_per_model(
    feats_l2: Dict[str, np.ndarray],
    *,
    topk: int,
) -> Dict[str, Dict[str, float]]:
    """各模型：范数分布熵、top-k 能量占比。"""
    out: Dict[str, Dict[str, float]] = {}
    for name, fn in feats_l2.items():
        fn = np.asarray(fn, dtype=np.float64).ravel()
        out[name] = {
            "entropy_of_normalized_norms": score_entropy_normalized(fn),
            "topk_energy_ratio": topk_energy_ratio(fn, topk),
        }
    return out


def save_summary_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pairwise_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def gt_contact_distance_stats(
    pc_common: np.ndarray,
    gt_grasps_17: np.ndarray,
    top_idx: np.ndarray,
    *,
    contact_radius: float = 0.02,
) -> Optional[Dict[str, float]]:
    """
    gt_grasps_17: (G,17) 行为 grasp，接触点近似取 row 中 translation（与 action2grasp 一致：索引 13:16）。
    返回 top-k 点到最近 GT 接触点的平均距离。
    """
    if gt_grasps_17 is None or gt_grasps_17.size == 0:
        return None
    g = np.asarray(gt_grasps_17, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] < 16:
        return None
    contacts = g[:, 13:16]
    pts = pc_common[top_idx]
    dmin = []
    for p in pts:
        dist = np.linalg.norm(contacts - p.reshape(1, 3), axis=1)
        dmin.append(float(np.min(dist)))
    dm = np.array(dmin, dtype=np.float64)
    return {
        "mean_dist_to_nearest_gt_contact": float(np.mean(dm)),
        "frac_within_radius": float(np.mean(dm < contact_radius)),
        "contact_radius": contact_radius,
    }
