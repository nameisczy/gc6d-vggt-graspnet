# -*- coding: utf-8 -*-
"""标量摘要：范数统计、top-k 重叠、可选 GT 接触距离。"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def map_to_gc6d_split(split: str) -> str:
    """GraspClutter6D 仅接受 train / test / all；管线里的 val 对应官方 train 划分中的场景。"""
    if split == "train":
        return "train"
    if split == "val":
        return "train"
    if split == "test":
        return "test"
    return "all"


def load_gt_grasps_from_gc6d_api(
    dataset_root: str,
    scene_id: int,
    ann_id: int,
    *,
    camera: str,
    split: str,
) -> Tuple[Optional[np.ndarray], str]:
    """
    通过 ``graspclutter6dAPI.GraspClutter6D.loadGrasp`` 加载场景标注帧的 GT 抓取 (G, 17)。
    平移在 ``[:, 13:16]``（与 ``GraspGroup.translations`` 一致）。
    """
    root = os.path.abspath(os.path.expanduser(dataset_root))
    if not os.path.isdir(root):
        print(f"[DEBUG] load_gt_grasps_from_gc6d_api: dataset_root is not a directory: {root!r}")
        return None, f"dataset_root_not_dir:{root}"

    try:
        from graspclutter6dAPI import GraspClutter6D
    except ImportError as e:
        print(f"[ERROR] graspclutter6dAPI import failed: {e!r}")
        return None, f"import_error:{e!s}"

    try:
        gc6d_split = map_to_gc6d_split(split)
        print(f"[DEBUG] Using GC6D split: {gc6d_split} (from input split={split})")
        g = GraspClutter6D(root, camera=camera, split=gc6d_split)
        grasp_group = g.loadGrasp(
            sceneId=int(scene_id),
            annId=int(ann_id),
            format="6d",
            camera=camera,
            fric_coef_thresh=0.2,
            remove_invisible=True,
        )
        gt_grasps = grasp_group.grasp_group_array.astype(np.float32)
        print(f"[DEBUG] Loaded GT grasps via API: shape={gt_grasps.shape}")
        if gt_grasps.size == 0 or gt_grasps.shape[0] == 0:
            return None, "api_empty_grasp_group"
        if gt_grasps.ndim != 2 or gt_grasps.shape[1] < 16:
            return None, f"api_bad_shape:{getattr(gt_grasps, 'shape', None)}"
        return gt_grasps, "ok:GraspClutter6D.loadGrasp"
    except Exception as e:
        print(f"[ERROR] GraspClutter6D.loadGrasp failed: {e!r}")
        return None, f"api_error:{e!s}"


def resolve_gt_grasps_17d(
    dataset_root: Optional[str],
    scene_id: int,
    ann_id: int,
    offline_npz_data: Any,
    *,
    camera: str = "realsense-d415",
    split: str = "val",
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    优先通过 ``GraspClutter6D.loadGrasp`` 从原始 GC6D 根目录加载 GT；失败时回退离线 npz 的 ``gt_grasp_group``。
    ``offline_npz_data`` 为 ``numpy.load`` 返回的对象。
    """
    print(f"[DEBUG] resolve_gt_grasps_17d: Trying to load GT grasps for scene={scene_id}, ann={ann_id}")
    dr = dataset_root if (dataset_root and str(dataset_root).strip()) else None
    print(f"[DEBUG] resolve_gt_grasps_17d: dataset_root = {dr!r}")
    _gc6d_split = map_to_gc6d_split(split)
    print(
        f"[DEBUG] resolve_gt_grasps_17d: GT via GraspClutter6D.loadGrasp("
        f"camera={camera!r}, pipeline_split={split!r} -> gc6d_split={_gc6d_split!r}, "
        f"fric_coef_thresh=0.2, remove_invisible=True)"
    )

    meta: Dict[str, Any] = {
        "source": None,
        "camera": camera,
        "split": split,
        "gc6d_api_split": _gc6d_split,
    }
    if dr:
        arr, note = load_gt_grasps_from_gc6d_api(
            dr, scene_id, ann_id, camera=camera, split=split
        )
        meta["dataset_attempt"] = note
        if arr is not None:
            meta["source"] = "gc6d_api_loadGrasp"
            print(f"[DEBUG] resolve_gt_grasps_17d: using API grasps, shape={arr.shape}")
            return arr, meta
        print(f"[DEBUG] resolve_gt_grasps_17d: API load failed or empty: {note!r}")

    off_files = list(getattr(offline_npz_data, "files", []))
    print(f"[DEBUG] resolve_gt_grasps_17d: offline npz keys: {off_files}")
    if "gt_grasp_group" in off_files:
        gg = np.asarray(offline_npz_data["gt_grasp_group"])
        print(f"[DEBUG] resolve_gt_grasps_17d: offline gt_grasp_group raw shape={getattr(gg, 'shape', None)}")
        if gg.size > 0 and gg.ndim == 2 and gg.shape[1] >= 16:
            meta["source"] = "offline_npz_gt_grasp_group"
            out = gg.astype(np.float64)
            print(f"[WARN] Falling back to offline npz GT grasp, shape={out.shape}")
            return out, meta
        print("[WARN] offline npz has gt_grasp_group but empty or wrong ndim/shape; cannot use")

    print("[ERROR] No GT grasps from GraspClutter6D API or offline npz")
    meta["source"] = "none"
    return None, meta


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


def topk_diff_grasp_region_overlap(
    abs_diff_per_point: np.ndarray,
    dist_to_gt_per_point: np.ndarray,
    *,
    top_fraction: float,
    grasp_dist_threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    衡量「表示变化最大」的点是否落在抓取相关空间：与 ``top_fraction_binary_masks`` 相同地取
    ``k = ceil(N * top_fraction)`` 个绝对差分最大的点；抓取区域为 ``dist_to_gt < grasp_dist_threshold``。

    随机基线 ``overlap_random_baseline = mean(1[dist_to_gt < τ])``，与在全体点上均匀抽样的期望落入 grasp 邻域比例一致；
    ``improvement_vs_random = overlap / baseline``（baseline 极小时为 None）。

    返回上述量及 ``k``、交集大小等；长度不一致或无效参数时返回 None。
    """
    d_diff = np.asarray(abs_diff_per_point, dtype=np.float64).ravel()
    d_gt = np.asarray(dist_to_gt_per_point, dtype=np.float64).ravel()
    if d_diff.shape != d_gt.shape or d_diff.size == 0:
        return None
    frac = float(top_fraction)
    if frac <= 0.0 or frac > 1.0:
        return None
    n = d_diff.size
    k = max(1, int(np.ceil(n * frac)))
    order = np.argsort(-d_diff, kind="stable")
    top_mask = np.zeros(n, dtype=bool)
    top_mask[order[:k]] = True
    grasp_mask = d_gt < float(grasp_dist_threshold)
    inter = int(np.logical_and(top_mask, grasp_mask).sum())
    overlap = float(inter) / float(k)
    baseline = float(np.mean(grasp_mask.astype(np.float64)))
    eps = 1e-12
    improvement = float(overlap / baseline) if baseline > eps else None
    return {
        "overlap": overlap,
        "k": int(k),
        "topk_in_grasp_region": int(inter),
        "top_fraction": float(frac),
        "grasp_dist_threshold_m": float(grasp_dist_threshold),
        "overlap_random_baseline": baseline,
        "improvement_vs_random": improvement,
        "definition": "|topk_abs_diff ∩ {i : min_dist_to_gt_translation(i) < τ}| / k",
        "baseline_definition": "mean over points of 1[dist_to_gt < τ]; uniform-random k-subset expectation ≈ baseline",
        "improvement_definition": "overlap / overlap_random_baseline; >1 means task-aligned vs chance",
    }


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
