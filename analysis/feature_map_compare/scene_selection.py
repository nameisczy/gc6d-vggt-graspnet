# -*- coding: utf-8 -*-
"""
从 GC6D offline_unified 索引 +（可选）原始数据集根目录中选一条「杂乱 + 部分遮挡 + 多抓取」样本。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class SelectionResult:
    index_row: int
    scene_id: int
    ann_id: int
    camera: str
    npz_path: str
    rgb_path: str
    score: float
    summary: Dict[str, Any] = field(default_factory=dict)


def _ann_id_to_img_id(ann_id: int, camera: str) -> int:
    img_id = ann_id * 4
    if camera == "realsense-d415":
        img_id += 1
    elif camera == "realsense-d435":
        img_id += 2
    elif camera == "azure-kinect":
        img_id += 3
    elif camera == "zivid":
        img_id += 4
    return img_id


def _load_scene_gt_object_indices(dataset_root: str, scene_id: int, img_id: int) -> List[int]:
    """scene_gt.json：当前帧可见物体列表（用于数物体个数）。"""
    path = os.path.join(dataset_root, "scenes", f"{scene_id:06d}", "scene_gt.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    key = str(img_id)
    if key not in gt:
        return []
    return list(range(len(gt[key])))


def _visibility_and_clutter_from_masks(
    dataset_root: str,
    scene_id: int,
    img_id: int,
    num_objects: int,
) -> Tuple[float, int, List[float], str]:
    """
    返回：(occlusion_score 聚合, 可见物体数估计, 每实例 visibility 列表, 说明)
    occlusion_score: 至少一个物体 strongly partial 时更高（min visibility 越低越好）
    """
    if cv2 is None:
        return 0.0, 0, [], "cv2_unavailable"
    vis_list: List[float] = []
    for i in range(num_objects):
        vpath = os.path.join(
            dataset_root, "scenes", f"{scene_id:06d}", "mask_visib", f"{img_id:06d}_{i:06d}.png"
        )
        apath = os.path.join(
            dataset_root, "scenes", f"{scene_id:06d}", "mask", f"{img_id:06d}_{i:06d}.png"
        )
        if not os.path.isfile(vpath) or not os.path.isfile(apath):
            continue
        vis = cv2.imread(vpath, cv2.IMREAD_UNCHANGED)
        amo = cv2.imread(apath, cv2.IMREAD_UNCHANGED)
        if vis is None or amo is None:
            continue
        if vis.ndim == 3:
            vis = vis[:, :, 0]
        if amo.ndim == 3:
            amo = amo[:, :, 0]
        s_vis = float(np.sum(vis > 0))
        s_amo = float(np.sum(amo > 0)) + 1e-6
        vis_list.append(s_vis / s_amo)
    if len(vis_list) < 2:
        return 0.0, len(vis_list), vis_list, "few_masks"
    # 至少 2 个物体、鼓励部分遮挡（最小可见比例小）
    min_vis = min(vis_list)
    mean_vis = float(np.mean(vis_list))
    # occlusion：min_vis 小表示至少一物体遮挡重
    occlusion_score = (1.0 - min_vis) + 0.25 * (1.0 - mean_vis)
    return occlusion_score, len(vis_list), vis_list, "ok"


def _load_grasp_test_scene_ids(dataset_root: str) -> List[int]:
    """与 ``eval_benchmark.py`` / GC6D API 一致：官方测试场景 ID 列表。"""
    p = os.path.join(os.path.abspath(os.path.expanduser(dataset_root)), "split_info", "grasp_test_scene_ids.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [int(x) for x in raw]


def _items_from_test_split_info_scan(
    data_dir: str,
    dataset_root: str,
    camera: str,
    max_items: int,
    *,
    max_npz_probe: int = 800_000,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    无 ``index_test_*.jsonl`` 时：按 ``grasp_test_scene_ids.json`` 筛 ``data_dir`` 下 ``.npz``，
    构造与 jsonl 行兼容的 dict 列表（供与 index 模式相同的打分循环使用）。

    使用 ``os.walk`` 顺序遍历，**不再**对排序后的路径列表做「只保留前 N 个」的截断，
    避免测试场景 npz 在字典序上偏后时永远扫不到的问题。
    若已收集 ``max_items`` 条测试样本或已打开探测 ``max_npz_probe`` 个 npz，则停止。
    """
    test_ids = set(_load_grasp_test_scene_ids(dataset_root))
    root = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    out: List[Dict[str, Any]] = []
    npz_checked = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for f in sorted(filenames):
            if not f.endswith(".npz"):
                continue
            if len(out) >= max_items:
                return out, {"npz_checked": npz_checked, "n_test_kept": len(out)}
            if npz_checked >= max_npz_probe:
                return out, {"npz_checked": npz_checked, "n_test_kept": len(out)}
            npz_checked += 1
            p = os.path.join(dirpath, f)
            try:
                data = np.load(p, allow_pickle=True)
            except Exception:
                continue
            if "sceneId" not in data.files or "annId" not in data.files:
                continue
            sid = int(data["sceneId"])
            if sid not in test_ids:
                continue
            rel = os.path.relpath(p, root)
            rgb_path = ""
            if "rgb_path" in data.files:
                try:
                    rp = data["rgb_path"]
                    rgb_path = str(rp.item()) if hasattr(rp, "item") else str(rp)
                except Exception:
                    rgb_path = ""
            out.append(
                {
                    "npz": rel,
                    "rgb_path": rgb_path,
                    "camera": camera,
                }
            )
    return out, {"npz_checked": npz_checked, "n_test_kept": len(out)}


def _npz_fallback_score(npz_path: str) -> Tuple[float, Dict[str, Any]]:
    """无 dataset_root 时：用 GT 抓取数 + 点云分散度作代理。"""
    data = np.load(npz_path, allow_pickle=True)
    meta: Dict[str, Any] = {}
    n_grasp = 0
    if "gt_grasp_group" in data:
        gg = np.asarray(data["gt_grasp_group"], dtype=np.float32)
        if gg.ndim == 2:
            n_grasp = int(gg.shape[0])
    pc = np.asarray(data["point_cloud"], dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        return 0.0, {"error": "bad_point_cloud", "n_grasp": n_grasp}
    # 覆盖范围（归一化后 spread）
    c = pc.mean(axis=0, keepdims=True)
    p0 = pc - c
    spread = float(np.percentile(np.abs(p0), 95))
    # 多抓取 + 非平凡场景
    score = min(n_grasp, 20) * 0.5 + spread * 2.0
    meta.update({"n_grasp_gt": n_grasp, "pc_spread_p95": spread, "mode": "npz_fallback"})
    return score, meta


def select_representative_example(
    data_dir: str,
    *,
    split: str = "val",
    camera: str = "realsense-d415",
    dataset_root: Optional[str] = None,
    max_candidates: int = 400,
    index_filename: Optional[str] = None,
) -> SelectionResult:
    """
    遍历索引中前 max_candidates 条，按
    clutter（>=2 物体）+ 遮挡（mask_visib/mask）+ 多抓取（GT 条数）排序，取最优。
    若无 dataset_root 或 mask 不可用，则退回 npz 启发式。

    ``index_filename``：默认 ``index_{split}_{camera}.jsonl``（位于 ``data_dir`` 下）；
    若传入相对文件名则相对于 ``data_dir``；若为绝对路径则直接使用。

    **测试集（``split=="test"``）**：若默认的 ``index_test_<camera>.jsonl`` 不存在、且未显式指定
    ``index_filename``、且提供了 ``dataset_root``（含 ``split_info/grasp_test_scene_ids.json``），
    则改为在 ``data_dir`` 下扫描 ``.npz``，仅保留 ``sceneId`` 属于官方测试集 的样本作为候选
    （与 ``eval_benchmark.py --split test`` 使用的测试场景定义一致，不依赖单独的 test index 文件）。
    """
    from data.dataset import load_index_jsonl

    explicit_index = index_filename is not None
    if index_filename is None:
        index_filename = f"index_{split}_{camera}.jsonl"
    expanded = os.path.expanduser(index_filename)
    if os.path.isabs(expanded):
        index_path = os.path.abspath(expanded)
    else:
        index_path = os.path.join(os.path.abspath(os.path.expanduser(data_dir)), expanded)

    index_source_note: str
    if os.path.isfile(index_path):
        items = load_index_jsonl(index_path)[:max_candidates]
        index_source_note = index_path
    elif (
        split == "test"
        and not explicit_index
        and dataset_root
        and os.path.isdir(os.path.abspath(os.path.expanduser(dataset_root)))
    ):
        try:
            items, scan_stats = _items_from_test_split_info_scan(
                data_dir,
                dataset_root,
                camera,
                max_items=max_candidates,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"未找到测试集索引 {index_path}，且无法从 dataset_root 读取 split_info（{e}）。"
                f"请放置 index_test_{camera}.jsonl，或提供完整 GC6D dataset_root。"
            ) from e
        if not items:
            raise RuntimeError(
                "未在 data_dir 下找到 sceneId 属于 grasp_test_scene_ids 的 .npz。"
                f"（本次共探测 {scan_stats.get('npz_checked', 0)} 个 npz 文件，命中测试集 {scan_stats.get('n_test_kept', 0)} 条）。"
                "常见原因：离线目录仅含 train/val、不含测试场景导出。"
                "请改用含测试样本的 data_dir、生成 index_test_*.jsonl，或使用 --index_jsonl 指向测试集列表；"
                "若仅需与验证集可比，可使用 --split val 与 --index_jsonl index_validation_....jsonl。"
            )
        index_source_note = (
            f"split_info:grasp_test_scene_ids.json + npz_scan_under({os.path.abspath(os.path.expanduser(data_dir))})"
        )
    else:
        raise FileNotFoundError(index_path)

    best: Optional[Tuple[float, int, Dict[str, Any]]] = None

    for idx, rec in enumerate(items):
        npz_field = rec.get("npz", "")
        if os.path.isabs(npz_field) and os.path.exists(npz_field):
            npz_path = npz_field
        else:
            npz_path = os.path.join(data_dir, npz_field)
        if not os.path.isfile(npz_path):
            continue

        data = np.load(npz_path, allow_pickle=True)
        scene_id = int(data["sceneId"])
        ann_id = int(data["annId"])
        cam = rec.get("camera", camera)

        clutter_score = 0.0
        occlusion_score = 0.0
        n_objects = 0
        per_obj_vis: List[float] = []
        detail: Dict[str, Any] = {}

        if dataset_root and os.path.isdir(dataset_root) and cv2 is not None:
            img_id = _ann_id_to_img_id(ann_id, cam)
            obj_indices = _load_scene_gt_object_indices(dataset_root, scene_id, img_id)
            n_objects = len(obj_indices)
            occlusion_score, n_vis, per_obj_vis, tag = _visibility_and_clutter_from_masks(
                dataset_root, scene_id, img_id, n_objects
            )
            clutter_score = float(n_objects) if n_objects >= 2 else 0.0
            detail["mask_mode"] = tag
            detail["n_scene_gt_objects"] = n_objects
            detail["per_object_visibility_ratio"] = per_obj_vis
        else:
            detail["mask_mode"] = "no_dataset_root_or_cv2"
            n_objects = 0

        n_grasp = 0
        if "gt_grasp_group" in data:
            gg = np.asarray(data["gt_grasp_group"])
            if gg.ndim == 2:
                n_grasp = int(gg.shape[0])

        fb_score, fb_meta = _npz_fallback_score(npz_path)
        detail.update(fb_meta)

        if clutter_score < 1.0 and dataset_root:
            # 严格模式未满足 2 物体：仍用 fallback 参与排序，但降低权重
            combined = 0.3 * fb_score + occlusion_score * 0.5 + min(n_grasp, 15) * 0.2
            detail["selection_note"] = "relaxed_clutter_not_verified"
        else:
            combined = (
                1.0 * clutter_score
                + 2.0 * occlusion_score
                + min(n_grasp, 20) * 0.15
                + 0.2 * fb_score
            )
            detail["selection_note"] = "full_criteria" if dataset_root else "npz_weighted"

        if best is None or combined > best[0]:
            rgb_path = rec.get("rgb_path", "")
            best = (
                combined,
                idx,
                {
                    "scene_id": scene_id,
                    "ann_id": ann_id,
                    "camera": cam,
                    "npz_path": npz_path,
                    "rgb_path": rgb_path,
                    "combined_score": combined,
                    "clutter_score": clutter_score,
                    "occlusion_score": occlusion_score,
                    "n_grasp_gt": n_grasp,
                    "detail": detail,
                },
            )

    if best is None:
        raise RuntimeError("未找到任何有效样本，请检查 data_dir / index。")

    _, row, summ = best
    summ = dict(summ)
    summ["index_source"] = index_source_note
    return SelectionResult(
        index_row=row,
        scene_id=int(summ["scene_id"]),
        ann_id=int(summ["ann_id"]),
        camera=str(summ["camera"]),
        npz_path=str(summ["npz_path"]),
        rgb_path=str(summ.get("rgb_path", "")),
        score=float(summ["combined_score"]),
        summary=summ,
    )
