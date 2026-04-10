# -*- coding: utf-8 -*-
"""
从深度反投影或 npz 点云构造统一的 pc_common（固定随机子采样）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


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


def _depth_scale(camera: str) -> float:
    if camera in ("realsense-d415", "realsense-d435"):
        return 1000.0
    if camera in ("azure-kinect", "zivid"):
        return 10000.0
    return 1000.0


def backproject_depth_to_points(
    dataset_root: str,
    scene_id: int,
    ann_id: int,
    camera: str,
    *,
    workspace_margin: float = 0.1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    与 GraspClutter6D.loadScenePointCloud(..., format='numpy') 一致的相机坐标系点云。
    返回 (points (M,3), colors (M,3) 或 None, meta)。
    """
    if cv2 is None:
        return None, None, {"error": "no_cv2"}

    img_id = _ann_id_to_img_id(ann_id, camera)
    scene_dir = os.path.join(dataset_root, "scenes", f"{scene_id:06d}")
    rgb_path = os.path.join(scene_dir, "rgb", f"{img_id:06d}.png")
    depth_path = os.path.join(scene_dir, "depth", f"{img_id:06d}.png")
    cam_path = os.path.join(scene_dir, "scene_camera.json")
    if not all(os.path.isfile(p) for p in (rgb_path, depth_path, cam_path)):
        return None, None, {"error": "missing_rgb_depth_or_camera", "img_id": img_id}

    colors = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if colors is None:
        return None, None, {"error": "rgb_read_fail"}
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depths = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depths is None:
        return None, None, {"error": "depth_read_fail"}

    with open(cam_path, "r", encoding="utf-8") as f:
        scene_camera = json.load(f)
    K = np.array(scene_camera[str(img_id)]["cam_K"], dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    s = _depth_scale(camera)

    h, w = depths.shape[:2]
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
    points_z = depths.astype(np.float64) / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)

    # workspace crop（与 API 类似）
    label_path = os.path.join(scene_dir, "label", f"{img_id:06d}.png")
    if os.path.isfile(label_path):
        mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            maskx = np.any(mask > 0, axis=0)
            masky = np.any(mask > 0, axis=1)
            if maskx.any() and masky.any():
                x1 = int(np.argmax(maskx))
                y1 = int(np.argmax(masky))
                x2 = len(maskx) - int(np.argmax(maskx[::-1]))
                y2 = len(masky) - int(np.argmax(masky[::-1]))
                wm = int(w * workspace_margin)
                hm = int(h * workspace_margin)
                x1 = max(0, x1 - wm)
                y1 = max(0, y1 - hm)
                x2 = min(w, x2 + wm)
                y2 = min(h, y2 + hm)
                points = points[y1:y2, x1:x2]
                colors = colors[y1:y2, x1:x2]

    valid = points[:, :, 2] > 0
    pts = points[valid].astype(np.float32)
    cols = colors[valid] if colors.ndim == 3 else colors

    meta = {"img_id": img_id, "num_points_raw": int(pts.shape[0]), "source": "depth_backproject"}
    return pts, cols, meta


def subsample_points(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    """points: (M,3) -> (n,3)"""
    rng = np.random.default_rng(seed)
    m = points.shape[0]
    if m <= n:
        idx = np.arange(m)
        if m < n:
            extra = rng.integers(0, m, size=n - m)
            idx = np.concatenate([idx, extra])
        return points[idx].astype(np.float32)
    idx = rng.choice(m, size=n, replace=False)
    return points[idx].astype(np.float32)


def build_pc_common(
    npz_path: str,
    *,
    dataset_root: Optional[str] = None,
    scene_id: Optional[int] = None,
    ann_id: Optional[int] = None,
    camera: Optional[str] = None,
    n_points: int = 2048,
    seed: int = 0,
    z_min: float = 0.0,
    z_max: float = 1.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    优先：dataset_root + 深度反投影 → 与 GC6D 一致坐标。
    否则：npz 点云过滤 workspace 后子采样。
    """
    meta: Dict[str, Any] = {"n_requested": n_points, "rng_seed": seed}
    pts: Optional[np.ndarray] = None

    data = np.load(npz_path, allow_pickle=True)
    if scene_id is None:
        scene_id = int(data["sceneId"])
    if ann_id is None:
        ann_id = int(data["annId"])

    if dataset_root and camera and os.path.isdir(dataset_root):
        p, _, m = backproject_depth_to_points(dataset_root, scene_id, ann_id, camera)
        meta["depth_attempt"] = m
        if p is not None and p.shape[0] > 100:
            pts = p

    if pts is None:
        pc = np.asarray(data["point_cloud"], dtype=np.float32)
        z = pc[:, 2]
        m = (z > z_min) & (z < z_max) & np.isfinite(pc).all(axis=1)
        pc = pc[m]
        meta["source"] = "npz_point_cloud_filtered"
        meta["fallback_reason"] = meta.get("depth_attempt", {})
        if pc.shape[0] < 50:
            pc = np.asarray(data["point_cloud"], dtype=np.float32)
        pts = pc

    pc_common = subsample_points(pts, n_points, seed)
    meta["num_source_points"] = int(pts.shape[0])
    meta["scene_id"] = int(scene_id)
    meta["ann_id"] = int(ann_id)
    return pc_common, meta
