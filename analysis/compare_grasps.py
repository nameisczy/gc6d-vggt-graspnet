#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline vs reranker grasp predictions against GT.

Outputs:
- analysis/score_distribution.png
- analysis/report.txt
- optional: analysis/grasp_points_3d.png
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
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


def img_id_to_ann_id(img_id: int, camera: str) -> int:
    offset_map = {
        "realsense-d415": 1,
        "realsense-d435": 2,
        "azure-kinect": 3,
        "zivid": 4,
    }
    if camera not in offset_map:
        raise ValueError(f"Unknown camera={camera}")
    offset = offset_map[camera]
    if img_id < offset:
        raise ValueError(f"img_id too small for camera {camera}: {img_id}")
    rem = (img_id - offset) % 4
    if rem != 0:
        raise ValueError(f"img_id={img_id} not aligned with camera={camera}")
    return (img_id - offset) // 4


def load_grasp_array(path: str) -> np.ndarray:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "grasp_group" in data.files:
            arr = data["grasp_group"]
        elif "pred" in data.files:
            arr = data["pred"]
        else:
            arr = data[data.files[0]]
    else:
        raise ValueError(f"Unsupported dump format: {path}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 17:
        raise ValueError(f"Expect grasp array shape (N,>=17), got {arr.shape} from {path}")
    return arr


def infer_scene_camera_ann_from_dump(dump_path: str) -> Tuple[int, str, int]:
    """
    Infer scene/camera/ann from path pattern:
    .../<scene_id>/<camera>/<img_id>.npy
    """
    p = os.path.abspath(os.path.expanduser(dump_path))
    parts = p.split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"Cannot infer scene/camera/ann from path: {dump_path}")
    stem = os.path.splitext(os.path.basename(p))[0]
    camera = parts[-2]
    scene_str = parts[-3]
    scene_id = int(scene_str)
    img_id = int(stem)
    ann_id = img_id_to_ann_id(img_id, camera)
    return scene_id, camera, ann_id


def load_gt_grasps(gt_root: str, scene_id: int, ann_id: int, camera: str, split: str) -> np.ndarray:
    try:
        from graspclutter6dAPI import GraspClutter6D
    except ImportError as exc:
        raise ImportError("graspclutter6dAPI is required to load GT grasps") from exc
    g = GraspClutter6D(os.path.abspath(os.path.expanduser(gt_root)), camera=camera, split=split)
    gg = g.loadGrasp(
        sceneId=int(scene_id),
        annId=int(ann_id),
        format="6d",
        camera=camera,
        fric_coef_thresh=0.2,
        remove_invisible=True,
    )
    arr = np.asarray(gg.grasp_group_array, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 17:
        raise ValueError(f"GT grasp array bad shape: {arr.shape}")
    return arr


def topk_by_score(arr: np.ndarray, k: int) -> np.ndarray:
    n = arr.shape[0]
    k = max(0, min(int(k), n))
    if k == 0:
        return arr[:0]
    idx = np.argpartition(-arr[:, 0], kth=k - 1)[:k]
    idx = idx[np.argsort(-arr[idx, 0])]
    return arr[idx]


def set_repr_rows(arr: np.ndarray, decimals: int = 6) -> set:
    q = np.round(arr[:, :17], decimals=decimals)
    return {tuple(x.tolist()) for x in q}


def topk_overlap_ratio(topk_a: np.ndarray, topk_b: np.ndarray, k: int) -> float:
    k_eff = min(int(k), topk_a.shape[0], topk_b.shape[0])
    if k_eff <= 0:
        return float("nan")
    sa = set_repr_rows(topk_a[:k_eff])
    sb = set_repr_rows(topk_b[:k_eff])
    return len(sa & sb) / float(k_eff)


def nearest_pred_distances(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    SAME as eval translation usage:
    translation index = [:, 13:16]
    """
    gt_t = gt[:, 13:16]
    pred_t = pred[:, 13:16]
    d = np.linalg.norm(gt_t[:, None, :] - pred_t[None, :, :], axis=2)  # (G, P)
    return np.min(d, axis=1)


def save_score_histogram(b_scores: np.ndarray, r_scores: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(b_scores, bins=50, alpha=0.6, label="baseline", color="tab:blue")
    plt.hist(r_scores, bins=50, alpha=0.6, label="reranker", color="tab:red")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Baseline vs Reranker Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_optional_3d_plot(gt: np.ndarray, b: np.ndarray, r: np.ndarray, out_path: str, top_k: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    gt_t = gt[:, 13:16]
    bt = topk_by_score(b, top_k)[:, 13:16]
    rt = topk_by_score(r, top_k)[:, 13:16]
    ax.scatter(gt_t[:, 0], gt_t[:, 1], gt_t[:, 2], c="green", s=8, label="GT", alpha=0.7)
    ax.scatter(bt[:, 0], bt[:, 1], bt[:, 2], c="blue", s=8, label="baseline", alpha=0.7)
    ax.scatter(rt[:, 0], rt[:, 1], rt[:, 2], c="red", s=8, label="reranker", alpha=0.7)
    ax.set_title(f"GT/Baseline/Reranker translations (top-{top_k})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_report_lines(
    *,
    topk_overlap: float,
    mean_b: float,
    mean_r: float,
    pct_improved: float,
    pct_worsened: float,
    b_dist: np.ndarray,
    r_dist: np.ndarray,
) -> List[str]:
    lines: List[str] = []
    lines.append(f"TopK overlap: {topk_overlap:.6f}")
    lines.append("")
    lines.append(f"Mean baseline distance: {mean_b:.6f}")
    lines.append(f"Mean reranker distance: {mean_r:.6f}")
    lines.append(f"% GT improved: {pct_improved:.2f}%")
    lines.append(f"% GT worsened: {pct_worsened:.2f}%")
    lines.append("")
    lines.append("Per-grasp comparison (top 10 GT by GT score):")
    gt_show = min(10, b_dist.shape[0])
    for i in range(gt_show):
        delta = r_dist[i] - b_dist[i]
        lines.append(f"GT {i}:")
        lines.append(f"  baseline dist: {b_dist[i]:.6f}")
        lines.append(f"  reranker dist: {r_dist[i]:.6f}")
        lines.append(f"  delta: {delta:+.6f}")
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs reranker grasp predictions")
    p.add_argument("--baseline_dump", required=True, type=str)
    p.add_argument("--reranker_dump", required=True, type=str)
    p.add_argument("--gt_root", required=True, type=str, help="GraspClutter6D root")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--scene_id", type=int, default=None)
    p.add_argument("--ann_id", type=int, default=None)
    p.add_argument("--camera", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=("train", "test", "all"))
    p.add_argument("--out_dir", type=str, default="analysis")
    p.add_argument("--plot_3d", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baseline = load_grasp_array(args.baseline_dump)
    reranker = load_grasp_array(args.reranker_dump)

    scene_id = args.scene_id
    ann_id = args.ann_id
    camera = args.camera
    if scene_id is None or ann_id is None or camera is None:
        s, c, a = infer_scene_camera_ann_from_dump(args.baseline_dump)
        if scene_id is None:
            scene_id = s
        if ann_id is None:
            ann_id = a
        if camera is None:
            camera = c

    gt = load_gt_grasps(args.gt_root, scene_id=scene_id, ann_id=ann_id, camera=camera, split=args.split)

    b_top = topk_by_score(baseline, args.top_k)
    r_top = topk_by_score(reranker, args.top_k)
    topk_overlap = topk_overlap_ratio(b_top, r_top, args.top_k)
    print("TopK overlap:", topk_overlap)

    score_png = os.path.join(args.out_dir, "score_distribution.png")
    save_score_histogram(baseline[:, 0], reranker[:, 0], score_png)

    # For "top 10 GT", sort GT by its own score descending first.
    gt_sorted = gt[np.argsort(-gt[:, 0])]
    b_dist = nearest_pred_distances(gt_sorted, b_top)
    r_dist = nearest_pred_distances(gt_sorted, r_top)

    mean_b = float(np.mean(b_dist))
    mean_r = float(np.mean(r_dist))
    improved = r_dist < b_dist
    worsened = r_dist > b_dist
    pct_improved = 100.0 * float(np.mean(improved))
    pct_worsened = 100.0 * float(np.mean(worsened))

    lines = build_report_lines(
        topk_overlap=topk_overlap,
        mean_b=mean_b,
        mean_r=mean_r,
        pct_improved=pct_improved,
        pct_worsened=pct_worsened,
        b_dist=b_dist,
        r_dist=r_dist,
    )

    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if args.plot_3d:
        save_optional_3d_plot(
            gt_sorted,
            baseline,
            reranker,
            out_path=os.path.join(args.out_dir, "grasp_points_3d.png"),
            top_k=args.top_k,
        )

    print(f"Saved score histogram: {score_png}")
    print(f"Saved report: {report_path}")
    print(f"Scene={scene_id}, ann={ann_id}, camera={camera}, img_id={ann_id_to_img_id(ann_id, camera)}")


if __name__ == "__main__":
    main()
