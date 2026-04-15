#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch compare baseline vs reranker grasp dumps across scenes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.compare_grasps import (
    img_id_to_ann_id,
    load_grasp_array,
    load_gt_grasps,
    nearest_pred_distances,
    topk_by_score,
    topk_overlap_ratio,
)


def _scene_id_from_name(name: str) -> Optional[int]:
    s = name.strip()
    if s.isdigit():
        return int(s)
    if s.startswith("scene_"):
        tail = s[len("scene_") :]
        if tail.isdigit():
            return int(tail)
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return int(digits)
    return None


def _collect_scene_names(gt_root: str) -> List[str]:
    scenes_dir = os.path.join(os.path.abspath(os.path.expanduser(gt_root)), "scenes")
    if not os.path.isdir(scenes_dir):
        raise FileNotFoundError(f"Missing scenes directory: {scenes_dir}")
    out: List[str] = []
    for n in sorted(os.listdir(scenes_dir)):
        p = os.path.join(scenes_dir, n)
        if os.path.isdir(p):
            out.append(n)
    return out


def _collect_matching_pairs_for_scene(
    baseline_root: str,
    reranker_root: str,
    scene_name: str,
) -> List[Tuple[str, str, str, int]]:
    """
    Returns list of (baseline_path, reranker_path, camera, ann_id) for one scene.
    """
    base_scene_dir = os.path.join(baseline_root, scene_name)
    if not os.path.isdir(base_scene_dir):
        return []
    baseline_files = sorted(glob.glob(os.path.join(base_scene_dir, "*", "*.npy")))
    pairs: List[Tuple[str, str, str, int]] = []
    for b in baseline_files:
        rel = os.path.relpath(b, baseline_root)
        r = os.path.join(reranker_root, rel)
        if not os.path.isfile(r):
            continue
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue
        camera = parts[1]
        img_name = os.path.splitext(os.path.basename(b))[0]
        if not img_name.isdigit():
            continue
        img_id = int(img_name)
        try:
            ann_id = img_id_to_ann_id(img_id, camera)
        except Exception:
            continue
        pairs.append((b, r, camera, ann_id))
    return pairs


def _compare_one_frame(
    baseline_path: str,
    reranker_path: str,
    gt: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    baseline = load_grasp_array(baseline_path)
    reranker = load_grasp_array(reranker_path)

    b_top = topk_by_score(baseline, top_k)
    r_top = topk_by_score(reranker, top_k)
    overlap = float(topk_overlap_ratio(b_top, r_top, top_k))

    gt_sorted = gt[np.argsort(-gt[:, 0])]
    b_dist = nearest_pred_distances(gt_sorted, b_top)
    r_dist = nearest_pred_distances(gt_sorted, r_top)

    improved = r_dist < b_dist
    worsened = r_dist > b_dist
    return {
        "topk_overlap": overlap,
        "mean_baseline_dist": float(np.mean(b_dist)),
        "mean_reranker_dist": float(np.mean(r_dist)),
        "improved_ratio": float(np.mean(improved)),
        "worsened_ratio": float(np.mean(worsened)),
        "distance_diff_mean": float(np.mean(r_dist - b_dist)),
    }


def _arr_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}


def _save_histogram(overlaps: List[float], dist_diffs: List[float], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(overlaps, bins=30, color="tab:purple", alpha=0.8)
    axes[0].set_title("TopK overlap distribution")
    axes[0].set_xlabel("overlap")
    axes[0].set_ylabel("count")
    axes[1].hist(dist_diffs, bins=30, color="tab:orange", alpha=0.8)
    axes[1].set_title("Distance diff distribution")
    axes[1].set_xlabel("reranker_dist - baseline_dist")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch compare baseline/reranker grasp dumps")
    p.add_argument("--baseline_dump_root", required=True, type=str)
    p.add_argument("--reranker_dump_root", required=True, type=str)
    p.add_argument("--gt_root", required=True, type=str)
    p.add_argument("--split", type=str, default="test", choices=("train", "test", "all"))
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="analysis")
    p.add_argument("--save_hist", action="store_true", default=False)
    p.add_argument(
        "--scene_filter",
        type=str,
        default=None,
        help="Comma-separated scene IDs, e.g. 000001,000002",
    )
    p.add_argument(
        "--max_frames_per_scene",
        type=int,
        default=None,
        help="Limit number of frames per scene for fast debugging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.scene_filter is not None:
        scene_filter = {x.strip() for x in args.scene_filter.split(",") if x.strip()}
    else:
        scene_filter = None

    baseline_root = os.path.abspath(os.path.expanduser(args.baseline_dump_root))
    reranker_root = os.path.abspath(os.path.expanduser(args.reranker_dump_root))
    gt_root = os.path.abspath(os.path.expanduser(args.gt_root))
    os.makedirs(args.out_dir, exist_ok=True)

    scene_names = _collect_scene_names(gt_root)

    gt_cache: Dict[Tuple[int, str, int], np.ndarray] = {}
    per_scene: Dict[str, Dict[str, float]] = {}
    all_frame_metrics: List[Dict[str, float]] = []

    for scene_name in scene_names:
        if scene_filter is not None and scene_name not in scene_filter:
            continue

        sid = _scene_id_from_name(scene_name)
        if sid is None:
            continue

        frame_files = _collect_matching_pairs_for_scene(baseline_root, reranker_root, scene_name)
        frame_files = sorted(frame_files)
        if args.max_frames_per_scene is not None:
            frame_files = frame_files[: args.max_frames_per_scene]

        print(f"[Scene {scene_name}] using {len(frame_files)} frames")
        if len(frame_files) == 0:
            print(f"[WARN] No frames found for scene {scene_name}")
            continue

        scene_frame_metrics: List[Dict[str, float]] = []
        for b_path, r_path, camera, ann_id in frame_files:
            key = (sid, camera, ann_id)
            if key not in gt_cache:
                try:
                    gt_cache[key] = load_gt_grasps(
                        gt_root=gt_root,
                        scene_id=sid,
                        ann_id=ann_id,
                        camera=camera,
                        split=args.split,
                    )
                except Exception:
                    continue
            gt = gt_cache[key]
            try:
                m = _compare_one_frame(b_path, r_path, gt, args.top_k)
            except Exception:
                continue
            scene_frame_metrics.append(m)
            all_frame_metrics.append(m)

        if not scene_frame_metrics:
            continue
        per_scene[scene_name] = {
            "num_frames": float(len(scene_frame_metrics)),
            "topk_overlap": float(np.mean([x["topk_overlap"] for x in scene_frame_metrics])),
            "mean_baseline_dist": float(np.mean([x["mean_baseline_dist"] for x in scene_frame_metrics])),
            "mean_reranker_dist": float(np.mean([x["mean_reranker_dist"] for x in scene_frame_metrics])),
            "improved_ratio": float(np.mean([x["improved_ratio"] for x in scene_frame_metrics])),
            "worsened_ratio": float(np.mean([x["worsened_ratio"] for x in scene_frame_metrics])),
        }

    overlaps = [x["topk_overlap"] for x in all_frame_metrics]
    b_dists = [x["mean_baseline_dist"] for x in all_frame_metrics]
    r_dists = [x["mean_reranker_dist"] for x in all_frame_metrics]
    improved = [x["improved_ratio"] for x in all_frame_metrics]
    worsened = [x["worsened_ratio"] for x in all_frame_metrics]
    dist_diffs = [x["distance_diff_mean"] for x in all_frame_metrics]

    global_summary = {
        "num_scenes": int(len(per_scene)),
        "num_frames": int(len(all_frame_metrics)),
        "overlap": _arr_stats(overlaps),
        "baseline_dist": _arr_stats(b_dists),
        "reranker_dist": _arr_stats(r_dists),
        "improved_ratio_mean": float(np.mean(improved)) if improved else float("nan"),
        "worsened_ratio_mean": float(np.mean(worsened)) if worsened else float("nan"),
    }

    print("=== GLOBAL SUMMARY ===")
    print(
        "TopK overlap: mean={:.4f} std={:.4f}".format(
            global_summary["overlap"]["mean"], global_summary["overlap"]["std"]
        )
    )
    print(
        "Baseline dist: mean={:.6f} std={:.6f}".format(
            global_summary["baseline_dist"]["mean"], global_summary["baseline_dist"]["std"]
        )
    )
    print(
        "Reranker dist: mean={:.6f} std={:.6f}".format(
            global_summary["reranker_dist"]["mean"], global_summary["reranker_dist"]["std"]
        )
    )
    print("Improved: {:.2f}%".format(100.0 * global_summary["improved_ratio_mean"]))
    print("Worsened: {:.2f}%".format(100.0 * global_summary["worsened_ratio_mean"]))

    report = {
        "config": {
            "baseline_dump_root": baseline_root,
            "reranker_dump_root": reranker_root,
            "gt_root": gt_root,
            "split": args.split,
            "top_k": int(args.top_k),
            "scene_filter": sorted(scene_filter) if scene_filter is not None else None,
            "max_frames_per_scene": args.max_frames_per_scene,
        },
        "global_summary": global_summary,
        "per_scene": per_scene,
    }
    report_path = os.path.join(args.out_dir, "batch_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {report_path}")

    if args.save_hist and all_frame_metrics:
        hist_path = os.path.join(args.out_dir, "batch_hist.png")
        _save_histogram(overlaps, dist_diffs, hist_path)
        print(f"Saved histogram: {hist_path}")


if __name__ == "__main__":
    main()
