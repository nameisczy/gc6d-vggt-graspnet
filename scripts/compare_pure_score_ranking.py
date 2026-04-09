#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较 pure GraspNet 训练前/后 checkpoint 的 score / ranking 语义。

输出：
- before_after_score_stats.json
- figures/objectness_hist.png
- figures/grasp_score_hist.png

说明：
- objectness / grasp_score 分布：直接来自模型 end_points
- top10/top50 collision / FC：基于已 dump 的 npy，通过 GraspClutter6DEval.eval_scene(return_list=True) 聚合
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, collate_gc6d
from utils.load_model import load_policy_from_checkpoint


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_hist(plt, a: np.ndarray, b: np.ndarray, labels: tuple[str, str], title: str, path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(a.ravel(), bins=100, density=True, alpha=0.4, label=labels[0])
    plt.hist(b.ravel(), bins=100, density=True, alpha=0.4, label=labels[1])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def _collect_score_stats(model, data_dir: str, camera: str, split: str, batch_size: int, num_batches: int, device) -> Dict[str, Any]:
    ds = GC6DOfflineUnifiedDataset(
        data_dir=data_dir,
        split=split,
        camera=camera,
        max_samples=batch_size * num_batches,
        load_gt_multi=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_gc6d, num_workers=0)
    objectness_vals = []
    grasp_score_vals = []
    top10_mean = []
    top50_mean = []
    model.eval()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        pcs, _actions, _rgb, metas = batch
        pcs = pcs.to(device)
        ep = model(point_cloud=pcs)
        obj = ep["objectness_score"].detach().float().cpu().numpy()  # (B,2,Ns)
        gs = ep["grasp_score_pred"].detach().float().cpu().numpy()   # (B,A,Ns,D)
        objectness_vals.append(obj)
        grasp_score_vals.append(gs)
        flat = gs.reshape(gs.shape[0], -1)
        for row in flat:
            srt = np.sort(row)[::-1]
            top10_mean.append(float(np.mean(srt[: min(10, len(srt))])))
            top50_mean.append(float(np.mean(srt[: min(50, len(srt))])))
    objectness = np.concatenate(objectness_vals, axis=0)
    grasp_score = np.concatenate(grasp_score_vals, axis=0)
    return {
        "objectness_score": objectness,
        "grasp_score_pred": grasp_score,
        "objectness_summary": {
            "mean": float(np.mean(objectness)),
            "std": float(np.std(objectness)),
        },
        "grasp_score_summary": {
            "mean": float(np.mean(grasp_score)),
            "std": float(np.std(grasp_score)),
        },
        "top10_mean_score": float(np.mean(top10_mean)),
        "top50_mean_score": float(np.mean(top50_mean)),
    }


def _aggregate_eval_lists(dump_dir: str, gc6d_root: str, camera: str, top_k: int) -> Dict[str, Any]:
    api_parent = os.path.abspath(os.path.expanduser(os.path.join(ROOT, "..", "graspclutter6dAPI")))
    if api_parent not in sys.path:
        sys.path.insert(0, api_parent)
    from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval

    ge = GraspClutter6DEval(root=gc6d_root, camera=camera, split="test")
    with open(os.path.join(gc6d_root, "split_info", "grasp_test_scene_ids.json"), "r", encoding="utf-8") as f:
        scene_ids = [int(x) for x in json.load(f)]
    top10_coll = []
    top50_coll = []
    top10_fc = []
    top50_fc = []
    for sid in scene_ids:
        scene_acc, _grasp_ll, score_ll, collision_ll = ge.eval_scene(
            sid, dump_dir, TOP_K=top_k, return_list=True, background_filter=True
        )
        _ = scene_acc
        for slist, clist in zip(score_ll, collision_ll):
            s = np.asarray(slist).ravel()
            c = np.asarray(clist).ravel()
            n10 = min(10, len(s))
            n50 = min(50, len(s))
            if n10 > 0:
                top10_coll.append(float(1.0 - np.mean(c[:n10].astype(np.float64))))
                top10_fc.append(float(np.sum(s[:n10] > 0)))
            if n50 > 0:
                top50_coll.append(float(1.0 - np.mean(c[:n50].astype(np.float64))))
                top50_fc.append(float(np.sum(s[:n50] > 0)))
    return {
        "top10_collision_free_rate": float(np.mean(top10_coll)) if top10_coll else None,
        "top50_collision_free_rate": float(np.mean(top50_coll)) if top50_coll else None,
        "top10_fc_success_count_mean": float(np.mean(top10_fc)) if top10_fc else None,
        "top50_fc_success_count_mean": float(np.mean(top50_fc)) if top50_fc else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before_ckpt", type=str, required=True)
    p.add_argument("--after_ckpt", type=str, required=True)
    p.add_argument("--before_dump_dir", type=str, required=True)
    p.add_argument("--after_dump_dir", type=str, required=True)
    p.add_argument("--gc6d_root", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--graspnet_ckpt", type=str, default=os.path.expanduser("~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"))
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt = _ensure_matplotlib()
    device = torch.device(args.device)

    before_model = load_policy_from_checkpoint(
        args.before_ckpt, device=str(device), graspnet_ckpt=args.graspnet_ckpt, graspnet_root=args.graspnet_root
    )
    after_model = load_policy_from_checkpoint(
        args.after_ckpt, device=str(device), graspnet_ckpt=args.graspnet_ckpt, graspnet_root=args.graspnet_root
    )

    before_stats = _collect_score_stats(before_model, args.data_dir, args.camera, "val", args.batch_size, args.num_batches, device)
    after_stats = _collect_score_stats(after_model, args.data_dir, args.camera, "val", args.batch_size, args.num_batches, device)
    before_eval = _aggregate_eval_lists(args.before_dump_dir, args.gc6d_root, args.camera, top_k=50)
    after_eval = _aggregate_eval_lists(args.after_dump_dir, args.gc6d_root, args.camera, top_k=50)

    _plot_hist(
        plt,
        before_stats["objectness_score"],
        after_stats["objectness_score"],
        ("before", "after"),
        "objectness_score distribution",
        os.path.join(fig_dir, "objectness_hist.png"),
    )
    _plot_hist(
        plt,
        before_stats["grasp_score_pred"],
        after_stats["grasp_score_pred"],
        ("before", "after"),
        "grasp_score_pred distribution",
        os.path.join(fig_dir, "grasp_score_hist.png"),
    )

    out = {
        "before": {
            "checkpoint": args.before_ckpt,
            "dump_dir": args.before_dump_dir,
            **before_stats["objectness_summary"],
            "grasp_score_mean": before_stats["grasp_score_summary"]["mean"],
            "grasp_score_std": before_stats["grasp_score_summary"]["std"],
            "top10_mean_score": before_stats["top10_mean_score"],
            "top50_mean_score": before_stats["top50_mean_score"],
            **before_eval,
        },
        "after": {
            "checkpoint": args.after_ckpt,
            "dump_dir": args.after_dump_dir,
            **after_stats["objectness_summary"],
            "grasp_score_mean": after_stats["grasp_score_summary"]["mean"],
            "grasp_score_std": after_stats["grasp_score_summary"]["std"],
            "top10_mean_score": after_stats["top10_mean_score"],
            "top50_mean_score": after_stats["top50_mean_score"],
            **after_eval,
        },
    }
    with open(os.path.join(args.out_dir, "before_after_score_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
