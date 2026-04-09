#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比：训练步中「整网 model.train()」vs「set_pure_graspnet_train_state」下 backbone
running_mean / running_var 是否变化。

用法:
  cd ~/gc6d_grasp_pipeline && conda activate gc6d
  python scripts/diagnose_pure_graspnet_bn.py --graspnet_ckpt ... --num_steps 30 --device cuda

输出: stdout 摘要 + 可选 --out_json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, collate_gc6d
from models.pure_graspnet import build_pure_graspnet_pipeline
from training.losses import compute_train_loss
from training.optim import apply_pure_graspnet_freeze, set_pure_graspnet_train_state


def snapshot_running_buffers(module: torch.nn.Module) -> dict:
    out = {}
    for name, buf in module.named_buffers():
        if "running_mean" in name or "running_var" in name:
            out[name] = buf.detach().float().cpu().clone()
    return out


def max_abs_diff(a: dict, b: dict) -> dict:
    keys = sorted(set(a.keys()) & set(b.keys()))
    per = {}
    all_max = 0.0
    for k in keys:
        d = (a[k] - b[k]).abs().max().item()
        per[k] = float(d)
        all_max = max(all_max, d)
    return {"per_buffer_max_abs": per, "global_max_abs": float(all_max), "n_buffers": len(keys)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--graspnet_ckpt", type=str, required=True)
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_steps", type=int, default=40)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    model = build_pure_graspnet_pipeline(
        graspnet_ckpt=args.graspnet_ckpt, graspnet_root=args.graspnet_root, device=device
    )
    apply_pure_graspnet_freeze(model, freeze_backbone=True, freeze_vpmodule=False)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    ds = GC6DOfflineUnifiedDataset(
        data_dir=args.data_dir,
        split="train",
        camera=args.camera,
        max_samples=max(args.batch_size * args.num_steps, args.batch_size * 4),
        load_gt_multi=True,
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_gc6d, num_workers=0)
    it = iter(loader)

    bb = model.grasp_net.view_estimator.backbone
    snap0 = snapshot_running_buffers(bb)

    # --- 旧行为：整网 train（backbone 仍为 train，BN 会更新 buffer）---
    for step in range(args.num_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        pcs, _, _, metas = batch
        pcs = pcs.to(device)
        model.train()
        opt.zero_grad()
        loss, _ = compute_train_loss(
            model=model,
            point_cloud=pcs,
            metas=metas,
            device=device,
            data_dir=args.data_dir,
        )
        loss.backward()
        opt.step()

    snap_buggy = snapshot_running_buffers(bb)
    diff_buggy = max_abs_diff(snap0, snap_buggy)

    # --- 重置模型与优化器，复现「修复后」行为 ---
    model2 = build_pure_graspnet_pipeline(
        graspnet_ckpt=args.graspnet_ckpt, graspnet_root=args.graspnet_root, device=device
    )
    apply_pure_graspnet_freeze(model2, freeze_backbone=True, freeze_vpmodule=False)
    opt2 = torch.optim.Adam([p for p in model2.parameters() if p.requires_grad], lr=1e-3)
    bb2 = model2.grasp_net.view_estimator.backbone
    snap0b = snapshot_running_buffers(bb2)
    it2 = iter(loader)
    for step in range(args.num_steps):
        try:
            batch = next(it2)
        except StopIteration:
            it2 = iter(loader)
            batch = next(it2)
        pcs, _, _, metas = batch
        pcs = pcs.to(device)
        set_pure_graspnet_train_state(model2)
        opt2.zero_grad()
        loss, _ = compute_train_loss(
            model=model2,
            point_cloud=pcs,
            metas=metas,
            device=device,
            data_dir=args.data_dir,
        )
        loss.backward()
        opt2.step()

    snap_fixed = snapshot_running_buffers(bb2)
    diff_fixed = max_abs_diff(snap0b, snap_fixed)

    report = {
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "backbone_running_buffer_count": len(snap0),
        "buggy_whole_model_train": {
            "description": "每步仅 model.train()，backbone 仍为 train 模式（与旧 train_alignment 一致）",
            "max_abs_change_running_buffers": diff_buggy["global_max_abs"],
            "n_buffers_compared": diff_buggy["n_buffers"],
            "top_5_buffers_by_max_abs": sorted(
                diff_buggy["per_buffer_max_abs"].items(), key=lambda x: -x[1]
            )[:5],
        },
        "fixed_backbone_eval_under_train": {
            "description": "每步 set_pure_graspnet_train_state（backbone.eval）",
            "max_abs_change_running_buffers": diff_fixed["global_max_abs"],
            "n_buffers_compared": diff_fixed["n_buffers"],
            "top_5_buffers_by_max_abs": sorted(
                diff_fixed["per_buffer_max_abs"].items(), key=lambda x: -x[1]
            )[:5],
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("Wrote", args.out_json, flush=True)


if __name__ == "__main__":
    main()
