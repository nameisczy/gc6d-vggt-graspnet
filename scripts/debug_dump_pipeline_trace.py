#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对单帧跑与 eval_benchmark 相同的 GraspNet dump 流水线，打印各阶段 grasp 数量；
可选与 repro 导出的 .npy 对比（仅统计条数与 score 分位数）。

示例：
  cd ~/gc6d_grasp_pipeline
  python scripts/debug_dump_pipeline_trace.py \\
    --checkpoint /path/to.pt \\
    --data_dir /path/to/offline_unified \\
    --dataset_root /path/to/GraspClutter6D \\
    --scene_id 42 --ann_id 0 --camera realsense-d435

  # 与 repro dump 对比（两路径均为 graspclutter6d 布局下同一 npy）
  python scripts/debug_dump_pipeline_trace.py \\
    --repro_npy /path/to/repro_dump/000042/realsense-d435/000002.npy \\
    --pipeline_npy /path/to/pipeline_dump/000042/realsense-d435/000002.npy
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stats_npy(path: str) -> None:
    import numpy as np

    g = np.load(path)
    if g.ndim != 2 or g.shape[1] != 17:
        print("  [skip] 非 (N,17):", g.shape)
        return
    n = g.shape[0]
    s = g[:, 0]
    print("  path:", path)
    print("  N=%d  score min/median/max=%.4f / %.4f / %.4f" % (n, float(np.min(s)), float(np.median(s)), float(np.max(s))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump 流水线各阶段数量 / 两份 npy 对比")
    ap.add_argument("--checkpoint", type=str, default=None, help="Encoder+GraspNet ckpt")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--dataset_root", type=str, default=os.environ.get("GC6D_ROOT"))
    ap.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    ap.add_argument("--camera", type=str, default="realsense-d435")
    ap.add_argument("--scene_id", type=int, default=None)
    ap.add_argument("--ann_id", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--max_dump_grasps", type=int, default=4096)
    ap.add_argument(
        "--pre_dump_collision_filter",
        dest="pre_dump_collision_filter",
        action="store_true",
        default=True,
        help="与 eval_benchmark 一致：dump 前 ModelFreeCollisionDetector",
    )
    ap.add_argument(
        "--no_pre_dump_collision_filter",
        dest="pre_dump_collision_filter",
        action="store_false",
        help="关闭预碰撞",
    )
    ap.add_argument("--collision_thresh", type=float, default=0.01)
    ap.add_argument("--collision_voxel_size", type=float, default=0.01)
    ap.add_argument("--graspnet_ckpt", type=str, default=None)
    ap.add_argument("--graspnet_root", type=str, default=None)
    ap.add_argument("--lift3d_root", type=str, default=None)
    ap.add_argument("--repro_npy", type=str, default=None, help="repro 侧 dump .npy")
    ap.add_argument("--pipeline_npy", type=str, default=None, help="pipeline 侧 dump .npy")
    args = ap.parse_args()

    if args.repro_npy or args.pipeline_npy:
        print("=== 静态 .npy 对比 ===")
        if args.repro_npy:
            print("[repro]")
            _stats_npy(os.path.abspath(os.path.expanduser(args.repro_npy)))
        if args.pipeline_npy:
            print("[pipeline]")
            _stats_npy(os.path.abspath(os.path.expanduser(args.pipeline_npy)))
        if not (args.checkpoint and args.scene_id is not None and args.ann_id is not None):
            return

    if args.scene_id is None or args.ann_id is None:
        print("未指定 --scene_id / --ann_id，跳过在线 trace。仅 npy 对比时无需指定。")
        if not (args.repro_npy or args.pipeline_npy):
            ap.print_help()
        return

    if args.checkpoint is None:
        print("需要 --checkpoint 以运行在线 trace")
        return
    if not args.data_dir:
        print("需要 --data_dir")
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from data import GC6DOfflineUnifiedDataset, collate_gc6d
    from utils import load_policy_from_checkpoint
    from utils.dump_pipeline_trace import trace_encoder_graspnet_dump

    ds = GC6DOfflineUnifiedDataset(
        data_dir=args.data_dir,
        split=args.split,
        camera=args.camera,
        max_samples=None,
        load_gt_multi=True,
    )
    target = (args.scene_id, args.ann_id)
    idx = None
    for i in range(len(ds)):
        meta = ds[i][3]
        if int(meta["sceneId"]) == target[0] and int(meta["annId"]) == target[1]:
            idx = i
            break
    if idx is None:
        print("在 index 中未找到 scene_id=%s ann_id=%s" % target)
        return

    loader = DataLoader(
        torch.utils.data.Subset(ds, [idx]),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_gc6d,
        num_workers=0,
    )
    batch = next(iter(loader))
    pcs, _, _, metas = batch
    device = torch.device(args.device)
    pcs = pcs.to(device)

    model = load_policy_from_checkpoint(
        args.checkpoint,
        device=args.device,
        lift3d_root=args.lift3d_root,
        graspnet_ckpt=args.graspnet_ckpt,
        graspnet_root=args.graspnet_root,
    )
    model.eval()
    if not hasattr(model, "grasp_net"):
        print("模型无 grasp_net，本脚本仅支持 EncoderAdapterGraspNet")
        return

    with torch.no_grad():
        end_points = model(pcs)
    pc_np = pcs[0].detach().cpu().numpy()

    api_split = "train" if args.split == "val" else args.split
    if api_split not in ("all", "train", "test"):
        api_split = "test"

    trace = trace_encoder_graspnet_dump(
        end_points,
        pc_np,
        device=device,
        max_dump_grasps=args.max_dump_grasps,
        pre_collision=args.pre_dump_collision_filter,
        collision_thresh=args.collision_thresh,
        collision_voxel_size=args.collision_voxel_size,
        graspnet_root=args.graspnet_root,
        dataset_root=args.dataset_root,
        camera=args.camera,
        scene_id=args.scene_id,
        ann_id=args.ann_id,
        api_split=api_split,
    )

    print("\n=== 在线 trace（与 eval_benchmark Encoder+GraspNet 路径一致）===")
    print("  scene_id=%s ann_id=%s camera=%s" % (args.scene_id, args.ann_id, args.camera))
    for k in (
        "raw_pred_decode",
        "after_sort_and_topk_pad_strip",
        "after_pre_dump_collision",
        "final_dump_rows",
        "after_api_foreground_filter",
    ):
        print("  %-32s %s" % (k, trace.get(k)))
    if trace.get("pre_dump_collision_error"):
        print("  pre_dump_collision_error:", trace["pre_dump_collision_error"])
    if trace.get("after_api_foreground_filter") == -1:
        print("  [warn] foreground 统计时物体数与 pose 数不一致，跳过")


if __name__ == "__main__":
    main()
