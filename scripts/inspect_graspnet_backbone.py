#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印 GraspNet 中 Pointnet2Backbone 的输入/输出张量形状（需能 import graspnet-baseline）。

用法（在已安装 torch、且能 import backbone 的环境中）:
  export GRASPNET_BASELINE=~/graspnet-baseline
  cd ~/gc6d_grasp_pipeline
  python scripts/inspect_graspnet_backbone.py
  python scripts/inspect_graspnet_backbone.py --batch 2 --num_points 20000
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--num_points", type=int, default=20000)
    ap.add_argument("--input_feature_dim", type=int, default=0, help="每点额外特征维，0 仅 xyz")
    ap.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = os.environ.get("GRASPNET_BASELINE", os.path.expanduser("~/graspnet-baseline"))
    root = os.path.abspath(root)
    models_dir = os.path.join(root, "models")
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    if root not in sys.path:
        sys.path.insert(0, root)
    for sub in ("pointnet2", "utils"):
        p = os.path.join(root, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from backbone import Pointnet2Backbone

    device = torch.device(args.device)
    B, N = args.batch, args.num_points
    C = 3 + args.input_feature_dim
    pc = torch.randn(B, N, C, device=device, dtype=torch.float32)

    net = Pointnet2Backbone(input_feature_dim=args.input_feature_dim).to(device)
    net.eval()
    with torch.no_grad():
        seed_features, seed_xyz, end_points = net(pc, end_points=None)

    print("========== Pointnet2Backbone（GraspNet view_estimator 内）==========")
    print("输入 pointcloud: (B, N, 3+input_feature_dim)")
    print(f"  本例: ({B}, {N}, {C})")
    print()
    print("返回值 (与 GraspNetStage1.forward 一致):")
    print("  1) seed_features  (B, 256, num_seed)  — 供 ApproachNet / 注入 cond 的通道维")
    print(f"      实际: tuple 或 tensor? -> {type(seed_features)} shape={getattr(seed_features, 'shape', 'N/A')}")
    print("  2) seed_xyz       (B, num_seed, 3)  — seed 点坐标，num_seed=1024 (sa2 点数)")
    print(f"      实际: {seed_xyz.shape}")
    print("  3) end_points     dict，含 input_xyz / fp2_xyz / fp2_features / sa* 等")
    print()
    for k in (
        "input_xyz",
        "fp2_xyz",
        "fp2_features",
        "sa2_xyz",
        "sa4_xyz",
    ):
        if k in end_points and end_points[k] is not None:
            t = end_points[k]
            print(f"  end_points['{k}']: {tuple(t.shape)}")
    print()
    print("说明: sa2 下采样到 npoint=1024，故 fp2_xyz / seed 数为 1024。")
    print("      EncoderAdapter 里 cond (B,256) 会 unsqueeze 成 (B,256,1) 与 seed_features 相加。")


if __name__ == "__main__":
    main()
