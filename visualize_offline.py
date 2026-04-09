#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, GC6DLIFT3DFormatDataset, collate_gc6d, collate_lift3d
from utils import action10_to_graspgroup, load_policy_from_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--camera", default="realsense-d415")
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--num_grasps", type=int, default=5)
    p.add_argument("--save_ply", action="store_true")
    p.add_argument("--no_render", action="store_true", help="skip Open3D window/screenshot (avoids segfault on headless)")
    p.add_argument("--lift3d_root", type=str, default=None, help="LIFT3D 根目录，用于加载 lift3d 的 ckpt")
    args = p.parse_args()

    ckpt = args.checkpoint or os.path.join(ROOT, "checkpoints", "gc6d_grasp_policy_one_sample.pt")
    out_dir = args.out_dir or os.path.join(ROOT, "vis_out")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    encoder_type = ckpt_meta.get("encoder_type", "placeholder")
    use_vggt = encoder_type in ("vggt_base", "vggt_ft")

    if use_vggt:
        dataset = GC6DLIFT3DFormatDataset(
            data_dir=args.data_dir, split=args.split, camera=args.camera,
            max_samples=args.max_samples, image_size=224,
        )
        loader = DataLoader(dataset, batch_size=args.max_samples, shuffle=False, collate_fn=collate_lift3d)
        batch = next(iter(loader))
        images, pcs, _, _, actions_gt, _, metas = batch
        images, pcs = images.to(args.device), pcs.to(args.device)
    else:
        dataset = GC6DOfflineUnifiedDataset(
            data_dir=args.data_dir, split=args.split, camera=args.camera, max_samples=args.max_samples,
        )
        loader = DataLoader(dataset, batch_size=args.max_samples, shuffle=False, collate_fn=collate_gc6d)
        batch = next(iter(loader))
        pcs, actions_gt, _, metas = batch
        pcs = pcs.to(args.device)
        images = None

    model = load_policy_from_checkpoint(ckpt, device=args.device, lift3d_root=args.lift3d_root)
    model.eval()
    with torch.no_grad():
        actions_pred = model(images) if use_vggt else model(pcs)
    action_np = actions_pred[0].cpu().numpy()
    action_gt_np = actions_gt[0].numpy()
    pc_np = pcs[0].cpu().numpy()

    # 预测 grasp（红色系）
    grasp_group = action10_to_graspgroup(
        action_np, pc_np, num_grasps=max(args.num_grasps, 10), score_mode="centroid"
    )
    gg_sub = grasp_group[: args.num_grasps] if len(grasp_group) > args.num_grasps else grasp_group
    gripper_geoms = gg_sub.to_open3d_geometry_list()

    # GT grasp（绿色）用于对比
    gt_grasp_group = action10_to_graspgroup(
        action_gt_np, pc_np, num_grasps=1, score_mode="centroid"
    )
    gt_geoms = gt_grasp_group.to_open3d_geometry_list()
    for g in gt_geoms:
        if hasattr(g, "paint_uniform_color"):
            g.paint_uniform_color([0.2, 0.8, 0.2])
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np.astype(np.float64))
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geoms = [pcd] + gripper_geoms + gt_geoms

    sample_id = metas[0].get("sample_id", "0")
    ply_path = os.path.join(out_dir, "scene_{}.ply".format(sample_id))
    o3d.io.write_point_cloud(ply_path, pcd)
    print("Saved", ply_path)

    # 无显示或指定 --no_render 时不开 Visualizer，避免 GLFW 段错误
    has_display = os.environ.get("DISPLAY") or os.environ.get("MESA_GL_VERSION_OVERRIDE")
    if args.no_render or not has_display:
        if not has_display:
            print("No DISPLAY/headless: skipping window render.")
        print("Open the ply in MeshLab or run on a machine with display for render.")
        return

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=960, height=720)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        img_path = os.path.join(out_dir, "render_{}.png".format(sample_id))
        vis.capture_screen_image(img_path)
        vis.destroy_window()
        print("Saved", img_path)
    except Exception as e:
        print("Offscreen render failed:", e)
        print("Open the ply in MeshLab or run on a machine with display.")


if __name__ == "__main__":
    main()
