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


def _grasp_arr_to_t_R_w(grasp_row):
    """GraspGroup 一行 [score,w,h,d, R9, t3, obj_id] -> t(3), R(3,3), width, depth."""
    t = np.asarray(grasp_row[13:16], dtype=np.float64)
    R = np.asarray(grasp_row[4:13], dtype=np.float64).reshape(3, 3)
    w = float(grasp_row[1])
    d = float(grasp_row[3])
    return t, R, w, d


def _draw_gripper_official(ax, t, R, width, depth, score=1.0, color=(1, 0, 0), edgecolor="darkred", alpha=0.8):
    try:
        from graspclutter6dAPI.utils.utils import plot_gripper_pro_max
    except ImportError:
        from graspclutter6dAPI.graspclutter6dAPI.utils.utils import plot_gripper_pro_max
    mesh = plot_gripper_pro_max(t, R, width, depth, score=score, color=color)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = verts[tris]
    coll = Poly3DCollection(faces, facecolors=color, edgecolors=edgecolor, linewidths=0.5, alpha=alpha)
    ax.add_collection3d(coll)


def _draw_gripper_wireframe(ax, t, R, width, depth, color="red", linewidth=2):
    half = width / 2
    left_base = t - half * R[:, 1]
    right_base = t + half * R[:, 1]
    left_tip = left_base + depth * R[:, 0]
    right_tip = right_base + depth * R[:, 0]
    for (a, b) in [
        (left_base, left_tip), (right_base, right_tip),
        (left_base, right_base), (left_tip, right_tip),
    ]:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=linewidth)


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
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--mode", default="sequence", choices=["sequence", "close"],
                   help="sequence: 逐帧加入每个 grasp; close: 对最佳 grasp 做闭合动画")
    p.add_argument("--use_official_gripper", action="store_true",
                   help="使用 GraspClutter6D 官方的 plot_gripper_pro_max 绘制夹爪（需 graspclutter6dAPI）")
    p.add_argument("--lift3d_root", type=str, default=None, help="LIFT3D 根目录，用于加载 lift3d 的 ckpt")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ckpt = args.checkpoint or os.path.join(ROOT, "checkpoints", "gc6d_grasp_policy_one_sample.pt")
    out_dir = args.out_dir or os.path.join(ROOT, "vis_out")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    use_vggt = ckpt_meta.get("encoder_type", "placeholder") in ("vggt_base", "vggt_ft")

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

    action_gt_np = actions_gt[0].numpy()
    model = load_policy_from_checkpoint(ckpt, device=args.device, lift3d_root=args.lift3d_root)
    model.eval()
    with torch.no_grad():
        actions_pred = model(images) if use_vggt else model(pcs)
    action_np = actions_pred[0].cpu().numpy()
    pc_np = pcs[0].cpu().numpy()

    grasp_group = action10_to_graspgroup(
        action_np, pc_np, num_grasps=max(args.num_grasps, 10), score_mode="centroid"
    )
    gg_arr = grasp_group.grasp_group_array
    n_show = min(args.num_grasps, len(gg_arr))
    # GT 单 grasp 用于每帧对比（绿色）
    gt_t, gt_R, gt_w, gt_d = _grasp_arr_to_t_R_w(
        action10_to_graspgroup(
            action_gt_np, pc_np, num_grasps=1, score_mode="centroid"
        ).grasp_group_array[0]
    )

    sample_id = metas[0].get("sample_id", "0")
    frames_dir = os.path.join(out_dir, "anim_frames_{}".format(sample_id))
    os.makedirs(frames_dir, exist_ok=True)

    # 点云范围，用于固定视角
    xyz_min = pc_np.min(axis=0)
    xyz_max = pc_np.max(axis=0)
    center = (xyz_min + xyz_max) / 2
    radius = max(np.ptp(pc_np, axis=0)) * 0.6

    def draw_grasp(ax, t, R, w, d, score=1.0, color="coral"):
        if args.use_official_gripper:
            if color == "green":
                rgb = (0.15, 1.0, 0.15)
                edgecolor = "darkgreen"
            else:
                rgb = (1, 0.4, 0.4)
                edgecolor = "darkred"
            _draw_gripper_official(ax, t, R, w, d, score=score, color=rgb, edgecolor=edgecolor)
        else:
            _draw_gripper_wireframe(ax, t, R, w, d, color=color)

    frames = []

    if args.mode == "sequence":
        for i in range(n_show + 1):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
                c="lightgray", s=1, alpha=0.4
            )
            for j in range(i):
                t, R, w, d = _grasp_arr_to_t_R_w(gg_arr[j])
                score = float(gg_arr[j][0])
                draw_grasp(ax, t, R, w, d, score=score, color="coral")
            draw_grasp(ax, gt_t, gt_R, gt_w, gt_d, score=1.0, color="green")
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Green=GT, Red=Pred | Grasp {} / {}".format(i, n_show) if i > 0 else "Green=GT")
            path = os.path.join(frames_dir, "frame_{:04d}.png".format(i))
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            frames.append(path)

    else:
        t, R, w, d = _grasp_arr_to_t_R_w(gg_arr[0])
        score = float(gg_arr[0][0])
        n_steps = 8
        for i in range(n_steps + 1):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                pc_np[:, 0], pc_np[:, 1], pc_np[:, 2],
                c="lightgray", s=1, alpha=0.4
            )
            frac = 1.0 - (i / n_steps)
            w_cur = w * (0.2 * frac + 1.0)
            draw_grasp(ax, t, R, w_cur, d, score=score, color="coral")
            draw_grasp(ax, gt_t, gt_R, gt_w, gt_d, score=1.0, color="green")
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Green=GT, Red=Pred close (step {}/{})".format(i, n_steps))
            path = os.path.join(frames_dir, "frame_{:04d}.png".format(i))
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            frames.append(path)

    # 导出 GIF
    try:
        import imageio
        gif_path = os.path.join(out_dir, "grasp_anim_{}.gif".format(sample_id))
        imageio.mimsave(gif_path, [imageio.imread(f) for f in frames], duration=1.0 / args.fps, loop=0)
        print("Saved GIF:", gif_path)
    except ImportError:
        print("Install imageio for GIF: pip install imageio")
        print("Frames saved in", frames_dir)

    print("Frames:", frames_dir)


if __name__ == "__main__":
    main()
