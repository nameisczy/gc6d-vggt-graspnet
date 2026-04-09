#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同一帧下对比 repro 与 pipeline 两份 dump 的 top-K 抓取（Open3D），用于检查偏移/落点。

依赖：GC6D_ROOT、graspclutter6dAPI、open3d。

示例：
  export GC6D_ROOT=/path/to/GraspClutter6D
  python scripts/visualize_compare_two_dumps.py \\
    --scene_id 42 --ann_id 0 --camera realsense-d435 \\
    --repro_npy /path/repro/000042/realsense-d435/000002.npy \\
    --pipeline_npy /path/pipe/000042/realsense-d435/000002.npy \\
    --top 20
"""
from __future__ import annotations

import argparse
import os
import sys


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


def _topk_colored_geometries(gg, top: int, color):
    """按 score 降序取前 top 条，统一着色。"""
    from graspclutter6dAPI.grasp import Grasp

    gg2 = gg.__class__(gg.grasp_group_array.copy())
    gg2.sort_by_score(reverse=False)  # API: reverse=False -> 高分在前
    n = min(int(top), len(gg2))
    geoms = []
    for i in range(n):
        g = Grasp(gg2.grasp_group_array[i])
        geoms.append(g.to_open3d_geometry(color=color))
    return geoms


def main() -> None:
    ap = argparse.ArgumentParser(description="对比两份 dump 的 top-K 抓取可视化")
    ap.add_argument("--gc6d_root", default=os.environ.get("GC6D_ROOT"), help="GraspClutter6D 数据集根目录")
    ap.add_argument("--scene_id", type=int, required=True)
    ap.add_argument("--ann_id", type=int, default=0)
    ap.add_argument("--camera", default="realsense-d435")
    ap.add_argument("--repro_npy", required=True)
    ap.add_argument("--pipeline_npy", required=True)
    ap.add_argument("--top", type=int, default=20, help="每侧显示最高分 grasp 条数")
    ap.add_argument("--max_width", type=float, default=0.14, help="与 eval_scene 一致，先 clip 宽度再可视化")
    args = ap.parse_args()

    if not args.gc6d_root or not os.path.isdir(args.gc6d_root):
        print("请设置 --gc6d_root 或环境变量 GC6D_ROOT")
        sys.exit(1)

    import numpy as np
    import open3d as o3d

    from graspclutter6dAPI import GraspClutter6D
    from graspclutter6dAPI.grasp import GraspGroup

    g6d = GraspClutter6D(args.gc6d_root, camera=args.camera, split="test")
    scene_pcd = g6d.loadScenePointCloud(
        sceneId=args.scene_id, camera=args.camera, annId=args.ann_id, align=False
    )

    def load_and_clip(path: str) -> GraspGroup:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        gg = GraspGroup().from_npy(path)
        arr = gg.grasp_group_array
        min_width_mask = arr[:, 1] < 0
        max_width_mask = arr[:, 1] > args.max_width
        arr = arr.copy()
        arr[min_width_mask, 1] = 0
        arr[max_width_mask, 1] = args.max_width
        gg.grasp_group_array = arr
        return gg

    repro = load_and_clip(args.repro_npy)
    pipe = load_and_clip(args.pipeline_npy)

    # 可选：与 eval_utils 类似做一次 NMS，便于看「去重后」分布（默认关）
    # repro = repro.nms(0.03, 15.0 / 180.0 * np.pi)

    color_repro = (1.0, 0.25, 0.15)  # 橙红
    color_pipe = (0.2, 0.55, 1.0)  # 蓝

    geoms_repro = _topk_colored_geometries(repro, args.top, color_repro)
    geoms_pipe = _topk_colored_geometries(pipe, args.top, color_pipe)

    img_num = ann_id_to_img_id(args.ann_id, args.camera)
    print(
        "scene=%06d ann_id=%d -> img_id=%06d | repro N=%d pipeline N=%d | 显示各 top-%d"
        % (args.scene_id, args.ann_id, img_num, len(repro), len(pipe), args.top)
    )
    print("窗口1: 场景点云 + [repro] 抓取（橙红）")
    o3d.visualization.draw_geometries([scene_pcd] + geoms_repro, window_name="repro dump top-%d" % args.top)
    print("窗口2: 场景点云 + [pipeline] 抓取（蓝）")
    o3d.visualization.draw_geometries([scene_pcd] + geoms_pipe, window_name="pipeline dump top-%d" % args.top)

    # 叠加（便于直接对比重叠情况）
    print("窗口3: 叠加 repro(橙红) + pipeline(蓝)")
    o3d.visualization.draw_geometries(
        [scene_pcd] + geoms_repro + geoms_pipe,
        window_name="repro + pipeline overlay",
    )


if __name__ == "__main__":
    main()
