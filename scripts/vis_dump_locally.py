#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地可视化 eval_benchmark 的 dump（预测抓取）。
用法：在本地安装 graspclutter6dAPI，设置 GC6D_ROOT 和 DUMP_FOLDER，然后：
  export GC6D_ROOT=/path/to/GraspClutter6D
  export DUMP_FOLDER=/path/to/dump_test
  python vis_dump_locally.py --scene_id 0 --camera realsense-d415 --ann_id 0 --num_grasp 30
"""
import os
import argparse


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    """与 GraspClutter6D API 一致。"""
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


def main():
    ap = argparse.ArgumentParser(description="可视化 dump 中的预测抓取（需 GC6D_ROOT + graspclutter6dAPI）")
    ap.add_argument("--dump_folder", default=os.environ.get("DUMP_FOLDER", "./dump_test"), help="dump 根目录")
    ap.add_argument("--scene_id", type=int, default=0, help="场景 id")
    ap.add_argument("--camera", default="realsense-d415")
    ap.add_argument("--ann_id", type=int, default=0, help="该场景下帧 id，0~12")
    ap.add_argument("--num_grasp", type=int, default=50, help="最多显示抓取数")
    ap.add_argument("--max_width", type=float, default=0.14)
    args = ap.parse_args()

    if "GC6D_ROOT" not in os.environ:
        print("请设置 GC6D_ROOT，例如: export GC6D_ROOT=/path/to/GraspClutter6D")
        return
    gc6d_root = os.environ["GC6D_ROOT"]
    dump_folder = os.path.abspath(os.path.expanduser(args.dump_folder))
    scene_name = "%06d" % args.scene_id
    img_num = ann_id_to_img_id(args.ann_id, args.camera)
    npy_path = os.path.join(dump_folder, scene_name, args.camera, "%06d.npy" % img_num)
    if not os.path.isfile(npy_path):
        print("Dump 文件不存在:", npy_path)
        return

    from graspclutter6dAPI import GraspClutter6D
    from graspclutter6dAPI.grasp import GraspGroup
    import numpy as np
    import open3d as o3d

    g = GraspClutter6D(gc6d_root, camera=args.camera, split="test")
    scene_pcd = g.loadScenePointCloud(sceneId=args.scene_id, camera=args.camera, annId=args.ann_id, align=False)
    gg = GraspGroup().from_npy(npy_path)
    gg = gg.nms(translation_thresh=0.03, rotation_thresh=15.0 / 180.0 * np.pi)
    w = gg.grasp_group_array[:, 1]
    gg.grasp_group_array = gg.grasp_group_array[w <= args.max_width]
    gg = gg[: args.num_grasp]
    geoms = [scene_pcd] + gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
