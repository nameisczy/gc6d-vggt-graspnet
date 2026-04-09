# -*- coding: utf-8 -*-
"""
单帧 dump 流程各阶段 grasp 数量统计，用于与 gc6d_graspnet_repro 对齐排查。
不修改 evaluator，仅复现：raw pred_decode → 去 pad →（可选）预碰撞 → API 内 foreground（≤5cm）逻辑。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def count_foreground_like_eval_scene(
    gg_array: np.ndarray,
    dataset_root: str,
    camera: str,
    scene_id: int,
    ann_id: int,
    split: str = "test",
    max_width: float = 0.14,
) -> int:
    """
    与 graspclutter6d_eval.eval_scene 中 background_filter 分支一致：
    宽度 clip 后，保留 grasp 中心到物体采样点最近距离 < 0.05m 的 grasp。
    """
    from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval
    from graspclutter6dAPI.utils.eval_utils import compute_point_distance, transform_points, voxel_sample_points

    arr = np.asarray(gg_array, dtype=np.float64)
    if arr.size == 0 or arr.shape[0] == 0:
        return 0
    min_width_mask = arr[:, 1] < 0
    max_width_mask = arr[:, 1] > max_width
    arr = arr.copy()
    arr[min_width_mask, 1] = 0
    arr[max_width_mask, 1] = max_width

    ge = GraspClutter6DEval(root=dataset_root, camera=camera, split=split)
    model_list, _, _ = ge.get_scene_models(sceneId=scene_id, annId=0)
    model_sampled_list = [voxel_sample_points(m, 0.008) for m in model_list]
    _, pose_list, _ = ge.get_model_poses(sceneId=scene_id, annId=ann_id)
    if len(model_sampled_list) != len(pose_list):
        return -1  # 与 API 假设不一致，调用方需知
    model_sampled_all = [
        transform_points(ms, pose) for ms, pose in zip(model_sampled_list, pose_list)
    ]
    model_sampled_all = np.concatenate(model_sampled_all, axis=0)
    dist = compute_point_distance(arr[:, 13:16], model_sampled_all)
    closest_dist = np.min(dist, axis=1)
    close_mask = closest_dist < 0.05
    return int(np.sum(close_mask))


def trace_encoder_graspnet_dump(
    end_points: dict,
    pc_np: np.ndarray,
    *,
    device: torch.device,
    max_dump_grasps: int = 4096,
    pre_collision: bool = True,
    collision_thresh: float = 0.01,
    collision_voxel_size: float = 0.01,
    graspnet_root: Optional[str] = None,
    dataset_root: Optional[str] = None,
    camera: str = "realsense-d415",
    scene_id: Optional[int] = None,
    ann_id: Optional[int] = None,
    api_split: str = "test",
) -> Dict[str, Any]:
    """
    返回各阶段数量；若提供 dataset_root + scene_id + ann_id，则额外计算「与 eval_scene 一致的 foreground 过滤后」数量。
    """
    from models.graspnet_adapter import (
        apply_model_free_collision_filter,
        pred_decode_17d,
        raw_pred_decode_num_grasps,
    )

    out: Dict[str, Any] = {}
    out["raw_pred_decode"] = int(raw_pred_decode_num_grasps(end_points))

    actions = pred_decode_17d(end_points, device=device, max_grasps=max(1, int(max_dump_grasps)))
    proposals = actions[0].detach().cpu().numpy()
    proposals = proposals[proposals[:, 0] > 1e-8]
    out["after_sort_and_topk_pad_strip"] = int(proposals.shape[0])

    if pre_collision and float(collision_thresh or 0) > 0:
        try:
            proposals = apply_model_free_collision_filter(
                proposals,
                pc_np,
                collision_thresh=float(collision_thresh),
                voxel_size=float(collision_voxel_size),
                graspnet_baseline_root=graspnet_root,
            )
            out["after_pre_dump_collision"] = int(proposals.shape[0])
        except Exception as e:
            out["after_pre_dump_collision"] = None
            out["pre_dump_collision_error"] = str(e)
    else:
        out["after_pre_dump_collision"] = int(proposals.shape[0])

    proposals = np.asarray(proposals, dtype=np.float32)
    if proposals.shape[0] > 0:
        idx = np.argsort(-proposals[:, 0])
        proposals = proposals[idx]
    out["final_dump_rows"] = int(proposals.shape[0])

    out["after_api_foreground_filter"] = None
    if (
        dataset_root
        and scene_id is not None
        and ann_id is not None
        and proposals.shape[0] > 0
    ):
        n_fg = count_foreground_like_eval_scene(
            proposals,
            dataset_root=dataset_root,
            camera=camera,
            scene_id=int(scene_id),
            ann_id=int(ann_id),
            split=api_split,
        )
        out["after_api_foreground_filter"] = n_fg

    # 「进入 eval_grasp」前的 grasp 数 = foreground 过滤后（与 eval_scene 读入 gg 后一致）
    return out
