#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC6D 评估：按 GraspClutter6D API 与 examples 的流程：
1) 用模型对 index 中样本逐条推理，将预测的 GraspGroup(17D) 按 API 要求的目录布局 dump 成 .npy；
2) 调用 GraspClutter6DEval.eval_all(dump_folder) [test] 或 eval_scene(dump_folder) 逐场景 [val] 得到 AP。

依赖：graspclutter6dAPI（pip 或 本机 graspclutter6dAPI 目录）、原始 GC6D 数据集 root（--dataset_root）。

诊断 AP≈0：使用 --eval_gt_dump 用 GT 17D 作为 dump 再跑同一套 API。若 GT 的 AP 很高（如 >30%），
说明评估逻辑与 17D 格式正确，问题在训练；若 GT 的 AP 也接近 0，则需检查评估/坐标系/路径等。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, GC6DLIFT3DFormatDataset, collate_gc6d, collate_lift3d
from utils import load_policy_from_checkpoint
from utils.action2grasp import action10_batch_to_graspgroup, action10_to_t_R_w, grasp_group_row_to_action10, proposals_11d_to_graspgroup
from graspclutter6dAPI.grasp import GraspGroup as GraspGroupAPI
from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    """与 GraspClutter6D API 一致：annId -> 图像编号。"""
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


def get_scene_name(scene_id: int) -> str:
    """与 API eval_utils.get_scene_name 一致：6 位字符串。"""
    return "%06d" % (scene_id,)


def _dump_alignment_dict(args, top_k: int, eval_background_filter: bool) -> dict:
    """写入 summary.json，便于与 repro 配置对照。"""
    return {
        "max_dump_grasps": int(args.max_dump_grasps),
        "top_k_eval": int(top_k),
        "pre_dump_collision_filter": bool(args.pre_dump_collision_filter),
        "collision_thresh": float(args.collision_thresh),
        "collision_voxel_size": float(getattr(args, "collision_voxel_size", 0.01)),
        "api_background_filter": bool(eval_background_filter),
    }


def main():
    parser = argparse.ArgumentParser(description="GC6D benchmark：dump 预测后调用 GraspClutter6D API 评估")
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--camera", type=str, default="realsense-d415")
    parser.add_argument("--max_samples", type=int, default=0, help="0=全部，>0 只评估前 N 条")
    parser.add_argument("--dataset_root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D",
                        help="原始 GC6D 数据集根目录（scenes/, models_m/, split_info/ 等），供 API 使用")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_k", type=int, default=50, help="API eval 的 TOP_K（GraspClutter6DEval.eval_scene / eval_all）")
    parser.add_argument(
        "--max_dump_grasps",
        type=int,
        default=4096,
        help="pred_decode 后写入 dump 的最大条数（与 repro baseline_infer 一致应足够大以保留碰撞过滤前的高分 grasp；仅截断排序后的前 N 条）",
    )
    parser.add_argument(
        "--pre_dump_collision_filter",
        dest="pre_dump_collision_filter",
        action="store_true",
        default=True,
        help="dump 前使用与 repro baseline_infer 相同的 ModelFreeCollisionDetector（默认开启以对齐 repro）",
    )
    parser.add_argument(
        "--no_pre_dump_collision_filter",
        dest="pre_dump_collision_filter",
        action="store_false",
        help="关闭 dump 前点云碰撞预滤除（旧行为：仅依赖 evaluator 内碰撞）",
    )
    parser.add_argument("--collision_thresh", type=float, default=0.01, help="与 repro baseline_infer 一致，0 表示关闭预碰撞")
    parser.add_argument("--collision_voxel_size", type=float, default=0.01, help="ModelFreeCollisionDetector voxel_size")
    parser.add_argument(
        "--no_background_filter",
        action="store_true",
        help="关闭 API eval_scene 内 foreground 过滤（默认与 repro 一致：background_filter=True）",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--lift3d_root", type=str, default=None)
    parser.add_argument("--graspnet_ckpt", type=str, default=None,
                        help="EncoderAdapterGraspNet 评估时若 ckpt 内未保存 graspnet_ckpt 则需传入")
    parser.add_argument("--graspnet_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--proc", type=int, default=2, help="eval_all 时并行进程数（仅 test）")
    parser.add_argument("--use_proposals", action="store_true")
    parser.add_argument("--no_use_proposals", action="store_true")
    parser.add_argument("--direct_17d", action="store_true")
    parser.add_argument("--depth", type=float, default=0.04)
    parser.add_argument("--width_max", type=float, default=0.12)
    parser.add_argument("--R_flatten", type=str, default="row", choices=("row", "col"))
    parser.add_argument("--R_permute", type=str, default="012")
    parser.add_argument("--eval_gt_dump", action="store_true",
                        help="用 GT 17D 作为 dump 内容再跑 API，用于验证评估流程：若 GT 的 AP 高则说明评估逻辑正确、问题在训练")
    parser.add_argument("--extra_stats", action="store_true",
                        help="输出 top-50 collision 剩余率 与 top-50 force-closure 成功数（需 API 支持 return_list=True）")
    args = parser.parse_args()
    if not (args.dataset_root and args.dataset_root.strip()):
        args.dataset_root = os.environ.get("GC6D_ROOT", "")
    dataset_root = os.path.abspath(os.path.expanduser(args.dataset_root or ""))
    if not dataset_root or not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            "原始 GC6D 数据集根目录不存在或未设置。请指定 --dataset_root=/path/to/GraspClutter6D（该目录下应有 split_info/, scenes/, models_m/ 等）"
        )
    split_info_dir = os.path.join(dataset_root, "split_info")
    if not os.path.isdir(split_info_dir):
        raise FileNotFoundError("--dataset_root 下无 split_info 目录: %s" % dataset_root)
    for name in ("grasp_train_scene_ids.json", "grasp_test_scene_ids.json"):
        p = os.path.join(split_info_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError("缺少 %s，请确认 dataset_root 为完整 GC6D 数据集: %s" % (name, dataset_root))
    args.dataset_root = dataset_root
    eval_background_filter = not getattr(args, "no_background_filter", False)

    if args.eval_gt_dump:
        use_vggt = False
        encoder_type = "eval_gt_dump"
    else:
        if args.checkpoint is None:
            args.checkpoint = os.path.join(ROOT, "checkpoints", "gc6d_grasp_policy_one_sample.pt")
        ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder_type = ckpt_meta.get("encoder_type", "placeholder")
        use_vggt = encoder_type in ("vggt_base", "vggt_ft")

    out_root = args.out_dir or os.path.join(ROOT, "eval_out")
    ckpt_basename = os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint else "default"
    eval_out_dir = os.path.join(out_root, encoder_type, ckpt_basename)
    os.makedirs(eval_out_dir, exist_ok=True)
    dump_folder = os.path.join(eval_out_dir, "dump_%s" % args.split)
    os.makedirs(dump_folder, exist_ok=True)

    load_gt_multi = True
    try:
        if use_vggt:
            dataset = GC6DLIFT3DFormatDataset(
                data_dir=args.data_dir, split=args.split, camera=args.camera,
                max_samples=args.max_samples if args.max_samples > 0 else None,
                image_size=224,
                load_gt_multi=load_gt_multi,
            )
            collate_fn = collate_lift3d
        else:
            dataset = GC6DOfflineUnifiedDataset(
                data_dir=args.data_dir, split=args.split, camera=args.camera,
                max_samples=args.max_samples if args.max_samples > 0 else None,
                load_gt_multi=load_gt_multi,
            )
            collate_fn = collate_gc6d
    except FileNotFoundError as e:
        if args.split == "test":
            raise FileNotFoundError(
                "测试集 index 不存在。请确认 --data_dir 下存在 index_test_%s.jsonl。%s" % (args.camera, e)
            ) from e
        raise

    n_total = len(dataset)
    if n_total == 0:
        print("index 无样本: split=%s camera=%s" % (args.split, args.camera))
        return
    print(
        "评估样本数: %d | dump_folder: %s\n  dump 对齐: max_dump_grasps=%d pre_collision=%s collision_thresh=%s voxel=%s api_background_filter=%s"
        % (
            n_total,
            dump_folder,
            int(args.max_dump_grasps),
            bool(args.pre_dump_collision_filter),
            args.collision_thresh,
            getattr(args, "collision_voxel_size", 0.01),
            eval_background_filter,
        ),
        flush=True,
    )
    if args.eval_gt_dump:
        print(" [eval_gt_dump] 使用 GT 17D 作为 dump，用于验证评估流程；若 GT 的 AP 高则评估逻辑正确、低 AP 来自训练")
        model = None
        use_proposals_eval = False
        direct_17d_eval = False
    else:
        print("加载 checkpoint 与模型（VGGT/GraspNet 等，首次可能较慢）...", flush=True)
        model = load_policy_from_checkpoint(
            args.checkpoint, device=args.device, lift3d_root=args.lift3d_root,
            graspnet_ckpt=args.graspnet_ckpt, graspnet_root=args.graspnet_root,
        )
        model.eval()
        print("模型加载完成，开始推理与 dump...", flush=True)
        use_proposals_eval = (args.use_proposals and not args.no_use_proposals and hasattr(model, "forward_proposals")) or hasattr(model, "grasp_net")
        direct_17d_eval = use_proposals_eval and args.direct_17d and hasattr(model, "forward_proposals_raw")
        is_adapter_graspnet = hasattr(model, "grasp_net")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    collected_scene_ids = set()

    for batch_idx, batch in enumerate(loader):
        if use_vggt:
            images, pcs, _, _, actions_gt, _, metas = batch
            images = images.to(args.device)
            pcs = pcs.to(args.device)
            if not args.eval_gt_dump:
                with torch.no_grad():
                    if is_adapter_graspnet:
                        from models.graspnet_adapter import pred_decode_17d
                        end_points = model(pcs, images=images)
                        actions_pred = pred_decode_17d(
                            end_points, device=pcs.device, max_grasps=max(1, int(args.max_dump_grasps or 4096))
                        )
                    elif direct_17d_eval:
                        actions_pred = model.forward_proposals_raw(images)
                    elif use_proposals_eval:
                        actions_pred = model.forward_proposals(images)
                    else:
                        actions_pred = model(images)
        else:
            pcs, actions_gt, _, metas = batch
            pcs = pcs.to(args.device)
            if not args.eval_gt_dump:
                with torch.no_grad():
                    if is_adapter_graspnet:
                        from models.graspnet_adapter import pred_decode_17d
                        end_points = model(pcs)
                        actions_pred = pred_decode_17d(
                            end_points, device=pcs.device, max_grasps=max(1, int(args.max_dump_grasps or 4096))
                        )
                    elif direct_17d_eval:
                        actions_pred = model.forward_proposals_raw(pcs)
                    elif use_proposals_eval:
                        actions_pred = model.forward_proposals(pcs)
                    else:
                        actions_pred = model(pcs)

        for i in range(pcs.shape[0]):
            scene_id = int(metas[i]["sceneId"])
            ann_id = int(metas[i]["annId"])
            camera = metas[i].get("camera", args.camera)
            collected_scene_ids.add(scene_id)

            object_id_from_gt = -1
            if "gt_grasp_group" in metas[i] and np.asarray(metas[i]["gt_grasp_group"]).ndim == 2 and metas[i]["gt_grasp_group"].shape[0] > 0:
                object_id_from_gt = int(metas[i]["gt_grasp_group"][0, 16])

            if args.eval_gt_dump:
                if "gt_grasp_group" not in metas[i]:
                    continue
                arr = np.asarray(metas[i]["gt_grasp_group"], dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 17:
                    continue
                gg_save = GraspGroupAPI(arr)
            elif use_proposals_eval:
                proposals_np = actions_pred[i].cpu().numpy()
                if proposals_np.ndim == 1:
                    proposals_np = proposals_np.reshape(1, -1)
                if proposals_np.shape[1] == 17:
                    # pred_decode_17d 已按 score 降序；去掉 pad 的零行，再按 score 排一次保证与旧 ckpt 兼容
                    if proposals_np.shape[0] > 0:
                        proposals_np = proposals_np[proposals_np[:, 0] > 1e-8]
                        if proposals_np.shape[0] > 0:
                            idx = np.argsort(-proposals_np[:, 0])
                            proposals_np = proposals_np[idx]
                    proposals_np = np.asarray(proposals_np, dtype=np.float32)
                    # 与 gc6d_graspnet_repro 的 baseline_infer 一致：dump 前点云碰撞预滤除
                    if getattr(args, "pre_dump_collision_filter", True) and float(getattr(args, "collision_thresh", 0.01) or 0) > 0:
                        try:
                            from models.graspnet_adapter import apply_model_free_collision_filter

                            pc_np = pcs[i].detach().cpu().numpy()
                            proposals_np = apply_model_free_collision_filter(
                                proposals_np,
                                pc_np,
                                collision_thresh=float(args.collision_thresh),
                                voxel_size=float(getattr(args, "collision_voxel_size", 0.01)),
                                graspnet_baseline_root=args.graspnet_root,
                            )
                        except Exception as ex:
                            print(
                                "[warn] pre_dump_collision_filter 失败，将跳过该步（与旧行为接近）: %s" % ex,
                                flush=True,
                            )
                    if proposals_np.shape[0] > 0:
                        idx = np.argsort(-proposals_np[:, 0])
                        proposals_np = proposals_np[idx]
                    proposals_np = np.asarray(proposals_np, dtype=np.float32)
                    if proposals_np.shape[0] > 0:
                        proposals_np[:, 16] = object_id_from_gt
                    grasp_group = GraspGroupAPI(proposals_np)
                elif direct_17d_eval and proposals_np.shape[1] == 11:
                    grasp_group = proposals_11d_to_graspgroup(
                        proposals_np, object_id=object_id_from_gt, depth=args.depth, width_max=args.width_max,
                    )
                else:
                    grasp_group = action10_batch_to_graspgroup(
                        proposals_np,
                        R_flatten=args.R_flatten,
                        object_id=object_id_from_gt,
                        R_permute=args.R_permute,
                        depth=args.depth,
                        width_max=args.width_max,
                    )
            else:
                action_np = actions_pred[i].cpu().numpy()
                if action_np.size == 17:
                    row17 = np.asarray(action_np, dtype=np.float32).reshape(1, 17)
                    row17[0, 16] = object_id_from_gt
                    grasp_group = GraspGroupAPI(row17)
                else:
                    grasp_group = action10_batch_to_graspgroup(
                        action_np.reshape(1, -1),
                        R_flatten=args.R_flatten,
                        object_id=object_id_from_gt,
                        R_permute=args.R_permute,
                        depth=args.depth,
                        width_max=args.width_max,
                    )

            if not args.eval_gt_dump:
                if not hasattr(grasp_group, "grasp_group_array"):
                    grasp_group = GraspGroupAPI(np.asarray(grasp_group, dtype=np.float32))
                arr = np.asarray(grasp_group.grasp_group_array, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 17:
                    continue
                gg_save = GraspGroupAPI(arr)
            img_num = ann_id_to_img_id(ann_id, camera)
            scene_dir = os.path.join(dump_folder, get_scene_name(scene_id), camera)
            os.makedirs(scene_dir, exist_ok=True)
            npy_path = os.path.join(scene_dir, "%06d.npy" % img_num)
            gg_save.save_npy(npy_path)

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            print("[dump] %d / %d" % (min((batch_idx + 1) * args.batch_size, n_total), n_total), flush=True)

    api_split = "train" if args.split == "val" else args.split
    if api_split not in ("all", "train", "test"):
        api_split = "test"
    ge = GraspClutter6DEval(root=args.dataset_root, camera=args.camera, split=api_split)
    top_k = args.top_k or 50
    extra_stats = getattr(args, "extra_stats", False)

    def _agg_extra_stats(grasp_list_list, score_list_list, collision_list_list, k):
        rates, fc_counts = [], []
        for glist, slist, clist in zip(grasp_list_list, score_list_list, collision_list_list):
            if not len(slist):
                continue
            slist = np.asarray(slist).ravel()
            clist = np.asarray(clist).ravel()
            n = min(len(slist), k)
            if n == 0:
                continue
            rates.append(1.0 - np.mean(clist[:n].astype(np.float64)))
            fc_counts.append(np.sum(slist[:n] > 0))
        return (np.mean(rates) if rates else None, np.mean(fc_counts) if fc_counts else None)

    if args.split == "test":
        if extra_stats:
            with open(os.path.join(args.dataset_root, "split_info", "grasp_test_scene_ids.json")) as f:
                scene_ids = [int(x) for x in json.load(f)]
            acc_list, collision_rates, fc_means = [], [], []
            for sid in scene_ids:
                ret = ge.eval_scene(
                    sid, dump_folder, TOP_K=top_k, return_list=True, background_filter=eval_background_filter
                )
                scene_acc, grasp_ll, score_ll, collision_ll = ret
                acc_list.append(scene_acc)
                r, fc = _agg_extra_stats(grasp_ll, score_ll, collision_ll, top_k)
                if r is not None:
                    collision_rates.append(r)
                if fc is not None:
                    fc_means.append(fc)
            res = np.array([[np.asarray(ann) for ann in scene_acc] for scene_acc in acc_list])
            _res = res.copy().transpose(3, 0, 1, 2).reshape(6, -1)
            _res = np.mean(_res, axis=1)
            ap = [np.mean(res), _res[1], _res[3]]
        else:
            res, ap = ge.eval_all(dump_folder, proc=args.proc)
        ap_mean, ap_04, ap_08 = 100.0 * ap[0], 100.0 * ap[1], 100.0 * ap[2]
        summary = {
            "encoder_type": encoder_type,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "camera": args.camera,
            "n_samples_dumped": n_total,
            "AP": round(ap_mean, 4),
            "AP0.4": round(ap_04, 4),
            "AP0.8": round(ap_08, 4),
            "dump_alignment": _dump_alignment_dict(args, top_k, eval_background_filter),
        }
        if extra_stats and collision_rates and fc_means:
            summary["top50_collision_remaining_rate"] = round(float(np.mean(collision_rates)), 4)
            summary["top50_force_closure_success_count_mean"] = round(float(np.mean(fc_means)), 2)
        print("\nEvaluation Result (GraspClutter6D API):\n  AP=%.2f  AP0.4=%.2f  AP0.8=%.2f" % (ap_mean, ap_04, ap_08))
        if extra_stats and collision_rates and fc_means:
            print("  top50_collision_remaining_rate=%.4f  top50_force_closure_success_count_mean=%.2f" % (np.mean(collision_rates), np.mean(fc_means)))
    else:
        scene_ids = sorted(collected_scene_ids)
        if not scene_ids:
            print("无有效 scene，跳过 API eval_scene")
            return
        acc_list, collision_rates, fc_means = [], [], []
        for sid in scene_ids:
            if extra_stats:
                ret = ge.eval_scene(
                    sid, dump_folder, TOP_K=top_k, return_list=True, background_filter=eval_background_filter
                )
                scene_acc, grasp_ll, score_ll, collision_ll = ret
                acc_list.append(scene_acc)
                r, fc = _agg_extra_stats(grasp_ll, score_ll, collision_ll, top_k)
                if r is not None:
                    collision_rates.append(r)
                if fc is not None:
                    fc_means.append(fc)
            else:
                acc = ge.eval_scene(sid, dump_folder, TOP_K=top_k, background_filter=eval_background_filter)
                acc_list.append(acc)
        if extra_stats:
            res = np.array([[np.asarray(ann) for ann in scene_acc] for scene_acc in acc_list])
        else:
            res = np.array(acc_list)
        _res = res.copy()
        _res = _res.transpose(3, 0, 1, 2).reshape(6, -1)
        _res = np.mean(_res, axis=1)
        ap = [np.mean(res), _res[1], _res[3]]
        ap_mean, ap_04, ap_08 = 100.0 * ap[0], 100.0 * ap[1], 100.0 * ap[2]
        summary = {
            "encoder_type": encoder_type,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "camera": args.camera,
            "n_samples_dumped": n_total,
            "n_scenes_eval": len(scene_ids),
            "AP": round(ap_mean, 4),
            "AP0.4": round(ap_04, 4),
            "AP0.8": round(ap_08, 4),
            "dump_alignment": _dump_alignment_dict(args, top_k, eval_background_filter),
        }
        if extra_stats and collision_rates and fc_means:
            summary["top50_collision_remaining_rate"] = round(float(np.mean(collision_rates)), 4)
            summary["top50_force_closure_success_count_mean"] = round(float(np.mean(fc_means)), 2)
        print("\nEvaluation Result (GraspClutter6D API eval_scene):\n  AP=%.2f  AP0.4=%.2f  AP0.8=%.2f" % (ap_mean, ap_04, ap_08))
        if extra_stats and collision_rates and fc_means:
            print("  top50_collision_remaining_rate=%.4f  top50_force_closure_success_count_mean=%.2f" % (np.mean(collision_rates), np.mean(fc_means)))

    summary_path = os.path.join(eval_out_dir, "summary_%s.json" % args.split)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Summary: %s" % summary_path)


if __name__ == "__main__":
    main()
