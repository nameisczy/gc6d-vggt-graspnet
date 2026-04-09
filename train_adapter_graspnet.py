#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoder + Adapter + 预训练 GraspNet 分阶段训练（不复用现有 train_stage* 逻辑）。
- Stage1: 冻结 encoder + 冻结 grasp_net，只训 adapter
- Stage2: 冻结 encoder，训 adapter + grasp_net
- Stage3: 冻结 grasp_net，训 encoder + adapter
- Stage4: 联合微调
- VGGT base: 只做 Stage1，然后 Stage2 用 (stage2_steps + stage4_steps) 步数

Loss: GraspNet 输出解码为 17D 后与 GT 17D 做 matching loss（action_loss_topk_matched_17d）。
数据: GC6D（point_cloud + 可选 images），load_gt_multi=True 取 gt_grasp_group。

训练与评估分离:
- 本脚本仅做训练：按你划分的 index_train / index_val 使用训练集训练、验证集做验证（17D matching loss）。
  单数据点=同一条上验证；小批量=同批+验证集；全量=验证集。不在此脚本内算 AP。
- 最后测试用 AP：训练完成后单独运行 eval_benchmark.py（指定 --checkpoint、--split test、--dataset_root）
  得到 AP/AP0.4/AP0.8。
"""

import argparse
import logging
import os
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, GC6DLIFT3DFormatDataset, collate_gc6d, collate_lift3d
from utils.loss import action_loss_topk_matched_17d, pad_gt_grasp_group_17d, ranking_align_loss
from models.graspnet_adapter import (
    build_encoder_adapter_graspnet,
    pred_decode_17d,
    pred_decode_17d_differentiable,
)

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


def _ann_id_to_img_id(ann_id: int, camera: str) -> int:
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


def _get_scene_name(scene_id: int) -> str:
    return "%06d" % (scene_id,)


def run_debug_rank_correlation(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_vggt: bool,
    max_batches: int = 4,
    max_grasps: int = 128,
    sort_in_train_decode: bool = True,
):
    """返回 (rank_corr, stats_dict)。sort_in_train_decode=False 时与实验 1 训练 decode 一致。"""
    model.eval()
    corrs = []
    all_scores, all_widths = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            if use_vggt:
                images, pcs, _, _, _, _, metas = batch
                images = images.to(device)
                pcs = pcs.to(device)
                end_points = model(point_cloud=pcs, images=images)
            else:
                pcs, _, _, metas = batch
                pcs = pcs.to(device)
                end_points = model(point_cloud=pcs, images=None)
            pred_train = pred_decode_17d_differentiable(
                end_points, device, max_grasps=max_grasps, sort_and_truncate=sort_in_train_decode
            )
            pred_bench = pred_decode_17d(end_points, device, max_grasps=max_grasps)
            all_scores.append(pred_train[:, :, 0].cpu().numpy().ravel())
            all_widths.append(pred_train[:, :, 1].cpu().numpy().ravel())
            B = pred_train.shape[0]
            for b in range(B):
                s_train = pred_train[b, :, 0].cpu().numpy().ravel()
                s_bench = pred_bench[b, :, 0].cpu().numpy().ravel()
                valid = np.isfinite(s_train) & np.isfinite(s_bench) & (s_bench > 0)
                if valid.sum() < 2:
                    continue
                s_bench_sorted = np.sort(s_bench[valid])[::-1]
                s_train_sorted = np.sort(s_train[valid])[::-1]
                std_t, std_b = np.nanstd(s_train_sorted), np.nanstd(s_bench_sorted)
                if std_t < 1e-9 or std_b < 1e-9:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.corrcoef(s_train_sorted, s_bench_sorted)[0, 1]
                if np.isfinite(r):
                    corrs.append(float(r))
    model.train()
    rank_corr = float(np.mean(corrs)) if corrs else float("nan")
    scores = np.concatenate(all_scores) if all_scores else np.array([0.0])
    widths = np.concatenate(all_widths) if all_widths else np.array([0.0])
    scores = scores[np.isfinite(scores)]
    widths = widths[np.isfinite(widths)]
    stats = {
        "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "score_std": float(np.std(scores)) if len(scores) > 1 else 0.0,
        "width_mean": float(np.mean(widths)) if len(widths) else float("nan"),
        "width_std": float(np.std(widths)) if len(widths) > 1 else 0.0,
    }
    return rank_corr, stats


def run_debug_benchmark_ap(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_vggt: bool,
    dataset_root: str,
    camera: str,
    top_k: int = 50,
) -> Optional[float]:
    """在固定子集上 dump 预测并调用 GraspClutter6D API 得到 AP（与 eval_benchmark 同流程）。无 dataset_root 时返回 None。"""
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root or ""))
    if not dataset_root or not os.path.isdir(dataset_root):
        return None
    split_info_dir = os.path.join(dataset_root, "split_info")
    if not os.path.isdir(split_info_dir):
        return None
    try:
        from graspclutter6dAPI.grasp import GraspGroup as GraspGroupAPI
        from graspclutter6dAPI import GraspClutter6DEval
    except ImportError:
        return None
    try:
        model.eval()
        with tempfile.TemporaryDirectory(prefix="gc6d_debug_dump_") as dump_folder:
            collected_scene_ids = set()
            with torch.no_grad():
                for batch in loader:
                    if use_vggt:
                        images, pcs, _, _, _, _, metas = batch
                        images = images.to(device)
                        pcs = pcs.to(device)
                        end_points = model(point_cloud=pcs, images=images)
                    else:
                        pcs, _, _, metas = batch
                        pcs = pcs.to(device)
                        end_points = model(point_cloud=pcs, images=None)
                    actions_pred = pred_decode_17d(end_points, device=pcs.device, max_grasps=top_k or 50)
                    for i in range(pcs.shape[0]):
                        scene_id = int(metas[i]["sceneId"])
                        ann_id = int(metas[i]["annId"])
                        cam = metas[i].get("camera", camera)
                        collected_scene_ids.add(scene_id)
                        object_id_from_gt = -1
                        if "gt_grasp_group" in metas[i] and np.asarray(metas[i]["gt_grasp_group"]).ndim == 2 and metas[i]["gt_grasp_group"].shape[0] > 0:
                            object_id_from_gt = int(metas[i]["gt_grasp_group"][0, 16])
                        proposals_np = actions_pred[i].cpu().numpy()
                        if proposals_np.ndim == 1:
                            proposals_np = proposals_np.reshape(1, -1)
                        if proposals_np.shape[1] == 17 and proposals_np.shape[0] > 0:
                            idx = np.argsort(-proposals_np[:, 0])
                            proposals_np = proposals_np[idx]
                        proposals_np = np.asarray(proposals_np, dtype=np.float32)
                        proposals_np[:, 16] = object_id_from_gt
                        grasp_group = GraspGroupAPI(proposals_np)
                        img_num = _ann_id_to_img_id(ann_id, cam)
                        scene_dir = os.path.join(dump_folder, _get_scene_name(scene_id), cam)
                        os.makedirs(scene_dir, exist_ok=True)
                        npy_path = os.path.join(scene_dir, "%06d.npy" % img_num)
                        grasp_group.save_npy(npy_path)
            scene_ids = sorted(collected_scene_ids)
            if not scene_ids:
                model.train()
                return None
            ge = GraspClutter6DEval(root=dataset_root, camera=camera, split="train")
            acc_list = [ge.eval_scene(sid, dump_folder, TOP_K=top_k) for sid in scene_ids]
            res = np.array(acc_list)
            _res = res.transpose(3, 0, 1, 2).reshape(6, -1)
            ap = [np.mean(res), _res[1], _res[3]]
        model.train()
        return float(100.0 * ap[0])
    except Exception as e:
        logging.getLogger(__name__).warning("run_debug_benchmark_ap 失败: %s", e)
        model.train()
        return None


def run_eval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_vggt: bool,
    loss_mode: str,
    loss_alpha: float,
    best_gt_weight: float,
    pred2gt_agg: str = "min",
) -> float:
    """在给定 DataLoader 上计算 17D matching loss 均值（eval 模式、无梯度）。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if use_vggt:
                images, pcs, _, _, _, _, metas = batch
                images = images.to(device)
                pcs = pcs.to(device)
            else:
                pcs, _, _, metas = batch
                pcs = pcs.to(device)
                images = None
            gt_17d = pad_gt_grasp_group_17d(metas, device)
            end_points = model(point_cloud=pcs, images=images)
            pred_17d = pred_decode_17d(end_points, device, max_grasps=128)
            loss = action_loss_topk_matched_17d(
                pred_17d, gt_17d,
                mode=loss_mode, alpha=loss_alpha, best_gt_weight=best_gt_weight,
                pred2gt_agg=pred2gt_agg,
            )
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def _setup_logging(log_file: Optional[str] = None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    root = logging.getLogger()
    if log_file:
        log_path = os.path.abspath(os.path.expanduser(log_file))
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)


logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Adapter+GraspNet 分阶段训练")
    p.add_argument("--data_dir", type=str, required=True, help="GC6D 数据目录（含 index_*.jsonl）")
    p.add_argument("--encoder", type=str, required=True,
                   choices=("lift3d", "lift3d_clip", "vggt_base", "vggt_ft"),
                   help="lift3d_clip 使用 LIFT3D 官方 lift3d_clip_base() 预训练")
    p.add_argument("--stage", type=int, required=True, choices=(1, 2, 3, 4))
    p.add_argument("--graspnet_ckpt", type=str, required=True, help="graspnet-baseline 预训练 checkpoint")
    p.add_argument("--graspnet_root", type=str, default=None)
    p.add_argument("--lift3d_root", type=str, default=None)
    p.add_argument("--lift3d_ckpt", type=str, default=None, help="LIFT3D PointNext 预训练 checkpoint，不设则 backbone 随机初始化")
    p.add_argument("--vggt_ckpt", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=0, help="0=全量，1=1 条，>0=小批量")
    p.add_argument("--steps", type=int, default=None, help="当前 stage 步数，不设则用默认（或由 epochs 推导）")
    p.add_argument("--epochs", type=int, default=None, help="全量/小批量时按 epoch 算 steps：steps=epochs*num_batches，与数据规模可比较")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr_scale", type=float, default=0.1,
                   help="encoder 参数学习率 = lr * encoder_lr_scale，避免冲坏预训练")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=None, help="固定随机种子便于复现，不设则不固定")
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--load_ckpt", type=str, default=None, help="从该 ckpt 续训（含 adapter/encoder）")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--encoder_feat_dim", type=int, default=256)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--lora_last_n_blocks", type=int, default=None,
                   help="仅对 backbone 最后 N 个 block 注入 LoRA，None=全注入，VGGT 建议 2~4")
    p.add_argument("--loss_alpha", type=float, default=0.7)
    p.add_argument("--loss_mode", type=str, default="bidir")
    p.add_argument("--loss_pred2gt_agg", type=str, default="min", choices=("min", "mean"),
                   help="预测→GT 分支对 K 个预测的聚合：min=取最优一个（过拟合可接近 0），mean=对 K 取平均")
    p.add_argument("--best_gt_weight", type=float, default=0.3)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--val_split", type=str, default="val",
                   help="训练过程中验证集 split，对应 index_{split}_{camera}.jsonl（最后测试 AP 请用单独指令 eval_benchmark.py --split test）")
    p.add_argument("--log_grad_norm", action="store_true", help="单样本过拟合诊断：每 val_every 步打一次梯度范数，若长期为 0 说明梯度断流")
    p.add_argument("--log_file", type=str, default=None, help="可选：将日志同时写入该文件（直接跑脚本时用；用 run 脚本时已有 run.log）")
    p.add_argument("--debug_benchmark_every", type=int, default=0,
                   help="每 N 步跑一次 benchmark 风格 AP + train/bench decode 排名相关，0=关闭，用于验证训练目标与 AP 是否一致")
    p.add_argument("--debug_benchmark_n", type=int, default=64, help="debug 用的固定样本数（前 N 条，不 shuffle）")
    p.add_argument("--debug_split", type=str, default="val", choices=("train", "val"),
                   help="中间 debug 用的数据划分：val=验证集（推荐，AP 更有参考意义），train=训练集前 N 条")
    p.add_argument("--debug_dataset_root", type=str, default="",
                   help="debug AP 用的 GC6D 根目录（scenes/split_info 等），空则只算 rank_corr 不算 AP")
    p.add_argument("--rank_align_weight", type=float, default=0.2,
                   help=">0 时加排序对齐 loss：让可微 decode 的 score 排序逼近 baseline decode，缓解目标错位；默认 0.2，可调 0.3~0.5")
    p.add_argument("--skip_stage1", action="store_true",
                   help="跳过 Stage1（仅 adapter）：主路径梯度常为 0，可直接从 Stage2 训 adapter+grasp_net")
    p.add_argument("--stage1_aux_cond_weight", type=float, default=1e-4,
                   help="Stage1 时 cond^2 辅助项系数，用于打通 adapter 梯度；0=仅当 loss 无梯度时自动加 1e-6")
    p.add_argument("--use_rank_weighted_loss", action="store_true",
                   help="pred2gt 分支按 rank 加权（前排权重大），不硬 top-K，避免早期 score 不准时坏 grasp 被强化")
    p.add_argument("--rank_weight_front", type=int, default=16, help="rank 1~front 权重 1.0")
    p.add_argument("--rank_weight_mid", type=int, default=48, help="rank front+1~mid 权重 0.5")
    p.add_argument("--rank_weight_rest_w", type=float, default=0.25, help="其余 rank 权重")
    p.add_argument("--train_head_only", action="store_true",
                   help="freeze encoder 与 adapter，只训 grasp head（诊断 baseline）")
    p.add_argument("--no_adapter", action="store_true", help="use_adapter=False，不加 conditioning")
    p.add_argument("--adapter_cond_coeff", type=float, default=0.25, help="cond 注入系数（当前推荐 baseline=0.25）")
    p.add_argument(
        "--adapter_cond_mode",
        type=str,
        default="additive",
        choices=("additive", "concat", "gated", "film"),
        help="concat: seed 维 concat 后 1x1 Conv1d 投影（见 EncoderAdapterGraspNet）",
    )
    p.add_argument("--no_sort_in_train_decode", action="store_true",
                   help="训练时可微 decode 不按 score 排序，仅 eval 时排序（实验 1）")
    p.add_argument("--use_collision_aux", action="store_true",
                   help="加入 collision-aware 辅助目标，与 --use_quality_aux 可单独或同时开")
    p.add_argument("--use_quality_aux", action="store_true",
                   help="加入 quality/force-closure-aware 辅助目标，与 --use_collision_aux 可单独或同时开")
    return p.parse_args()


# 各 stage 默认步数：全量时 encoder 阶段多给步数便于对比不同 encoder
DEFAULT_STEPS_FULL = {1: 1000, 2: 4000, 3: 5000, 4: 4000}  # Stage3/4 训 encoder，多 step 便于对比
DEFAULT_STEPS_SMALL = {1: 200, 2: 400, 3: 200, 4: 200}   # 小批量几百 step
DEFAULT_STEPS_1SAMPLE = {1: 50, 2: 80, 3: 50, 4: 50}     # 单样本几十 step


def _default_steps_for_stage(stage: int, max_samples: int, encoder: str) -> int:
    """根据 max_samples 和 stage 返回该 stage 默认步数。"""
    if max_samples <= 0:
        tab = DEFAULT_STEPS_FULL
    elif max_samples == 1:
        tab = DEFAULT_STEPS_1SAMPLE
    else:
        tab = DEFAULT_STEPS_SMALL
    steps = tab.get(stage, 200)
    if encoder == "vggt_base" and stage == 2:
        # 不微调 encoder，Stage2 仅 adapter+head，固定 4k+2k=6k
        steps = tab.get(2, 400) + 2000
    return steps


def main():
    args = parse_args()
    _setup_logging(args.log_file)
    if getattr(args, "seed", None) is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info("Random seed set to %d", args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_vggt = args.encoder in ("vggt_base", "vggt_ft")

    max_samples = args.max_samples
    max_samples_for_data = None if max_samples <= 0 else max_samples
    val_every = args.val_every
    load_gt_multi = True

    if use_vggt:
        dataset = GC6DLIFT3DFormatDataset(
            data_dir=args.data_dir,
            split="train",
            camera=args.camera,
            max_samples=max_samples_for_data,
            image_size=224,
            load_gt_multi=load_gt_multi,
        )
        collate_fn = collate_lift3d
    else:
        dataset = GC6DOfflineUnifiedDataset(
            data_dir=args.data_dir,
            split="train",
            camera=args.camera,
            max_samples=max_samples_for_data,
            load_gt_multi=load_gt_multi,
        )
        collate_fn = collate_gc6d

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    n_train = len(dataset)
    num_batches = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    if args.epochs is not None and args.epochs > 0:
        steps = args.epochs * num_batches
        logger.info("按 epoch 算 steps: epochs=%d, num_batches=%d -> steps=%d", args.epochs, num_batches, steps)
    elif args.steps is None:
        steps = _default_steps_for_stage(args.stage, max_samples, args.encoder)
    else:
        steps = args.steps
    step_schedule = "1sample(几十)" if max_samples == 1 else ("small(几百)" if max_samples and max_samples > 0 else "full(几千)")
    logger.info("Train samples: %d, stage=%d, steps=%d (%s)", n_train, args.stage, steps, step_schedule)
    if steps < 200 and val_every >= steps:
        val_every = max(1, steps // 5)
        logger.info("steps=%d 较小，val_every 调整为 %d", steps, val_every)

    # 评估数据：单点=同一条；小批量=同批+验证集；全量=仅验证集（按你划分的 train/val，最后测试 AP 用单独指令）
    eval_same_loader = None  # 与训练同分布/同数据
    eval_val_loader = None   # 验证集
    if args.max_samples == 1:
        eval_same_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
        logger.info("Eval: 单数据点过拟合，评估用同一数据点")
    elif args.max_samples and args.max_samples > 0:
        eval_same_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        logger.info("Eval: 小批量，同一批数据 + 验证集")
    else:
        logger.info("Eval: 全量，仅验证集")

    val_index_path = os.path.join(args.data_dir, f"index_{args.val_split}_{args.camera}.jsonl")
    if os.path.isfile(val_index_path):
        if use_vggt:
            val_dataset = GC6DLIFT3DFormatDataset(
                args.data_dir, split=args.val_split, camera=args.camera,
                image_size=224, load_gt_multi=True,
            )
        else:
            val_dataset = GC6DOfflineUnifiedDataset(
                args.data_dir, split=args.val_split, camera=args.camera,
                load_gt_multi=True,
            )
        eval_val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        logger.info("验证集 %s: %d 样本", args.val_split, len(val_dataset))
    else:
        eval_val_loader = None
        if (args.max_samples and args.max_samples > 0) or args.max_samples == 0:
            logger.warning("验证集 index 不存在 %s，仅做同批/训练集评估", val_index_path)

    if args.max_samples == 0 and eval_val_loader is None:
        logger.warning("全量训练且无验证集 index，将不进行验证")

    debug_loader = None
    if getattr(args, "debug_benchmark_every", 0) > 0:
        from torch.utils.data import Subset
        debug_split = getattr(args, "debug_split", "val")
        if debug_split == "val" and eval_val_loader is not None:
            base_dataset = eval_val_loader.dataset
            n_debug = min(getattr(args, "debug_benchmark_n", 64), len(base_dataset))
            debug_dataset = Subset(base_dataset, list(range(n_debug)))
            logger.info(
                "Debug 诊断: 每 %d 步跑 rank_corr + benchmark_AP(若 --debug_dataset_root)，固定 %d 条 **验证集**",
                args.debug_benchmark_every, n_debug,
            )
        else:
            n_debug = min(getattr(args, "debug_benchmark_n", 64), len(dataset))
            debug_dataset = Subset(dataset, list(range(n_debug)))
            logger.info(
                "Debug 诊断: 每 %d 步跑 rank_corr + benchmark_AP(若 --debug_dataset_root)，固定 %d 条 **训练集**",
                args.debug_benchmark_every, n_debug,
            )
        debug_loader = DataLoader(
            debug_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    use_adapt = not getattr(args, "no_adapter", False)
    model = build_encoder_adapter_graspnet(
        encoder_type=args.encoder,
        graspnet_ckpt=args.graspnet_ckpt,
        encoder_feat_dim=args.encoder_feat_dim,
        graspnet_root=args.graspnet_root,
        lift3d_root=args.lift3d_root,
        lift3d_ckpt=getattr(args, "lift3d_ckpt", None),
        vggt_ckpt=args.vggt_ckpt,
        lora_r=getattr(args, "lora_r", 8),
        lora_scale=getattr(args, "lora_scale", 1.0),
        lora_last_n_blocks=getattr(args, "lora_last_n_blocks", None),
        device=device,
        use_adapter=use_adapt,
        adapter_cond_coeff=getattr(args, "adapter_cond_coeff", 0.25),
        adapter_cond_mode=getattr(args, "adapter_cond_mode", "additive"),
    )
    model.to(device)

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded %s", args.load_ckpt)

    # 互斥的 stage 冻结逻辑，与文档一致：S1=仅 adapter，S2=adapter+grasp_net，S3=encoder+adapter、grasp_net 冻结，S4=联合
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.grasp_net.parameters():
        p.requires_grad = False
    if model.adapter is not None:
        for p in model.adapter.parameters():
            p.requires_grad = True

    if getattr(args, "train_head_only", False):
        if model.adapter is not None:
            for p in model.adapter.parameters():
                p.requires_grad = False
        for p in model.grasp_net.parameters():
            p.requires_grad = True
        logger.info("train_head_only: 仅 grasp_net 可训")
    elif args.stage == 1:
        pass  # 仅 adapter 已设
    elif args.stage == 2:
        for p in model.grasp_net.parameters():
            p.requires_grad = True
    elif args.stage == 3:
        # Stage3：只训 encoder(+adapter/pt_mlp)，grasp_net 保持冻结
        last_n = getattr(args, "lora_last_n_blocks", None)
        if hasattr(model.encoder, "set_backbone_lora_trainable"):
            model.encoder.set_backbone_lora_trainable(True, last_n_blocks=last_n)
            if hasattr(model.encoder, "set_adapter_trainable"):
                model.encoder.set_adapter_trainable(True)
            if hasattr(model.encoder, "pt_mlp"):
                for p in model.encoder.pt_mlp.parameters():
                    p.requires_grad = True
        elif hasattr(model.encoder, "get_lora_params") and callable(model.encoder.get_lora_params):
            for p in model.encoder.get_lora_params():
                p.requires_grad = True
            if hasattr(model.encoder, "adapter"):
                for p in model.encoder.adapter.parameters():
                    p.requires_grad = True
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
        # grasp_net 不训
    else:
        # Stage4：联合微调
        last_n = getattr(args, "lora_last_n_blocks", None)
        if hasattr(model.encoder, "set_backbone_lora_trainable"):
            model.encoder.set_backbone_lora_trainable(True, last_n_blocks=last_n)
            if hasattr(model.encoder, "set_adapter_trainable"):
                model.encoder.set_adapter_trainable(True)
            if hasattr(model.encoder, "pt_mlp"):
                for p in model.encoder.pt_mlp.parameters():
                    p.requires_grad = True
        elif hasattr(model.encoder, "get_lora_params") and callable(model.encoder.get_lora_params):
            for p in model.encoder.get_lora_params():
                p.requires_grad = True
            if hasattr(model.encoder, "adapter"):
                for p in model.encoder.adapter.parameters():
                    p.requires_grad = True
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
        for p in model.grasp_net.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    if args.log_grad_norm or os.environ.get("GC6D_LOG_TRAINABLE") == "1":
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info("Stage %d trainable (%d): %s", args.stage, len(trainable_names), trainable_names[:20] if len(trainable_names) > 20 else trainable_names)
    n_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_gn = sum(p.numel() for p in model.grasp_net.parameters() if p.requires_grad)
    n_ad = sum(p.numel() for p in (model.adapter.parameters() if model.adapter is not None else [])) if model.adapter is not None else 0
    logger.info("Stage %d: encoder=%d adapter=%d grasp_net=%d trainable", args.stage, n_enc, n_ad, n_gn)
    enc_param_ids = {id(p) for p in model.encoder.parameters()}
    ad_param_ids = {id(p) for p in (model.adapter.parameters() if model.adapter is not None else [])}
    enc_params = [p for p in params if id(p) in enc_param_ids]
    ad_params = [p for p in params if id(p) in ad_param_ids]
    head_params = [p for p in params if id(p) not in enc_param_ids and id(p) not in ad_param_ids]
    if enc_params or ad_params or head_params:
        groups = []
        if ad_params:
            groups.append({"params": ad_params, "lr": args.lr})
        if head_params:
            groups.append({"params": head_params, "lr": args.lr})
        if enc_params:
            groups.append({"params": enc_params, "lr": args.lr * args.encoder_lr_scale})
        optimizer = torch.optim.Adam(groups)
        logger.info("Optimizer: adapter=%d lr=%s, head=%d lr=%s, encoder=%d lr=%s",
                    len(ad_params), args.lr, len(head_params), args.lr, len(enc_params), args.lr * args.encoder_lr_scale)
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr)

    save_name = args.save_name or f"gc6d_{args.encoder}_adapter_graspnet_s{args.stage}"
    # 仅在使用默认 save_name 时加后缀，避免 run 脚本已带 _1sample 时出现双重后缀
    if args.save_name is None or args.save_name == "":
        if args.max_samples == 1:
            save_name += "_1sample"
        elif args.max_samples > 0:
            save_name += f"_small{args.max_samples}"
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(ROOT, "checkpoints", f"{save_name}.pt")

    _sort_in_train_decode = not getattr(args, "no_sort_in_train_decode", False)

    # rank-weighted loss：前排权重大，多 grasp 参与，避免硬 top-K
    rank_weights_tensor = None
    if getattr(args, "use_rank_weighted_loss", False):
        K = 128
        f = getattr(args, "rank_weight_front", 16)
        m = getattr(args, "rank_weight_mid", 48)
        rw = getattr(args, "rank_weight_rest_w", 0.25)
        w = torch.ones(K, dtype=torch.float32)
        w[f:m] = 0.5
        w[m:K] = rw
        rank_weights_tensor = w.to(device)
        logger.info("use_rank_weighted_loss: front=%d mid=%d rest_w=%.2f", f, m, rw)

    model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if use_vggt:
                images, pcs, _, _, _, _, metas = batch
                images = images.to(device)
                pcs = pcs.to(device)
            else:
                pcs, _actions, _rgb_paths, metas = batch
                pcs = pcs.to(device)
                images = None

            gt_17d = pad_gt_grasp_group_17d(metas, device)
            optimizer.zero_grad()
            # graspnet-baseline 的 backbone 用 FPS/group，梯度从 point_cloud 传到 fp2_xyz/input_xyz；若 pcs 无 grad，整条链到 grasp_score_pred 都无 grad
            if pcs.requires_grad is False and (args.stage == 1 or args.stage == 3):
                pcs = pcs.detach().requires_grad_(True)
            end_points = model(point_cloud=pcs, images=images)
            sort_in_train = not getattr(args, "no_sort_in_train_decode", False)
            pred_17d = pred_decode_17d_differentiable(end_points, device, max_grasps=128, sort_and_truncate=sort_in_train)
            loss = action_loss_topk_matched_17d(
                pred_17d,
                gt_17d,
                mode=args.loss_mode,
                alpha=args.loss_alpha,
                best_gt_weight=args.best_gt_weight,
                pred2gt_agg=args.loss_pred2gt_agg,
                rank_weights=rank_weights_tensor,
            )
            rank_w = getattr(args, "rank_align_weight", 0.0)
            if rank_w > 0:
                pred_bench = pred_decode_17d(end_points, device, max_grasps=128)
                score_train = pred_17d[:, :, 0]
                score_bench = pred_bench[:, :, 0].detach()
                loss_rank = ranking_align_loss(score_train, score_bench)
                loss = loss + rank_w * loss_rank
            need_aux = not loss.requires_grad or loss.grad_fn is None
            aux_w = getattr(args, "stage1_aux_cond_weight", 1e-4) if args.stage == 1 else 0.0
            if need_aux:
                which = []
                for k, v in end_points.items():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        which.append(k)
                logger.warning(
                    "loss 无梯度 (grad_fn=None)，主路径断。end_points 中 require_grad 的 key: %s；用 cond^2 辅助项打通 adapter 梯度",
                    which or "无",
                )
                if "_cond" in end_points:
                    cond = end_points["_cond"]
                    loss = loss + (aux_w if aux_w > 0 else 1e-6) * cond.pow(2).sum()
                else:
                    raise RuntimeError(
                        "loss 无梯度且无 _cond，无法 backward。请检查 GraspNet/encoder 是否在 forward 中 detach 或 no_grad。"
                    )
            elif args.stage == 1 and aux_w > 0 and "_cond" in end_points:
                # Stage1 可选：始终加 cond^2 辅助项，增强 adapter 梯度
                loss = loss + aux_w * end_points["_cond"].pow(2).sum()
            if getattr(args, "use_collision_aux", False):
                # TODO: 接入 collision 信号（数据或 API），实现 collision-aware 辅助 loss
                loss_collision_aux = pred_17d.new_zeros(1).squeeze(0)
                loss = loss + loss_collision_aux
            if getattr(args, "use_quality_aux", False):
                # TODO: 接入 quality/force-closure 信号，实现 quality-aware 辅助 loss
                loss_quality_aux = pred_17d.new_zeros(1).squeeze(0)
                loss = loss + loss_quality_aux
            loss.backward()
            if args.log_grad_norm and step > 0 and step % val_every == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                logger.info("[Stage%d] step %d grad_norm=%.6f (若长期≈0 则梯度断流)", args.stage, step, total_norm)
            optimizer.step()
            step += 1
            if step % 50 == 0 or step == 1:
                logger.info("[Stage%d] step %d/%d loss=%.6f", args.stage, step, steps, loss.item())
            if val_every > 0 and step % val_every == 0:
                eval_parts = []
                if eval_same_loader is not None:
                    loss_same = run_eval(
                        model, eval_same_loader, device, use_vggt,
                        args.loss_mode, args.loss_alpha, args.best_gt_weight,
                        args.loss_pred2gt_agg,
                    )
                    eval_parts.append("train_same=%.6f" % loss_same)
                if eval_val_loader is not None:
                    loss_val = run_eval(
                        model, eval_val_loader, device, use_vggt,
                        args.loss_mode, args.loss_alpha, args.best_gt_weight,
                        args.loss_pred2gt_agg,
                    )
                    eval_parts.append("val=%.6f" % loss_val)
                if eval_parts:
                    logger.info("[Stage%d] step %d %s", args.stage, step, " ".join(eval_parts))
            if debug_loader is not None and step % getattr(args, "debug_benchmark_every", 0) == 0:
                rank_corr, debug_stats = run_debug_rank_correlation(model, debug_loader, device, use_vggt, sort_in_train_decode=_sort_in_train_decode)
                debug_root = getattr(args, "debug_dataset_root", "") or os.environ.get("GC6D_ROOT", "")
                ap_val = run_debug_benchmark_ap(
                    model, debug_loader, device, use_vggt,
                    dataset_root=debug_root,
                    camera=args.camera,
                    top_k=50,
                )
                ap_str = "%.2f" % ap_val if ap_val is not None else "N/A"
                logger.info(
                    "[Stage%d] step %d debug rank_corr=%.4f debug_AP=%s | score mean=%.4f std=%.4f width mean=%.4f std=%.4f",
                    args.stage, step, rank_corr, ap_str,
                    debug_stats.get("score_mean", float("nan")), debug_stats.get("score_std", 0),
                    debug_stats.get("width_mean", float("nan")), debug_stats.get("width_std", 0),
                )
            if step >= steps:
                break
        if step >= steps:
            break

    # 训练结束前最后一次验证（17D loss）
    if val_every > 0 and (eval_same_loader is not None or eval_val_loader is not None):
        final_parts = []
        if eval_same_loader is not None:
            final_parts.append("train_same=%.6f" % run_eval(
                model, eval_same_loader, device, use_vggt,
                args.loss_mode, args.loss_alpha, args.best_gt_weight,
                args.loss_pred2gt_agg,
            ))
        if eval_val_loader is not None:
            final_parts.append("val=%.6f" % run_eval(
                model, eval_val_loader, device, use_vggt,
                args.loss_mode, args.loss_alpha, args.best_gt_weight,
                args.loss_pred2gt_agg,
            ))
        if final_parts:
            logger.info("[Stage%d] final %s", args.stage, " ".join(final_parts))

    if debug_loader is not None:
        rank_corr, debug_stats = run_debug_rank_correlation(
            model, debug_loader, device, use_vggt,
            sort_in_train_decode=_sort_in_train_decode,
        )
        debug_root = getattr(args, "debug_dataset_root", "") or os.environ.get("GC6D_ROOT", "")
        ap_val = run_debug_benchmark_ap(
            model, debug_loader, device, use_vggt,
            dataset_root=debug_root, camera=args.camera, top_k=50,
        )
        ap_str = "%.2f" % ap_val if ap_val is not None else "N/A"
        logger.info("[Stage%d] final debug rank_corr=%.4f debug_AP=%s | score mean=%.4f std=%.4f width mean=%.4f std=%.4f",
                    args.stage, rank_corr, ap_str,
                    debug_stats.get("score_mean", float("nan")), debug_stats.get("score_std", 0),
                    debug_stats.get("width_mean", float("nan")), debug_stats.get("width_std", 0))
        # 目标错位简要结论（看 log 时一眼能判断）
        if not np.isfinite(rank_corr):
            logger.info("[目标错位诊断] rank_corr 无法计算（可能同秩 score 方差为 0）；若 step 200 时曾 >0.3 则目标已对齐")
        elif rank_corr < 0.3:
            logger.info("[目标错位诊断] rank_corr=%.3f < 0.3 → 训练 decode 与 benchmark decode 排序不一致，存在目标错位风险", rank_corr)
        else:
            logger.info("[目标错位诊断] rank_corr=%.3f 尚可；若训练中 loss↓ 而 debug_AP 不升反降，仍可能存在目标错位", rank_corr)

    torch.save({
        "model": model.state_dict(),
        "encoder_type": args.encoder,
        "stage": args.stage,
        "graspnet_ckpt": args.graspnet_ckpt,
        "graspnet_root": getattr(args, "graspnet_root", None),
        "lora_r": getattr(args, "lora_r", None),
        "lora_scale": getattr(args, "lora_scale", None),
        "lora_last_n_blocks": getattr(args, "lora_last_n_blocks", None),
        "use_adapter": getattr(model, "use_adapter", True),
        "adapter_cond_coeff": getattr(model, "adapter_cond_coeff", 0.25),
        "adapter_cond_mode": getattr(model, "adapter_cond_mode", "additive"),
    }, ckpt_path)
    logger.info("Saved %s", ckpt_path)


if __name__ == "__main__":
    main()
