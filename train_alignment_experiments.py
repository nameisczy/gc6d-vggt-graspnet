#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五主实验统一训练（alignment v2）：

  pure_graspnet | vggt_replacement | lift3d_replacement_clip | lift3d_replacement_dinov2 | vggt_fusion_normalized

- LIFT3D：CLIP 用官方 ``lift3d_clip_base``（预训练）；DINOv2 用 ``torch.hub`` ViT-B/14 + BEV 栅格（非裸 PointNext）。
- 不修改 benchmark 口径；可选训练后调用 eval_benchmark_rewrite.py。
- 结果表见 --results_table（含 encoder_type）。
- pure_graspnet：冻结 backbone 时，训练步内对 ``view_estimator.backbone`` 使用 ``eval()``（``set_pure_graspnet_train_state``），避免整网 ``model.train()`` 导致 BN running_mean/var 仍被更新。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, collate_gc6d
from models.dinov2_replacement_pipeline import build_dinov2_replacement_graspnet
from models.lift3d_clip_replacement_pipeline import build_lift3d_clip_replacement_graspnet
from models.lift3d_progressive_replacement_pipeline import (
    build_dinov2_progressive_replacement_graspnet,
    build_lift3d_clip_progressive_replacement_graspnet,
)
from models.lift3d_replacement_distill_pipeline import (
    build_dinov2_replacement_distill_graspnet,
    build_lift3d_clip_replacement_distill_graspnet,
)
from models.pure_graspnet import build_pure_graspnet_pipeline
from models.vggt_backbone_fusion_pipeline import (
    build_vggt_fusion_distill_graspnet,
    build_vggt_fusion_progressive_graspnet,
)
from models.vggt_fusion_normalized_pipeline import build_vggt_fusion_normalized_graspnet
from models.vggt_progressive_replacement_pipeline import build_vggt_progressive_replacement_graspnet
from models.vggt_replacement_distill_pipeline import build_vggt_replacement_distill_graspnet
from models.vggt_replacement_pipeline import build_vggt_replacement_graspnet
from training.losses import compute_train_loss, evaluate_mean_val_loss
from training.optim import apply_alignment_freeze, build_optimizer, set_alignment_train_state


def _setup_logging(log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)


def _save_checkpoint(path: str, model: torch.nn.Module, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "encoder_type": meta.get("encoder_type", meta.get("model_mode", "alignment")),
        "model_mode": meta.get("model_mode"),
        "graspnet_ckpt": meta.get("graspnet_ckpt"),
        "graspnet_root": meta.get("graspnet_root"),
        "lift3d_root": meta.get("lift3d_root"),
        "vggt_ckpt": meta.get("vggt_ckpt"),
        "vggt_sample_k": meta.get("vggt_sample_k"),
        "replacement_align_mode": meta.get("replacement_align_mode"),
        "replacement_affine_init_scale": meta.get("replacement_affine_init_scale"),
        "replacement_adapter_hidden": meta.get("replacement_adapter_hidden"),
        "replacement_adapter_depth": meta.get("replacement_adapter_depth"),
        "replacement_scale_mode": meta.get("replacement_scale_mode"),
        "replacement_fixed_alpha": meta.get("replacement_fixed_alpha"),
        "replacement_learnable_scale_init": meta.get("replacement_learnable_scale_init"),
        "pure_score_calibration_mode": meta.get("pure_score_calibration_mode"),
        "pure_score_delta_scale": meta.get("pure_score_delta_scale"),
        "pure_score_calibration_hidden": meta.get("pure_score_calibration_hidden"),
        "pure_train_score_calibrator_only": meta.get("pure_train_score_calibrator_only"),
        "quality_aux_mode": meta.get("quality_aux_mode"),
        "ranking_aux": meta.get("ranking_aux"),
        "ranking_aux_weight": meta.get("ranking_aux_weight"),
        "ranking_margin": meta.get("ranking_margin"),
        "fuse_residual": meta.get("fuse_residual"),
        "fuse_alpha": meta.get("fuse_alpha"),
        "lora_r": meta.get("lora_r"),
        "lora_scale": meta.get("lora_scale"),
        "lora_last_n_blocks": meta.get("lora_last_n_blocks"),
        "progressive_alpha": meta.get("progressive_alpha"),
        "progressive_score_calibration_mode": meta.get("progressive_score_calibration_mode"),
        "progressive_score_delta_scale": meta.get("progressive_score_delta_scale"),
        "progressive_score_calibration_hidden": meta.get("progressive_score_calibration_hidden"),
        "distill_loss_type": meta.get("distill_loss_type"),
        "distill_weight": meta.get("distill_weight"),
        "distill_task_weight": meta.get("distill_task_weight"),
        "vggt_train_encoder": meta.get("vggt_train_encoder"),
        "vggt_train_encoder_last_n_blocks": meta.get("vggt_train_encoder_last_n_blocks"),
        "encoder_lr_scale": meta.get("encoder_lr_scale"),
        "pipeline": meta.get("pipeline", "alignment_experiment_v1"),
        "train_meta": meta,
    }
    torch.save(payload, path)


def _backbone_used_tag(mode: str) -> str:
    return {
        "pure_graspnet": "pure",
        "vggt_replacement": "replaced",
        "vggt_progressive_replacement": "replaced",
        "vggt_replacement_distill": "replaced",
        "lift3d_replacement_clip": "replaced",
        "lift3d_replacement_dinov2": "replaced",
        "vggt_fusion_normalized": "fused",
        "vggt_progressive_replacement": "replaced",
        "vggt_replacement_distill": "replaced",
        "lift3d_progressive_replacement_clip": "replaced",
        "lift3d_progressive_replacement_dinov2": "replaced",
        "lift3d_replacement_distill_clip": "replaced",
        "lift3d_replacement_distill_dinov2": "replaced",
        "vggt_progressive_fusion": "replaced",
        "vggt_fusion_distill": "replaced",
    }.get(mode, mode)


def _encoder_type_for_table(model: torch.nn.Module, model_mode: str) -> str:
    return getattr(model, "encoder_type", model_mode)


def _trainable_param_stats(model: torch.nn.Module) -> tuple[int, int, int]:
    """返回 (encoder 可训元素数, 非 encoder 可训元素数, 总可训元素数)。"""
    enc = getattr(model, "encoder", None)
    enc_ids = {id(p) for p in enc.parameters()} if enc is not None else set()
    enc_n = 0
    other_n = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if id(p) in enc_ids:
            enc_n += n
        else:
            other_n += n
    return enc_n, other_n, enc_n + other_n


def _append_results_table(path: str, row: Dict[str, Any]) -> None:
    rows: list = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                rows = []
        except Exception:
            rows = []
    rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser(description="Alignment 五主实验统一训练")
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--max_samples", type=int, default=0, help="0=全量")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--graspnet_ckpt", type=str, required=True)
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument(
        "--lift3d_root",
        type=str,
        default=os.path.expanduser("~/LIFT3D"),
        help="LIFT3D-CLIP：仓库路径（lift3d_clip_base 预训练）",
    )
    p.add_argument("--vggt_ckpt", type=str, default=None, help="VGGT 微调权重；不设则用 VGGT 默认预训练")
    p.add_argument("--vggt_sample_k", type=int, default=1024)
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--loss_mode", type=str, default="bidir")
    p.add_argument("--loss_alpha", type=float, default=0.7)
    p.add_argument("--best_gt_weight", type=float, default=0.3)
    p.add_argument("--loss_pred2gt_agg", type=str, default="min", choices=("min", "mean"))
    p.add_argument("--collision_aux", dest="collision_aux", action="store_true", default=True)
    p.add_argument("--no_collision_aux", dest="collision_aux", action="store_false")
    p.add_argument("--collision_aux_weight", type=float, default=0.1)
    p.add_argument("--collision_margin", type=float, default=0.01)
    p.add_argument("--quality_aux", action="store_true", default=False)
    p.add_argument("--quality_aux_weight", type=float, default=0.0)
    p.add_argument("--quality_aux_mode", type=str, default="quality_mse", choices=("quality_mse",))
    p.add_argument("--ranking_aux", action="store_true", default=False)
    p.add_argument("--ranking_aux_weight", type=float, default=0.0)
    p.add_argument("--ranking_margin", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--lora_last_n_blocks", type=int, default=None)
    p.add_argument(
        "--replacement_scale_mode",
        type=str,
        default="none",
        choices=("none", "fixed", "ln_learnable"),
        help="仅 vggt_replacement：projector 后、vpmodule 前的 scale 对齐模式",
    )
    p.add_argument(
        "--replacement_fixed_alpha",
        type=float,
        default=1.0,
        help="仅 replacement_scale_mode=fixed 时生效",
    )
    p.add_argument(
        "--replacement_learnable_scale_init",
        type=float,
        default=1.0,
        help="仅 replacement_scale_mode=ln_learnable 时，learnable scalar 初始化值",
    )
    p.add_argument(
        "--replacement_align_mode",
        type=str,
        default="none",
        choices=(
            "none",
            "layernorm",
            "layernorm_affine",
            "adapter",
            "ln_adapter",
            "layernorm_adapter",
            "layernorm_affine_adapter",
            "layernorm_affine_deep_adapter",
        ),
        help="仅 vggt_replacement：projector 后、vpmodule 前的特征对齐方式",
    )
    p.add_argument(
        "--replacement_affine_init_scale",
        type=float,
        default=1.0,
        help="仅 replacement_align_mode=layernorm_affine 时，gamma 初始化",
    )
    p.add_argument(
        "--replacement_adapter_hidden",
        type=int,
        default=256,
        help="仅 replacement_align_mode=adapter/ln_adapter 时 adapter 隐层宽度",
    )
    p.add_argument(
        "--replacement_adapter_depth",
        type=int,
        default=2,
        help="仅 replacement adapter 类模式：adapter 堆叠层数（2=默认，3=更强）",
    )
    p.add_argument(
        "--model_mode",
        type=str,
        required=True,
        choices=(
            "pure_graspnet",
            "vggt_replacement",
            "vggt_progressive_replacement",
            "vggt_replacement_distill",
            "lift3d_replacement_clip",
            "lift3d_replacement_dinov2",
            "lift3d_progressive_replacement_clip",
            "lift3d_progressive_replacement_dinov2",
            "lift3d_replacement_distill_clip",
            "lift3d_replacement_distill_dinov2",
            "vggt_progressive_fusion",
            "vggt_fusion_distill",
            "vggt_fusion_normalized",
        ),
    )
    p.add_argument("--fuse_residual", action="store_true", default=False, help="仅 vggt_fusion_normalized")
    p.add_argument("--fuse_alpha", type=float, default=0.1, help="fuse_residual 时有效")
    p.add_argument("--out_dir", type=str, default=None, help="默认 checkpoints/alignment_experiments/<exp_name>_<ts>")
    p.add_argument("--exp_name", type=str, default="alignment")
    p.add_argument("--results_table", type=str, default=None, help="汇总表 JSON 路径；默认 out_dir/../alignment_runs_table.json")
    p.add_argument("--run_eval_after", action="store_true")
    p.add_argument("--dataset_root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D")
    p.add_argument("--eval_rewrite", type=str, default=os.path.join(ROOT, "eval_benchmark_rewrite.py"))
    p.add_argument("--eval_extra_stats", action="store_true", help="rewrite eval 传 --extra_stats")
    p.add_argument("--eval_max_scenes", type=int, default=0, help="0=全量 test")
    p.add_argument(
        "--pure_freeze_vpmodule",
        action="store_true",
        help="仅 pure_graspnet：冻结 vpmodule，只训 grasp_generator（与默认「训 vpmodule+head」对照）",
    )
    p.add_argument(
        "--run_eval_before_train",
        action="store_true",
        help="仅 pure_graspnet：训练前跑一次 rewrite eval（仅 graspnet_ckpt，无 pipeline ckpt）",
    )
    p.add_argument(
        "--freeze_head",
        action="store_true",
        help="仅用于 VGGT replacement / fusion：冻结 vpmodule + grasp_generator，只训对齐/融合模块",
    )
    p.add_argument(
        "--pure_only_train_last_score_head",
        action="store_true",
        help="仅 pure_graspnet：仅训练 grasp_generator.operation.conv3 与 tolerance.conv3",
    )
    p.add_argument(
        "--pure_score_calibration_mode",
        type=str,
        default="none",
        choices=("none", "residual"),
        help="仅 pure_graspnet：保留 pretrained score，并学习小幅 delta_score",
    )
    p.add_argument("--pure_score_delta_scale", type=float, default=0.1, help="residual delta 的最大幅度系数")
    p.add_argument("--pure_score_calibration_hidden", type=int, default=12, help="residual calibration 隐层宽度")
    p.add_argument(
        "--pure_train_score_calibrator_only",
        action="store_true",
        help="仅 pure_graspnet：只训练 residual score calibrator，冻结原 grasp head",
    )
    p.add_argument(
        "--run_eval_each_epoch",
        action="store_true",
        help="每个 epoch 结束后跑一次 rewrite eval 并记录 AP/collision/FC 轨迹",
    )
    p.add_argument(
        "--progressive_alpha",
        type=float,
        default=0.5,
        help="vggt_progressive_replacement / vggt_progressive_fusion：混合系数 α",
    )
    p.add_argument(
        "--progressive_train_head",
        action="store_true",
        help="仅 vggt_progressive_replacement：训 vpmodule+grasp_generator（与 --freeze_head 互斥）",
    )
    p.add_argument(
        "--progressive_score_calibration_mode",
        type=str,
        default="none",
        choices=("none", "residual"),
        help="仅 vggt_progressive_replacement：训 head 时对 grasp_score 做 residual cal",
    )
    p.add_argument("--progressive_score_delta_scale", type=float, default=0.1)
    p.add_argument("--progressive_score_calibration_hidden", type=int, default=12)
    p.add_argument(
        "--distill_loss_type",
        type=str,
        default="l2",
        choices=("l2", "cosine"),
        help="vggt_replacement_distill / vggt_fusion_distill：teacher/student 对齐损失",
    )
    p.add_argument("--distill_weight", type=float, default=1.0, help="distill 项权重")
    p.add_argument(
        "--distill_task_weight",
        type=float,
        default=1.0,
        help="distill 模式：17D 主任务权重（默认 1.0；设 0 为纯 distill）",
    )
    p.add_argument(
        "--fusion_distill_alpha",
        type=float,
        default=0.2,
        help="仅 vggt_fusion_distill：vpmodule 输入中 student 权重 α，(1-α)·teacher+α·student",
    )
    p.add_argument(
        "--vggt_train_encoder",
        action="store_true",
        help="仅 vggt_progressive_replacement / vggt_replacement_distill：解冻 VGGT backbone 内 LoRA 微调（小步长见 --encoder_lr_scale）",
    )
    p.add_argument(
        "--vggt_train_encoder_last_n_blocks",
        type=int,
        default=4,
        help="与 --vggt_train_encoder 联用：只训最后 N 个 block 的 LoRA",
    )
    p.add_argument(
        "--encoder_lr_scale",
        type=float,
        default=0.1,
        help="可训 encoder（含 LoRA）相对主 lr 的倍率；VGGT 微调建议 0.05~0.1",
    )
    p.add_argument(
        "--two_stage_training",
        action="store_true",
        help="与 --vggt_train_encoder 联用：两阶段训练（Stage1 冻结 encoder；Stage2 LoRA 微调）。总 epoch = stage1+stage2",
    )
    p.add_argument("--stage1_epochs", type=int, default=5, help="两阶段训练时 Stage1（冻结 encoder）epoch 数")
    p.add_argument("--stage2_epochs", type=int, default=10, help="两阶段训练时 Stage2（LoRA）epoch 数")
    return p.parse_args()


def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(ROOT, "checkpoints", "alignment_experiments", f"{args.exp_name}_{ts}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    _setup_logging(log_path)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    max_samples = None if args.max_samples <= 0 else args.max_samples

    train_ds = GC6DOfflineUnifiedDataset(
        data_dir=args.data_dir,
        split="train",
        camera=args.camera,
        max_samples=max_samples,
        load_gt_multi=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_gc6d,
        num_workers=0,
    )
    val_index = os.path.join(args.data_dir, f"index_{args.val_split}_{args.camera}.jsonl")
    val_loader = None
    if os.path.isfile(val_index):
        val_ds = GC6DOfflineUnifiedDataset(
            data_dir=args.data_dir,
            split=args.val_split,
            camera=args.camera,
            load_gt_multi=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_gc6d, num_workers=0
        )
        logger.info("Val: %s (%d samples)", val_index, len(val_ds))

    # 构建模型
    mm = args.model_mode
    if mm == "pure_graspnet":
        model = build_pure_graspnet_pipeline(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            score_calibration_mode=args.pure_score_calibration_mode,
            score_delta_scale=args.pure_score_delta_scale,
            score_calibration_hidden=args.pure_score_calibration_hidden,
            device=device,
        )
    elif mm == "lift3d_replacement_clip":
        model = build_lift3d_clip_replacement_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            lift3d_root=args.lift3d_root,
            device=device,
        )
    elif mm == "lift3d_replacement_dinov2":
        model = build_dinov2_replacement_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            device=device,
        )
    elif mm == "lift3d_progressive_replacement_clip":
        model = build_lift3d_clip_progressive_replacement_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            lift3d_root=args.lift3d_root,
            progressive_alpha=args.progressive_alpha,
            device=device,
        )
    elif mm == "lift3d_progressive_replacement_dinov2":
        model = build_dinov2_progressive_replacement_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            progressive_alpha=args.progressive_alpha,
            device=device,
        )
    elif mm == "lift3d_replacement_distill_clip":
        model = build_lift3d_clip_replacement_distill_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            lift3d_root=args.lift3d_root,
            distill_loss_type=args.distill_loss_type,
            device=device,
        )
    elif mm == "lift3d_replacement_distill_dinov2":
        model = build_dinov2_replacement_distill_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            distill_loss_type=args.distill_loss_type,
            device=device,
        )
    elif mm == "vggt_replacement":
        model = build_vggt_replacement_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            vggt_ckpt=args.vggt_ckpt,
            feat_dim=256,
            sample_k=args.vggt_sample_k,
            lora_r=args.lora_r,
            lora_scale=args.lora_scale,
            lora_last_n_blocks=args.lora_last_n_blocks,
            replacement_align_mode=args.replacement_align_mode,
            replacement_affine_init_scale=args.replacement_affine_init_scale,
            replacement_adapter_hidden=args.replacement_adapter_hidden,
            replacement_adapter_depth=args.replacement_adapter_depth,
            replacement_scale_mode=args.replacement_scale_mode,
            replacement_fixed_alpha=args.replacement_fixed_alpha,
            replacement_learnable_scale_init=args.replacement_learnable_scale_init,
            device=device,
        )
    elif mm == "vggt_progressive_replacement":
        model = build_vggt_progressive_replacement_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            vggt_ckpt=args.vggt_ckpt,
            feat_dim=256,
            sample_k=args.vggt_sample_k,
            lora_r=args.lora_r,
            lora_scale=args.lora_scale,
            lora_last_n_blocks=args.lora_last_n_blocks,
            progressive_alpha=args.progressive_alpha,
            score_calibration_mode=args.progressive_score_calibration_mode,
            score_delta_scale=args.progressive_score_delta_scale,
            score_calibration_hidden=args.progressive_score_calibration_hidden,
            device=device,
        )
    elif mm == "vggt_replacement_distill":
        model = build_vggt_replacement_distill_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            vggt_ckpt=args.vggt_ckpt,
            feat_dim=256,
            sample_k=args.vggt_sample_k,
            lora_r=args.lora_r,
            lora_scale=args.lora_scale,
            lora_last_n_blocks=args.lora_last_n_blocks,
            distill_loss_type=args.distill_loss_type,
            device=device,
        )
    elif mm == "vggt_progressive_fusion":
        model = build_vggt_fusion_progressive_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            vggt_ckpt=args.vggt_ckpt,
            feat_dim=256,
            sample_k=args.vggt_sample_k,
            lora_r=args.lora_r,
            lora_scale=args.lora_scale,
            lora_last_n_blocks=args.lora_last_n_blocks,
            progressive_alpha=args.progressive_alpha,
            device=device,
        )
    elif mm == "vggt_fusion_distill":
        model = build_vggt_fusion_distill_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            vggt_ckpt=args.vggt_ckpt,
            feat_dim=256,
            sample_k=args.vggt_sample_k,
            lora_r=args.lora_r,
            lora_scale=args.lora_scale,
            lora_last_n_blocks=args.lora_last_n_blocks,
            distill_loss_type=args.distill_loss_type,
            fusion_distill_alpha=float(args.fusion_distill_alpha),
            device=device,
        )
    elif mm == "vggt_fusion_normalized":
        model = build_vggt_fusion_normalized_graspnet(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            vggt_ckpt=args.vggt_ckpt,
            feat_dim=256,
            sample_k=args.vggt_sample_k,
            lora_r=args.lora_r,
            lora_scale=args.lora_scale,
            lora_last_n_blocks=args.lora_last_n_blocks,
            fuse_residual=args.fuse_residual,
            fuse_alpha=args.fuse_alpha,
            device=device,
        )
    else:
        raise ValueError("未知 model_mode: %s" % mm)

    model.to(device)

    freeze_head_eff = bool(args.freeze_head)
    if mm == "vggt_progressive_replacement":
        if args.progressive_train_head:
            if freeze_head_eff:
                raise ValueError("不要同时指定 --progressive_train_head 与 --freeze_head")
            freeze_head_eff = False
        else:
            freeze_head_eff = True
    elif mm == "vggt_replacement_distill":
        freeze_head_eff = True
    elif mm == "vggt_progressive_fusion":
        freeze_head_eff = True
    elif mm == "vggt_fusion_distill":
        freeze_head_eff = True
    elif mm in ("lift3d_progressive_replacement_clip", "lift3d_progressive_replacement_dinov2"):
        freeze_head_eff = True
    elif mm in ("lift3d_replacement_distill_clip", "lift3d_replacement_distill_dinov2", "lift3d_clip_replacement_distill"):
        freeze_head_eff = True

    if args.vggt_train_encoder and mm not in ("vggt_progressive_replacement", "vggt_replacement_distill"):
        raise ValueError("--vggt_train_encoder 仅用于 vggt_progressive_replacement 或 vggt_replacement_distill")

    use_two_stage = (
        bool(getattr(args, "two_stage_training", False))
        and bool(args.vggt_train_encoder)
        and mm in ("vggt_progressive_replacement", "vggt_replacement_distill")
    )
    if bool(getattr(args, "two_stage_training", False)) and not bool(args.vggt_train_encoder):
        logger.info("--two_stage_training 仅在同时指定 --vggt_train_encoder 时生效；本次按单阶段训练。")

    total_epochs = (
        int(args.stage1_epochs) + int(args.stage2_epochs) if use_two_stage else int(args.epochs)
    )
    encoder_lr_scale_stage2_effective: Optional[float] = None
    if use_two_stage:
        encoder_lr_scale_stage2_effective = min(float(args.encoder_lr_scale), 0.05)
        if encoder_lr_scale_stage2_effective < float(args.encoder_lr_scale):
            logger.info(
                "两阶段 Stage2：encoder_lr_scale 从 %s 限制为 %s（上限 0.05）",
                args.encoder_lr_scale,
                encoder_lr_scale_stage2_effective,
            )
        logger.info(
            "两阶段训练：stage1=%d + stage2=%d = %d 总 epoch（忽略 --epochs=%d）",
            int(args.stage1_epochs),
            int(args.stage2_epochs),
            total_epochs,
            int(args.epochs),
        )
        apply_alignment_freeze(
            model,
            mm,
            pure_freeze_vpmodule=bool(getattr(args, "pure_freeze_vpmodule", False)),
            pure_only_train_last_score_head=bool(getattr(args, "pure_only_train_last_score_head", False)),
            pure_train_score_calibrator_only=bool(getattr(args, "pure_train_score_calibrator_only", False)),
            freeze_head=freeze_head_eff,
            vggt_train_encoder=False,
            vggt_train_encoder_last_n_blocks=int(args.vggt_train_encoder_last_n_blocks),
        )
        optimizer = build_optimizer(model, lr=args.lr, encoder_lr_scale=args.encoder_lr_scale)
        e_n, o_n, t_n = _trainable_param_stats(model)
        print("=== Stage 1: Frozen encoder ===")
        print(
            "Trainable params: total=%d, encoder=%d, non-encoder=%d"
            % (t_n, e_n, o_n)
        )
        print("encoder_lr=0.0 (frozen), head_lr=%s" % (args.lr,))
    else:
        apply_alignment_freeze(
            model,
            mm,
            pure_freeze_vpmodule=bool(getattr(args, "pure_freeze_vpmodule", False)),
            pure_only_train_last_score_head=bool(getattr(args, "pure_only_train_last_score_head", False)),
            pure_train_score_calibrator_only=bool(getattr(args, "pure_train_score_calibrator_only", False)),
            freeze_head=freeze_head_eff,
            vggt_train_encoder=bool(args.vggt_train_encoder),
            vggt_train_encoder_last_n_blocks=int(args.vggt_train_encoder_last_n_blocks),
        )
        optimizer = build_optimizer(model, lr=args.lr, encoder_lr_scale=args.encoder_lr_scale)

    meta_train: Dict[str, Any] = {
        "exp_name": args.exp_name,
        "model_mode": mm,
        "encoder_type": getattr(model, "encoder_type", mm),
        "data_dir": args.data_dir,
        "camera": args.camera,
        "epochs": total_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "graspnet_ckpt": args.graspnet_ckpt,
        "graspnet_root": args.graspnet_root,
        "lift3d_root": args.lift3d_root,
        "vggt_ckpt": args.vggt_ckpt,
        "vggt_sample_k": args.vggt_sample_k,
        "replacement_align_mode": args.replacement_align_mode if mm == "vggt_replacement" else None,
        "replacement_affine_init_scale": args.replacement_affine_init_scale if mm == "vggt_replacement" else None,
        "replacement_adapter_hidden": args.replacement_adapter_hidden if mm == "vggt_replacement" else None,
        "replacement_adapter_depth": args.replacement_adapter_depth if mm == "vggt_replacement" else None,
        "replacement_scale_mode": args.replacement_scale_mode if mm == "vggt_replacement" else None,
        "replacement_fixed_alpha": args.replacement_fixed_alpha if mm == "vggt_replacement" else None,
        "replacement_learnable_scale_init": (
            args.replacement_learnable_scale_init if mm == "vggt_replacement" else None
        ),
        "pure_score_calibration_mode": args.pure_score_calibration_mode if mm == "pure_graspnet" else None,
        "pure_score_delta_scale": args.pure_score_delta_scale if mm == "pure_graspnet" else None,
        "pure_score_calibration_hidden": args.pure_score_calibration_hidden if mm == "pure_graspnet" else None,
        "pure_train_score_calibrator_only": bool(getattr(args, "pure_train_score_calibrator_only", False)),
        "fuse_residual": args.fuse_residual if mm == "vggt_fusion_normalized" else None,
        "fuse_alpha": args.fuse_alpha if mm == "vggt_fusion_normalized" else None,
        "collision_aux": args.collision_aux,
        "quality_aux": args.quality_aux,
        "quality_aux_mode": args.quality_aux_mode,
        "ranking_aux": args.ranking_aux,
        "ranking_aux_weight": args.ranking_aux_weight,
        "ranking_margin": args.ranking_margin,
        "lora_r": args.lora_r,
        "lora_scale": args.lora_scale,
        "lora_last_n_blocks": args.lora_last_n_blocks,
        "progressive_alpha": args.progressive_alpha
        if mm
        in (
            "vggt_progressive_replacement",
            "vggt_progressive_fusion",
            "lift3d_progressive_replacement_clip",
            "lift3d_progressive_replacement_dinov2",
        )
        else None,
        "progressive_train_head": bool(args.progressive_train_head) if mm == "vggt_progressive_replacement" else None,
        "progressive_score_calibration_mode": (
            args.progressive_score_calibration_mode if mm == "vggt_progressive_replacement" else None
        ),
        "progressive_score_delta_scale": (
            args.progressive_score_delta_scale if mm == "vggt_progressive_replacement" else None
        ),
        "progressive_score_calibration_hidden": (
            args.progressive_score_calibration_hidden if mm == "vggt_progressive_replacement" else None
        ),
        "distill_loss_type": args.distill_loss_type
        if mm
        in (
            "vggt_replacement_distill",
            "vggt_fusion_distill",
            "lift3d_replacement_distill_clip",
            "lift3d_replacement_distill_dinov2",
        )
        else None,
        "distill_weight": args.distill_weight
        if mm
        in (
            "vggt_replacement_distill",
            "vggt_fusion_distill",
            "lift3d_replacement_distill_clip",
            "lift3d_replacement_distill_dinov2",
        )
        else None,
        "distill_task_weight": args.distill_task_weight
        if mm
        in (
            "vggt_replacement_distill",
            "vggt_fusion_distill",
            "lift3d_replacement_distill_clip",
            "lift3d_replacement_distill_dinov2",
        )
        else None,
        "fusion_distill_alpha": float(args.fusion_distill_alpha) if mm == "vggt_fusion_distill" else None,
        "vggt_train_encoder": bool(args.vggt_train_encoder)
        if mm in ("vggt_progressive_replacement", "vggt_replacement_distill")
        else None,
        "vggt_train_encoder_last_n_blocks": int(args.vggt_train_encoder_last_n_blocks)
        if mm in ("vggt_progressive_replacement", "vggt_replacement_distill")
        else None,
        "encoder_lr_scale": args.encoder_lr_scale,
        "two_stage_training": bool(use_two_stage),
        "stage1_epochs": int(args.stage1_epochs) if use_two_stage else None,
        "stage2_epochs": int(args.stage2_epochs) if use_two_stage else None,
        "encoder_lr_scale_stage2_effective": encoder_lr_scale_stage2_effective,
        "pipeline": "alignment_experiment_v1",
        "encoder_frozen": (
            "two_stage"
            if use_two_stage
            else (
                "partial_lora"
                if bool(args.vggt_train_encoder) and mm in ("vggt_progressive_replacement", "vggt_replacement_distill")
                else "yes"
            )
        ),
        "backbone_used": _backbone_used_tag(mm),
        "freeze_head": freeze_head_eff,
        "pure_only_train_last_score_head": bool(getattr(args, "pure_only_train_last_score_head", False)),
    }

    ckpt_path = os.path.join(out_dir, f"{args.exp_name}_{mm}_{ts}.pt")
    final_val_loss: Optional[float] = None

    eval_before_train: Dict[str, Any] = {}
    if getattr(args, "run_eval_before_train", False) and mm == "pure_graspnet":
        dump_tag = "%s_%s_%s_before_train" % (args.exp_name, mm, ts)
        dump_dir = os.path.join(ROOT, "eval_out_rewrite", dump_tag)
        cmd = [
            sys.executable,
            args.eval_rewrite,
            "--gc6d_root",
            args.dataset_root,
            "--graspnet_ckpt",
            args.graspnet_ckpt,
            "--graspnet_root",
            args.graspnet_root,
            "--camera",
            args.camera,
            "--dump_dir",
            dump_dir,
            "--tag",
            dump_tag,
        ]
        if args.eval_max_scenes > 0:
            cmd.extend(["--max_scenes", str(args.eval_max_scenes)])
        if args.eval_extra_stats:
            cmd.append("--extra_stats")
        logger.info("Eval BEFORE train (pure pretrained only): %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
        summ_path = os.path.join(dump_dir, "summary_rewrite.json")
        if os.path.isfile(summ_path):
            with open(summ_path, "r", encoding="utf-8") as f:
                eval_before_train = json.load(f)
        with open(os.path.join(out_dir, "eval_before_train_summary.json"), "w", encoding="utf-8") as f:
            json.dump(eval_before_train, f, indent=2, ensure_ascii=False)

    epoch_eval_rows: list[dict] = []

    def run_one_epoch(epoch: int, total_ep: int, optim: torch.optim.Optimizer) -> None:
        nonlocal final_val_loss
        set_alignment_train_state(
            model,
            mm,
            pure_freeze_vpmodule=bool(getattr(args, "pure_freeze_vpmodule", False)),
            freeze_head=freeze_head_eff,
        )
        ep_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            pcs, _, _, metas = batch
            pcs = pcs.to(device)
            optim.zero_grad()
            loss, log = compute_train_loss(
                model=model,
                point_cloud=pcs,
                metas=metas,
                device=device,
                loss_mode=args.loss_mode,
                loss_alpha=args.loss_alpha,
                best_gt_weight=args.best_gt_weight,
                pred2gt_agg=args.loss_pred2gt_agg,
                collision_aux=args.collision_aux,
                collision_aux_weight=args.collision_aux_weight,
                collision_margin=args.collision_margin,
                quality_aux=args.quality_aux,
                quality_aux_weight=args.quality_aux_weight,
                quality_aux_mode=args.quality_aux_mode,
                ranking_aux=args.ranking_aux,
                ranking_aux_weight=args.ranking_aux_weight,
                ranking_margin=args.ranking_margin,
                data_dir=args.data_dir,
                distill_weight=args.distill_weight,
                distill_task_weight=args.distill_task_weight,
            )
            loss.backward()
            optim.step()
            ep_loss += log["loss_total"]
            n_batches += 1
        logger.info("Epoch %d/%d train_loss_mean=%.6f", epoch + 1, total_ep, ep_loss / max(n_batches, 1))

        if val_loader is not None and (epoch + 1) % max(args.val_every, 1) == 0:
            final_val_loss = evaluate_mean_val_loss(
                model,
                val_loader,
                device,
                loss_mode=args.loss_mode,
                loss_alpha=args.loss_alpha,
                best_gt_weight=args.best_gt_weight,
                pred2gt_agg=args.loss_pred2gt_agg,
                data_dir=args.data_dir,
                distill_weight=args.distill_weight,
                distill_task_weight=args.distill_task_weight,
            )
            logger.info("Epoch %d val_loss=%.6f", epoch + 1, final_val_loss)
            set_alignment_train_state(
                model,
                mm,
                pure_freeze_vpmodule=bool(getattr(args, "pure_freeze_vpmodule", False)),
                freeze_head=freeze_head_eff,
            )
        if getattr(args, "run_eval_each_epoch", False):
            epoch_ckpt = os.path.join(out_dir, f"epoch_{epoch + 1:02d}.pt")
            _save_checkpoint(epoch_ckpt, model, meta={**meta_train, "model_mode": mm, "epoch": epoch + 1})
            dump_tag = "%s_%s_%s_epoch%02d" % (args.exp_name, mm, ts, epoch + 1)
            dump_dir = os.path.join(ROOT, "eval_out_rewrite", dump_tag)
            cmd = [
                sys.executable,
                args.eval_rewrite,
                "--gc6d_root",
                args.dataset_root,
                "--graspnet_ckpt",
                args.graspnet_ckpt,
                "--graspnet_root",
                args.graspnet_root,
                "--pipeline_checkpoint",
                epoch_ckpt,
                "--camera",
                args.camera,
                "--dump_dir",
                dump_dir,
                "--tag",
                dump_tag,
            ]
            if args.eval_max_scenes > 0:
                cmd.extend(["--max_scenes", str(args.eval_max_scenes)])
            if args.eval_extra_stats:
                cmd.append("--extra_stats")
            logger.info("Epoch %d eval: %s", epoch + 1, " ".join(cmd))
            subprocess.run(cmd, check=False)
            summ_path = os.path.join(dump_dir, "summary_rewrite.json")
            row = {"epoch": epoch + 1, "train_loss_mean": ep_loss / max(n_batches, 1), "val_loss": final_val_loss}
            if os.path.isfile(summ_path):
                with open(summ_path, "r", encoding="utf-8") as f:
                    js = json.load(f)
                row.update(
                    {
                        "AP": js.get("AP"),
                        "AP0.4": js.get("AP0.4"),
                        "AP0.8": js.get("AP0.8"),
                        "top50_collision_remaining_rate": js.get("top50_collision_remaining_rate"),
                        "top50_force_closure_success_count_mean": js.get("top50_force_closure_success_count_mean"),
                        "summary_path": summ_path,
                    }
                )
            epoch_eval_rows.append(row)

    if use_two_stage:
        for epoch in range(int(args.stage1_epochs)):
            run_one_epoch(epoch, total_epochs, optimizer)
        assert encoder_lr_scale_stage2_effective is not None
        apply_alignment_freeze(
            model,
            mm,
            pure_freeze_vpmodule=bool(getattr(args, "pure_freeze_vpmodule", False)),
            pure_only_train_last_score_head=bool(getattr(args, "pure_only_train_last_score_head", False)),
            pure_train_score_calibrator_only=bool(getattr(args, "pure_train_score_calibrator_only", False)),
            freeze_head=freeze_head_eff,
            vggt_train_encoder=True,
            vggt_train_encoder_last_n_blocks=int(args.vggt_train_encoder_last_n_blocks),
        )
        optimizer = build_optimizer(
            model, lr=args.lr, encoder_lr_scale=float(encoder_lr_scale_stage2_effective)
        )
        e_n2, o_n2, t_n2 = _trainable_param_stats(model)
        print("=== Stage 2: LoRA finetuning ===")
        print(
            "Trainable params: total=%d, encoder=%d, non-encoder=%d"
            % (t_n2, e_n2, o_n2)
        )
        enc_lr_s2 = args.lr * float(encoder_lr_scale_stage2_effective)
        print("encoder_lr=%s, head_lr=%s" % (enc_lr_s2, args.lr))
        s1 = int(args.stage1_epochs)
        for local_i in range(int(args.stage2_epochs)):
            run_one_epoch(s1 + local_i, total_epochs, optimizer)
    else:
        for epoch in range(args.epochs):
            run_one_epoch(epoch, total_epochs, optimizer)

    if val_loader is not None and final_val_loss is None:
        final_val_loss = evaluate_mean_val_loss(
            model,
            val_loader,
            device,
            loss_mode=args.loss_mode,
            loss_alpha=args.loss_alpha,
            best_gt_weight=args.best_gt_weight,
            pred2gt_agg=args.loss_pred2gt_agg,
            data_dir=args.data_dir,
            distill_weight=args.distill_weight,
            distill_task_weight=args.distill_task_weight,
        )
        set_alignment_train_state(
            model,
            mm,
            pure_freeze_vpmodule=bool(getattr(args, "pure_freeze_vpmodule", False)),
            freeze_head=freeze_head_eff,
        )

    meta_train["final_val_loss"] = final_val_loss
    meta_train["steps_epochs"] = "%d epochs" % total_epochs
    meta_train["pure_freeze_vpmodule"] = bool(getattr(args, "pure_freeze_vpmodule", False))
    if eval_before_train:
        meta_train["eval_before_train_AP"] = eval_before_train.get("AP")
        meta_train["eval_before_train_dump_dir"] = eval_before_train.get("dump_dir")
    _save_checkpoint(ckpt_path, model, meta={**meta_train, "model_mode": mm})
    logger.info("Saved %s", ckpt_path)

    summary_path = os.path.join(out_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(meta_train, f, indent=2, ensure_ascii=False)
    if epoch_eval_rows:
        with open(os.path.join(out_dir, "epoch_eval_trace.json"), "w", encoding="utf-8") as f:
            json.dump(epoch_eval_rows, f, indent=2, ensure_ascii=False)

    # 评估（rewrite）
    eval_summary: Dict[str, Any] = {}
    if args.run_eval_after:
        dump_tag = "%s_%s_%s" % (args.exp_name, mm, ts)
        dump_dir = os.path.join(ROOT, "eval_out_rewrite", dump_tag)
        cmd = [
            sys.executable,
            args.eval_rewrite,
            "--gc6d_root",
            args.dataset_root,
            "--graspnet_ckpt",
            args.graspnet_ckpt,
            "--graspnet_root",
            args.graspnet_root,
            "--pipeline_checkpoint",
            ckpt_path,
            "--camera",
            args.camera,
            "--dump_dir",
            dump_dir,
            "--tag",
            dump_tag,
        ]
        if args.eval_max_scenes > 0:
            cmd.extend(["--max_scenes", str(args.eval_max_scenes)])
        if args.eval_extra_stats:
            cmd.append("--extra_stats")
        logger.info("Running eval: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
        summ_path = os.path.join(dump_dir, "summary_rewrite.json")
        if os.path.isfile(summ_path):
            with open(summ_path, "r", encoding="utf-8") as f:
                eval_summary = json.load(f)

    results_table = args.results_table or os.path.join(ROOT, "checkpoints", "alignment_runs_table.json")
    row = {
        "exp": args.exp_name,
        "model_mode": mm,
        "encoder_type": _encoder_type_for_table(model, mm),
        "samples": len(train_ds),
        "epochs_steps": meta_train["steps_epochs"],
        "lr": args.lr,
        "seed": args.seed,
        "encoder_frozen": meta_train.get("encoder_frozen", "yes"),
        "backbone_used": _backbone_used_tag(mm),
        "replacement_scale_mode": args.replacement_scale_mode if mm == "vggt_replacement" else None,
        "replacement_fixed_alpha": args.replacement_fixed_alpha if mm == "vggt_replacement" else None,
        "final_val_loss": final_val_loss,
        "AP": eval_summary.get("AP"),
        "AP0.4": eval_summary.get("AP0.4"),
        "AP0.8": eval_summary.get("AP0.8"),
        "top50_collision_remaining_rate": eval_summary.get("top50_collision_remaining_rate"),
        "top50_force_closure_success_count_mean": eval_summary.get("top50_force_closure_success_count_mean"),
        "checkpoint": ckpt_path,
        "eval_dump_dir": eval_summary.get("dump_dir"),
        "timestamp": ts,
    }
    _append_results_table(results_table, row)
    logger.info("Appended row to %s", results_table)


if __name__ == "__main__":
    main()
