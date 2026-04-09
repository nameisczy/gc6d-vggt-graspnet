#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIFT3D baseline 训练（全新入口，不依赖旧 Stage1/2/3/4 脚本）。

默认（``--fusion_mode additive``）：冻结 LIFT3D encoder，只训练 adapter + 预训练 GraspNet；additive，coeff=1.0；
无 rank_weighted；collision_aux 可开；quality_aux 默认关。

**局部融合**（``--fusion_mode concat_proj`` / ``residual_proj``）：冻结 LIFT3D backbone，训练 ``lift3d_seed_proj`` + 融合头 + GraspNet；
见 ``models/lift3d_local_fusion.py``、``docs/LIFT3D_PIPELINE.md``。

评估 AP / top-50 统计：训练结束后请用 **eval_benchmark.py**（不改评估逻辑）。

**pure_graspnet 实验 1（仅预训练、不训练）**：``--model_mode pure_graspnet --epochs 0`` 会跳过优化、保存带 ``encoder_type`` 的 pipeline checkpoint，便于 eval 摘要；或直接 ``eval_benchmark.py --checkpoint /path/to/checkpoint-rs.tar``（由 ``load_policy_from_checkpoint`` 识别为原始 GraspNet 权重）。

**pure_graspnet 实验 2**：``--model_mode pure_graspnet`` 且 ``epochs>=1``，默认 ``--freeze_graspnet_backbone`` 只训 vpmodule+grasp_generator。
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
from models.lift3d_grasp_pipeline import build_lift3d_grasp_pipeline
from models.lift3d_local_fusion import build_lift3d_local_fusion_graspnet
from models.pure_graspnet import build_pure_graspnet_pipeline
from training.losses import compute_train_loss, evaluate_mean_val_loss
from training.optim import apply_lift3d_baseline_freeze, apply_pure_graspnet_freeze, build_optimizer


def _setup_logging(log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    meta: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "encoder_type": meta.get("encoder_type", "lift3d"),
        "model_mode": meta.get("model_mode", "lift3d"),
        "graspnet_ckpt": meta.get("graspnet_ckpt"),
        "graspnet_root": meta.get("graspnet_root"),
        "lift3d_root": meta.get("lift3d_root"),
        "lift3d_ckpt": meta.get("lift3d_ckpt"),
        "use_adapter": getattr(model, "use_adapter", True),
        "adapter_cond_coeff": getattr(model, "adapter_cond_coeff", 1.0),
        "adapter_cond_mode": getattr(model, "adapter_cond_mode", "additive"),
        "fusion_mode": meta.get("fusion_mode"),
        "residual_alpha": meta.get("residual_alpha"),
        "lora_r": meta.get("lora_r"),
        "lora_scale": meta.get("lora_scale"),
        "lora_last_n_blocks": meta.get("lora_last_n_blocks"),
        "pipeline": meta.get("pipeline", "lift3d_baseline_v1"),
        "train_meta": meta,
    }
    torch.save(payload, path)


def parse_args():
    p = argparse.ArgumentParser(description="LIFT3D + adapter + GraspNet baseline 训练")
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--max_samples", type=int, default=0, help="0=全量 index_train")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10, help="0=跳过训练，仅保存当前模型 checkpoint（用于 pure_graspnet 预训练直评）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--graspnet_ckpt", type=str, required=True)
    p.add_argument("--graspnet_root", type=str, default=None)
    p.add_argument("--lift3d_root", type=str, default=os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D")))
    p.add_argument("--lift3d_ckpt", type=str, default=None)
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--val_every", type=int, default=1, help="每多少个 epoch 验证一次 val loss")
    p.add_argument("--loss_mode", type=str, default="bidir")
    p.add_argument("--loss_alpha", type=float, default=0.7)
    p.add_argument("--best_gt_weight", type=float, default=0.3)
    p.add_argument("--loss_pred2gt_agg", type=str, default="min", choices=("min", "mean"))
    p.add_argument(
        "--model_mode",
        type=str,
        default="lift3d",
        choices=("lift3d", "pure_graspnet"),
        help="lift3d: 现有 LIFT3D 分支；pure_graspnet: 不使用 encoder/adapter/cond，仅 GraspNet",
    )
    p.add_argument(
        "--fusion_mode",
        type=str,
        default="additive",
        choices=("additive", "concat_proj", "residual_proj"),
        help="additive=E1 全局 cond+additive；concat_proj=E2；residual_proj=E3（与 seed_features 残差融合）",
    )
    p.add_argument(
        "--residual_alpha",
        type=float,
        default=1.0,
        help="仅 fusion_mode=residual_proj：seed_features += alpha * proj(lift3d_seed)",
    )
    p.add_argument("--adapter_cond_mode", type=str, default="additive", choices=("additive", "concat"))
    p.add_argument("--adapter_cond_coeff", type=float, default=1.0)
    p.add_argument(
        "--collision_aux",
        dest="collision_aux",
        action="store_true",
        default=True,
        help="清障代理 loss（默认开）",
    )
    p.add_argument("--no_collision_aux", dest="collision_aux", action="store_false")
    p.add_argument("--collision_aux_weight", type=float, default=0.1)
    p.add_argument("--collision_margin", type=float, default=0.01)
    p.add_argument("--quality_aux", action="store_true", default=False, help="默认关；接口预留")
    p.add_argument("--quality_aux_weight", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--lora_last_n_blocks", type=int, default=None)
    p.add_argument("--save_dir", type=str, default=None, help="默认 checkpoints/lift3d_pipeline/")
    p.add_argument("--exp_name", type=str, default="lift3d_baseline")
    p.add_argument("--run_eval_after", action="store_true", help="训练结束后调用 eval_benchmark.py（需本机路径正确）")
    p.add_argument("--dataset_root", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D", help="供 eval_benchmark 使用")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument(
        "--eval_extra_stats",
        "--extra_stats",
        dest="eval_extra_stats",
        action="store_true",
        help="训练结束后 eval_benchmark 时传入 --extra_stats（与 eval_benchmark 命名一致）",
    )
    p.add_argument(
        "--freeze_graspnet_backbone",
        dest="freeze_graspnet_backbone",
        action="store_true",
        default=True,
        help="仅 model_mode=pure_graspnet 生效：默认冻结 grasp backbone，仅训 head",
    )
    p.add_argument(
        "--no_freeze_graspnet_backbone",
        dest="freeze_graspnet_backbone",
        action="store_false",
        help="仅 pure_graspnet：允许训练整个 grasp_net",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    save_dir = args.save_dir or os.path.join(ROOT, "checkpoints", "lift3d_pipeline")
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"{args.exp_name}_{ts}.log")
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
    else:
        logger.warning("No val index: %s", val_index)

    if args.model_mode == "pure_graspnet":
        model = build_pure_graspnet_pipeline(
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            device=device,
        )
    else:
        if args.fusion_mode == "additive":
            model = build_lift3d_grasp_pipeline(
                graspnet_ckpt=args.graspnet_ckpt,
                graspnet_root=args.graspnet_root,
                lift3d_root=args.lift3d_root,
                lift3d_ckpt=args.lift3d_ckpt,
                encoder_feat_dim=256,
                adapter_cond_coeff=args.adapter_cond_coeff,
                adapter_cond_mode=args.adapter_cond_mode,
                use_adapter=True,
                lora_r=args.lora_r,
                lora_scale=args.lora_scale,
                lora_last_n_blocks=args.lora_last_n_blocks,
                device=device,
            )
        else:
            model = build_lift3d_local_fusion_graspnet(
                fusion_mode=args.fusion_mode,
                graspnet_ckpt=args.graspnet_ckpt,
                graspnet_root=args.graspnet_root,
                lift3d_root=args.lift3d_root,
                lift3d_ckpt=args.lift3d_ckpt,
                residual_alpha=args.residual_alpha,
                encoder_feat_dim=256,
                lora_r=args.lora_r,
                lora_scale=args.lora_scale,
                lora_last_n_blocks=args.lora_last_n_blocks,
                device=device,
            )
    model.to(device)
    if args.model_mode == "pure_graspnet":
        apply_pure_graspnet_freeze(model, freeze_backbone=args.freeze_graspnet_backbone)
    else:
        apply_lift3d_baseline_freeze(model)
    optimizer = None
    if args.epochs > 0:
        optimizer = build_optimizer(model, lr=args.lr, encoder_lr_scale=0.1)
    else:
        logger.info("epochs=0：跳过训练与优化器构建")

    if args.model_mode == "pure_graspnet":
        pipeline_name = "pure_graspnet_v1"
    else:
        pipeline_name = "lift3d_baseline_v1" if args.fusion_mode == "additive" else "lift3d_local_fusion_v1"
    meta_train: Dict[str, Any] = {
        "exp_name": args.exp_name,
        "data_dir": args.data_dir,
        "camera": args.camera,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "model_mode": args.model_mode,
        "fusion_mode": args.fusion_mode,
        "residual_alpha": args.residual_alpha if args.fusion_mode == "residual_proj" else None,
        "freeze_graspnet_backbone": args.freeze_graspnet_backbone if args.model_mode == "pure_graspnet" else None,
        "adapter_mode": args.adapter_cond_mode,
        "adapter_cond_coeff": args.adapter_cond_coeff,
        "collision_aux": args.collision_aux,
        "quality_aux": args.quality_aux,
        "train_samples": len(train_ds),
        "graspnet_ckpt": args.graspnet_ckpt,
        "graspnet_root": args.graspnet_root,
        "lift3d_root": args.lift3d_root,
        "lift3d_ckpt": args.lift3d_ckpt,
        "lora_r": args.lora_r,
        "lora_scale": args.lora_scale,
        "lora_last_n_blocks": args.lora_last_n_blocks,
        "pipeline": pipeline_name,
    }
    ckpt_path = os.path.join(save_dir, f"{args.exp_name}_{ts}.pt")

    final_val_loss: Optional[float] = None
    if args.epochs > 0:
        assert optimizer is not None
        for epoch in range(args.epochs):
            model.train()
            ep_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                pcs, _, _, metas = batch
                pcs = pcs.to(device)
                optimizer.zero_grad()
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
                    data_dir=args.data_dir,
                )
                loss.backward()
                optimizer.step()
                ep_loss += log["loss_total"]
                n_batches += 1
            ep_mean = ep_loss / max(n_batches, 1)
            logger.info("Epoch %d/%d train_loss_mean=%.6f", epoch + 1, args.epochs, ep_mean)

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
                )
                logger.info("Epoch %d val_loss=%.6f", epoch + 1, final_val_loss)

    if args.epochs > 0 and val_loader is not None and final_val_loss is None:
        final_val_loss = evaluate_mean_val_loss(
            model,
            val_loader,
            device,
            loss_mode=args.loss_mode,
            loss_alpha=args.loss_alpha,
            best_gt_weight=args.best_gt_weight,
            pred2gt_agg=args.loss_pred2gt_agg,
            data_dir=args.data_dir,
        )
        logger.info("Final val_loss=%.6f", final_val_loss)

    meta_train["final_val_loss"] = final_val_loss
    _save_checkpoint(
        ckpt_path,
        model,
        meta={
            **meta_train,
            "graspnet_ckpt": args.graspnet_ckpt,
            "graspnet_root": args.graspnet_root,
            "lift3d_root": args.lift3d_root,
            "lift3d_ckpt": args.lift3d_ckpt,
            "model_mode": args.model_mode,
            "encoder_type": "pure_graspnet" if args.model_mode == "pure_graspnet" else "lift3d",
            "fusion_mode": args.fusion_mode if args.model_mode == "lift3d" and args.fusion_mode != "additive" else None,
            "residual_alpha": args.residual_alpha if args.fusion_mode == "residual_proj" else None,
            "lora_r": args.lora_r,
            "lora_scale": args.lora_scale,
            "lora_last_n_blocks": args.lora_last_n_blocks,
            "pipeline": pipeline_name,
        },
    )
    logger.info("Saved %s", ckpt_path)

    summary_path = os.path.join(save_dir, f"{args.exp_name}_{ts}_train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(meta_train, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", summary_path)

    if args.run_eval_after:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "eval_benchmark.py"),
            "--data_dir",
            args.data_dir,
            "--dataset_root",
            args.dataset_root,
            "--checkpoint",
            ckpt_path,
            "--split",
            args.eval_split,
            "--camera",
            args.camera,
            "--top_k",
            "50",
            "--graspnet_ckpt",
            args.graspnet_ckpt,
        ]
        if args.graspnet_root:
            cmd.extend(["--graspnet_root", args.graspnet_root])
        if args.eval_extra_stats:
            cmd.append("--extra_stats")
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
