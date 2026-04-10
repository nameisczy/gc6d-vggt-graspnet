#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure GraspNet + LoRA head + ResidualReranker 训练入口（不修改 pred_decode / 评估 dump）。

experiment_mode:
  - baseline: 无 LoRA、无 reranker；loss_type=regression（全头冻结时仅作 sanity，需自行设可训参数）
  - lora_only: LoRA + regression
  - reranker_only: 冻结 GraspNet，仅 reranker + ranking loss
  - lora_reranker: LoRA + reranker + ranking loss

loss_type: regression | ranking（ranking 需 --reranker_enabled）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, collate_gc6d
from models.pure_graspnet import build_pure_graspnet_pipeline
from training.losses import compute_train_loss, evaluate_mean_val_loss


def _setup_logging(log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)


def _trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _save_checkpoint(path: str, model: torch.nn.Module, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "train_meta": meta,
        "encoder_type": "pure_graspnet",
        "model_mode": "pure_graspnet_residual",
    }
    torch.save(payload, path)


def parse_args():
    p = argparse.ArgumentParser(description="Residual reranker / LoRA head 训练")
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--graspnet_ckpt", type=str, required=True)
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="runs/residual_rerank")
    p.add_argument("--experiment_mode", type=str, default="lora_reranker", choices=(
        "baseline", "lora_only", "reranker_only", "lora_reranker",
    ))
    p.add_argument("--loss_type", type=str, default="ranking", choices=("regression", "ranking"))
    p.add_argument("--use_lora_head", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--inject_view_estimator_last", action="store_true", default=False)
    p.add_argument("--reranker_enabled", action="store_true", default=False)
    p.add_argument("--reranker_type", type=str, default="ranking", choices=("ranking", "quality"))
    p.add_argument("--reranker_fusion", type=str, default="add", choices=("add", "mul"))
    p.add_argument("--reranker_lambda", type=float, default=0.1)
    p.add_argument(
        "--reranker_unbounded",
        action="store_true",
        default=False,
        help="禁用 tanh 有界 residual（恢复旧融合：add 为 raw residual，mul 为 sigmoid）",
    )
    p.add_argument(
        "--reranker_3d_only",
        action="store_true",
        default=False,
        help="仅 score/width/tolerance 三维输入（小 MLP，兼容旧 checkpoint 结构）",
    )
    p.add_argument(
        "--no_normalize_reranker_center",
        action="store_true",
        default=False,
        help="9 维模式下不对 grasp 中心做场景尺度归一化",
    )
    p.add_argument("--ranking_margin", type=float, default=0.1)
    p.add_argument("--ranking_top_k", type=int, default=50)
    p.add_argument("--ranking_pos_dist_thresh", type=float, default=0.05)
    p.add_argument("--ranking_neg_samples_per_pos", type=int, default=3)
    p.add_argument("--ranking_max_pairs", type=int, default=2048)
    p.add_argument("--loss_mode", type=str, default="bidir")
    p.add_argument("--loss_alpha", type=float, default=0.7)
    p.add_argument("--best_gt_weight", type=float, default=0.3)
    p.add_argument("--loss_pred2gt_agg", type=str, default="min", choices=("min", "mean"))
    return p.parse_args()


def _apply_experiment_mode(model: torch.nn.Module, mode: str) -> None:
    """根据模式冻结/解冻（在 build 之后调用）。reranker_only 不调用 LoRA 解冻逻辑。"""
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "baseline":
        return
    rrk = getattr(model, "reranker", None)
    gn = getattr(model, "grasp_net", None)
    if mode == "reranker_only":
        if rrk is None:
            raise ValueError("experiment_mode=reranker_only 需要 reranker（--reranker_enabled）")
        for p in rrk.parameters():
            p.requires_grad_(True)
        return
    if gn is None:
        return
    from models.graspnet_head_lora import freeze_all_original_grasp_params

    freeze_all_original_grasp_params(gn)
    if mode == "lora_only":
        if rrk is not None:
            for p in rrk.parameters():
                p.requires_grad_(False)
        return
    if mode == "lora_reranker":
        if rrk is None:
            raise ValueError("experiment_mode=lora_reranker 需要 reranker（--reranker_enabled）")
        for p in rrk.parameters():
            p.requires_grad_(True)
        return


def main():
    args = parse_args()
    _setup_logging()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    use_lora = args.use_lora_head or args.experiment_mode in ("lora_only", "lora_reranker")
    reranker_on = args.reranker_enabled or args.experiment_mode in ("reranker_only", "lora_reranker")
    if args.loss_type == "ranking" and not reranker_on:
        raise SystemExit("loss_type=ranking 需要 reranker（--reranker_enabled 或 experiment_mode=reranker_only|lora_reranker）")

    model = build_pure_graspnet_pipeline(
        graspnet_ckpt=args.graspnet_ckpt,
        graspnet_root=args.graspnet_root,
        device=device,
        use_lora_head=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        inject_view_estimator_last=args.inject_view_estimator_last,
        reranker_enabled=reranker_on,
        reranker_extended_features=not args.reranker_3d_only,
    )
    _apply_experiment_mode(model, args.experiment_mode)

    params = _trainable_params(model)
    if not params:
        logging.warning("无可训练参数，请检查 experiment_mode / use_lora_head / reranker")
    opt = torch.optim.Adam(params, lr=args.lr) if params else None

    train_ds = GC6DOfflineUnifiedDataset(
        args.data_dir, split="train", camera=args.camera, max_samples=args.max_samples or None
    )
    val_ds = GC6DOfflineUnifiedDataset(
        args.data_dir, split=args.val_split, camera=args.camera, max_samples=args.max_samples or None
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_gc6d)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_gc6d)

    os.makedirs(args.out_dir, exist_ok=True)
    meta = vars(args).copy()
    meta["timestamp"] = datetime.utcnow().isoformat() + "Z"

    for epoch in range(args.epochs):
        model.train()
        if getattr(model, "grasp_net", None) is not None:
            from training.optim import set_pure_graspnet_train_state

            set_pure_graspnet_train_state(model)
        losses_ep = []
        for batch in train_loader:
            pcs, _, _, metas = batch
            pcs = pcs.to(device)
            if opt is None:
                break
            opt.zero_grad()
            loss, log = compute_train_loss(
                model=model,
                point_cloud=pcs,
                metas=metas,
                device=device,
                loss_mode=args.loss_mode,
                loss_alpha=args.loss_alpha,
                best_gt_weight=args.best_gt_weight,
                pred2gt_agg=args.loss_pred2gt_agg,
                max_grasps_decode=128,
                sort_and_truncate_decode=True,
                loss_type=args.loss_type,
                experiment_mode=args.experiment_mode,
                reranker_type=args.reranker_type,
                reranker_fusion=args.reranker_fusion,
                reranker_lambda=args.reranker_lambda,
                ranking_top_k=args.ranking_top_k,
                ranking_margin=args.ranking_margin,
                ranking_pos_dist_thresh=args.ranking_pos_dist_thresh,
                ranking_neg_samples_per_pos=args.ranking_neg_samples_per_pos,
                ranking_max_pairs=args.ranking_max_pairs,
                reranker_bounded=not args.reranker_unbounded,
                reranker_normalize_center=not args.no_normalize_reranker_center,
                reranker_extended_features=None,
            )
            loss.backward()
            opt.step()
            losses_ep.append(log.get("loss_total", float(loss.detach().item())))

        if opt is None:
            logging.warning("训练跳过（无可训练参数）")
            break

        mean_tr = sum(losses_ep) / max(len(losses_ep), 1)
        logging.info("epoch %d train loss_total mean=%.6f", epoch, mean_tr)

        if (epoch + 1) % args.val_every == 0:
            val_loss = evaluate_mean_val_loss(
                model,
                val_loader,
                device,
                loss_mode=args.loss_mode,
                loss_alpha=args.loss_alpha,
                best_gt_weight=args.best_gt_weight,
                pred2gt_agg=args.loss_pred2gt_agg,
                max_grasps=128,
                loss_type=args.loss_type,
                experiment_mode=args.experiment_mode,
                reranker_type=args.reranker_type,
                reranker_fusion=args.reranker_fusion,
                reranker_lambda=args.reranker_lambda,
                ranking_top_k=args.ranking_top_k,
                ranking_margin=args.ranking_margin,
                ranking_pos_dist_thresh=args.ranking_pos_dist_thresh,
                ranking_neg_samples_per_pos=args.ranking_neg_samples_per_pos,
                ranking_max_pairs=args.ranking_max_pairs,
                reranker_bounded=not args.reranker_unbounded,
                reranker_normalize_center=not args.no_normalize_reranker_center,
                reranker_extended_features=None,
            )
            logging.info("epoch %d val loss=%.6f", epoch, val_loss)

    ckpt_path = os.path.join(args.out_dir, "last.pt")
    _save_checkpoint(ckpt_path, model, meta)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info("saved %s", ckpt_path)


if __name__ == "__main__":
    main()
