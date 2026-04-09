#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VGGT 微调 Stage1：冻结 encoder，只训 adapter + head。与 LIFT3D 流程对齐。"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DLIFT3DFormatDataset, collate_lift3d
from models import build_vggt_ft_policy
from utils.train_logging import setup_train_logging
from utils.loss import action_loss_topk_matched, action_loss_topk_matched_17d, pad_actions_multi, pad_gt_grasp_group_17d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--split", default="train")
    p.add_argument("--camera", default="realsense-d415")
    p.add_argument("--max_samples", type=int, default=1, help="0=全量")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--save_name", default="gc6d_vggt_ft_stage1.pt")
    p.add_argument("--log_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_every", type=int, default=0, help="每 N 步在验证集上算 loss；0=不做")
    p.add_argument("--val_split", default="val")
    p.add_argument("--val_max_batches", type=int, default=50)
    p.add_argument("--lora_r", type=int, default=8, help="LoRA 秩，与 LIFT3D/VGGT base 对齐")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--adapter_dropout", type=float, default=0.0)
    p.add_argument("--grasp_head_type", type=str, default="simple", choices=("simple", "simple_17d", "mature", "lift3d_action", "mature_17d", "graspnet"))
    p.add_argument("--num_proposals", type=int, default=4, help="graspnet head 的 proposal 数量")
    p.add_argument("--match_mode", type=str, default="bidir", choices=("bidir", "min", "hungarian"))
    p.add_argument("--alpha", type=float, default=0.7, help="match_mode=bidir 时 loss = alpha*(预测→GT) + (1-alpha)*(GT→预测)")
    p.add_argument("--pred2gt_top_frac", type=float, default=1.0, help="pred→gt 只对 cost 最小的前 frac 反传，如 0.25")
    p.add_argument("--loss_best_gt_weight", type=float, default=0.0, help="至少一个 pred 逼近主 GT 的 loss 权重，推荐 0.2~0.4")
    p.add_argument("--loss_17d", action="store_true", help="在 17D 空间算 loss，与 eval 一致；需 GraspNet head")
    args = p.parse_args()
    logger = setup_train_logging(args.log_dir, args.save_name)
    torch.manual_seed(args.seed)

    dataset = GC6DLIFT3DFormatDataset(
        data_dir=args.data_dir, split=args.split, camera=args.camera,
        max_samples=args.max_samples, image_size=224, load_gt_multi=True,
        train_color_augment=True, train_random_resized_crop=False)
    n_samples = len(dataset)
    batch_size = args.batch_size if args.batch_size is not None else (args.max_samples if args.max_samples > 0 else 32)
    shuffle = batch_size < n_samples
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_lift3d)

    model = build_vggt_ft_policy(
        encoder_feat_dim=256, ckpt_path=None, freeze_backbone=True,
        lora_r=args.lora_r, lora_scale=args.lora_scale,
        head_dropout=args.head_dropout, adapter_dropout=args.adapter_dropout,
        grasp_head_type=args.grasp_head_type,
        num_proposals=args.num_proposals,
    ).to(args.device)
    trainable = list(model.encoder.adapter.parameters()) + list(model.grasp_head.parameters())
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    val_loader = None
    if args.val_every > 0:
        try:
            val_ds = GC6DLIFT3DFormatDataset(
                data_dir=args.data_dir, split=args.val_split, camera=args.camera,
                max_samples=None, image_size=224, load_gt_multi=True,
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_lift3d)
            logger.info("Validation every %d steps, split=%s, n=%d", args.val_every, args.val_split, len(val_ds))
        except FileNotFoundError as e:
            logger.warning("No val index, skip validation: %s", e)
    model.train()
    data_iter = iter(loader)
    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        images, _, _, _, actions_gt, _, metas = batch
        images, actions_gt = images.to(args.device), actions_gt.to(args.device)
        use_loss_17d = getattr(args, "loss_17d", False)
        if use_loss_17d:
            gt_multi_17d = pad_gt_grasp_group_17d(metas, args.device)
            pred_k = model.forward_proposals_raw(images) if hasattr(model, "forward_proposals_raw") else model.forward_proposals(images)
        else:
            gt_multi = pad_actions_multi(metas, actions_gt, args.device)
            pred_k = model.forward_proposals(images)
        optim.zero_grad()
        if use_loss_17d:
            loss = action_loss_topk_matched_17d(pred_k, gt_multi_17d, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
        else:
            loss = action_loss_topk_matched(pred_k, actions_gt, gt_multi, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
        loss.backward()
        optim.step()
        if (step + 1) % 100 == 0 or step == 0:
            logger.info("[VGGT ft Stage1] step %d/%d loss=%.6f", step + 1, args.max_steps, loss.item())
        if val_loader and (step + 1) % args.val_every == 0:
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for vi, batch in enumerate(val_loader):
                    if vi >= args.val_max_batches:
                        break
                    images, _, _, _, actions_gt, _, metas = batch
                    images = images.to(args.device)
                    actions_gt = actions_gt.to(args.device)
                    if use_loss_17d:
                        gt_multi_17d_v = pad_gt_grasp_group_17d(metas, args.device)
                        pred_k_v = model.forward_proposals_raw(images) if hasattr(model, "forward_proposals_raw") else model.forward_proposals(images)
                        val_loss_sum += action_loss_topk_matched_17d(pred_k_v, gt_multi_17d_v, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight).item() * images.shape[0]
                    else:
                        gt_multi_v = pad_actions_multi(metas, actions_gt, args.device)
                        pred_k_v = model.forward_proposals(images)
                        val_loss_sum += action_loss_topk_matched(pred_k_v, actions_gt, gt_multi_v, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight).item() * images.shape[0]
                    val_n += images.shape[0]
            model.train()
            if val_n > 0:
                logger.info("[VGGT ft Stage1] step %d val_loss=%.6f (n=%d)", step + 1, val_loss_sum / val_n, val_n)
        if loss.item() < 1e-6:
            break

    save_dir = args.save_dir or os.path.join(ROOT, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, args.save_name)
    torch.save({"model": model.state_dict(), "encoder_type": "vggt_ft", "grasp_head_type": getattr(model, "grasp_head_type", "simple"), "grasp_head_num_proposals": getattr(model, "grasp_head_num_proposals", None), "lora_r": args.lora_r, "lora_scale": args.lora_scale, "stage": 1, "step": step + 1, "loss": loss.item()}, path)
    logger.info("Saved %s", path)

if __name__ == "__main__":
    main()
