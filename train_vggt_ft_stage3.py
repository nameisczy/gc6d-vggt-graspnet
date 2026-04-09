#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DLIFT3DFormatDataset, collate_lift3d
from models import build_vggt_ft_policy
from utils.train_logging import setup_train_logging
from utils.loss import action_loss_topk_matched, action_loss_topk_matched_17d, pad_actions_multi, pad_gt_grasp_group_17d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--camera", type=str, default="realsense-d415")
    parser.add_argument("--max_samples", type=int, default=1, help="0=全量")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=400, help="Stage3 步数 200~400 防 joint 爆")
    parser.add_argument("--lr", type=float, default=None, help="encoder(LoRA) lr，默认=lr_head*0.03 或 0.1")
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="gc6d_vggt_ft_stage3.pt")
    parser.add_argument("--ckpt_stage2", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=0, help="每 N 步在验证集上算一次 loss；0=不做 validation")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--val_max_batches", type=int, default=50, help="每次 validation 最多跑多少 batch")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩，与 LIFT3D/VGGT base 对齐")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--lora_last_n_blocks", type=int, default=2, help="Stage3 只训最后 n 个 block 的 LoRA，0=全开")
    parser.add_argument("--match_mode", type=str, default="bidir", choices=("bidir", "min", "hungarian"), help="bidir=双向最近邻(默认)，min=仅GT→预测，hungarian=一对一")
    parser.add_argument("--alpha", type=float, default=0.7, help="match_mode=bidir 时 loss = alpha*(预测→GT) + (1-alpha)*(GT→预测)")
    parser.add_argument("--pred2gt_top_frac", type=float, default=1.0, help="pred→gt 只对 cost 最小的前 frac 反传，如 0.25")
    parser.add_argument("--loss_best_gt_weight", type=float, default=0.0, help="至少一个 pred 逼近主 GT 的 loss 权重，推荐 0.2~0.4")
    parser.add_argument("--loss_17d", action="store_true", help="在 17D 空间算 loss，与 eval 一致；需 GraspNet head")
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--save_best_ckpt", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_val_worse", type=int, default=0, help="val 连续几次变差就停；0=不 early stop，1=第一次变差即停")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = args.lr_head * 0.03  # 0.03 或 0.1，防 joint 爆

    logger = setup_train_logging(args.log_dir, args.save_name)
    if not args.ckpt_stage2:
        args.ckpt_stage2 = os.path.join(ROOT, "checkpoints", "gc6d_vggt_ft_stage2.pt")
    if not os.path.isfile(args.ckpt_stage2):
        raise FileNotFoundError(f"Stage2 ckpt not found: {args.ckpt_stage2}")

    ckpt2_meta = torch.load(args.ckpt_stage2, map_location="cpu", weights_only=False)
    grasp_head_type = ckpt2_meta.get("grasp_head_type", "simple")
    num_proposals = ckpt2_meta.get("grasp_head_num_proposals", 4)

    torch.manual_seed(args.seed)

    dataset = GC6DLIFT3DFormatDataset(
        data_dir=args.data_dir,
        split=args.split,
        camera=args.camera,
        max_samples=args.max_samples,
        image_size=224,
        train_color_augment=True,
        train_random_resized_crop=False,
        load_gt_multi=True,
    )
    n_samples = len(dataset)
    batch_size = args.batch_size if args.batch_size is not None else (args.max_samples if args.max_samples > 0 else 32)
    shuffle = batch_size < n_samples
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_lift3d,
    )

    model = build_vggt_ft_policy(
        encoder_feat_dim=256,
        ckpt_path=None,
        freeze_backbone=False,
        lora_r=args.lora_r,
        lora_scale=args.lora_scale,
        head_dropout=args.head_dropout,
        adapter_dropout=args.adapter_dropout,
        grasp_head_type=grasp_head_type,
        num_proposals=num_proposals,
    ).to(args.device)
    model.load_state_dict(ckpt2_meta.get("model", ckpt2_meta), strict=True)

    model.encoder.set_backbone_lora_trainable(True, last_n_blocks=args.lora_last_n_blocks or None)
    model.encoder.set_adapter_trainable(True)
    for p in model.grasp_head.parameters():
        p.requires_grad = True

    encoder_params = list(model.encoder.get_backbone_lora_params(last_n_blocks=args.lora_last_n_blocks or None))
    head_params = list(model.encoder.adapter.parameters()) + list(model.grasp_head.parameters())
    optim = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
    ])
    # top-K 预测 + 多 GT matching（Hungarian 或 min）
    def _criterion(metas, actions_gt, device):
        gt_multi = pad_actions_multi(metas, actions_gt, device)
        return lambda pred_k: action_loss_topk_matched(pred_k, actions_gt, gt_multi, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)

    val_loader = None
    if args.val_every > 0:
        try:
            val_ds = GC6DLIFT3DFormatDataset(
                data_dir=args.data_dir, split=args.val_split, camera=args.camera,
                max_samples=None, image_size=224, load_gt_multi=True,
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_lift3d)
            logger.info("Validation: every %d steps, split=%s, %d samples", args.val_every, args.val_split, len(val_ds))
        except FileNotFoundError as e:
            logger.warning("No val index found, skip validation: %s", e)

    model.train()
    data_iter = iter(loader)
    best_val_loss = float("inf")
    no_improve_count = 0
    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        images, _, _, _, actions_gt, _, metas = batch
        images = images.to(args.device)
        actions_gt = actions_gt.to(args.device)

        use_loss_17d = getattr(args, "loss_17d", False)
        optim.zero_grad()
        if use_loss_17d:
            gt_multi_17d = pad_gt_grasp_group_17d(metas, args.device)
            pred_k = model.forward_proposals_raw(images) if hasattr(model, "forward_proposals_raw") else model.forward_proposals(images)
            loss = action_loss_topk_matched_17d(pred_k, gt_multi_17d, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
        else:
            pred_k = model.forward_proposals(images)
            loss = _criterion(metas, actions_gt, args.device)(pred_k)
        loss.backward()
        optim.step()

        if (step + 1) % 100 == 0 or step == 0:
            logger.info("[VGGT ft Stage3] step %d/%d loss=%.6f", step + 1, args.max_steps, loss.item())

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
                        L = action_loss_topk_matched_17d(pred_k_v, gt_multi_17d_v, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
                    else:
                        pred_k_v = model.forward_proposals(images)
                        L = _criterion(metas, actions_gt, args.device)(pred_k_v)
                    val_loss_sum += L.item() * images.shape[0]
                    val_n += images.shape[0]
            model.train()
            if val_n > 0:
                val_loss = val_loss_sum / val_n
                logger.info("[VGGT ft Stage3] step %d val_loss=%.6f (n=%d)", step + 1, val_loss, val_n)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if args.early_stop_val_worse > 0 and no_improve_count >= args.early_stop_val_worse:
                        logger.info("Early stop: val worsened %d time(s), stop at step %d", args.early_stop_val_worse, step + 1)
                        break

        if loss.item() < 1e-6:
            logger.info("loss < 1e-6 at step %d, done.", step + 1)
            break

    save_dir = args.save_dir or os.path.join(ROOT, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, args.save_name)
    torch.save({
        "model": model.state_dict(),
        "encoder_type": "vggt_ft",
        "grasp_head_type": getattr(model, "grasp_head_type", "simple"),
        "grasp_head_num_proposals": getattr(model, "grasp_head_num_proposals", None),
        "lora_r": args.lora_r,
        "lora_scale": args.lora_scale,
        "stage": 3,
        "step": step + 1,
        "loss": loss.item(),
    }, ckpt_path)
    logger.info("Saved to %s", ckpt_path)


if __name__ == "__main__":
    main()