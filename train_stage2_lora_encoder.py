#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2：冻结 grasp head 与 encoder.adapter，只训练 LIFT3D backbone 内的 LoRA 参数。
需先跑 Stage1 得到 ckpt，再本脚本加载后仅解冻 LoRA 训练。
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, GC6DLIFT3DFormatDataset, collate_gc6d, collate_lift3d
from models import build_lift3d_clip_policy, build_lift3d_clip_policy_multimodal
from utils.train_logging import setup_train_logging
from utils.loss import action_loss_topk_matched, action_loss_topk_matched_17d, pad_actions_multi, pad_gt_grasp_group_17d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--camera", type=str, default="realsense-d415")
    parser.add_argument("--max_samples", type=int, default=1, help="0=全量")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="gc6d_lift3d_stage2.pt")
    parser.add_argument("--ckpt_stage1", type=str, default=None, help="Stage1 保存的 checkpoint")
    parser.add_argument("--lift3d_root", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=0, help="每 N 步在验证集上算 loss；0=不做")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--val_max_batches", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩，与 VGGT base/ft 对齐")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--use_images", action="store_true", help="与 Stage1 一致：点云+图像双模态")
    parser.add_argument("--match_mode", type=str, default="bidir", choices=("bidir", "min", "hungarian"))
    parser.add_argument("--alpha", type=float, default=0.7, help="match_mode=bidir 时 loss = alpha*(预测→GT) + (1-alpha)*(GT→预测)")
    parser.add_argument("--pred2gt_top_frac", type=float, default=1.0, help="pred→gt 只对 cost 最小的前 frac 反传，如 0.25")
    parser.add_argument("--loss_best_gt_weight", type=float, default=0.0, help="至少一个 pred 逼近主 GT 的 loss 权重，推荐 0.2~0.4")
    parser.add_argument("--loss_17d", action="store_true", help="在 17D 空间算 loss，与 eval 一致；需 GraspNet head")
    args = parser.parse_args()

    logger = setup_train_logging(args.log_dir, args.save_name)
    if not args.ckpt_stage1:
        args.ckpt_stage1 = os.path.join(ROOT, "checkpoints", "gc6d_lift3d_stage1.pt")
    if not os.path.isfile(args.ckpt_stage1):
        raise FileNotFoundError(f"Stage1 ckpt not found: {args.ckpt_stage1}")

    ckpt1_meta = torch.load(args.ckpt_stage1, map_location="cpu", weights_only=False)
    grasp_head_type = ckpt1_meta.get("grasp_head_type", "simple")
    num_proposals = ckpt1_meta.get("grasp_head_num_proposals", 4)
    use_images = args.use_images or (ckpt1_meta.get("encoder_type") == "lift3d_clip_multimodal")

    torch.manual_seed(args.seed)

    if use_images:
        dataset = GC6DLIFT3DFormatDataset(
            data_dir=args.data_dir, split=args.split, camera=args.camera,
            max_samples=args.max_samples, image_size=224,
            train_color_augment=True, train_random_resized_crop=False, load_gt_multi=True,
        )
        collate_fn = collate_lift3d
    else:
        dataset = GC6DOfflineUnifiedDataset(
            data_dir=args.data_dir, split=args.split, camera=args.camera,
            max_samples=args.max_samples, load_gt_multi=True,
        )
        collate_fn = collate_gc6d
    n_samples = len(dataset)
    batch_size = args.batch_size if args.batch_size is not None else (args.max_samples if args.max_samples > 0 else 32)
    shuffle = batch_size < n_samples
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    if use_images:
        model = build_lift3d_clip_policy_multimodal(
            encoder_feat_dim=256, lift3d_root=args.lift3d_root, freeze_backbone=True,
            lora_r=args.lora_r, lora_scale=args.lora_scale,
            head_dropout=args.head_dropout, adapter_dropout=args.adapter_dropout,
            grasp_head_type=grasp_head_type, num_proposals=num_proposals,
            freeze_image_encoder=True,
        ).to(args.device)
    else:
        model = build_lift3d_clip_policy(
            encoder_feat_dim=256, lift3d_root=args.lift3d_root, freeze_backbone=True,
            lora_r=args.lora_r, lora_scale=args.lora_scale,
            head_dropout=args.head_dropout, adapter_dropout=args.adapter_dropout,
            grasp_head_type=grasp_head_type, num_proposals=num_proposals,
        ).to(args.device)

    sd = ckpt1_meta.get("model", ckpt1_meta)
    model.load_state_dict(sd, strict=True)

    # 冻结 head、adapter、fusion（多模态时）
    for p in model.grasp_head.parameters():
        p.requires_grad = False
    for p in model.encoder.adapter.parameters():
        p.requires_grad = False
    if use_images:
        for p in model.fusion.parameters():
            p.requires_grad = False
    # 只训 backbone 内 LoRA
    model.encoder.set_backbone_lora_trainable(True)

    lora_params = model.encoder.get_backbone_lora_params()
    if not lora_params:
        logger.warning("No LoRA params in backbone; training nothing. Check LIFT3D encoder.")
    optim = torch.optim.AdamW([p for p in lora_params if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    val_loader = None
    if args.val_every > 0:
        try:
            if use_images:
                val_ds = GC6DLIFT3DFormatDataset(
                    data_dir=args.data_dir, split=args.val_split, camera=args.camera, max_samples=None, image_size=224, load_gt_multi=True,
                )
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_lift3d)
            else:
                val_ds = GC6DOfflineUnifiedDataset(
                    data_dir=args.data_dir, split=args.val_split, camera=args.camera, max_samples=None, load_gt_multi=True,
                )
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_gc6d)
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
        if use_images:
            images, pcs, _, _, actions_gt, _, metas = batch
            images = images.to(args.device)
        else:
            pcs, actions_gt, _, metas = batch
        pcs = pcs.to(args.device)
        actions_gt = actions_gt.to(args.device)
        use_loss_17d = getattr(args, "loss_17d", False)
        if use_loss_17d:
            gt_multi_17d = pad_gt_grasp_group_17d(metas, args.device)
            pred_k = (model.forward_proposals_raw(pcs, images) if use_images else model.forward_proposals_raw(pcs)) if hasattr(model, "forward_proposals_raw") else (model.forward_proposals(pcs, images) if use_images else model.forward_proposals(pcs))
        else:
            gt_multi = pad_actions_multi(metas, actions_gt, args.device)
            pred_k = model.forward_proposals(pcs, images) if use_images else model.forward_proposals(pcs)

        optim.zero_grad()
        if use_loss_17d:
            loss = action_loss_topk_matched_17d(pred_k, gt_multi_17d, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
        else:
            loss = action_loss_topk_matched(pred_k, actions_gt, gt_multi, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
        loss.backward()
        optim.step()

        if (step + 1) % 100 == 0 or step == 0:
            logger.info("[Stage2] step %d/%d loss=%.6f", step + 1, args.max_steps, loss.item())
        if val_loader and (step + 1) % args.val_every == 0:
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for vi, batch in enumerate(val_loader):
                    if vi >= args.val_max_batches:
                        break
                    if use_images:
                        images_v, pcs_v, _, _, actions_gt_v, _, metas_v = batch
                        images_v = images_v.to(args.device)
                    else:
                        pcs_v, actions_gt_v, _, metas_v = batch
                    pcs_v, actions_gt_v = pcs_v.to(args.device), actions_gt_v.to(args.device)
                    if use_loss_17d:
                        gt_multi_17d_v = pad_gt_grasp_group_17d(metas_v, args.device)
                        pred_k_v = (model.forward_proposals_raw(pcs_v, images_v) if use_images else model.forward_proposals_raw(pcs_v)) if hasattr(model, "forward_proposals_raw") else (model.forward_proposals(pcs_v, images_v) if use_images else model.forward_proposals(pcs_v))
                        val_loss_sum += action_loss_topk_matched_17d(pred_k_v, gt_multi_17d_v, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight).item() * pcs_v.shape[0]
                    else:
                        gt_multi_v = pad_actions_multi(metas_v, actions_gt_v, args.device)
                        pred_k_v = model.forward_proposals(pcs_v, images_v) if use_images else model.forward_proposals(pcs_v)
                        val_loss_sum += action_loss_topk_matched(pred_k_v, actions_gt_v, gt_multi_v, mode=args.match_mode, alpha=args.alpha, pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight).item() * pcs_v.shape[0]
                    val_n += pcs_v.shape[0]
            model.train()
            if val_n > 0:
                logger.info("[Stage2] step %d val_loss=%.6f (n=%d)", step + 1, val_loss_sum / val_n, val_n)
        if loss.item() < 1e-6:
            logger.info("loss < 1e-6 at step %d, done.", step + 1)
            break

    save_dir = args.save_dir or os.path.join(ROOT, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, args.save_name)
    torch.save({
        "model": model.state_dict(),
        "encoder_type": "lift3d_clip_multimodal" if use_images else "lift3d_clip",
        "grasp_head_type": getattr(model, "grasp_head_type", "simple"),
        "grasp_head_num_proposals": getattr(model, "grasp_head_num_proposals", None),
        "lora_r": args.lora_r,
        "lora_scale": args.lora_scale,
        "stage": 2,
        "step": step + 1,
        "loss": loss.item(),
    }, ckpt_path)
    logger.info("Saved to %s", ckpt_path)


if __name__ == "__main__":
    main()
