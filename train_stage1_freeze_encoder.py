#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1：冻结 LIFT3D encoder（backbone），只训练 adapter + grasp head。
从 1 个数据点跑通开始，再可扩展到全量。
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


def _get_grasp_head_first_weight_param(model):
    """First Linear weight in grasp_head (for grad/weight debug)."""
    head = model.grasp_head
    if hasattr(head, "feature_extractor") and isinstance(head.feature_extractor, nn.Sequential):
        for m in head.feature_extractor:
            if isinstance(m, nn.Linear):
                return m.weight
    if hasattr(head, "fc"):
        return head.fc.weight
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--camera", type=str, default="realsense-d415")
    parser.add_argument("--max_samples", type=int, default=1, help="前 N 条；0=全量")
    parser.add_argument("--batch_size", type=int, default=None, help="未设则：max_samples>0 时为 max_samples，否则 32")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="head/adapter L2 正则")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="gc6d_lift3d_stage1.pt")
    parser.add_argument("--lift3d_root", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None, help="若设置则写入 log_dir/<save_name>.log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=0, help="每 N 步在验证集上算 loss；0=不做")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--val_max_batches", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩，与 VGGT base/ft 对齐")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--grasp_head_type", type=str, default="simple", choices=("simple", "simple_17d", "mature", "lift3d_action", "mature_17d", "graspnet"), help="simple=单层; simple_17d=单层输出17D; mature/lift3d_action=LIFT3D action head; mature_17d=成熟头输出17D; graspnet=GraspNet proposal")
    parser.add_argument("--num_proposals", type=int, default=4, help="graspnet head 的 proposal 数量")
    parser.add_argument("--use_images", action="store_true", help="点云+图像双模态，用 LIFT3D 格式数据与 VGGT 图像 encoder 融合")
    parser.add_argument("--match_mode", type=str, default="bidir", choices=("bidir", "min", "hungarian"), help="bidir=双向最近邻(默认)，min=仅GT→预测，hungarian=一对一")
    parser.add_argument("--alpha", type=float, default=0.7, help="match_mode=bidir 时 loss = alpha*(预测→GT) + (1-alpha)*(GT→预测)")
    parser.add_argument("--debug_grad", action="store_true", help="打印 optimizer 参数、梯度、权重变化，用于 overfit 诊断")
    parser.add_argument("--loss_components_log", action="store_true", help="打印 loss 分量 mse_t / mse_rot6d / mse_w（matched GT 平均）")
    parser.add_argument("--action_weights", type=str, default=None, help="t/rot6d/width 权重，如 '1,0.2,0.5'；不设则等权")
    parser.add_argument("--use_smooth_l1", action="store_true", help="用 SmoothL1 替代 MSE")
    parser.add_argument("--pred2gt_top_frac", type=float, default=1.0, help="pred→gt 只对 cost 最小的前 frac 反传，如 0.25；1=全部")
    parser.add_argument("--loss_best_gt_weight", type=float, default=0.0, help="加一项「至少一个 pred 逼近主 GT」权重，缓解多 GT 时 loss 停滞；推荐 0.2~0.4")
    parser.add_argument("--loss_17d", action="store_true", help="在 17D 空间算 loss：pred 11D 可微转 17D 与 GT 17D 比较，与 eval 一致；需 GraspNet head")
    args = parser.parse_args()
    if args.action_weights:
        args.action_weights = tuple(float(x) for x in args.action_weights.split(","))
    else:
        args.action_weights = None

    logger = setup_train_logging(args.log_dir, args.save_name)
    torch.manual_seed(args.seed)

    use_images = args.use_images
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
            grasp_head_type=args.grasp_head_type, num_proposals=args.num_proposals,
            freeze_image_encoder=True,
        ).to(args.device)
        trainable = list(model.encoder.adapter.parameters()) + list(model.fusion.parameters()) + list(model.grasp_head.parameters())
    else:
        model = build_lift3d_clip_policy(
            encoder_feat_dim=256, lift3d_root=args.lift3d_root, freeze_backbone=True,
            lora_r=args.lora_r, lora_scale=args.lora_scale,
            head_dropout=args.head_dropout, adapter_dropout=args.adapter_dropout,
            grasp_head_type=args.grasp_head_type, num_proposals=args.num_proposals,
        ).to(args.device)
        trainable = list(model.encoder.adapter.parameters()) + list(model.grasp_head.parameters())
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # (A) Debug: optimizer param groups and trainable count
    if args.debug_grad:
        total_params = sum(p.numel() for p in trainable)
        trainable_count = sum(1 for p in trainable if p.requires_grad)
        logger.info("[debug_grad] Optimizer: %d param groups, total trainable params %d (%d with requires_grad=True)",
                    len(optim.param_groups), total_params, trainable_count)
        for i, pg in enumerate(optim.param_groups):
            n_params = len(pg["params"])
            n_numel = sum(p.numel() for p in pg["params"])
            logger.info("[debug_grad]   param_group[%d]: len=%d, numel=%d, lr=%.2e", i, n_params, n_numel, pg["lr"])
        first_weight = _get_grasp_head_first_weight_param(model)
        if first_weight is None:
            logger.warning("[debug_grad] No grasp head first weight found (head type may differ)")
        else:
            logger.info("[debug_grad] Grasp head first weight: shape=%s, requires_grad=%s",
                        tuple(first_weight.shape), first_weight.requires_grad)

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
    w_prev = None  # for (C) weight change between steps
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

        first_weight = _get_grasp_head_first_weight_param(model) if args.debug_grad else None
        if args.debug_grad and first_weight is not None and step <= 1:
            w_before = first_weight.data.clone()

        optim.zero_grad()
        return_comp = args.loss_components_log and not use_loss_17d
        if use_loss_17d:
            loss = action_loss_topk_matched_17d(
                pred_k, gt_multi_17d,
                mode=args.match_mode, alpha=args.alpha,
                pred2gt_top_frac=args.pred2gt_top_frac,
                best_gt_weight=args.loss_best_gt_weight,
            )
        else:
            out = action_loss_topk_matched(
                pred_k, actions_gt, gt_multi,
                mode=args.match_mode, alpha=args.alpha,
                action_weights=args.action_weights,
                return_components=return_comp,
                use_smooth_l1=args.use_smooth_l1,
                pred2gt_top_frac=args.pred2gt_top_frac,
                best_gt_weight=args.loss_best_gt_weight,
            )
        if return_comp:
            loss, comp = out
            if (step + 1) % 100 == 0 or step == 0:
                logger.info("[loss_components] mse_t=%.2e mse_rot6d=%.2e mse_w=%.2e", comp["mse_t"], comp["mse_rot6d"], comp["mse_w"])
        elif not use_loss_17d:
            loss = out
        loss.backward()

        # (B) Debug: gradient after backward
        if args.debug_grad and first_weight is not None and step <= 1:
            g = first_weight.grad
            if g is None:
                logger.info("[debug_grad] step %d after backward: grasp_head first weight grad is None (backward broken or requires_grad=False)", step + 1)
            else:
                logger.info("[debug_grad] step %d after backward: grad.abs().mean()=%.2e, grad.norm()=%.2e", step + 1, g.abs().mean().item(), g.norm().item())

        optim.step()

        # (C) Debug: weight change after step
        if args.debug_grad and first_weight is not None and step <= 1:
            w_after = first_weight.data
            w_abs_mean = w_after.abs().mean().item()
            if w_prev is not None:
                diff_norm = (w_after - w_prev).norm().item()
                logger.info("[debug_grad] step %d after step: weight.abs().mean()=%.2e, ||w_t - w_{t-1}||=%.2e", step + 1, w_abs_mean, diff_norm)
            else:
                logger.info("[debug_grad] step %d after step: weight.abs().mean()=%.2e (no w_prev yet)", step + 1, w_abs_mean)
            w_prev = w_after.clone()

        if (step + 1) % 100 == 0 or step == 0:
            logger.info("[Stage1] step %d/%d loss=%.6f", step + 1, args.max_steps, loss.item())
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
                        v = action_loss_topk_matched_17d(pred_k_v, gt_multi_17d_v, mode=args.match_mode, alpha=args.alpha,
                                    pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
                    else:
                        gt_multi_v = pad_actions_multi(metas_v, actions_gt_v, args.device)
                        pred_k_v = model.forward_proposals(pcs_v, images_v) if use_images else model.forward_proposals(pcs_v)
                        v = action_loss_topk_matched(pred_k_v, actions_gt_v, gt_multi_v, mode=args.match_mode, alpha=args.alpha,
                                action_weights=args.action_weights, use_smooth_l1=args.use_smooth_l1,
                                pred2gt_top_frac=args.pred2gt_top_frac, best_gt_weight=args.loss_best_gt_weight)
                    val_loss_sum += v.item() * pcs_v.shape[0]
                    val_n += pcs_v.shape[0]
            model.train()
            if val_n > 0:
                logger.info("[Stage1] step %d val_loss=%.6f (n=%d)", step + 1, val_loss_sum / val_n, val_n)
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
        "stage": 1,
        "step": step + 1,
        "loss": loss.item(),
    }, ckpt_path)
    logger.info("Saved to %s", ckpt_path)


if __name__ == "__main__":
    main()
