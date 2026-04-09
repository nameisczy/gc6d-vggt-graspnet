#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断「loss 无梯度」断点：跑一次 forward + backward，打印 end_points 各 key 的 requires_grad，
以及 backward 后哪些参数收到了 grad。用于确认梯度是在 backbone / vpmodule / grasp_generator 哪一截断的。

用法（在 gc6d_grasp_pipeline 根目录）:
  python scripts/check_grad_flow.py --data_dir /path/to/offline_unified --encoder lift3d --graspnet_ckpt /path/to/checkpoint-rs.tar
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from data import GC6DOfflineUnifiedDataset, collate_gc6d
from utils.loss import action_loss_topk_matched_17d, pad_gt_grasp_group_17d
from models.graspnet_adapter import (
    build_encoder_adapter_graspnet,
    pred_decode_17d_differentiable,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--encoder", type=str, default="lift3d", choices=("lift3d", "lift3d_clip", "vggt_base", "vggt_ft"))
    p.add_argument("--graspnet_ckpt", type=str, required=True)
    p.add_argument("--graspnet_root", type=str, default=None)
    p.add_argument("--lift3d_root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_vggt = args.encoder in ("vggt_base", "vggt_ft")

    dataset = GC6DOfflineUnifiedDataset(
        args.data_dir, split="train", camera="realsense-d415",
        max_samples=4, load_gt_multi=True,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_gc6d, num_workers=0)
    batch = next(iter(loader))
    pcs = batch[0].to(device)
    metas = batch[3]
    images = None
    if use_vggt:
        from data import GC6DLIFT3DFormatDataset, collate_lift3d
        dataset = GC6DLIFT3DFormatDataset(
            args.data_dir, split="train", camera="realsense-d415",
            max_samples=4, image_size=224, load_gt_multi=True,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_lift3d, num_workers=0)
        batch = next(iter(loader))
        images, pcs = batch[0].to(device), batch[1].to(device)
        metas = batch[6]

    model = build_encoder_adapter_graspnet(
        encoder_type=args.encoder,
        graspnet_ckpt=args.graspnet_ckpt,
        encoder_feat_dim=256,
        graspnet_root=args.graspnet_root,
        lift3d_root=args.lift3d_root,
        device=device,
    )
    model.to(device)
    model.train()
    # Stage1 配置：只 adapter 可训，便于复现「主路径断」
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.grasp_net.parameters():
        p.requires_grad = False
    for p in model.adapter.parameters():
        p.requires_grad = True

    gt_17d = pad_gt_grasp_group_17d(metas, device)
    end_points = model(point_cloud=pcs, images=images)
    pred_17d = pred_decode_17d_differentiable(end_points, device, max_grasps=128)
    loss = action_loss_topk_matched_17d(pred_17d, gt_17d, mode="bidir", alpha=0.7, best_gt_weight=0.3)

    print("=" * 60)
    print("end_points 各 key 的 requires_grad（True=该 tensor 在计算图中）:")
    print("=" * 60)
    for k in sorted(end_points.keys()):
        v = end_points[k]
        if isinstance(v, torch.Tensor):
            rg = v.requires_grad
            print("  %s: requires_grad=%s  shape=%s" % (k, rg, tuple(v.shape)))
        else:
            print("  %s: (not Tensor)" % k)

    print()
    print("loss.requires_grad = %s, loss.grad_fn = %s" % (loss.requires_grad, loss.grad_fn))
    if not loss.requires_grad or loss.grad_fn is None:
        print(">>> 结论: 主路径断。上面 requires_grad=False 的 key 中，与 pred_decode 相关的是断点上游（grasp_generator 输出未带 grad）。")
    else:
        print(">>> 主路径有梯度。")

    print()
    print("Backward 后，各模块是否有参数收到 grad:")
    if loss.requires_grad and loss.grad_fn is not None:
        loss.backward()
    else:
        # loss 无梯度时 backward 会报错，用 cond^2 做一次 dummy backward 以验证 adapter 能收梯度
        if "_cond" in end_points:
            aux = end_points["_cond"].pow(2).sum() * 1e-6
            aux.backward()
            print("  (loss 无梯度，已用 cond^2 做 dummy backward)")
        else:
            print("  (loss 无梯度且无 _cond，跳过 backward)")
    has_enc = any(p.grad is not None and p.grad.abs().sum() != 0 for p in model.encoder.parameters())
    has_gn = any(p.grad is not None and p.grad.abs().sum() != 0 for p in model.grasp_net.parameters())
    has_adapter = any(p.grad is not None and p.grad.abs().sum() != 0 for p in model.adapter.parameters())
    print("  encoder 有非零 grad: %s" % has_enc)
    print("  grasp_net 有非零 grad: %s" % has_gn)
    print("  adapter 有非零 grad: %s" % has_adapter)
    print()
    print("若 adapter=True 且 encoder/grasp_net=False，说明只有 cond^2 辅助项在反传；主 loss 未传到 grasp_net 输出。")
    print("断点应在 graspnet-baseline 的 grasp_generator 内（其输出 grasp_score_pred 等未参与计算图）。")


if __name__ == "__main__":
    main()
