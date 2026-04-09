#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单数据点训练：无正则，MSE loss，应收敛到 0。
用于验证 pipeline 从数据加载 -> 前向 -> 反传 -> 保存。
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 项目根为 gc6d_grasp_pipeline
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, collate_gc6d
from models import GC6DGraspPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--camera", type=str, default="realsense-d415")
    parser.add_argument("--max_samples", type=int, default=1, help="1 = 单数据点")
    parser.add_argument("--max_steps", type=int, default=2000, help="单样本过拟合步数")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = GC6DOfflineUnifiedDataset(
        data_dir=args.data_dir,
        split=args.split,
        camera=args.camera,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.max_samples, shuffle=False, collate_fn=collate_gc6d)

    model = GC6DGraspPolicy(encoder_feat_dim=256).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 单数据点：无正则，应 loss -> 0
    model.train()
    for step in range(args.max_steps):
        batch = next(iter(loader))
        pcs, actions_gt, _, metas = batch
        pcs = pcs.to(args.device)
        actions_gt = actions_gt.to(args.device)

        optim.zero_grad()
        actions_pred = model(pcs)
        loss = criterion(actions_pred, actions_gt)
        loss.backward()
        optim.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1}/{args.max_steps} loss={loss.item():.6f}")

        if loss.item() < 1e-6:
            print(f"loss < 1e-6 at step {step+1}, done.")
            break

    save_dir = args.save_dir or os.path.join(ROOT, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "gc6d_grasp_policy_one_sample.pt")
    torch.save({"model": model.state_dict(), "step": step + 1, "loss": loss.item()}, ckpt_path)
    print(f"Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
