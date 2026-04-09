#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三阶段训练：LIFT3D encoder（LoRA）+ GC6D grasp head。

Stage 1: 不接我们的 head；用 LIFT3D 格式数据 (images, point_clouds, robot_states, raw_states, actions, texts)
          单独训练 encoder（仅 LoRA），用 LIFT3D 自带的 GraspHead 只算 loss，不更新 head。
Stage 2: 冻结 encoder，接我们的 GC6D grasp head，只训练 head。
Stage 3: encoder 与 head 一起训练。
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


def _ensure_lift3d_path(lift3d_root=None):
    if lift3d_root is None:
        lift3d_root = os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
    lift3d_root = os.path.abspath(os.path.expanduser(lift3d_root))
    if not os.path.isdir(lift3d_root):
        raise FileNotFoundError(f"LIFT3D root not found: {lift3d_root}")
    if lift3d_root not in sys.path:
        sys.path.insert(0, lift3d_root)
    return lift3d_root


def build_stage1_model(lift3d_root, lora_r=8, lora_scale=1.0, device="cuda"):
    """
    Stage 1 专用：PointNext + LoRA（无 adapter）+ LIFT3D GraspHead(512)。
    仅用 LIFT3D head 算 loss，不接我们的 GC6D head。
    """
    from models.lift3d_encoder import _load_point_next
    from models.lora import inject_lora, get_lora_params

    lift3d_root = _ensure_lift3d_path(lift3d_root)
    backbone = _load_point_next(lift3d_root, "point_next.yaml")
    inject_lora(backbone, r=lora_r, scale=lora_scale)

    from lift3d.models.grasp_head import GraspHead
    head = GraspHead(input_dim=512, hidden_dims=[512, 512, 256], width_min=0.01, width_max=0.12)

    class Stage1Wrapper(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, point_clouds):
            # 点云归一化（与 LIFT3DEncoder 一致）
            center = point_clouds.mean(dim=1, keepdim=True)
            pc = point_clouds - center
            scale = pc.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
            pc = pc / scale
            x = pc[:, :, :3].contiguous()
            with torch.amp.autocast(device_type="cuda", enabled=x.is_cuda):
                feat = self.backbone(x)
            return self.head(feat)

        def get_lora_params(self):
            return get_lora_params(self.backbone)

    model = Stage1Wrapper(backbone, head)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--camera", type=str, default="realsense-d415")
    parser.add_argument("--max_samples", type=int, default=1, help="先单数据点跑通")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="Stage1 LoRA 学习率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lift3d_root", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_scale", type=float, default=1.0)

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="1=encoder only (LoRA, LIFT3D-format data); 2=freeze encoder train head; 3=encoder+head",
    )
    parser.add_argument(
        "--stage3_unfreeze_lora_only",
        action="store_true",
        help="Stage3 只解冻 LoRA，默认解冻全部",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Stage2 加载 Stage1 的 ckpt；Stage3 加载 Stage2 的 ckpt",
    )
    parser.add_argument("--val_every", type=int, default=0, help="每 N 步在验证集上算 loss；0=不做")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--val_max_batches", type=int, default=50)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.stage == 1:
        # ---------- Stage 1: LIFT3D 格式数据，单独训 encoder（仅 LoRA），不接我们的 head ----------
        from data import GC6DLIFT3DFormatDataset, collate_lift3d

        dataset = GC6DLIFT3DFormatDataset(
            data_dir=args.data_dir,
            split=args.split,
            camera=args.camera,
            max_samples=args.max_samples,
            robot_state_dim=0,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.max_samples,
            shuffle=False,
            collate_fn=collate_lift3d,
        )
        model = build_stage1_model(
            args.lift3d_root,
            lora_r=args.lora_r,
            lora_scale=args.lora_scale,
            device=args.device,
        )
        lora_params = model.get_lora_params()
        for p in model.parameters():
            p.requires_grad = False
        for p in lora_params:
            p.requires_grad = True
        optim = torch.optim.Adam(lora_params, lr=args.lr_lora)
        criterion = nn.MSELoss()
        val_loader_s1 = None
        if args.val_every > 0:
            try:
                val_ds = GC6DLIFT3DFormatDataset(
                    data_dir=args.data_dir, split=args.val_split, camera=args.camera,
                    max_samples=None, robot_state_dim=0,
                )
                val_loader_s1 = DataLoader(val_ds, batch_size=args.max_samples, shuffle=False, collate_fn=collate_lift3d)
                print(f"Validation every {args.val_every} steps, split={args.val_split}, n={len(val_ds)}")
            except FileNotFoundError as e:
                print(f"No val index, skip validation: {e}")
        model.train()
        print("Stage 1: encoder only (LoRA), LIFT3D-format data; head used only for loss, frozen")

        for step in range(args.max_steps):
            batch = next(iter(loader))
            images, point_clouds, robot_states, raw_states, actions_gt, texts = batch
            point_clouds = point_clouds.to(args.device)
            actions_gt = actions_gt.to(args.device)

            optim.zero_grad()
            actions_pred = model(point_clouds)
            loss = criterion(actions_pred, actions_gt)
            loss.backward()
            optim.step()

            if (step + 1) % 100 == 0 or step == 0:
                print(f"step {step+1}/{args.max_steps} loss={loss.item():.6f}")
            if val_loader_s1 and (step + 1) % args.val_every == 0:
                model.eval()
                val_loss_sum, val_n = 0.0, 0
                with torch.no_grad():
                    for vi, batch in enumerate(val_loader_s1):
                        if vi >= args.val_max_batches:
                            break
                        _, point_clouds_v, _, _, actions_gt_v, _ = batch
                        point_clouds_v = point_clouds_v.to(args.device)
                        actions_gt_v = actions_gt_v.to(args.device)
                        val_loss_sum += criterion(model(point_clouds_v), actions_gt_v).item() * point_clouds_v.shape[0]
                        val_n += point_clouds_v.shape[0]
                model.train()
                if val_n > 0:
                    print(f"step {step+1} val_loss={val_loss_sum/val_n:.6f} (n={val_n})")
            if loss.item() < 1e-6:
                print(f"loss < 1e-6 at step {step+1}, done.")
                break

        save_dir = args.save_dir or os.path.join(ROOT, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "gc6d_lift3d_stage1.pt")
        torch.save(
            {
                "encoder_backbone": model.backbone.state_dict(),
                "encoder_type": "lift3d",
                "stage": 1,
                "step": step + 1,
                "loss": loss.item(),
            },
            ckpt_path,
        )
        print(f"Saved to {ckpt_path} (encoder_backbone only)")
        return

    # ---------- Stage 2 / 3: 我们的 policy（encoder + adapter + GC6D head） ----------
    from data import GC6DOfflineUnifiedDataset, collate_gc6d
    from models import build_lift3d_policy

    dataset = GC6DOfflineUnifiedDataset(
        data_dir=args.data_dir,
        split=args.split,
        camera=args.camera,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.max_samples,
        shuffle=False,
        collate_fn=collate_gc6d,
    )

    model = build_lift3d_policy(
        encoder_feat_dim=256,
        width_min=0.01,
        width_max=0.12,
        lift3d_root=args.lift3d_root,
        use_lora=True,
        lora_r=args.lora_r,
        lora_scale=args.lora_scale,
        normalize_pc=True,
    ).to(args.device)

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=args.device)
        if "encoder_backbone" in ckpt:
            model.encoder.backbone.load_state_dict(ckpt["encoder_backbone"], strict=True)
            print(f"Loaded encoder_backbone from Stage1: {args.ckpt}")
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            print(f"Loaded full model from {args.ckpt} (step={ckpt.get('step')})")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded state_dict from {args.ckpt}")

    if args.stage == 2:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.grasp_head.parameters():
            p.requires_grad = True
        train_params = list(model.grasp_head.parameters())
        optim = torch.optim.Adam(train_params, lr=args.lr)
        print("Stage 2: freeze encoder, train grasp head only")
    else:
        if args.stage3_unfreeze_lora_only:
            for p in model.encoder.parameters():
                p.requires_grad = False
            for p in model.encoder.get_lora_params():
                p.requires_grad = True
            for p in model.grasp_head.parameters():
                p.requires_grad = True
            train_params = model.encoder.get_lora_params() + list(model.grasp_head.parameters())
        else:
            for p in model.parameters():
                p.requires_grad = True
            train_params = list(model.parameters())
        optim = torch.optim.Adam(train_params, lr=args.lr)
        print("Stage 3: encoder + head" + (" (LoRA only)" if args.stage3_unfreeze_lora_only else " (full)"))

    criterion = nn.MSELoss()
    val_loader = None
    if args.val_every > 0:
        try:
            val_ds = GC6DOfflineUnifiedDataset(
                data_dir=args.data_dir, split=args.val_split, camera=args.camera, max_samples=None,
            )
            val_loader = DataLoader(val_ds, batch_size=args.max_samples, shuffle=False, collate_fn=collate_gc6d)
            print(f"Validation every {args.val_every} steps, split={args.val_split}, n={len(val_ds)}")
        except FileNotFoundError as e:
            print(f"No val index, skip validation: {e}")
    model.train()

    for step in range(args.max_steps):
        batch = next(iter(loader))
        pcs, actions_gt, _, _ = batch
        pcs = pcs.to(args.device)
        actions_gt = actions_gt.to(args.device)

        optim.zero_grad()
        actions_pred = model(pcs)
        loss = criterion(actions_pred, actions_gt)
        loss.backward()
        optim.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1}/{args.max_steps} loss={loss.item():.6f}")
        if val_loader and (step + 1) % args.val_every == 0:
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for vi, batch in enumerate(val_loader):
                    if vi >= args.val_max_batches:
                        break
                    pcs_v, actions_gt_v, _, _ = batch
                    pcs_v = pcs_v.to(args.device)
                    actions_gt_v = actions_gt_v.to(args.device)
                    val_loss_sum += criterion(model(pcs_v), actions_gt_v).item() * pcs_v.shape[0]
                    val_n += pcs_v.shape[0]
            model.train()
            if val_n > 0:
                print(f"step {step+1} val_loss={val_loss_sum/val_n:.6f} (n={val_n})")
        if loss.item() < 1e-6:
            print(f"loss < 1e-6 at step {step+1}, done.")
            break

    save_dir = args.save_dir or os.path.join(ROOT, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = f"gc6d_lift3d_stage{args.stage}.pt"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    torch.save(
        {
            "model": model.state_dict(),
            "encoder_type": "lift3d",
            "grasp_head_type": getattr(model, "grasp_head_type", "simple"),
            "grasp_head_num_proposals": getattr(model, "grasp_head_num_proposals", None),
            "stage": args.stage,
            "step": step + 1,
            "loss": loss.item(),
        },
        ckpt_path,
    )
    print(f"Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
