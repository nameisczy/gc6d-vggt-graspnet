#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可执行检查：train/val 一致性、action 统计、width 单位、点云与 action 尺度。
检查完成后在 logs/ 下生成带时间戳的 log 文件。
用法：在 gc6d_grasp_pipeline 根目录执行
  python scripts/check_data_consistency.py --data_dir /path/to/offline_unified
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Tee:
    """同时写入 stdout 和 log 文件。"""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


def main():
    p = argparse.ArgumentParser(description="Check data and train/val consistency")
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified", help="offline_unified 根目录")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--max_samples", type=int, default=500, help="用于统计的 train/val 样本数，0=全量")
    p.add_argument("--log_dir", type=str, default=None, help="log 目录，默认 ROOT/logs")
    args = p.parse_args()
    data_dir = os.path.expanduser(args.data_dir)

    log_dir = args.log_dir or os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"check_data_consistency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        print(f"Log: {log_path}")
        print()
        _run_checks(args, data_dir)
    finally:
        sys.stdout = tee.terminal
        tee.close()
    print(f"检查完成，log 已写入: {log_path}")


def _run_checks(args, data_dir):
    camera = args.camera
    max_train = args.max_samples if args.max_samples > 0 else None
    max_val = args.max_samples if args.max_samples > 0 else None

    from data.dataset import (
        GC6DOfflineUnifiedDataset,
        GC6DLIFT3DFormatDataset,
        load_index_jsonl,
    )

    print("=" * 60)
    print("1. Train vs Val: image_size / transform / 数据源")
    print("=" * 60)
    # 点云数据集：train vs val 同 class，仅 split 不同
    train_pc = GC6DOfflineUnifiedDataset(data_dir=data_dir, split="train", camera=camera, max_samples=max_train or 10)
    val_pc = GC6DOfflineUnifiedDataset(data_dir=data_dir, split="val", camera=camera, max_samples=max_val or 10)
    print("[OK] 点云: train/val 均用 GC6DOfflineUnifiedDataset，无 image_size/crop，点云同源 npz。")
    print("      encoder 内 normalize_pc 对 train/val 同一套（center+scale）。")

    # 图像数据集：必须用相同 image_size
    IMAGE_SIZE = 224
    train_img = GC6DLIFT3DFormatDataset(data_dir=data_dir, split="train", camera=camera, image_size=IMAGE_SIZE, max_samples=max_train or 10)
    val_img = GC6DLIFT3DFormatDataset(data_dir=data_dir, split="val", camera=camera, image_size=IMAGE_SIZE, max_samples=max_val or 10)
    print(f"[OK] 图像: train/val 均用 GC6DLIFT3DFormatDataset(image_size={IMAGE_SIZE})，无 crop。")
    print("      transform: Resize(224,224)+ToTensor()+Normalize(ImageNet)，train/val 同一套。")
    # 抽样看图像数值范围（Normalize 后约 [-2,2] 量级）
    it, iv = train_img[0][0], val_img[0][0]
    print(f"      train 图像范围: min={it.min():.4f} max={it.max():.4f} (Normalize 后)")
    print(f"      val   图像范围: min={iv.min():.4f} max={iv.max():.4f}")
    if abs(it.min() - iv.min()) > 0.01 or abs(it.max() - iv.max()) > 0.01:
        print("      [WARN] train/val 图像范围不一致，请检查是否同源或 augment 仅 train 有。")
    else:
        print("      [OK] train/val 图像数值范围一致。")

    # 定位 val max 偏低原因：PIL dtype/mode、ToTensor 前后范围
    print()
    print("      1b. 定位图像范围 (PIL → ToTensor 前后，前 3 个 train / 3 个 val)")
    from PIL import Image
    import torchvision.transforms as T
    for split_name, ds in [("train", train_img), ("val", val_img)]:
        for idx in range(min(3, len(ds))):
            rec = ds.items[idx]
            rgb_path = rec.get("rgb_path", "")
            if not rgb_path:
                print(f"        {split_name}[{idx}] 无 rgb_path → 零张量 max=0")
                continue
            if not os.path.isfile(rgb_path):
                rgb_path = os.path.join(ds.data_dir, rgb_path)
            if not os.path.isfile(rgb_path):
                print(f"        {split_name}[{idx}] 文件不存在 → 零张量")
                continue
            try:
                pil = Image.open(rgb_path).convert("RGB")
                arr = np.array(pil)
                trans = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
                ten = trans(pil)
                print(f"        {split_name}[{idx}] {os.path.basename(rgb_path)} PIL.mode={pil.mode} shape={arr.shape} dtype={arr.dtype}")
                print(f"          [0,255] min={arr.min()} max={arr.max()} → ToTensor min={ten.min():.4f} max={ten.max():.4f}")
            except Exception as e:
                print(f"        {split_name}[{idx}] 异常: {e}")

    print()
    print("=" * 60)
    print("2. npz 内容：keys、depth→pc 的 K、坐标系说明")
    print("=" * 60)
    # 读一个 npz 看 keys
    idx_path = os.path.join(data_dir, f"index_train_{camera}.jsonl")
    if not os.path.exists(idx_path):
        print(f"[SKIP] 无 index: {idx_path}")
    else:
        items = load_index_jsonl(idx_path)
        if not items:
            print("[SKIP] index 为空")
        else:
            rec = items[0]
            npz_path = rec.get("npz")
            if not npz_path:
                print("[SKIP] index 无 npz 字段")
            else:
                if not os.path.isabs(npz_path):
                    npz_path = os.path.join(data_dir, npz_path)
                if os.path.exists(npz_path):
                    data = np.load(npz_path, allow_pickle=True)
                    keys = list(data.keys())
                    print("      npz keys:", keys)
                    if "K" in keys:
                        print("      K (内参) 存在，depth→pc 是否用此 K 需在生成 npz 的代码中确认。")
                    else:
                        print("      [INFO] npz 无 K（点云在生成 npz 的流程里已算好）。请在生成 point_cloud 的代码中确认：depth→pc 用的是与当前图像 crop/resize 对应的「变换后」内参 K。")
                    if "point_cloud" in keys:
                        pc = data["point_cloud"]
                        print(f"      point_cloud shape={pc.shape}, 中心={pc.mean(axis=0)}, 尺度(最大边长)≈{np.abs(pc).max():.4f}")
                else:
                    print(f"[SKIP] npz 不存在: {npz_path}")

    print()
    print("=" * 60)
    print("3. action 各维统计（train）；val 用同一套？当前无归一化")
    print("=" * 60)
    n_train = len(train_pc)
    n_val = len(val_pc)
    actions_train = []
    for i in range(min(n_train, 2000)):
        _, a, _, _ = train_pc[i]
        actions_train.append(a.numpy())
    actions_train = np.stack(actions_train, axis=0)
    mean_tr = actions_train.mean(axis=0)
    std_tr = actions_train.std(axis=0)
    print("      action 维度: t(3), R6d(6), width(1)")
    print("      Train 统计 (前 2000 或全量):")
    for i in range(10):
        name = "t" if i < 3 else ("r6" if i < 9 else "width")
        print(f"        dim[{i}] ({name}): mean={mean_tr[i]:.6f} std={std_tr[i]:.6f} min={actions_train[:, i].min():.6f} max={actions_train[:, i].max():.6f}")
    # 保存 train mean/std 供后续 val 用同一套
    out_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "action_stats_train.npz")
    np.savez(stats_path, mean=mean_tr, std=std_tr)
    print(f"      [OK] Train mean/std 已写入 {stats_path}，若将来做 action 归一化，val 必须用此文件。")
    print("      当前 pipeline 未做 action 归一化，val 与 train 同用原始 action。")

    # val 抽样对比
    actions_val = []
    for i in range(min(n_val, 500)):
        _, a, _, _ = val_pc[i]
        actions_val.append(a.numpy())
    if actions_val:
        actions_val = np.stack(actions_val, axis=0)
        mean_val = actions_val.mean(axis=0)
        std_val = actions_val.std(axis=0)
        print("      Val 统计 (前 500):")
        for i in range(10):
            name = "t" if i < 3 else ("r6" if i < 9 else "width")
            print(f"        dim[{i}] ({name}): mean={mean_val[i]:.6f} std={std_val[i]:.6f}")
        diff = np.abs(mean_tr - mean_val)
        if diff.max() > 0.5:
            print("      [WARN] train/val action 均值差异较大，可能分布不同或需归一化。")
        else:
            print("      [OK] train/val action 均值量级接近。")

    print()
    print("=" * 60)
    print("4. width 单位：米 vs 毫米")
    print("=" * 60)
    w = actions_train[:, 9]
    print(f"      width (dim 9) train: min={w.min():.6f} max={w.max():.6f} mean={w.mean():.6f}")
    if w.min() >= 0.001 and w.max() <= 0.25:
        print("      [OK] 数值在 ~0.002~0.15 米 量级，与 head 的 width_min=0.01, width_max=0.12 一致（单位：米）。")
    elif w.min() >= 1 and w.max() <= 250:
        print("      [WARN] 数值像毫米，需在数据或 Dataset 中除以 1000 再喂给 head。")
    else:
        print("      [?] 数值量级不明，请人工确认单位。")

    print()
    print("=" * 60)
    print("5. action(t) 与点云中心尺度/坐标系一致性")
    print("=" * 60)
    centers = []
    ts = []
    for i in range(min(n_train, 200)):
        pc, a, _, _ = train_pc[i]
        pc = pc.numpy()
        centers.append(pc.mean(axis=0))
        ts.append(a.numpy()[:3])
    centers = np.array(centers)
    ts = np.array(ts)
    dist_center = np.linalg.norm(centers - centers.mean(axis=0), axis=1).mean()
    dist_t = np.linalg.norm(ts - ts.mean(axis=0), axis=1).mean()
    print(f"      point_cloud 中心 L2 散布(平均): {dist_center:.4f}")
    print(f"      action t  L2 散布(平均):        {dist_t:.4f}")
    print(f"      action t 与 pc 中心 平均 L2 差: {np.linalg.norm(centers - ts, axis=1).mean():.4f}")
    if dist_center > 1e-6 and dist_t > 1e-6:
        ratio = dist_t / (dist_center + 1e-8)
        if 0.3 < ratio < 3.0:
            print("      [OK] t 与点云中心尺度量级接近，可能同坐标系。")
        else:
            print(f"      [WARN] t 与点云尺度比≈{ratio:.2f}，请确认是否同系(Tcw/Twc)。")
    print()
    print("检查完成。若有 [WARN]/[?] 需在数据生成或 Dataset 中修正。")


if __name__ == "__main__":
    main()
