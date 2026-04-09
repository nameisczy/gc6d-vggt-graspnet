#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化 replacement checkpoint 的特征数量级是否与 GraspNet backbone seed_features 对齐。

设计原则：
- 只做前向推理，不训练，不复用旧实验脚本逻辑
- 对每个 replacement checkpoint：
  1) 加载模型并冻结全部参数
  2) 在 vpmodule 前挂 hook，抓取真正送入 head 的 replacement features
  3) 用同一批点云跑纯预训练 GraspNet，得到 reference backbone seed_features
  4) 输出 JSON / CSV 统计与可视化

默认分析三类 replacement：
- VGGT replacement
- LIFT3D CLIP replacement
- DINOv2 replacement
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, collate_gc6d
from models.graspnet_adapter import load_graspnet_pretrained
from utils.batch_images import load_images_batch
from utils.load_model import load_policy_from_checkpoint


DEFAULT_VGGT_REP = os.path.join(
    ROOT,
    "checkpoints/alignment_runs/exp_vggt_rep/exp_vggt_rep_vggt_replacement_20260328_163036.pt",
)
DEFAULT_L3CLIP_REP = os.path.join(
    ROOT,
    "checkpoints/alignment_runs/exp_l3clip_rep/exp_l3clip_rep_lift3d_replacement_clip_20260328_164446.pt",
)
DEFAULT_L3DINO_REP = os.path.join(
    ROOT,
    "checkpoints/alignment_runs/exp_l3dino_rep/exp_l3dino_rep_lift3d_replacement_dinov2_20260328_164455.pt",
)


@dataclass
class CkptSpec:
    name: str
    path: str


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _global_stats(x: np.ndarray) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "abs_mean": float(np.mean(np.abs(a))),
        "l2": float(np.linalg.norm(a)),
    }


def _per_channel_std(x_bcs: np.ndarray) -> np.ndarray:
    # x: (B, C, S)
    cflat = np.transpose(x_bcs, (1, 0, 2)).reshape(x_bcs.shape[1], -1)
    return np.std(cflat, axis=1)


def _per_seed_std(x_bcs: np.ndarray) -> np.ndarray:
    # x: (B, C, S) -> (B*S,)
    seed_flat = np.transpose(x_bcs, (0, 2, 1)).reshape(-1, x_bcs.shape[1])
    return np.std(seed_flat, axis=1)


def _channel_cosine_matrix(ref_flat_nc: np.ndarray, other_flat_nc: np.ndarray) -> np.ndarray:
    # 输入: (N, C)
    na = np.linalg.norm(ref_flat_nc, axis=0) + 1e-12
    nb = np.linalg.norm(other_flat_nc, axis=0) + 1e-12
    m = (ref_flat_nc.T @ other_flat_nc) / (na[:, None] * nb[None, :])
    return np.clip(m, -1.0, 1.0)


def _plot_hist_compare(plt, arrays: Dict[str, np.ndarray], title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    for name, arr in arrays.items():
        vals = np.asarray(arr).ravel()
        plt.hist(vals, bins=100, density=True, alpha=0.4, label=name)
    plt.title(title)
    plt.xlabel("feature value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_box(plt, items: Dict[str, np.ndarray], ylabel: str, title: str, out_path: str) -> None:
    plt.figure(figsize=(11, 5))
    labels = list(items.keys())
    data = [np.asarray(v).ravel() for v in items.values()]
    plt.boxplot(data, labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_heatmap(plt, mat: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("replacement channel")
    plt.ylabel("graspnet seed channel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _model_requires_images(model: nn.Module) -> bool:
    enc = getattr(model, "encoder", None)
    if enc is not None and type(enc).__name__ == "VGGTEncoder":
        return True
    return bool(getattr(model, "requires_images", False))


def _freeze_all(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def _register_vpmodule_capture(model: nn.Module) -> Tuple[Any, Dict[str, List[torch.Tensor]]]:
    store: Dict[str, List[torch.Tensor]] = {"seed_features": [], "seed_xyz": []}
    vpmodule = model.grasp_net.view_estimator.vpmodule

    def _pre_hook(_mod, inputs):
        if len(inputs) >= 2:
            store["seed_xyz"].append(inputs[0])
            store["seed_features"].append(inputs[1])

    handle = vpmodule.register_forward_pre_hook(_pre_hook)
    return handle, store


@torch.no_grad()
def _forward_replacement_features(
    model: nn.Module,
    pcs: torch.Tensor,
    metas: List[dict],
    data_dir: str,
    device: torch.device,
) -> np.ndarray:
    images = None
    if _model_requires_images(model):
        images = load_images_batch(metas, data_dir, device)
    handle, store = _register_vpmodule_capture(model)
    try:
        _ = model(point_cloud=pcs, images=images)
    finally:
        handle.remove()
    if not store["seed_features"]:
        raise RuntimeError("未捕获到 vpmodule 输入特征")
    feat = store["seed_features"][-1].detach().float().cpu().numpy()
    return feat


@torch.no_grad()
def _forward_reference_seed_features(grasp_net: nn.Module, pcs: torch.Tensor) -> np.ndarray:
    end_points = {"point_clouds": pcs}
    ve = grasp_net.view_estimator
    seed_features, _seed_xyz, _ = ve.backbone(pcs, end_points)
    return seed_features.detach().float().cpu().numpy()


def _collect_features(
    *,
    ckpt_specs: List[CkptSpec],
    data_dir: str,
    camera: str,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    graspnet_ckpt: str,
    graspnet_root: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    ds = GC6DOfflineUnifiedDataset(
        data_dir=data_dir,
        split="val",
        camera=camera,
        max_samples=batch_size * num_batches,
        load_gt_multi=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_gc6d, num_workers=0)

    ref_graspnet = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    ref_graspnet.eval()
    _freeze_all(ref_graspnet)

    models: Dict[str, nn.Module] = {}
    metas_out: Dict[str, Dict[str, Any]] = {}
    for spec in ckpt_specs:
        if not spec.path:
            continue
        model = load_policy_from_checkpoint(
            spec.path,
            device=str(device),
            graspnet_ckpt=graspnet_ckpt,
            graspnet_root=graspnet_root,
        )
        _freeze_all(model)
        models[spec.name] = model
        ck = torch.load(spec.path, map_location="cpu", weights_only=False)
        metas_out[spec.name] = {
            "checkpoint_path": spec.path,
            "model_mode": ck.get("model_mode"),
            "encoder_type": ck.get("encoder_type"),
            "model_class_name": type(model).__name__,
        }

    ref_list: List[np.ndarray] = []
    rep_lists: Dict[str, List[np.ndarray]] = {k: [] for k in models}

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        pcs, _actions, _rgbs, metas = batch
        pcs = pcs.to(device)
        ref_list.append(_forward_reference_seed_features(ref_graspnet, pcs))
        for name, model in models.items():
            rep_lists[name].append(_forward_replacement_features(model, pcs, list(metas), data_dir, device))

    if not ref_list:
        raise RuntimeError("没有收集到任何 batch")

    ref_arr = np.concatenate(ref_list, axis=0)
    rep_arrs = {name: np.concatenate(v, axis=0) for name, v in rep_lists.items()}
    return ref_arr, rep_arrs, metas_out


def _make_summary_row(
    *,
    name: str,
    meta: Dict[str, Any],
    ref_arr: np.ndarray,
    rep_arr: np.ndarray,
) -> Dict[str, Any]:
    ref_g = _global_stats(ref_arr)
    rep_g = _global_stats(rep_arr)
    ref_ch_std = _per_channel_std(ref_arr)
    rep_ch_std = _per_channel_std(rep_arr)
    ref_seed_std = _per_seed_std(ref_arr)
    rep_seed_std = _per_seed_std(rep_arr)
    ratio = float(rep_g["std"] / (ref_g["std"] + 1e-12))
    return {
        "name": name,
        "checkpoint_path": meta["checkpoint_path"],
        "model_mode": meta["model_mode"],
        "encoder_type": meta["encoder_type"],
        "model_class_name": meta["model_class_name"],
        "ref_global_std": ref_g["std"],
        "replacement_global_std": rep_g["std"],
        "global_std_ratio_replacement_over_ref": ratio,
        "ref_per_channel_std_mean": float(np.mean(ref_ch_std)),
        "replacement_per_channel_std_mean": float(np.mean(rep_ch_std)),
        "ref_per_seed_std_mean": float(np.mean(ref_seed_std)),
        "replacement_per_seed_std_mean": float(np.mean(rep_seed_std)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Measure replacement feature scale vs GraspNet seed reference")
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--graspnet_ckpt",
        type=str,
        default=os.path.expanduser("~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"),
    )
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--vggt_replacement_ckpt", type=str, default=DEFAULT_VGGT_REP)
    p.add_argument("--lift3d_clip_replacement_ckpt", type=str, default=DEFAULT_L3CLIP_REP)
    p.add_argument("--dinov2_replacement_ckpt", type=str, default=DEFAULT_L3DINO_REP)
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(ROOT, "outputs", "replacement_feature_scale"),
    )
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt = _ensure_matplotlib()

    specs = [
        CkptSpec("vggt_replacement", os.path.abspath(os.path.expanduser(args.vggt_replacement_ckpt))),
        CkptSpec("lift3d_replacement_clip", os.path.abspath(os.path.expanduser(args.lift3d_clip_replacement_ckpt))),
        CkptSpec("lift3d_replacement_dinov2", os.path.abspath(os.path.expanduser(args.dinov2_replacement_ckpt))),
    ]
    specs = [s for s in specs if os.path.isfile(s.path)]
    if not specs:
        raise FileNotFoundError("未找到任何 replacement checkpoint")

    device = torch.device(args.device)
    ref_arr, rep_arrs, metas = _collect_features(
        ckpt_specs=specs,
        data_dir=args.data_dir,
        camera=args.camera,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=device,
        graspnet_ckpt=args.graspnet_ckpt,
        graspnet_root=args.graspnet_root,
    )

    ref_global = _global_stats(ref_arr)
    ref_ch_std = _per_channel_std(ref_arr)
    ref_seed_std = _per_seed_std(ref_arr)
    ref_flat_nc = np.transpose(ref_arr, (0, 2, 1)).reshape(-1, ref_arr.shape[1])

    detail_json: Dict[str, Any] = {
        "config": {
            "data_dir": args.data_dir,
            "camera": args.camera,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "graspnet_ckpt": args.graspnet_ckpt,
            "graspnet_root": args.graspnet_root,
        },
        "reference_graspnet_seed_features": {
            "global": ref_global,
            "per_channel_std": ref_ch_std.tolist(),
            "per_seed_std": ref_seed_std.tolist(),
        },
        "checkpoints": {},
    }
    summary_rows: List[Dict[str, Any]] = []

    hist_arrays: Dict[str, np.ndarray] = {"graspnet_ref": ref_arr.ravel()}
    box_channel: Dict[str, np.ndarray] = {"graspnet_ref": ref_ch_std}
    box_seed: Dict[str, np.ndarray] = {"graspnet_ref": ref_seed_std}

    for name, rep_arr in rep_arrs.items():
        rep_global = _global_stats(rep_arr)
        rep_ch_std = _per_channel_std(rep_arr)
        rep_seed_std = _per_seed_std(rep_arr)
        rep_flat_nc = np.transpose(rep_arr, (0, 2, 1)).reshape(-1, rep_arr.shape[1])
        ch_cos = _channel_cosine_matrix(ref_flat_nc, rep_flat_nc)
        ratio = float(rep_global["std"] / (ref_global["std"] + 1e-12))

        detail_json["checkpoints"][name] = {
            **metas[name],
            "global": rep_global,
            "per_channel_std": rep_ch_std.tolist(),
            "per_seed_std": rep_seed_std.tolist(),
            "global_std_ratio_replacement_over_ref": ratio,
            "per_channel_std_mean_ratio_over_ref": float(np.mean(rep_ch_std) / (np.mean(ref_ch_std) + 1e-12)),
            "per_seed_std_mean_ratio_over_ref": float(np.mean(rep_seed_std) / (np.mean(ref_seed_std) + 1e-12)),
        }
        summary_rows.append(_make_summary_row(name=name, meta=metas[name], ref_arr=ref_arr, rep_arr=rep_arr))

        hist_arrays[name] = rep_arr.ravel()
        box_channel[name] = rep_ch_std
        box_seed[name] = rep_seed_std
        _plot_heatmap(
            plt,
            ch_cos,
            f"Channel Cosine vs GraspNet Ref: {name}",
            os.path.join(fig_dir, f"heatmap_channel_cosine_{name}.png"),
        )

    with open(os.path.join(out_dir, "replacement_feature_scale_detail.json"), "w", encoding="utf-8") as f:
        json.dump(detail_json, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "replacement_feature_scale_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "checkpoint_path",
                "model_mode",
                "encoder_type",
                "model_class_name",
                "ref_global_std",
                "replacement_global_std",
                "global_std_ratio_replacement_over_ref",
                "ref_per_channel_std_mean",
                "replacement_per_channel_std_mean",
                "ref_per_seed_std_mean",
                "replacement_per_seed_std_mean",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with open(os.path.join(out_dir, "replacement_feature_scale_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    _plot_hist_compare(
        plt,
        hist_arrays,
        "Replacement Feature Value Histogram vs GraspNet Reference",
        os.path.join(fig_dir, "hist_values_compare.png"),
    )
    _plot_box(
        plt,
        box_channel,
        ylabel="per-channel std",
        title="Per-channel std vs GraspNet reference",
        out_path=os.path.join(fig_dir, "box_per_channel_std.png"),
    )
    _plot_box(
        plt,
        box_seed,
        ylabel="per-seed std",
        title="Per-seed std vs GraspNet reference",
        out_path=os.path.join(fig_dir, "box_per_seed_std.png"),
    )

    report = {
        "summary_csv": csv_path,
        "detail_json": os.path.join(out_dir, "replacement_feature_scale_detail.json"),
        "summary_json": os.path.join(out_dir, "replacement_feature_scale_summary.json"),
        "figures_dir": fig_dir,
    }
    with open(os.path.join(out_dir, "README.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
