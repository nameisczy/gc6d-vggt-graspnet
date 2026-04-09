#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraspNet seed_features vs LIFT3D / VGGT encoder 特征对齐分析（不改模型结构、不训练主模型）。

输出：数值表 JSON、图表目录、REPORT.md 摘要。
用法示例：
  conda activate gc6d
  cd ~/gc6d_grasp_pipeline
  python scripts/analyze_encoder_graspnet_feature_alignment.py \\
    --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \\
    --camera realsense-d415 \\
    --graspnet_ckpt ~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar \\
    --num_batches 8 --batch_size 2 \\
    --lift3d_ckpt /path/to/lift3d.pt \\
    --vggt_encoder_ckpt /path/to/vggt_ft.pt \\
    --out_dir outputs/feature_alignment_run1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def global_stats(t: np.ndarray) -> Dict[str, float]:
    t = np.asarray(t, dtype=np.float64).ravel()
    if t.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "max", "median", "q25", "q75", "l2", "l1", "abs_mean")}
    return {
        "mean": float(np.mean(t)),
        "std": float(np.std(t)),
        "min": float(np.min(t)),
        "max": float(np.max(t)),
        "median": float(np.median(t)),
        "q25": float(np.quantile(t, 0.25)),
        "q75": float(np.quantile(t, 0.75)),
        "l2": float(np.linalg.norm(t)),
        "l1": float(np.sum(np.abs(t))),
        "abs_mean": float(np.mean(np.abs(t))),
    }


def per_channel_stats(x_bc_s: np.ndarray) -> Dict[str, Any]:
    """x: (B, C, S)"""
    if x_bc_s.ndim != 3:
        raise ValueError(x_bc_s.shape)
    B, C, S = x_bc_s.shape
    # per channel over B*S
    ch = x_bc_s.transpose(1, 0, 2).reshape(C, -1)
    ch_mean = np.mean(ch, axis=1)
    ch_std = np.std(ch, axis=1)
    ch_min = np.min(ch, axis=1)
    ch_max = np.max(ch, axis=1)
    ch_abs_mean = np.mean(np.abs(ch), axis=1)
    ch_norm = np.linalg.norm(ch, axis=1)
    return {
        "C": C,
        "channel_mean": ch_mean.tolist(),
        "channel_std": ch_std.tolist(),
        "channel_min": ch_min.tolist(),
        "channel_max": ch_max.tolist(),
        "channel_abs_mean": ch_abs_mean.tolist(),
        "channel_norm": ch_norm.tolist(),
        "summary_channel_mean_of_means": float(np.mean(ch_mean)),
        "summary_channel_std_of_stds": float(np.mean(ch_std)),
        "summary_channel_norm_mean": float(np.mean(ch_norm)),
    }


def per_seed_l2_array(x_bcs: np.ndarray) -> np.ndarray:
    """(B,C,S) -> (B*S,) per-seed L2 over C"""
    B, C, S = x_bcs.shape
    t = np.transpose(x_bcs, (0, 2, 1)).reshape(-1, C)
    return np.linalg.norm(t, axis=1)


def per_seed_stats(x_bcs: np.ndarray) -> Dict[str, float]:
    """(B,C,S) -> per-seed L2 over C"""
    norms = per_seed_l2_array(x_bcs)
    B, C, S = x_bcs.shape
    t = np.transpose(x_bcs, (0, 2, 1)).reshape(-1, C)
    l1 = np.sum(np.abs(t), axis=1)
    return {
        "per_seed_l2_mean": float(np.mean(norms)),
        "per_seed_l2_std": float(np.std(norms)),
        "per_seed_l2_q25": float(np.quantile(norms, 0.25)),
        "per_seed_l2_q75": float(np.quantile(norms, 0.75)),
        "per_seed_l1_mean": float(np.mean(l1)),
    }


def per_channel_std_ratio(
    base_bcs: np.ndarray, other_bcs: np.ndarray
) -> Dict[str, Any]:
    """逐通道 std 比值 other/base，及 |log ratio| 摘要。"""
    if base_bcs.shape != other_bcs.shape or base_bcs.ndim != 3:
        return {"error": "shape_mismatch"}
    sb = per_channel_stats(base_bcs)["channel_std"]
    so = per_channel_stats(other_bcs)["channel_std"]
    sb = np.maximum(np.asarray(sb, dtype=np.float64), 1e-12)
    so = np.asarray(so, dtype=np.float64)
    r = so / sb
    lr = np.log(r + 1e-12)
    return {
        "ratio_mean": float(np.mean(r)),
        "ratio_median": float(np.median(r)),
        "ratio_min": float(np.min(r)),
        "ratio_max": float(np.max(r)),
        "abs_log_ratio_mean": float(np.mean(np.abs(lr))),
        "channels_where_other_gt_2x_base": int(np.sum(r > 2.0)),
        "channels_where_base_gt_2x_other": int(np.sum(r < 0.5)),
    }


def ratio_summary(a: Dict[str, float], b: Dict[str, float], keys: Tuple[str, ...]) -> Dict[str, Optional[float]]:
    out = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None or (isinstance(vb, float) and abs(vb) < 1e-12):
            out[k] = None
        else:
            out[k] = float(va / (vb + 1e-12))
    return out


# ---------------------------------------------------------------------------
# 特征抽取（不修改 nn.Module，仅前向子步骤）
# ---------------------------------------------------------------------------


def extract_graspnet_seed_only(grasp_net: nn.Module, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    end_points = {"point_clouds": pc}
    ve = grasp_net.view_estimator
    sf, sxyz, ep = ve.backbone(pc, end_points)
    return sf, sxyz


def extract_encoder_adapter(
    model: nn.Module, pc: torch.Tensor, images: Optional[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """EncoderAdapterGraspNet 中间量。"""
    B = pc.shape[0]
    if model.use_adapter:
        if images is not None:
            enc_feat = model.encoder(images)
        else:
            enc_feat = model.encoder(pc)
        cond = model.adapter(enc_feat.float())
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = cond[:, :256]
        if cond.shape[1] < 256:
            cond = torch.cat([cond, cond.new_zeros(B, 256 - cond.shape[1])], dim=1)
    else:
        cond = pc.new_zeros(B, 256)

    end_points = {"point_clouds": pc}
    seed_features, seed_xyz, end_points = model.grasp_net.view_estimator.backbone(pc, end_points)
    S = seed_features.shape[2]
    c = model.adapter_cond_coeff * cond.unsqueeze(2)
    cond_expand = c.expand(-1, -1, S)

    out: Dict[str, torch.Tensor] = {
        "cond": cond,
        "cond_expand": cond_expand,
        "c_raw": c,
        "seed_features": seed_features,
        "seed_xyz": seed_xyz,
    }

    mode = model.adapter_cond_mode
    if mode == "concat" and model.concat_proj is not None:
        sf_cat = torch.cat([seed_features, cond_expand], dim=1)
        out["concat_pre_proj"] = sf_cat
        out["seed_after_fusion"] = model.concat_proj(sf_cat)
    elif mode == "gated" and model.cond_gate is not None:
        gate = torch.sigmoid(model.cond_gate(cond)).unsqueeze(2)
        out["gate"] = gate
        out["seed_after_fusion"] = seed_features + gate * c
    elif mode == "film" and model.film_proj is not None:
        gb = model.film_proj(cond)
        half = gb.shape[1] // 2
        gamma, beta = gb[:, :half].unsqueeze(2), gb[:, half:].unsqueeze(2)
        out["film_gamma"] = gamma
        out["film_beta"] = beta
        if gamma.shape[1] == seed_features.shape[1]:
            out["seed_after_fusion"] = gamma * seed_features + beta
        else:
            out["seed_after_fusion"] = seed_features + c
    else:
        out["seed_after_fusion"] = seed_features + c

    return out


def extract_lift3d_local_fusion(model: nn.Module, pc: torch.Tensor) -> Dict[str, torch.Tensor]:
    from models.lift3d_local_fusion import nearest_neighbor_gather_features

    end_points = {"point_clouds": pc}
    ve = model.grasp_net.view_estimator
    seed_features, seed_xyz, end_points = ve.backbone(pc, end_points)
    seed_xyz_norm = model.encoder.normalize_seed_xyz(pc, seed_xyz)
    p_list, f_list = model.encoder.forward_seg_feat(pc)
    p_last, f_last = p_list[-1], f_list[-1]
    if f_last.dim() == 4:
        f_last = f_last.squeeze(-1)
    lift3d_raw = nearest_neighbor_gather_features(seed_xyz_norm, p_last, f_last).float()
    lift3d_seed = model.lift3d_seed_proj(lift3d_raw)
    out: Dict[str, torch.Tensor] = {
        "seed_features": seed_features,
        "lift3d_raw": lift3d_raw,
        "lift3d_seed": lift3d_seed,
        "seed_xyz": seed_xyz,
    }
    if model.fusion_mode == "concat_proj":
        fused = torch.cat([seed_features, lift3d_seed], dim=1)
        out["fused_pre_proj"] = fused
        out["seed_after_fusion"] = model.fusion_concat_proj(fused)
    else:
        delta = model.fusion_residual_proj(lift3d_seed)
        out["residual_delta"] = delta
        out["seed_after_fusion"] = seed_features + model.residual_alpha * delta
    return out


# ---------------------------------------------------------------------------
# 线性对齐（ridge）
# ---------------------------------------------------------------------------


def linear_map_mse_cosine(
    X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3
) -> Dict[str, float]:
    """
    X, Y: (N, D) 行对齐样本，拟合 Y ≈ XW，W 为 (D,D) 或最小二乘每列。
    这里用多输出 ridge: W = (X'X + λI)^{-1} X' Y
    """
    N, d_in = X.shape
    _, d_out = Y.shape
    XtX = X.T @ X + ridge * np.eye(d_in)
    Xty = X.T @ Y
    try:
        W = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(XtX) @ Xty
    Y_hat = X @ W
    mse = float(np.mean((Y - Y_hat) ** 2))
    # cosine per row
    yf = Y.reshape(-1)
    yh = Y_hat.reshape(-1)
    cos = float(np.dot(yf, yh) / (np.linalg.norm(yf) * np.linalg.norm(yh) + 1e-12))
    return {"mse": mse, "cosine_global": cos, "fro_W": float(np.linalg.norm(W))}


def channel_cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A,B: (N,C) 列=通道，返回 C×C 通道间余弦相似度。"""
    # M_ij = (A[:,i]^T B[:,j]) / (||A[:,i]|| ||B[:,j]||)
    na = np.linalg.norm(A, axis=0) + 1e-12
    nb = np.linalg.norm(B, axis=0) + 1e-12
    M = (A.T @ B) / (na[:, None] * nb[None, :])
    return np.clip(M, -1.0, 1.0)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_histogram_compare(
    plt,
    arrays: Dict[str, np.ndarray],
    title: str,
    path: str,
    bins: int = 80,
    log_density: bool = False,
):
    plt.figure(figsize=(10, 5))
    for name, arr in arrays.items():
        a = np.asarray(arr).ravel()
        a = a[np.isfinite(a)]
        plt.hist(a, bins=bins, density=True, alpha=0.45, label=name[:40])
    plt.legend()
    plt.title(title)
    plt.xlabel("value")
    if log_density:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_box_channel_metric(plt, data: Dict[str, np.ndarray], metric_name: str, path: str):
    """data: name -> (256,) per-channel values"""
    plt.figure(figsize=(12, 5))
    labels, vals = [], []
    for k, v in data.items():
        labels.append(k[:20])
        vals.append(np.asarray(v).ravel())
    plt.boxplot(vals, labels=labels)
    plt.title("per-channel %s" % metric_name)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_heatmap(plt, M: np.ndarray, title: str, path: str, vmax: float = 1.0, vmin: float = -1.0):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(M, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("encoder channel j")
    plt.ylabel("GraspNet seed channel i")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pca2d(plt, feats: Dict[str, np.ndarray], path: str, max_points_per: int = 4000):
    from sklearn.decomposition import PCA

    plt.figure(figsize=(9, 7))
    parts: List[np.ndarray] = []
    labels: List[str] = []
    for name, X in feats.items():
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        X = X[:max_points_per]
        parts.append(X)
        labels.extend([name[:50]] * len(X))
    Xall = np.vstack(parts)
    Z = PCA(n_components=2, random_state=42).fit_transform(Xall)
    labs = np.array(labels)
    for name in sorted(set(labels)):
        m = labs == name
        plt.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.35, label=name[:35])
    plt.legend()
    plt.title("PCA-2D (pooled rows, 256-dim)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_violin_norms(plt, per_seed_l2s: Dict[str, np.ndarray], path: str):
    plt.figure(figsize=(11, 5))
    labels, data = [], []
    for k, v in per_seed_l2s.items():
        labels.append(k[-28:])
        data.append(np.asarray(v).ravel())
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=20)
    plt.ylabel("per-seed L2 norm")
    plt.title("Per-seed L2 norm distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_box_per_seed_norm(plt, per_seed_l2s: Dict[str, np.ndarray], path: str):
    plt.figure(figsize=(11, 5))
    labels, data = [], []
    for k, v in per_seed_l2s.items():
        labels.append(k[-28:])
        data.append(np.asarray(v).ravel())
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=18)
    plt.ylabel("per-seed L2")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def try_tsne(plt, feats: Dict[str, np.ndarray], path: str, max_points_per: int = 2500):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return False
    parts, labels = [], []
    for name, X in feats.items():
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        X = X[:max_points_per]
        parts.append(X)
        labels.extend([name[:40]] * len(X))
    Xall = np.vstack(parts)
    Z = TSNE(n_components=2, random_state=42, perplexity=min(30, Xall.shape[0] - 1)).fit_transform(Xall)
    plt.figure(figsize=(9, 7))
    labs = np.array(labels)
    for name in sorted(set(labels)):
        m = labs == name
        plt.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.35, label=name[:35])
    plt.legend()
    plt.title("t-SNE-2D")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True


def offline_normalize_compare(x_bcs: np.ndarray) -> Dict[str, Any]:
    """x: (B,C,S) — LayerNorm / channel standardize 后全局统计。

    LayerNorm 对每个 (batch, seed) 上的 C 维向量归一化：输入须为 (B,S,C)，使
    normalized_shape=(C,) 与最后一维对齐。
    """
    t = torch.from_numpy(x_bcs.astype(np.float32))
    B, C, S = t.shape
    t_bsc = t.permute(0, 2, 1).contiguous()  # (B, S, C)
    ln = torch.nn.functional.layer_norm(t_bsc, (C,), eps=1e-5)
    # channel-wise z-score over B*S
    flat = t.permute(0, 2, 1).reshape(-1, C)
    mu = flat.mean(0, keepdim=True)
    sd = flat.std(0, keepdim=True)
    z = (flat - mu) / (sd + 1e-6)
    z = z.reshape(B, S, C).permute(0, 2, 1).numpy()
    return {
        "layer_norm_global": global_stats(ln.detach().numpy()),
        "channel_zscore_global": global_stats(z),
    }


def try_umap(plt, feats: Dict[str, np.ndarray], path: str, max_points: int = 4000):
    try:
        import umap
    except ImportError:
        return False
    Xs, labels = [], []
    for name, X in feats.items():
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        X = X[: max_points // max(len(feats), 1)]
        Xs.append(X)
        labels.extend([name] * len(X))
    Xall = np.vstack(Xs)
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(Xall)
    plt.figure(figsize=(8, 7))
    for name in feats.keys():
        m = np.array(labels) == name
        plt.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.4, label=name[:30])
    plt.legend()
    plt.title("UMAP-2D")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_images_for_batch(metas: List[dict], data_dir: str, device: torch.device, image_size: int = 224):
    """与 GC6DLIFT3DFormatDataset 一致的 ImageNet normalize。"""
    from PIL import Image
    from torchvision import transforms as T

    tfm = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    batch = []
    for m in metas:
        rp = m.get("rgb_path", "")
        if not rp:
            batch.append(torch.zeros(3, image_size, image_size))
            continue
        p = rp if os.path.isfile(rp) else os.path.join(data_dir, rp)
        if not os.path.isfile(p):
            batch.append(torch.zeros(3, image_size, image_size))
            continue
        img = Image.open(p).convert("RGB")
        batch.append(tfm(img))
    return torch.stack(batch, dim=0).to(device)


@dataclass
class RunConfig:
    data_dir: str
    camera: str
    batch_size: int
    num_batches: int
    graspnet_ckpt: str
    graspnet_root: str
    device: str
    out_dir: str
    lift3d_ckpt: Optional[str] = None
    lift3d_root: Optional[str] = None
    adapter_lift3d_pt: Optional[str] = None
    vggt_pt: Optional[str] = None
    lift3d_fusion_pt: Optional[str] = None
    fusion_mode: str = "concat_proj"
    seed: int = 42


def run(cfg: RunConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    fig_dir = os.path.join(cfg.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    from torch.utils.data import DataLoader

    from data import GC6DOfflineUnifiedDataset, collate_gc6d
    from models.graspnet_adapter import build_encoder_adapter_graspnet, load_graspnet_pretrained

    ds = GC6DOfflineUnifiedDataset(
        data_dir=cfg.data_dir,
        split="val",
        camera=cfg.camera,
        max_samples=cfg.batch_size * cfg.num_batches + 5,
        load_gt_multi=True,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_gc6d,
        num_workers=0,
    )

    grasp_net = load_graspnet_pretrained(cfg.graspnet_ckpt, device, cfg.graspnet_root, is_training=False)
    grasp_net.eval()

    models_info: List[Tuple[str, nn.Module, Callable]] = []

    if cfg.adapter_lift3d_pt and os.path.isfile(cfg.adapter_lift3d_pt):
        from utils.load_model import load_policy_from_checkpoint

        m = load_policy_from_checkpoint(
            cfg.adapter_lift3d_pt,
            device=str(device),
            graspnet_ckpt=cfg.graspnet_ckpt,
            graspnet_root=cfg.graspnet_root,
            lift3d_root=cfg.lift3d_root,
        )
        m.eval()

        def fn_lift(pc, metas_):
            imgs = None
            if getattr(m.encoder, "__class__", type).__name__ == "VGGTEncoder":
                imgs = load_images_for_batch(metas_, cfg.data_dir, device)
            return extract_encoder_adapter(m, pc, imgs)

        models_info.append(("adapter_ckpt", m, fn_lift))

    if cfg.vggt_pt and os.path.isfile(cfg.vggt_pt):
        from utils.load_model import load_policy_from_checkpoint

        m = load_policy_from_checkpoint(
            cfg.vggt_pt,
            device=str(device),
            graspnet_ckpt=cfg.graspnet_ckpt,
            graspnet_root=cfg.graspnet_root,
        )
        m.eval()

        def fn_vggt(pc, metas_):
            imgs = load_images_for_batch(metas_, cfg.data_dir, device)
            return extract_encoder_adapter(m, pc, imgs)

        models_info.append(("vggt_adapter_ckpt", m, fn_vggt))

    if cfg.lift3d_fusion_pt and os.path.isfile(cfg.lift3d_fusion_pt):
        from models.lift3d_local_fusion import build_lift3d_local_fusion_graspnet

        ck = torch.load(cfg.lift3d_fusion_pt, map_location=device, weights_only=False)
        st = ck.get("model", ck)
        # 需与 ckpt 一致参数；此处从 ckpt meta 读
        fm = ck.get("fusion_mode", cfg.fusion_mode)
        m = build_lift3d_local_fusion_graspnet(
            fusion_mode=fm if fm in ("concat_proj", "residual_proj") else cfg.fusion_mode,
            graspnet_ckpt=cfg.graspnet_ckpt,
            graspnet_root=cfg.graspnet_root,
            lift3d_root=cfg.lift3d_root or os.path.expanduser("~/LIFT3D"),
            lift3d_ckpt=cfg.lift3d_ckpt,
            device=device,
        )
        m.load_state_dict(st, strict=False)
        m.eval()

        def fn_fusion(pc, metas_):
            return extract_lift3d_local_fusion(m, pc)

        models_info.append(("lift3d_local_fusion", m, fn_fusion))

    # 聚合容器（单遍 dataloader：同一 batch 上比较 GraspNet 与各 encoder）
    buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    scene_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    batch_stability: Dict[str, List[float]] = defaultdict(list)

    def push(name: str, t: torch.Tensor, scene_key: str = "", bi: int = -1):
        buckets[name].append(_to_numpy(t))
        if scene_key:
            g = global_stats(_to_numpy(t))
            for k, v in g.items():
                if np.isfinite(v):
                    scene_stats[scene_key][f"{name}__{k}"].append(v)
        if bi >= 0:
            g2 = global_stats(_to_numpy(t))
            batch_stability[f"{name}__l2_batch"].append(g2["l2"])

    for bi, batch in enumerate(loader):
        if bi >= cfg.num_batches:
            break
        pcs, _, _, metas = batch
        pcs = pcs.to(device)
        scene_key = ",".join(str(int(m.get("sceneId", -1))) for m in metas[:3])

        with torch.no_grad():
            sf, _ = extract_graspnet_seed_only(grasp_net, pcs)
            push("graspnet_seed_features", sf, scene_key, bi)
            if bi == 0:
                print("[info] seed_features shape:", tuple(sf.shape), flush=True)
            for tag, _, fn in models_info:
                d = fn(pcs, metas)
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        push("%s__%s" % (tag, k), v, scene_key, bi)

    # 合并 batch
    merged: Dict[str, np.ndarray] = {}
    for k, parts in buckets.items():
        merged[k] = np.concatenate([p for p in parts], axis=0) if parts[0].ndim >= 1 else parts[0]

    # 写 stats JSON（所有 3D 张量做全套；2D 仅 global）
    report_stats: Dict[str, Any] = {
        "global": {},
        "per_channel_summary": {},
        "per_seed": {},
        "per_channel_distributions": {},
        "ratios": {},
        "batch_stability_l2": {k: {"min": float(np.min(v)), "max": float(np.max(v)), "mean": float(np.mean(v))} for k, v in batch_stability.items() if len(v)},
    }

    key_sf = "graspnet_seed_features"
    for k, arr in merged.items():
        if not isinstance(arr, np.ndarray):
            continue
        report_stats["global"][k] = global_stats(arr)
        if arr.ndim == 3:
            pcs_ = per_channel_stats(arr)
            report_stats["per_channel_summary"][k] = pcs_
            report_stats["per_seed"][k] = per_seed_stats(arr)
            report_stats["per_channel_distributions"][k] = {
                "channel_mean_hist": np.histogram(np.array(pcs_["channel_mean"]), bins=32, density=True)[0].tolist(),
                "channel_std_hist": np.histogram(np.array(pcs_["channel_std"]), bins=32, density=True)[0].tolist(),
                "channel_norm_hist": np.histogram(np.array(pcs_["channel_norm"]), bins=32, density=True)[0].tolist(),
            }
        elif arr.ndim == 2:
            report_stats["per_seed"][k] = {
                "per_sample_l2_mean": float(np.mean(np.linalg.norm(arr, axis=1))),
                "per_sample_l2_std": float(np.std(np.linalg.norm(arr, axis=1))),
            }

    # ratio vs graspnet seed
    base_g = report_stats["global"].get(key_sf, {})
    for k in report_stats["global"]:
        if k != key_sf:
            report_stats["ratios"]["%s_vs_%s" % (k, key_sf)] = ratio_summary(
                report_stats["global"][k],
                base_g,
                ("mean", "std", "abs_mean", "l2"),
            )

    # 逐通道 std 比值（与 graspnet 同形状的张量）
    base_arr = merged.get(key_sf)
    if isinstance(base_arr, np.ndarray) and base_arr.ndim == 3:
        report_stats["per_channel_std_ratio_vs_graspnet"] = {}
        for k, arr in merged.items():
            if k == key_sf or not isinstance(arr, np.ndarray) or arr.ndim != 3:
                continue
            if arr.shape == base_arr.shape:
                report_stats["per_channel_std_ratio_vs_graspnet"][k] = per_channel_std_ratio(base_arr, arr)

    # 跨 scene 的稳定性（每个 scene 内对 tensor 算了 global，再跨 scene 汇总）
    report_stats["scene_stability"] = {}
    for sk, inner in scene_stats.items():
        report_stats["scene_stability"][sk] = {}
        for metric_key, vals in inner.items():
            if not vals:
                continue
            report_stats["scene_stability"][sk][metric_key] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "n": len(vals),
            }
    # 关键量跨 scene 的波动范围
    scene_keys = list(report_stats["scene_stability"].keys())
    if scene_keys:
        pivot: Dict[str, List[float]] = defaultdict(list)
        for sk in scene_keys:
            for mk, agg in report_stats["scene_stability"][sk].items():
                if mk.endswith("__l2"):
                    pivot[mk].append(agg["mean"])
        report_stats["cross_scene_l2_range"] = {
            k: {"min": float(np.min(v)), "max": float(np.max(v)), "spread": float(np.max(v) - np.min(v))}
            for k, v in pivot.items()
            if v
        }

    # 离线归一化（LayerNorm / 通道 z-score）后全局统计 — 对比原空间与归一化后是否更接近 graspnet
    report_stats["offline_normalize"] = {}
    if isinstance(base_arr, np.ndarray) and base_arr.ndim == 3:
        report_stats["offline_normalize"][key_sf] = offline_normalize_compare(base_arr)
        for k, arr in merged.items():
            if k == key_sf or not isinstance(arr, np.ndarray) or arr.ndim != 3:
                continue
            if arr.shape != base_arr.shape:
                continue
            report_stats["offline_normalize"][k] = offline_normalize_compare(arr)
        # 归一化后 l2 与 graspnet 归一化后 l2 的比值（衡量 scale 是否拉近）
        ref_ln = report_stats["offline_normalize"][key_sf].get("layer_norm_global", {})
        ref_l2 = ref_ln.get("l2", float("nan"))
        report_stats["offline_normalize_l2_ratio_vs_graspnet_ln"] = {}
        for k, od in report_stats["offline_normalize"].items():
            if k == key_sf:
                continue
            ol2 = od.get("layer_norm_global", {}).get("l2")
            if ref_l2 and ol2 is not None and np.isfinite(ref_l2) and np.isfinite(ol2) and ref_l2 > 0:
                report_stats["offline_normalize_l2_ratio_vs_graspnet_ln"][k] = float(ol2 / ref_l2)

    with open(os.path.join(cfg.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(report_stats, f, indent=2, ensure_ascii=False, default=str)

    plt = _ensure_matplotlib()

    # 直方图对比
    hdict = {key_sf: merged[key_sf]}
    for k in merged:
        if "cond_expand" in k or "lift3d_seed" in k:
            hdict[k[-40:]] = merged[k]
    if len(hdict) > 1:
        plot_histogram_compare(plt, hdict, "feature value histogram", os.path.join(fig_dir, "hist_values.png"))
        plot_histogram_compare(
            plt,
            {a: np.abs(b) for a, b in hdict.items()},
            "|feature| histogram",
            os.path.join(fig_dir, "hist_abs.png"),
        )

    # per-channel std 分布
    ch_std_map = {}
    for k in report_stats["per_channel_summary"]:
        ch_std_map[k] = np.array(report_stats["per_channel_summary"][k]["channel_std"], dtype=np.float64)
    if len(ch_std_map) > 1:
        plot_box_channel_metric(plt, ch_std_map, "std", os.path.join(fig_dir, "box_per_channel_std.png"))

    # 通道余弦热图：graspnet vs cond_expand / lift3d_seed 等同形状张量
    ce_key = None
    for k in merged:
        if "cond_expand" in k:
            ce_key = k
            break
    l3_key = None
    for k in merged:
        if "lift3d_seed" in k:
            l3_key = k
            break

    def _heatmap_pair(tag: str, b_key: Optional[str]):
        if not b_key or key_sf not in merged:
            return
        A, B = merged[key_sf], merged[b_key]
        if A.shape != B.shape:
            return
        N = min(8000, A.shape[0] * A.shape[2])
        a = A.transpose(0, 2, 1).reshape(-1, A.shape[1])[:N]
        b = B.transpose(0, 2, 1).reshape(-1, B.shape[1])[:N]
        M = channel_cosine_matrix(a, b)
        plot_heatmap(
            plt,
            M,
            "channel cosine %s" % tag,
            os.path.join(fig_dir, "heatmap_ch_cosine_%s.png" % tag),
        )

    if ce_key:
        _heatmap_pair("cond_expand", ce_key)
    if l3_key:
        _heatmap_pair("lift3d_seed", l3_key)

    # PCA / UMAP / t-SNE（键名用完整前缀，避免混淆）
    pca_feats: Dict[str, np.ndarray] = {key_sf: merged[key_sf]}
    for k in sorted(merged.keys()):
        if k == key_sf:
            continue
        if "lift3d_seed" in k or "cond_expand" in k:
            pca_feats[k] = merged[k]
        elif "seed_features" in k and isinstance(merged[k], np.ndarray) and merged[k].ndim == 3:
            pca_feats[k] = merged[k]
    if len(pca_feats) >= 1:
        plot_pca2d(plt, pca_feats, os.path.join(fig_dir, "pca2d.png"))
        if not try_umap(plt, pca_feats, os.path.join(fig_dir, "umap2d.png")):
            pass
        if not try_tsne(plt, pca_feats, os.path.join(fig_dir, "tsne2d.png")):
            pass

    # per-seed L2 分布：violin + boxplot
    per_seed_l2s: Dict[str, np.ndarray] = {}
    for k, arr in merged.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[1] == merged[key_sf].shape[1]:
            per_seed_l2s[k[-48:]] = per_seed_l2_array(arr)
    if len(per_seed_l2s) > 1:
        plot_violin_norms(plt, per_seed_l2s, os.path.join(fig_dir, "violin_per_seed_l2.png"))
        plot_box_per_seed_norm(plt, per_seed_l2s, os.path.join(fig_dir, "box_per_seed_l2.png"))

    # 整体 norm 的 boxplot（各张量展平后 L2 在 batch 维上无意义；用每个 batch 的 tensor L2 来自 batch_stability）
    norm_box_data = {}
    for k, v in batch_stability.items():
        if k.endswith("__l2_batch") and v:
            norm_box_data[k.replace("__l2_batch", "")] = np.array(v, dtype=np.float64)
    if len(norm_box_data) > 1:
        plt.figure(figsize=(12, 5))
        labels = list(norm_box_data.keys())
        plt.boxplot([norm_box_data[x] for x in labels], labels=[x[-35:] for x in labels])
        plt.xticks(rotation=20)
        plt.ylabel("batch tensor L2 (global)")
        plt.title("L2 norm per batch across tensors")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "box_batch_l2.png"), dpi=150)
        plt.close()

    # 线性 ridge：encoder 3D 特征 → graspnet seed
    linear_results: Dict[str, Any] = {}
    Y_full = merged.get(key_sf)
    if isinstance(Y_full, np.ndarray) and Y_full.ndim == 3:
        Y = Y_full.transpose(0, 2, 1).reshape(-1, Y_full.shape[1])
        n = min(20000, Y.shape[0])

        def _ridge_pair(name: str, X_full: np.ndarray):
            if X_full.shape != Y_full.shape:
                return
            X = X_full.transpose(0, 2, 1).reshape(-1, X_full.shape[1])
            idx = np.random.choice(X.shape[0], min(n, X.shape[0]), replace=False)
            linear_results["%s_to_graspnet_seed_ridge" % name] = linear_map_mse_cosine(X[idx], Y[idx])
            Xn = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
            Yn = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
            linear_results["%s_to_graspnet_seed_ridge_normalized" % name] = linear_map_mse_cosine(
                Xn[idx], Yn[idx]
            )

        if ce_key:
            _ridge_pair("cond_expand", merged[ce_key])
        if l3_key:
            _ridge_pair("lift3d_seed", merged[l3_key])

    with open(os.path.join(cfg.out_dir, "linear_probe.json"), "w", encoding="utf-8") as f:
        json.dump(linear_results, f, indent=2)

    # REPORT.md：结构化三部分 + 自动结论要点
    rlines = [
        "# Encoder vs GraspNet seed_features 对齐分析\n",
        "\n## 一、数值统计表（完整见 `stats.json`）\n",
        "- `global`：各张量 mean/std/min/max/median/q25/q75/l2/l1/abs_mean\n",
        "- `per_channel_summary`：256 通道 mean/std/min/max/abs_mean/norm 及汇总\n",
        "- `per_seed`：per-seed L2/L1 分位数\n",
        "- `per_channel_distributions`：通道 mean/std/norm 的直方图计数\n",
        "- `ratios`：相对 `graspnet_seed_features` 的全局比值\n",
        "- `per_channel_std_ratio_vs_graspnet`：逐通道 std 比值摘要\n",
        "- `batch_stability_l2` / `scene_stability` / `cross_scene_l2_range`：batch 与 scene 稳定性\n",
        "- `offline_normalize`：LayerNorm 与通道 z-score 后的全局统计；`offline_normalize_l2_ratio_vs_graspnet_ln`\n",
        "\n## 二、可视化图集（`figures/`）\n",
        "- `hist_values.png` / `hist_abs.png`：值与绝对值密度\n",
        "- `box_per_channel_std.png`：各通道 std 的箱线图\n",
        "- `heatmap_ch_cosine_*.png`：通道余弦矩阵（cond_expand / lift3d_seed）\n",
        "- `pca2d.png` / `umap2d.png` / `tsne2d.png`：流形可视化\n",
        "- `violin_per_seed_l2.png` / `box_per_seed_l2.png`：per-seed L2\n",
        "- `box_batch_l2.png`：各张量按 batch 的全局 L2\n",
        "\n## 三、线性对齐探针（`linear_probe.json`）\n",
        json.dumps(linear_results, indent=2, ensure_ascii=False),
        "\n## 四、结论（根据本次运行自动摘要，需结合图核对）\n",
    ]

    def _scale_verdict() -> List[str]:
        out: List[str] = []
        ratios = report_stats.get("ratios", {})
        chrat = report_stats.get("per_channel_std_ratio_vs_graspnet", {})
        for rk, rv in ratios.items():
            l2v = rv.get("l2") if isinstance(rv, dict) else None
            if l2v is not None and np.isfinite(l2v):
                out.append(
                    "- **l2 比值** `%s`: %.4f（encoder 相对 graspnet）" % (rk, float(l2v))
                )
        for ck, cv in chrat.items():
            if isinstance(cv, dict) and "ratio_median" in cv:
                out.append(
                    "- **逐通道 std 中位比** `%s`: median=%.4f, 通道数>2x: %d, <0.5x: %d"
                    % (
                        ck,
                        cv["ratio_median"],
                        cv.get("channels_where_other_gt_2x_base", -1),
                        cv.get("channels_where_base_gt_2x_other", -1),
                    )
                )
        return out or ["- （无 ratio 数据：可能未加载 encoder ckpt）"]

    def _dist_verdict() -> List[str]:
        o: List[str] = []
        off = report_stats.get("offline_normalize_l2_ratio_vs_graspnet_ln", {})
        for k, v in off.items():
            o.append("- LayerNorm 后 L2 与 graspnet 之比 `%s`: %.4f" % (k, v))
        return o or ["- （无 offline_normalize 对比）"]

    def _sem_verdict() -> List[str]:
        o: List[str] = []
        for key, val in linear_results.items():
            if "ridge" in key and isinstance(val, dict):
                o.append(
                    "- `%s`: MSE=%.6f, cosine=%.4f, ||W||_F=%.2f"
                    % (key, val.get("mse", float("nan")), val.get("cosine_global", float("nan")), val.get("fro_W", float("nan")))
                )
        o.append("- 通道余弦热图：若**非对角**且 ridge MSE 仍高 → 支持 channel semantic mismatch。")
        return o

    rlines.extend(["### Scale mismatch\n"] + _scale_verdict())
    rlines.extend(["\n### Distribution mismatch\n"] + _dist_verdict())
    rlines.extend(["\n### Channel semantic mismatch（线性可对齐性 + 热图）\n"] + _sem_verdict())
    rlines.extend(
        [
            "\n### 综合与后续建议\n",
            "- 若 **l2/abs_mean 比值**远离 1 且逐通道 std 比两极化 → **scale** 问题突出，优先 **LayerNorm / 可学习 1×1 / channel-wise scale**。\n",
            "- 若直方图形状差异大、但 LayerNorm 后统计接近 → **distribution** 多为尺度/偏移，可先 **normalization** 再 fusion。\n",
            "- 若 ridge（尤其归一化后）仍高 MSE、热图杂乱 → **channel semantic** 突出，优先 **concat + 投影、FiLM、或与 graspnet 分支并行再融合**。\n",
        ]
    )

    with open(os.path.join(cfg.out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rlines))

    print("Done. Output:", cfg.out_dir, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--graspnet_ckpt", type=str, default=os.path.expanduser("~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"))
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--lift3d_root", type=str, default=os.path.expanduser("~/LIFT3D"))
    p.add_argument("--lift3d_ckpt", type=str, default=None, help="LIFT3D 预训练，用于 local fusion 构建")
    p.add_argument("--adapter_lift3d_pt", type=str, default=None, help="train_adapter 或 lift3d pipeline 的 .pt（含 LIFT3D+adapter）")
    p.add_argument("--vggt_pt", type=str, default=None, help="VGFT+adapter checkpoint")
    p.add_argument("--lift3d_fusion_pt", type=str, default=None, help="Lift3D local fusion 的 .pt")
    p.add_argument("--fusion_mode", type=str, default="concat_proj")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "outputs", "feature_alignment"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(
        RunConfig(
            data_dir=a.data_dir,
            camera=a.camera,
            batch_size=a.batch_size,
            num_batches=a.num_batches,
            graspnet_ckpt=a.graspnet_ckpt,
            graspnet_root=a.graspnet_root,
            device=a.device,
            out_dir=os.path.abspath(a.out_dir),
            lift3d_ckpt=a.lift3d_ckpt,
            lift3d_root=a.lift3d_root,
            adapter_lift3d_pt=a.adapter_lift3d_pt,
            vggt_pt=a.vggt_pt,
            lift3d_fusion_pt=a.lift3d_fusion_pt,
            fusion_mode=a.fusion_mode,
            seed=a.seed,
        )
    )
