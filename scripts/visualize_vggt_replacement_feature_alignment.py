#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT replacement：在进入 frozen GraspNet head（vpmodule）之前的特征 vs GraspNet backbone reference 对齐分析。

抽取张量（与训练/评估一致）：
- **reference（graspnet backbone）**：`view_estimator.vpmodule` 的 forward_pre_hook 捕获的 **第二个输入**，
  即 `seed_features`，与 `view_estimator.backbone(point_cloud, end_points)` 输出的 `seed_features` 相同。
- **vggt_replacement（任意对齐模式）**：投影 + scale/对齐模块之后、进入 vpmodule 之前的张量，
  同样通过 **同一 vpmodule 的 forward_pre_hook** 捕获第二项 `seed_features`。

因此四种情况比较的都是「真正送入 vpmodule / frozen head 前」的 256 维 seed feature map，形状 (B, 256, N_seed)。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import GC6DOfflineUnifiedDataset, collate_gc6d
from utils.batch_images import load_images_batch
from utils.load_model import load_policy_from_checkpoint


KEY_REF = "graspnet_reference"
KEY_LN_AFFINE = "vggt_replacement_layernorm_affine"
KEY_LN_AFFINE_ADAPTER = "vggt_replacement_ln_affine_adapter"
KEY_LN_ADAPTER = "vggt_replacement_ln_adapter"

ALIGN_MODES = {
    KEY_LN_AFFINE: "layernorm_affine",
    KEY_LN_AFFINE_ADAPTER: "layernorm_affine_adapter",
    KEY_LN_ADAPTER: "layernorm_adapter",
}


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _model_requires_images(model: nn.Module) -> bool:
    enc = getattr(model, "encoder", None)
    if enc is not None and type(enc).__name__ == "VGGTEncoder":
        return True
    return bool(getattr(model, "requires_images", False))


def _freeze_eval(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def _register_vpmodule_seed_capture(model: nn.Module) -> Tuple[Any, List[torch.Tensor]]:
    """在 vpmodule 前向入口处捕获第二个参数 seed_features (B, C, N)."""
    store: List[torch.Tensor] = []
    vpmodule = model.grasp_net.view_estimator.vpmodule

    def _pre_hook(_mod, inputs):
        if len(inputs) >= 2 and isinstance(inputs[1], torch.Tensor):
            store.append(inputs[1].detach())

    handle = vpmodule.register_forward_pre_hook(_pre_hook)
    return handle, store


@torch.no_grad()
def _forward_capture_seed_features(
    model: nn.Module,
    pcs: torch.Tensor,
    metas: List[dict],
    data_dir: str,
    device: torch.device,
) -> np.ndarray:
    """返回该 batch 最后一次进入 vpmodule 的 seed_features，numpy (B, C, N)."""
    handle, store = _register_vpmodule_seed_capture(model)
    try:
        images = None
        if _model_requires_images(model):
            images = load_images_batch(metas, data_dir, device)
        _ = model(point_cloud=pcs, images=images)
    finally:
        handle.remove()
    if not store:
        raise RuntimeError("未捕获到 vpmodule 输入 seed_features")
    feat = store[-1].float().cpu().numpy()
    return feat


def _global_stats(x: np.ndarray) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "abs_mean": float(np.mean(np.abs(a))),
        "l2": float(np.linalg.norm(a)),
    }


def _per_channel_mean(x_bcs: np.ndarray) -> np.ndarray:
    # (B, C, S) -> (C,)
    cflat = np.transpose(x_bcs, (1, 0, 2)).reshape(x_bcs.shape[1], -1)
    return np.mean(cflat, axis=1)


def _per_channel_std(x_bcs: np.ndarray) -> np.ndarray:
    cflat = np.transpose(x_bcs, (1, 0, 2)).reshape(x_bcs.shape[1], -1)
    return np.std(cflat, axis=1)


def _per_seed_l2(x_bcs: np.ndarray) -> np.ndarray:
    # (B, C, S) -> (B*S,) L2 over C per seed
    t = np.transpose(x_bcs, (0, 2, 1)).reshape(-1, x_bcs.shape[1])
    return np.linalg.norm(t, axis=1)


def _channel_cosine_matrix(ref_flat_nc: np.ndarray, other_flat_nc: np.ndarray) -> np.ndarray:
    """ref_flat_nc, other_flat_nc: (N, C) -> (C, C) 通道余弦相似度矩阵。"""
    na = np.linalg.norm(ref_flat_nc, axis=0) + 1e-12
    nb = np.linalg.norm(other_flat_nc, axis=0) + 1e-12
    m = (ref_flat_nc.T @ other_flat_nc) / (na[:, None] * nb[None, :])
    return np.clip(m, -1.0, 1.0)


def _channel_pearson_matrix(ref_flat_nc: np.ndarray, other_flat_nc: np.ndarray) -> np.ndarray:
    """每列中心化后算 Pearson 相关，形状 (C, C)。"""
    r = ref_flat_nc - np.mean(ref_flat_nc, axis=0, keepdims=True)
    o = other_flat_nc - np.mean(other_flat_nc, axis=0, keepdims=True)
    cr = r / (np.linalg.norm(r, axis=0, keepdims=True) + 1e-12)
    co = o / (np.linalg.norm(o, axis=0, keepdims=True) + 1e-12)
    return np.clip(cr.T @ co / max(ref_flat_nc.shape[0], 1), -1.0, 1.0)


def _flat_nc(x_bcs: np.ndarray) -> np.ndarray:
    return np.transpose(x_bcs, (0, 2, 1)).reshape(-1, x_bcs.shape[1])


def _wasserstein_1d(a: np.ndarray, b: np.ndarray, n_bins: int = 200) -> float:
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a.ravel(), b.ravel()))
    except Exception:
        return float("nan")


def _discover_ckpt(align_mode: str, search_roots: List[str]) -> Optional[str]:
    """在 search_roots 下找 model_mode=vggt_replacement 且 replacement_align_mode 匹配的最近 checkpoint。"""
    candidates: List[Tuple[float, str]] = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        pattern = os.path.join(os.path.expanduser(root), "**", "*.pt")
        for path in glob.glob(pattern, recursive=True):
            try:
                st = os.path.getmtime(path)
                ck = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            meta = ck.get("train_meta") if isinstance(ck.get("train_meta"), dict) else {}
            mm = meta.get("model_mode") or ck.get("model_mode")
            ram = meta.get("replacement_align_mode") or ck.get("replacement_align_mode")
            if mm != "vggt_replacement":
                continue
            if ram != align_mode:
                continue
            candidates.append((st, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _collect_all_features(
    *,
    data_dir: str,
    camera: str,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    graspnet_ckpt: str,
    graspnet_root: str,
    ckpt_map: Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    ds = GC6DOfflineUnifiedDataset(
        data_dir=data_dir,
        split="val",
        camera=camera,
        max_samples=batch_size * num_batches,
        load_gt_multi=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_gc6d, num_workers=0)

    models: Dict[str, nn.Module] = {}
    paths: Dict[str, str] = {}

    # reference: pure pipeline（与 eval 一致）
    from models.pure_graspnet import build_pure_graspnet_pipeline

    ref_model = build_pure_graspnet_pipeline(
        graspnet_ckpt=graspnet_ckpt,
        graspnet_root=graspnet_root,
        device=device,
    )
    _freeze_eval(ref_model)
    models[KEY_REF] = ref_model
    paths[KEY_REF] = graspnet_ckpt

    for key, align_mode in ALIGN_MODES.items():
        ckpt_path = ckpt_map.get(key)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("缺少或无效 checkpoint: %s -> %s" % (key, ckpt_path))
        m = load_policy_from_checkpoint(
            ckpt_path,
            device=str(device),
            graspnet_ckpt=graspnet_ckpt,
            graspnet_root=graspnet_root,
        )
        _freeze_eval(m)
        models[key] = m
        paths[key] = os.path.abspath(ckpt_path)

    arrs: Dict[str, List[np.ndarray]] = {k: [] for k in models}

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        pcs, _a, _r, metas = batch
        pcs = pcs.to(device)
        mlist = list(metas)
        for name, model in models.items():
            arrs[name].append(_forward_capture_seed_features(model, pcs, mlist, data_dir, device))

    out: Dict[str, np.ndarray] = {}
    for k, lst in arrs.items():
        if not lst:
            raise RuntimeError("没有收集到特征: %s" % k)
        out[k] = np.concatenate(lst, axis=0)
    return out, paths


def _subsample_rows(flat: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = flat.shape[0]
    if n <= max_rows:
        return flat
    idx = rng.choice(n, size=max_rows, replace=False)
    return flat[idx]


def _plot_embeddings(
    plt,
    embeddings: Dict[str, np.ndarray],
    title: str,
    out_path: str,
    colors: Dict[str, str],
) -> None:
    plt.figure(figsize=(9, 7))
    for name, xy in embeddings.items():
        plt.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.35, c=colors.get(name, "#666666"), label=name)
    plt.legend(markerscale=3, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_report(
    path: str,
    stats: Dict[str, Any],
    best_scale: str,
    best_w1: str,
    best_pca: str,
    best_hm: str,
    ln_vs_lnadapter: str,
    affine_vs_affine_adapter: str,
    gap_note: str,
) -> None:
    lines = [
        "# VGGT replacement vs GraspNet reference（vpmodule 输入）\n",
        "\n",
        "## 摘要（自动根据数值指标给出）\n",
        "\n",
        "### 哪一种在 **scale（全局 std 比值接近 1）** 上最接近 backbone？\n",
        "\n",
        "**%s**\n" % best_scale,
        "\n",
        "### 哪一种在 **整体分布形状（与 reference 的一维 Wasserstein 距离更小）** 上最接近？\n",
        "\n",
        "**%s**\n" % best_w1,
        "\n",
        "### 哪一种在 **PCA 空间（样本到 reference 质心的平均距离更小）** 上更接近？\n",
        "\n",
        "**%s**\n" % best_pca,
        "\n",
        "### 通道余弦热图（对角线均值更高通常表示通道对齐更好）上哪一种更好？\n",
        "\n",
        "**%s**\n" % best_hm,
        "\n",
        "### LayerNorm / affine / adapter 的对比（脚本内启发式）\n",
        "\n",
        "- **layernorm_affine vs layernorm_adapter（看 affine 是否关键）**：\n",
        "  %s\n" % ln_vs_lnadapter,
        "- **layernorm_affine vs layernorm_affine_adapter（看 adapter 是否有帮助）**：\n",
        "  %s\n" % affine_vs_affine_adapter,
        "- **与 GraspNet backbone 是否仍有明显 gap**：\n",
        "  %s\n" % gap_note,
        "\n",
        "## 指标表（摘录）\n",
        "\n",
        "```json\n",
        json.dumps(stats, indent=2, ensure_ascii=False)[:12000],
        "\n```\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(
        description="可视化并统计 VGGT replacement 进入 vpmodule 前特征 vs GraspNet backbone reference"
    )
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--graspnet_ckpt", type=str, default=os.path.expanduser("~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"))
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--out_dir", type=str, default=os.path.expanduser("~/gc6d_grasp_pipeline/analysis/vggt_replacement_feature_compare"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_embed_rows", type=int, default=4000, help="PCA/t-SNE/UMAP 子采样行数上限")
    p.add_argument(
        "--search_roots",
        type=str,
        nargs="*",
        default=[
            os.path.expanduser("~/gc6d_grasp_pipeline/checkpoints/alignment_runs"),
            os.path.expanduser("~/gc6d_grasp_pipeline/checkpoints/alignment_experiments"),
        ],
        help="--auto_discover 时在这些目录下递归搜 .pt",
    )
    p.add_argument(
        "--auto_discover",
        action="store_true",
        help="按 train_meta.replacement_align_mode 自动选各模式最新 checkpoint",
    )
    p.add_argument("--ckpt_vggt_ln_affine", type=str, default=None)
    p.add_argument("--ckpt_vggt_ln_affine_adapter", type=str, default=None)
    p.add_argument("--ckpt_vggt_ln_adapter", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    fig_dir = os.path.join(out_dir, "figures")
    stat_dir = os.path.join(out_dir, "stats")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    ckpt_map: Dict[str, str] = {}
    if args.auto_discover:
        for key, mode in ALIGN_MODES.items():
            found = _discover_ckpt(mode, args.search_roots)
            if not found:
                raise FileNotFoundError("auto_discover 未找到 align_mode=%s，请显式传参或检查 search_roots" % mode)
            ckpt_map[key] = found
        with open(os.path.join(stat_dir, "auto_discovered_checkpoints.json"), "w", encoding="utf-8") as f:
            json.dump({ALIGN_MODES[k]: ckpt_map[k] for k in ALIGN_MODES}, f, indent=2, ensure_ascii=False)
    else:
        if not args.ckpt_vggt_ln_affine or not args.ckpt_vggt_ln_affine_adapter or not args.ckpt_vggt_ln_adapter:
            raise ValueError("请提供三个 --ckpt_vggt_* 或使用 --auto_discover")
        ckpt_map[KEY_LN_AFFINE] = os.path.expanduser(args.ckpt_vggt_ln_affine)
        ckpt_map[KEY_LN_AFFINE_ADAPTER] = os.path.expanduser(args.ckpt_vggt_ln_affine_adapter)
        ckpt_map[KEY_LN_ADAPTER] = os.path.expanduser(args.ckpt_vggt_ln_adapter)

    feats, paths = _collect_all_features(
        data_dir=args.data_dir,
        camera=args.camera,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=device,
        graspnet_ckpt=os.path.expanduser(args.graspnet_ckpt),
        graspnet_root=os.path.expanduser(args.graspnet_root),
        ckpt_map=ckpt_map,
    )

    ref = feats[KEY_REF]
    ref_flat = _flat_nc(ref)

    order = [KEY_REF, KEY_LN_AFFINE, KEY_LN_AFFINE_ADAPTER, KEY_LN_ADAPTER]
    colors = {
        KEY_REF: "#1f77b4",
        KEY_LN_AFFINE: "#ff7f0e",
        KEY_LN_AFFINE_ADAPTER: "#2ca02c",
        KEY_LN_ADAPTER: "#d62728",
    }
    labels_short = {
        KEY_REF: "graspnet_ref",
        KEY_LN_AFFINE: "ln_affine",
        KEY_LN_AFFINE_ADAPTER: "ln_affine_adapter",
        KEY_LN_ADAPTER: "ln_adapter",
    }

    # ---------- 数值统计 ----------
    summary: Dict[str, Any] = {
        "tensor_note": (
            "reference: vpmodule forward_pre_hook 第二输入 seed_features（与 backbone 输出一致）。"
            "replacement: 对齐模块之后、进入 vpmodule 前的 seed_features。"
        ),
        "checkpoint_paths": paths,
        "modes": {k: ALIGN_MODES[k] for k in ALIGN_MODES},
        "per_feature": {},
    }

    ref_g = _global_stats(ref)
    ref_ch_std = _per_channel_std(ref)
    ref_ps_l2 = _per_seed_l2(ref)
    rep_keys = [KEY_LN_AFFINE, KEY_LN_AFFINE_ADAPTER, KEY_LN_ADAPTER]

    w1_scores: Dict[str, float] = {}
    pca_dists: Dict[str, float] = {}
    hm_diag: Dict[str, float] = {}

    for name in order:
        x = feats[name]
        g = _global_stats(x)
        ch_m = _per_channel_mean(x)
        ch_s = _per_channel_std(x)
        ps = _per_seed_l2(x)
        row = {
            "global": g,
            "per_channel_mean": ch_m.tolist(),
            "per_channel_std": ch_s.tolist(),
            "per_seed_l2_mean": float(np.mean(ps)),
            "per_seed_l2_std": float(np.std(ps)),
        }
        if name != KEY_REF:
            ratio_std = float(g["std"] / (ref_g["std"] + 1e-12))
            ratio_l2 = float(g["l2"] / (ref_g["l2"] + 1e-12))
            row["std_ratio_vs_ref"] = ratio_std
            row["l2_ratio_vs_ref"] = ratio_l2
            row["per_channel_std_mean_ratio_vs_ref"] = float(np.mean(ch_s) / (np.mean(ref_ch_std) + 1e-12))
            xf = _flat_nc(x)
            w1_scores[name] = _wasserstein_1d(ref_flat, xf)
            row["wasserstein_1d_vs_ref"] = w1_scores[name]
            # heatmap diagonal mean
            hm = _channel_cosine_matrix(ref_flat, xf)
            dmean = float(np.mean(np.diag(hm)))
            hm_diag[name] = dmean
            row["channel_cosine_diag_mean_vs_ref"] = dmean
        summary["per_feature"][name] = row

    # PCA / t-SNE / UMAP（子采样避免内存爆炸；seed 按名称固定，避免 hash() 跨进程不稳定）
    all_names = order
    flats = {
        n: _subsample_rows(
            _flat_nc(feats[n]),
            args.max_embed_rows,
            args.seed + sum(ord(c) for c in n) % 100000,
        )
        for n in all_names
    }
    X_list = [flats[n] for n in all_names]
    y_labels = np.concatenate([np.full(len(flats[n]), i, dtype=np.int32) for i, n in enumerate(all_names)])

    X_all = np.vstack(X_list)
    X_all = X_all - np.mean(X_all, axis=0, keepdims=True)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=2, random_state=args.seed)
    Z_pca = pca.fit_transform(X_all)
    pca_by_name: Dict[str, np.ndarray] = {}
    off = 0
    for i, n in enumerate(all_names):
        le = len(flats[n])
        pca_by_name[n] = Z_pca[off : off + le]
        off += le

    c0_pca = np.mean(pca_by_name[KEY_REF], axis=0)
    for n in rep_keys:
        # 每个 replacement 样本到 reference 云在 PCA 空间质心的平均 L2 距离
        pca_dists[n] = float(np.mean(np.linalg.norm(pca_by_name[n] - c0_pca, axis=1)))

    plt = _ensure_matplotlib()
    _plot_embeddings(
        plt,
        pca_by_name,
        "PCA-2D (combined fit)",
        os.path.join(fig_dir, "pca2d.png"),
        colors,
    )

    max_tsne = min(8000, len(X_all))
    X_tsne = X_all[:max_tsne]
    y_tsne = y_labels[:max_tsne]
    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto")
    Z_ts = tsne.fit_transform(X_tsne)
    tsne_by_name: Dict[str, np.ndarray] = {}
    for i, n in enumerate(all_names):
        mask = y_tsne == i
        tsne_by_name[n] = Z_ts[mask]

    _plot_embeddings(
        plt,
        tsne_by_name,
        "t-SNE-2D",
        os.path.join(fig_dir, "tsne2d.png"),
        colors,
    )

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        Z_um = reducer.fit_transform(X_tsne)
        labels_um = y_tsne[: Z_um.shape[0]]
        umap_by_name: Dict[str, np.ndarray] = {}
        for i, n in enumerate(all_names):
            umap_by_name[n] = Z_um[labels_um == i]
        _plot_embeddings(
            plt,
            umap_by_name,
            "UMAP-2D",
            os.path.join(fig_dir, "umap2d.png"),
            colors,
        )
    except Exception:
        with open(os.path.join(stat_dir, "umap_skipped.txt"), "w", encoding="utf-8") as f:
            f.write("umap-learn 未安装或运行失败，已跳过 umap2d.png\n")

    # Histograms
    plt.figure(figsize=(10, 5))
    for name in order:
        v = feats[name].ravel()
        plt.hist(v, bins=120, density=True, alpha=0.35, color=colors[name], label=labels_short[name])
    plt.xlabel("feature value")
    plt.ylabel("density")
    plt.title("hist_values (vpmodule input)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_values.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for name in order:
        v = np.abs(feats[name].ravel())
        plt.hist(v, bins=120, density=True, alpha=0.35, color=colors[name], label=labels_short[name])
    plt.xlabel("|feature|")
    plt.ylabel("density")
    plt.title("hist_abs_values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hist_abs_values.png"), dpi=160)
    plt.close()

    # per-channel std box + line
    plt.figure(figsize=(11, 5))
    plt.boxplot(
        [_per_channel_std(feats[n]) for n in order],
        labels=[labels_short[n] for n in order],
    )
    plt.ylabel("per-channel std")
    plt.title("box_per_channel_std")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "box_per_channel_std.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    ch = ref.shape[1]
    x_axis = np.arange(ch)
    for name in order:
        plt.plot(x_axis, _per_channel_std(feats[name]), color=colors[name], alpha=0.8, linewidth=1.0, label=labels_short[name])
    plt.xlabel("channel")
    plt.ylabel("std")
    plt.title("line_per_channel_std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "line_per_channel_std.png"), dpi=160)
    plt.close()

    # per-seed L2
    plt.figure(figsize=(11, 5))
    plt.boxplot(
        [_per_seed_l2(feats[n]) for n in order],
        labels=[labels_short[n] for n in order],
    )
    plt.ylabel("per-seed L2 norm")
    plt.title("box_per_seed_l2")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "box_per_seed_l2.png"), dpi=160)
    plt.close()

    # violin
    plt.figure(figsize=(11, 5))
    parts = plt.violinplot(
        [_per_seed_l2(feats[n]) for n in order],
        positions=range(len(order)),
        showmeans=True,
        showmedians=True,
    )
    plt.xticks(range(len(order)), [labels_short[n] for n in order], rotation=15)
    plt.ylabel("per-seed L2")
    plt.title("violin_per_seed_l2")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "violin_per_seed_l2.png"), dpi=160)
    plt.close()

    # heatmaps
    for name in [KEY_LN_AFFINE, KEY_LN_AFFINE_ADAPTER, KEY_LN_ADAPTER]:
        xf = _flat_nc(feats[name])
        hm = _channel_cosine_matrix(ref_flat, xf)
        plt.figure(figsize=(8, 7))
        plt.imshow(hm, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        plt.colorbar()
        plt.title("channel cosine vs graspnet ref: %s" % labels_short[name])
        plt.xlabel("replacement channel")
        plt.ylabel("graspnet ref channel")
        fn = "heatmap_ch_cosine_vs_graspnet_%s.png" % labels_short[name]
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, fn), dpi=160)
        plt.close()

        phm = _channel_pearson_matrix(ref_flat, xf)
        plt.figure(figsize=(8, 7))
        plt.imshow(phm, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        plt.colorbar()
        plt.title("channel Pearson vs graspnet ref: %s" % labels_short[name])
        plt.xlabel("replacement channel")
        plt.ylabel("graspnet ref channel")
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, "heatmap_ch_pearson_vs_graspnet_%s.png" % labels_short[name]),
            dpi=160,
        )
        plt.close()

    # JSON + CSV
    with open(os.path.join(stat_dir, "features_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(stat_dir, "per_channel_std.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel_index"] + [labels_short[n] for n in order])
        for c in range(ref.shape[1]):
            w.writerow([c] + [float(_per_channel_std(feats[n])[c]) for n in order])

    glob_csv = os.path.join(stat_dir, "global_stats_table.csv")
    with open(glob_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "mean",
                "std",
                "min",
                "max",
                "abs_mean",
                "l2_global",
                "std_ratio_vs_ref",
                "l2_ratio_vs_ref",
            ]
        )
        for n in order:
            g = summary["per_feature"][n]["global"]
            row = [
                labels_short[n],
                g["mean"],
                g["std"],
                g["min"],
                g["max"],
                g["abs_mean"],
                g["l2"],
            ]
            if n == KEY_REF:
                row.extend(["", ""])
            else:
                row.append(summary["per_feature"][n].get("std_ratio_vs_ref", ""))
                row.append(summary["per_feature"][n].get("l2_ratio_vs_ref", ""))
            w.writerow(row)

    # best picks
    def _best_min(d: Dict[str, float], keys: List[str]) -> str:
        best = None
        bv = float("inf")
        for k in keys:
            if k not in d:
                continue
            v = d[k]
            if math.isnan(v):
                continue
            if v < bv:
                bv = v
                best = k
        return best or "n/a"

    def _best_max(d: Dict[str, float], keys: List[str]) -> str:
        best = None
        bv = float("-inf")
        for k in keys:
            if k not in d:
                continue
            v = d[k]
            if math.isnan(v):
                continue
            if v > bv:
                bv = v
                best = k
        return best or "n/a"

    std_ratios = {k: summary["per_feature"][k]["std_ratio_vs_ref"] for k in rep_keys}
    best_scale = _best_min({k: abs(1.0 - std_ratios[k]) for k in rep_keys}, rep_keys)
    if best_scale != "n/a":
        best_scale = "%s (|std_ratio-1| 最小, ratio=%.4f)" % (labels_short[best_scale], std_ratios[best_scale])

    best_w1_key = _best_min(w1_scores, rep_keys)
    best_w1 = (
        "%s (W1=%.6f)" % (labels_short[best_w1_key], w1_scores[best_w1_key]) if best_w1_key != "n/a" else "n/a"
    )

    best_pca_key = _best_min(pca_dists, rep_keys)
    best_pca = (
        "%s (dist=%.6f)" % (labels_short[best_pca_key], pca_dists[best_pca_key])
        if best_pca_key != "n/a"
        else "n/a"
    )

    best_hm_key = _best_max(hm_diag, rep_keys)
    best_hm = (
        "%s (diag_mean=%.4f)" % (labels_short[best_hm_key], hm_diag[best_hm_key])
        if best_hm_key != "n/a"
        else "n/a"
    )

    def _score_win(a: str, b: str) -> Tuple[int, int]:
        """在 |std_ratio-1|、W1、PCA 距离上越小越好；cosine 对角线越大越好。"""
        sa = abs(1.0 - std_ratios[a])
        sb = abs(1.0 - std_ratios[b])
        wa = w1_scores[a]
        wb = w1_scores[b]
        pa = pca_dists[a]
        pb = pca_dists[b]
        ha = hm_diag[a]
        hb = hm_diag[b]
        ca = sum(
            [
                sa < sb,
                wa < wb,
                pa < pb,
                ha > hb,
            ]
        )
        cb = sum(
            [
                sb < sa,
                wb < wa,
                pb < pa,
                hb > ha,
            ]
        )
        return ca, cb

    ca1, cb1 = _score_win(KEY_LN_AFFINE, KEY_LN_ADAPTER)
    ca2, cb2 = _score_win(KEY_LN_AFFINE, KEY_LN_AFFINE_ADAPTER)

    ln_vs_lnadapter = (
        "在 |std_ratio-1|、W1、PCA 距离（越小越好）与通道余弦对角线（越大越好）四项上，"
        "layernorm_affine 胜 %d 项，layernorm_adapter 胜 %d 项。"
        "若 affine 明显占优，更支持「LayerNorm+affine 对对齐关键」；若接近则需结合图看。" % (ca1, cb1)
    )
    affine_vs_affine_adapter = (
        "同上四项：layernorm_affine 胜 %d 项，layernorm_affine_adapter 胜 %d 项。"
        "若 adapter 版本在多项上更好，支持「adapter 有额外帮助」。" % (ca2, cb2)
    )

    dev_std = max(abs(1.0 - v) for v in std_ratios.values())
    dev_w1 = max(w1_scores.values())
    dev_pca = max(pca_dists.values())
    gap_note = (
        "当前三种 replacement 相对 reference：max|std_ratio-1|=%.4f，max W1=%.6f，max PCA 均距=%.6f。"
        "若这些量仍显著大于 0，则通常认为 **仍存在明显 gap**（具体阈值请结合业务敏感度）。"
        % (dev_std, dev_w1, dev_pca)
    )

    _write_report(
        os.path.join(out_dir, "REPORT.md"),
        {
            "std_ratio_vs_ref": std_ratios,
            "wasserstein_1d_vs_ref": w1_scores,
            "pca_mean_distance_to_ref_centroid": pca_dists,
            "channel_cosine_diag_mean": hm_diag,
        },
        best_scale,
        best_w1,
        best_pca,
        best_hm,
        ln_vs_lnadapter,
        affine_vs_affine_adapter,
        gap_note,
    )

    print("Done. Output:", out_dir)
    print("Figures:", fig_dir)
    print("Stats:", stat_dir)
    print("Report:", os.path.join(out_dir, "REPORT.md"))


if __name__ == "__main__":
    main()
