# -*- coding: utf-8 -*-
"""点云标量着色 + RGB/Depth/Mask 预览（固定视角）。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _ensure_plt():
    if plt is None:
        raise RuntimeError("需要 matplotlib")
    return plt


def per_point_l2_diff_to_reference(
    feats_np: Dict[str, np.ndarray],
    ref_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    逐点 ``||f_model - f_ref||_2``，与 ``feats_np`` 中参考模型对齐（同 N、同特征维）。
    参考模型自身为全零向量，便于与其它面板同图展示。
    """
    meta: Dict[str, Any] = {"reference": ref_name, "skipped": {}, "error": None}
    ref = feats_np.get(ref_name)
    if ref is None:
        meta["error"] = f"参考模型 {ref_name!r} 不在 feats_np 中"
        return {}, meta
    ref = np.asarray(ref, dtype=np.float64)
    if ref.ndim != 2:
        meta["error"] = "参考特征须为 (N, D)"
        return {}, meta
    n, d = ref.shape
    out: Dict[str, np.ndarray] = {}
    for name, f in feats_np.items():
        f = np.asarray(f, dtype=np.float64)
        if f.ndim != 2 or f.shape[0] != n or f.shape[1] != d:
            meta["skipped"][name] = f"shape {tuple(f.shape)} vs ref {(n, d)}"
            continue
        if name == ref_name:
            out[name] = np.zeros(n, dtype=np.float64)
        else:
            out[name] = np.linalg.norm(f - ref, axis=1)
    return out, meta


def relative_l2_diff_maps(
    feats_np: Dict[str, np.ndarray],
    ref_name: str,
    abs_diff_maps: Dict[str, np.ndarray],
    *,
    epsilon: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    逐点相对差分：``||f_m - f_raw|| / (||f_raw|| + epsilon)``（参考模型为全零便于对齐展示）。
    """
    ref = feats_np.get(ref_name)
    if ref is None:
        return {}
    ref = np.asarray(ref, dtype=np.float64)
    if ref.ndim != 2:
        return {}
    ref_row_norm = np.linalg.norm(ref, axis=1)
    denom = ref_row_norm + float(epsilon)
    out: Dict[str, np.ndarray] = {}
    for name, abs_d in abs_diff_maps.items():
        a = np.asarray(abs_d, dtype=np.float64).ravel()
        if name == ref_name:
            out[name] = np.zeros_like(a, dtype=np.float64)
        else:
            out[name] = a / denom
    return out


def top_fraction_binary_masks(
    diff_maps: Dict[str, np.ndarray],
    ref_name: str,
    *,
    top_fraction: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    对每个非参考模型，按逐点 diff 取最高的 ``ceil(N * top_fraction)`` 个点标为 1，其余为 0。
    参考模型不生成 mask（不写入字典）。
    """
    out: Dict[str, np.ndarray] = {}
    frac = float(top_fraction)
    if frac <= 0.0 or frac > 1.0:
        return out
    for name, sc in diff_maps.items():
        if name == ref_name:
            continue
        s = np.asarray(sc, dtype=np.float64).ravel()
        n = s.size
        if n == 0:
            continue
        k = max(1, int(np.ceil(n * frac)))
        mask = np.zeros(n, dtype=np.float64)
        order = np.argsort(-s, kind="stable")
        mask[order[:k]] = 1.0
        out[name] = mask
    return out


# 表示空间族：仅族内与各自 RAW 参考比较 diff（语义可解释）。
DIFF_FAMILY_SPECS: List[Dict[str, Any]] = [
    {
        "diff_family": "vggt",
        "reference_model": "vggt_raw",
        "members": [
            "vggt_raw",
            "vggt_progressive_alpha05",
            "vggt_distill",
            "vggt_fusion_progressive_alpha05",
            "vggt_prog_enc_lora",
        ],
    },
    {
        "diff_family": "lift3d_clip",
        "reference_model": "lift3d_clip_raw",
        "members": [
            "lift3d_clip_raw",
            "lift3d_clip_progressive_alpha05",
        ],
    },
    {
        "diff_family": "lift3d_dino",
        "reference_model": "lift3d_dinov2_raw",
        "members": [
            "lift3d_dinov2_raw",
            "lift3d_dinov2_progressive_alpha05",
        ],
    },
    {
        "diff_family": "graspnet",
        "reference_model": None,
        "members": ["graspnet_backbone"],
    },
]


def model_diff_family(model_name: str) -> Optional[str]:
    """返回 diff_family 键（vggt / lift3d_clip / lift3d_dino / graspnet），未知模型为 None。"""
    for spec in DIFF_FAMILY_SPECS:
        if model_name in spec["members"]:
            return str(spec["diff_family"])
    return None


def _excluded_models_for_family(
    feats_np: Dict[str, np.ndarray],
    current_members: Any,
) -> Dict[str, Dict[str, str]]:
    """
    当前族 diff 面板不包含的其它 feats 键：标注 skipped_reason（cross_family / unknown_family）。
    """
    mem = frozenset(current_members)
    out: Dict[str, Dict[str, str]] = {}
    for name in feats_np:
        if name in mem:
            continue
        fam = model_diff_family(name)
        if fam is None:
            out[name] = {"skipped_reason": "unknown_family"}
        else:
            out[name] = {"skipped_reason": "cross_family"}
    return out


def representation_aware_family_diffs(
    feats_np: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Any]]], Dict[str, Any]]:
    """
    按表示族分组，仅在族内对各自 RAW 参考计算 ``||f_m - f_ref||_2``。

    返回:
      - family_results: diff_family -> (diff_maps, meta)
      - report: graspnet 说明、全局模型族标签等

    GraspNet 仅单模型且无 raw 参考：不生成 diff_maps，meta 中说明原因。
    """
    report: Dict[str, Any] = {
        "model_to_diff_family": {m: model_diff_family(m) for m in feats_np},
    }

    family_results: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = {}

    for spec in DIFF_FAMILY_SPECS:
        df = str(spec["diff_family"])
        ref = spec["reference_model"]
        members = list(spec["members"])

        excluded = _excluded_models_for_family(feats_np, members)

        if ref is None:
            present_g = [m for m in members if m in feats_np]
            family_results[df] = (
                {},
                {
                    "diff_family": df,
                    "reference_model": None,
                    "error": "no_intra_family_raw_reference",
                    "skipped": {},
                    "excluded_from_this_family": excluded,
                    "members_present": present_g,
                    "note": "graspnet: single backbone entry; no representation-aware diff vs raw",
                },
            )
            continue

        present = [m for m in members if m in feats_np]
        base_meta: Dict[str, Any] = {
            "diff_family": df,
            "reference_model": ref,
            "skipped": {},
            "error": None,
            "excluded_from_this_family": excluded,
        }

        if ref not in feats_np:
            base_meta["error"] = f"参考模型 {ref!r} 不在 feats_np 中"
            family_results[df] = ({}, base_meta)
            continue

        if len(present) < 2:
            base_meta["error"] = "only_reference_or_empty_family"
            base_meta["members_present"] = present
            family_results[df] = ({}, base_meta)
            continue

        sub = {k: feats_np[k] for k in present}
        diff_maps, sub_meta = per_point_l2_diff_to_reference(sub, ref)
        base_meta["skipped"] = dict(sub_meta.get("skipped") or {})
        if sub_meta.get("error"):
            base_meta["error"] = sub_meta["error"]
            family_results[df] = (diff_maps, base_meta)
            continue

        family_results[df] = (diff_maps, base_meta)

    return family_results, report


def _binary_mask_colors(mask: np.ndarray) -> np.ndarray:
    """mask (N,) 0/1 -> RGBA 或 RGB for scatter (matplotlib accepts (N,4) rgba)"""
    m = np.asarray(mask, dtype=np.float64).ravel()
    # 背景灰、高亮琥珀
    r = 0.22 + 0.78 * m
    g = 0.22 + 0.55 * m
    b = 0.22 + 0.12 * m
    return np.stack([r, g, b], axis=1)


def normalize_scores(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).ravel()
    lo, hi = float(np.min(s)), float(np.max(s))
    if hi - lo < 1e-12:
        return np.zeros_like(s, dtype=np.float64)
    return (s - lo) / (hi - lo)


def scatter_pc_color(
    xyz: np.ndarray,
    color: np.ndarray,
    *,
    elev: float = 20.0,
    azim: float = -60.0,
    title: str = "",
    figsize: Tuple[float, float] = (3.5, 3.5),
):
    """xyz (N,3), color (N,) 0-1 for colormap"""
    pl = _ensure_plt()
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, cmap="turbo", s=2, alpha=0.9)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    pl.colorbar(p, ax=ax, shrink=0.5, fraction=0.08)
    ax.set_box_aspect((1, 1, 1))
    return fig, ax


def save_rgb_depth_mask_previews(
    dataset_root: str,
    scene_id: int,
    ann_id: int,
    camera: str,
    out_dir: str,
) -> Dict[str, Optional[str]]:
    """写出 rgb_preview.png / depth_preview.png / mask_preview.png（若文件存在）。"""
    try:
        import cv2
    except ImportError:
        return {"rgb": None, "depth": None, "mask": None, "error": "no_cv2"}

    def _img_id():
        i = ann_id * 4
        if camera == "realsense-d415":
            i += 1
        elif camera == "realsense-d435":
            i += 2
        elif camera == "azure-kinect":
            i += 3
        elif camera == "zivid":
            i += 4
        return i

    img_id = _img_id()
    scene = os.path.join(dataset_root, "scenes", f"{scene_id:06d}")
    paths = {
        "rgb": os.path.join(scene, "rgb", f"{img_id:06d}.png"),
        "depth": os.path.join(scene, "depth", f"{img_id:06d}.png"),
        "mask": os.path.join(scene, "label", f"{img_id:06d}.png"),
    }
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Optional[str]] = {}
    for k, p in paths.items():
        if os.path.isfile(p):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                out[k] = None
                continue
            if k == "rgb":
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            dst = os.path.join(out_dir, f"{k}_preview.png")
            if k == "depth":
                d = im.astype(np.float32)
                d = (d - d.min()) / (d.max() - d.min() + 1e-6)
                d = (d * 255).astype(np.uint8)
                cv2.imwrite(dst, d)
            elif k == "mask":
                if im.ndim == 3:
                    im = im[:, :, 0]
                cv2.imwrite(dst, (im % 256).astype(np.uint8))
            else:
                cv2.imwrite(dst, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            out[k] = dst
        else:
            out[k] = None
    return out


def save_comparison_grid(
    panels: Sequence[Tuple],
    out_path: str,
    *,
    elev: float = 20.0,
    azim: float = -60.0,
    ncols: int = 4,
    figsize_per: Tuple[float, float] = (3.2, 3.0),
):
    """
    panels 每项为以下之一：
    - ("pointcloud", xyz, score, title)
    - ("pointcloud_binary", xyz, mask, title)  mask: (N,) 0/1 二值高亮
    - ("image", img, title)  img: HxW 或 HxWx3 float32/float64 uint8
    """
    pl = _ensure_plt()
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig_w = figsize_per[0] * min(ncols, n)
    fig_h = figsize_per[1] * nrows
    fig = pl.figure(figsize=(fig_w, fig_h))
    for i, pan in enumerate(panels):
        kind = pan[0]
        if kind == "pointcloud":
            _, xyz, sc, title = pan
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            c = normalize_scores(np.asarray(sc).ravel())
            p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, cmap="turbo", s=1.5, alpha=0.85)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=8)
            pl.colorbar(p, ax=ax, shrink=0.55, fraction=0.06)
            ax.set_box_aspect((1, 1, 1))
        elif kind == "pointcloud_binary":
            _, xyz, mask, title = pan
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            m = np.asarray(mask, dtype=np.float64).ravel()
            rgb = _binary_mask_colors(m)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=rgb,
                s=1.8,
                alpha=0.9,
            )
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=8)
            ax.set_box_aspect((1, 1, 1))
        elif kind == "image":
            _, img, title = pan
            ax = fig.add_subplot(nrows, ncols, i + 1)
            im = np.asarray(img)
            if im.ndim == 2:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(np.clip(im, 0, 1) if im.dtype != np.uint8 else im)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        else:
            raise ValueError(pan)
    pl.tight_layout()
    fig.savefig(out_path, dpi=160)
    pl.close(fig)
