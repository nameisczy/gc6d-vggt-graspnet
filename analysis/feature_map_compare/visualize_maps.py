# -*- coding: utf-8 -*-
"""点云标量着色 + RGB/Depth/Mask 预览（固定视角）。"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

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
