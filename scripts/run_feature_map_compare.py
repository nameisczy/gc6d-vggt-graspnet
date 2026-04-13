#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单例 GC6D 样本：多模型特征图（L2 / PCA-1）在统一 pc_common 上的对比。

示例：
  cd ~/gc6d_grasp_pipeline
  python scripts/run_feature_map_compare.py \\
    --data_dir /path/to/offline_unified \\
    --dataset_root /path/to/GraspClutter6D \\
    --checkpoint_manifest /path/to/manifest.json \\
    --graspnet_ckpt /path/to/checkpoint-rs.tar

manifest 支持「字符串 ckpt 路径」或 ``{"type":"pretrained"|"checkpoint",...}``；
示例见 analysis/feature_map_compare/checkpoint_manifest.example.json

特征差分：默认按表示族内与各自 RAW 比较，输出绝对 ``compare_diff_family_<族>_l2.png``、相对 ``*_rel.png``、
top 比例二值高亮 ``*_mask_top*pct*_l2.png`` 与 ``feature_diff_by_family.json``（含每模型
``topk_diff_grasp_overlap``：含 ``overlap``、``overlap_random_baseline``（全点落入 grasp 邻域比例）、
``improvement_vs_random = overlap / baseline``，见 ``--grasp_dist_threshold`` 与 ``--diff_top_fraction``）；
``--enable_cross_model_diff`` 可额外生成全局单参考 legacy 版图（含 graspnet，非跨族可比）。

Grasp 对齐：优先从 ``dataset_root/scenes/<scene>/label/<ann>.npz`` 读全量 ``grasps`` (N×17)，否则回退离线 npz 的
``gt_grasp_group``；输出 ``compare_grasp_aligned_featnorm.png`` 与 ``grasp_aligned_featnorm_meta.json``。

``--global_norm_for_featvis``：L2 与 grasp 对齐图中特征范数色标在已加载模型间共用 min–max（子图标题带 ``[global ||·||]``）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.feature_map_compare.feature_extract import (
    extract_by_model_name,
    load_model_from_manifest,
    normalize_checkpoint_manifest_entry,
)
from analysis.feature_map_compare.point_cloud_common import build_pc_common
from analysis.feature_map_compare.scene_selection import select_representative_example
from analysis.feature_map_compare.summary_stats import (
    concentration_metrics_per_model,
    gt_nearest_pc_common_mapping,
    l2_norm_per_point,
    pairwise_pearson_feature_norms,
    pca_first_component,
    per_point_min_dist_to_gt_translations,
    resolve_gt_grasps_17d,
    save_summary_json,
    summarize_models,
    topk_diff_grasp_region_overlap,
    topk_distance_to_nearest_gt_translation,
)
from analysis.feature_map_compare.visualize_maps import (
    global_feat_norm_range_from_models,
    per_point_l2_diff_to_reference,
    relative_l2_diff_maps,
    representation_aware_family_diffs,
    save_comparison_grid,
    top_fraction_binary_masks,
    save_rgb_depth_mask_previews,
)
from utils.batch_images import load_images_batch


MODEL_PANEL_ORDER = [
    "graspnet_backbone",
    "vggt_raw",
    "vggt_progressive_alpha05",
    "vggt_distill",
    "vggt_fusion_progressive_alpha05",
    "vggt_prog_enc_lora",
    "lift3d_clip_raw",
    "lift3d_clip_progressive_alpha05",
    "lift3d_dinov2_raw",
    "lift3d_dinov2_progressive_alpha05",
]

MODEL_DISPLAY_SHORT = {
    "graspnet_backbone": "GraspNet",
    "vggt_raw": "VGGT raw",
    "vggt_progressive_alpha05": "VGGT prog.",
    "vggt_distill": "VGGT distill",
    "vggt_fusion_progressive_alpha05": "VGGT fusion",
    "vggt_prog_enc_lora": "VGGT enc-LoRA",
    "lift3d_clip_raw": "L3D-CLIP",
    "lift3d_clip_progressive_alpha05": "L3D-CLIP prog.",
    "lift3d_dinov2_raw": "L3D-DINO",
    "lift3d_dinov2_progressive_alpha05": "L3D-DINO prog.",
}

DIFF_FAMILY_SUPTITLE = {
    "vggt": "Feature Difference Analysis (VGGT Family)",
    "lift3d_clip": "Feature Difference Analysis (Lift3D CLIP Family)",
    "lift3d_dino": "Feature Difference Analysis (Lift3D DINO Family)",
}

SUBGROUPS = {
    "group_A": ["graspnet_backbone", "vggt_raw", "vggt_progressive_alpha05", "vggt_distill"],
    "group_B": ["lift3d_clip_raw", "lift3d_clip_progressive_alpha05"],
    "group_C": ["lift3d_dinov2_raw", "lift3d_dinov2_progressive_alpha05"],
    "group_D": ["vggt_raw", "vggt_fusion_progressive_alpha05", "graspnet_backbone"],
}


def _short_model_title(model_name: str) -> str:
    return MODEL_DISPLAY_SHORT.get(model_name, model_name)


def _ann_id_to_img_id(ann_id: int, camera: str) -> int:
    """与 GraspClutter6D / scene_selection 一致。"""
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


def _rgb_path_from_dataset_root(
    dataset_root: Optional[str], scene_id: int, ann_id: int, camera: str
) -> Optional[str]:
    """索引里无 rgb_path 时，从原始 GC6D 目录拼 scenes/<scene>/rgb/<img>.png。"""
    if not dataset_root or not os.path.isdir(dataset_root):
        return None
    iid = _ann_id_to_img_id(ann_id, camera)
    rp = os.path.join(dataset_root, "scenes", f"{scene_id:06d}", "rgb", f"{iid:06d}.png")
    return rp if os.path.isfile(rp) else None


def _load_rgb_depth_for_figure(dataset_root: str, scene_id: int, ann_id: int, camera: str):
    try:
        import cv2
    except ImportError:
        return None, None, None

    iid = _ann_id_to_img_id(ann_id, camera)
    scene = os.path.join(dataset_root, "scenes", f"{scene_id:06d}")
    rp = os.path.join(scene, "rgb", f"{iid:06d}.png")
    dp = os.path.join(scene, "depth", f"{iid:06d}.png")
    rgb, depth = None, None
    if os.path.isfile(rp):
        bgr = cv2.imread(rp, cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    if os.path.isfile(dp):
        d = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if d is not None:
            depth = d.astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return rgb, depth, iid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, default=None, help="GraspClutter6D 根目录（mask/depth/相机）")
    ap.add_argument(
        "--split",
        type=str,
        default="val",
        help="与默认索引名 index_{split}_{camera}.jsonl 对应；测试集用 test（优先 index_test_*.jsonl，缺失时见 scene_selection 从 split_info+npz 回退）",
    )
    ap.add_argument("--camera", type=str, default="realsense-d415")
    ap.add_argument(
        "--index_jsonl",
        type=str,
        default=None,
        help="覆盖默认的 data_dir/index_{split}_{camera}.jsonl；可写文件名（相对 data_dir）或绝对路径",
    )
    ap.add_argument(
        "--checkpoint_manifest",
        type=str,
        required=True,
        help="JSON：模型名 -> ckpt 路径字符串，或 {\"type\":\"checkpoint\",\"path\":...} / {\"type\":\"pretrained\"}",
    )
    ap.add_argument(
        "--graspnet_ckpt",
        type=str,
        default=None,
        help="GraspNet 预训练（如 checkpoint-rs.tar）：alignment / 预训练 raw 模型加载骨干时用；若 manifest 未列 graspnet_backbone，也会自动用该路径跑骨干对比",
    )
    ap.add_argument("--graspnet_root", type=str, default=None)
    ap.add_argument("--lift3d_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="默认 analysis_outputs/feature_map_compare/<scene>_<ann>/")
    ap.add_argument("--pc_num_points", type=int, default=2048)
    ap.add_argument("--pc_seed", type=int, default=0)
    ap.add_argument("--max_candidates", type=int, default=500)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--topk_stats", type=int, default=64)
    ap.add_argument(
        "--grasp_dist_threshold",
        type=float,
        default=0.02,
        help="top-k 点到最近 GT 抓取平移的距离阈值（米级，与点云单位一致）",
    )
    ap.add_argument(
        "--vggt_feature_mode",
        type=str,
        default="seed256",
        choices=("dense768", "seed256"),
        help="VGGT：seed256=默认，与训练一致：backbone(LoRA)→中间768→NN→replacement_projector→"
        "(vggt_raw 含 align/scale；progressive/distill/fusion 等)→vpmodule 前 256 维任务适配特征；"
        "dense768=仅全图 pt_mlp768（无 projector/混合，纯几何对照）",
    )
    ap.add_argument(
        "--diff_ref_model",
        type=str,
        default="vggt_raw",
        help="仅当启用 --enable_cross_model_diff 时有效：全局单参考 diff=||f_m-f_ref||_2（含 graspnet，非表示一致）",
    )
    ap.add_argument(
        "--enable_cross_model_diff",
        action="store_true",
        help="额外输出「全局单参考」差分图（compare_diff_legacy_*.png），默认关闭；主输出为按族 RAW 参考的表示一致 diff",
    )
    ap.add_argument(
        "--diff_rel_epsilon",
        type=float,
        default=1e-8,
        help="相对差分分母：||f-f_raw|| / (||f_raw||+epsilon)",
    )
    ap.add_argument(
        "--diff_top_fraction",
        type=float,
        default=0.1,
        help="按逐点绝对 L2 差分取最高的 ceil(N*frac) 个点做二值高亮（0–1，默认 0.1 即约 top 10%%）",
    )
    ap.add_argument(
        "--global_norm_for_featvis",
        action="store_true",
        help="特征范数可视化（L2 主图、子组、grasp 对齐）在 MODEL_PANEL_ORDER 已加载模型上共用 min–max 色标，而非逐子图拉伸",
    )
    ap.add_argument("--vggt_dense_max_points", type=int, default=80000)
    args = ap.parse_args()

    device = torch.device(args.device)
    sel = select_representative_example(
        args.data_dir,
        split=args.split,
        camera=args.camera,
        dataset_root=args.dataset_root,
        max_candidates=args.max_candidates,
        index_filename=args.index_jsonl,
    )
    scene_id, ann_id = sel.scene_id, sel.ann_id
    out_dir = args.out_dir or os.path.join(
        ROOT, "analysis_outputs", "feature_map_compare", f"{scene_id:06d}_{ann_id:02d}"
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": args.data_dir,
                "dataset_root": args.dataset_root,
                "split": args.split,
                "camera": args.camera,
                "index_jsonl": args.index_jsonl,
                "checkpoint_manifest": args.checkpoint_manifest,
                "graspnet_ckpt": args.graspnet_ckpt,
                "pc_num_points": args.pc_num_points,
                "pc_seed": args.pc_seed,
                "vggt_feature_mode": args.vggt_feature_mode,
                "vggt_dense_max_points": args.vggt_dense_max_points,
                "grasp_dist_threshold": args.grasp_dist_threshold,
                "diff_ref_model": args.diff_ref_model,
                "enable_cross_model_diff": args.enable_cross_model_diff,
                "diff_rel_epsilon": args.diff_rel_epsilon,
                "diff_top_fraction": args.diff_top_fraction,
                "global_norm_for_featvis": args.global_norm_for_featvis,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(out_dir, "selected_example.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene_id": scene_id,
                "ann_id": ann_id,
                "camera": sel.camera,
                "npz_path": sel.npz_path,
                "rgb_path": sel.rgb_path,
                "score": sel.score,
                "summary": sel.summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    pc_common, pc_meta = build_pc_common(
        sel.npz_path,
        dataset_root=args.dataset_root,
        scene_id=scene_id,
        ann_id=ann_id,
        camera=args.camera,
        n_points=args.pc_num_points,
        seed=args.pc_seed,
    )
    np.save(os.path.join(out_dir, "pc_common.npy"), pc_common)
    with open(os.path.join(out_dir, "pc_common_meta.json"), "w", encoding="utf-8") as f:
        json.dump(pc_meta, f, indent=2, ensure_ascii=False)

    if args.dataset_root and os.path.isdir(args.dataset_root):
        save_rgb_depth_mask_previews(args.dataset_root, scene_id, ann_id, sel.camera, out_dir)

    data = np.load(sel.npz_path, allow_pickle=True)
    point_cloud = torch.from_numpy(data["point_cloud"].astype(np.float32)).unsqueeze(0).to(device)
    meta = {
        "sceneId": int(data["sceneId"]),
        "annId": int(data["annId"]),
        "camera": sel.camera,
        "sample_id": None,
        "rgb_path": sel.rgb_path,
    }
    if not (meta.get("rgb_path") or "").strip():
        fb_rgb = _rgb_path_from_dataset_root(args.dataset_root, scene_id, ann_id, sel.camera)
        if fb_rgb:
            meta["rgb_path"] = fb_rgb
    images = None
    if (meta.get("rgb_path") or "").strip():
        images = load_images_batch([meta], args.data_dir, device)

    pc_t = torch.from_numpy(pc_common.astype(np.float32)).to(device)

    manifest_path = os.path.abspath(os.path.expanduser(args.checkpoint_manifest))
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"checkpoint_manifest 不是有效文件: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = raw.lstrip("\ufeff").strip()
    if not raw:
        raise ValueError(
            f"checkpoint_manifest 为空或仅空白，无法解析 JSON: {manifest_path}"
        )
    try:
        manifest_raw = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"checkpoint_manifest 不是合法 JSON（请检查逗号/引号/是否存盘为空）: {manifest_path}\n"
            f"  {e}"
        ) from e
    if not isinstance(manifest_raw, dict):
        raise ValueError(f"checkpoint_manifest 顶层必须是 JSON 对象 {{...}}，当前为 {type(manifest_raw).__name__}")

    manifest: Dict[str, Dict[str, Any]] = {}
    for k, v in manifest_raw.items():
        try:
            manifest[k] = normalize_checkpoint_manifest_entry(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"checkpoint_manifest 中键 {k!r} 格式无效: {e}") from e

    # manifest 未写 graspnet_backbone 时：用 --graspnet_ckpt（通常为 checkpoint-rs.tar）跑纯 GraspNet 骨干对比
    if "graspnet_backbone" not in manifest and args.graspnet_ckpt:
        gcp = os.path.abspath(os.path.expanduser(args.graspnet_ckpt))
        if os.path.isfile(gcp):
            manifest["graspnet_backbone"] = normalize_checkpoint_manifest_entry(gcp)
            print(f"[manifest] graspnet_backbone 未在 JSON 中声明，已使用 --graspnet_ckpt: {gcp}")

    feats_np: Dict[str, np.ndarray] = {}
    feats_l2: Dict[str, np.ndarray] = {}
    feats_pca: Dict[str, np.ndarray] = {}
    logs: List[Dict[str, Any]] = []

    def run_one(name: str, entry: Optional[Dict[str, Any]]) -> None:
        if entry is None:
            print(f"[WARN] 跳过 {name}：manifest 中无此项")
            logs.append(
                {
                    "model": name,
                    "load_type": None,
                    "source": None,
                    "checkpoint": None,
                    "status": "missing_manifest",
                }
            )
            return
        if entry.get("type") == "checkpoint":
            ckpt = entry.get("path")
            if not ckpt or not os.path.isfile(ckpt):
                print(f"[WARN] 跳过 {name}：checkpoint 文件不存在: {ckpt}")
                logs.append(
                    {
                        "model": name,
                        "load_type": "checkpoint",
                        "source": ckpt,
                        "checkpoint": ckpt,
                        "status": "missing",
                    }
                )
                return
        try:
            model, load_info = load_model_from_manifest(
                entry,
                name,
                device=device,
                graspnet_ckpt=args.graspnet_ckpt,
                graspnet_root=args.graspnet_root,
                lift3d_root=args.lift3d_root,
            )
        except Exception as e:
            print(f"[WARN] 加载 {name} 失败: {e}")
            logs.append(
                {
                    "model": name,
                    "load_type": entry.get("type"),
                    "source": entry.get("path") if entry.get("type") == "checkpoint" else None,
                    "checkpoint": entry.get("path") if entry.get("type") == "checkpoint" else None,
                    "status": "load_error",
                    "error": str(e),
                }
            )
            return
        lt = load_info.get("load_type")
        src = load_info.get("source")
        ckpt_path = load_info.get("checkpoint_path")
        print(f"[load] {name}  load_type={lt}  source={src}")
        meta_load = {"load_type": lt, "source": src}
        need_img = bool(getattr(model, "requires_images", False))
        if need_img and images is None:
            print(f"[WARN] {name} 需要 RGB，但未能加载图像，跳过")
            logs.append(
                {
                    "model": name,
                    **meta_load,
                    "checkpoint": ckpt_path,
                    "status": "no_image",
                }
            )
            return
        try:
            out = extract_by_model_name(
                model,
                name,
                point_cloud,
                images if need_img else None,
                pc_t,
                vggt_mode=args.vggt_feature_mode,
                vggt_dense_max_points=args.vggt_dense_max_points,
            )
        except Exception as e:
            print(f"[WARN] 特征提取 {name} 失败: {e}")
            logs.append(
                {
                    "model": name,
                    **meta_load,
                    "checkpoint": ckpt_path,
                    "status": "extract_error",
                    "error": str(e),
                }
            )
            return
        feat = out.get("features")
        meta_ex = out.get("meta") or {}
        if feat is None:
            logs.append(
                {
                    "model": name,
                    **meta_load,
                    "checkpoint": ckpt_path,
                    "status": "no_features",
                    "meta": meta_ex,
                }
            )
            return
        fn = feat.detach().float().cpu().numpy()
        feats_np[name] = fn
        feats_l2[name] = l2_norm_per_point(fn)
        try:
            feats_pca[name] = pca_first_component(fn)
        except Exception:
            feats_pca[name] = np.zeros((fn.shape[0],), dtype=np.float64)
        logs.append(
            {
                "model": name,
                **meta_load,
                "checkpoint": ckpt_path,
                "status": "ok",
                "meta": meta_ex,
            }
        )

    for mname in MODEL_PANEL_ORDER:
        run_one(mname, manifest.get(mname))

    with open(os.path.join(out_dir, "extraction_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    gt_resolved, gt_resolve_meta = resolve_gt_grasps_17d(args.dataset_root, scene_id, ann_id, data)

    # 统计
    if feats_np:
        summ = summarize_models(feats_np, topk=args.topk_stats)
        summ["gt_grasp_resolution"] = gt_resolve_meta
        gt = gt_resolved

        # 1) Grasp alignment：GT 平移 ↔ pc_common 最近点；各模型 top-k ↔ 最近 GT 平移
        grasp_align: Dict[str, Any] = {}
        if gt is not None and gt.size > 0:
            m_gt = gt_nearest_pc_common_mapping(pc_common, gt)
            if m_gt:
                grasp_align["gt_translation_nearest_point_in_pc_common"] = m_gt
            per_topk: Dict[str, Any] = {}
            for mn in summ["topk_indices"].keys():
                idx = np.asarray(summ["topk_indices"][mn], dtype=np.int64)
                st = topk_distance_to_nearest_gt_translation(
                    pc_common,
                    gt,
                    idx,
                    dist_threshold=args.grasp_dist_threshold,
                )
                if st:
                    per_topk[mn] = st
            if per_topk:
                grasp_align["per_model_topk_to_nearest_gt_translation"] = per_topk
                # 与旧版字段名兼容（数值与 grasp_alignment 中一致）
                summ["gt_contact_distance_topk"] = {
                    mn: {
                        "mean_dist_to_nearest_gt_contact": v["mean_dist_topk_to_nearest_gt_translation"],
                        "frac_within_radius": v["frac_topk_within_dist_threshold"],
                        "contact_radius": v["dist_threshold"],
                    }
                    for mn, v in per_topk.items()
                }
        summ["grasp_alignment_analysis"] = grasp_align if grasp_align else None

        # 2) 特征范数 Pearson 相关（逐点，长度 N）
        summ["feature_norm_pearson"] = pairwise_pearson_feature_norms(feats_l2)

        # 3) 集中度：熵 + top-k 能量占比
        summ["concentration"] = {
            "per_model": concentration_metrics_per_model(feats_l2, topk=args.topk_stats),
            "top_k_used_for_energy_ratio": int(args.topk_stats),
            "entropy_definition": "shannon_entropy_natural_log_of_normalized_nonnegative_scores",
        }

        summ["metrics_notes"] = {
            "feature_norm_pearson": "Pearson r between per-point L2 feature norms (same N as pc_common).",
            "concentration_entropy": "Entropy of p_i = norm_i / sum(norm), over all points.",
            "concentration_topk_energy": "sum of top-k norms / sum of all norms (same k as top_k_stats).",
        }

        save_summary_json(os.path.join(out_dir, "summary_stats.json"), summ)

        pear = summ.get("feature_norm_pearson") or {}
        plist = pear.get("pairwise_list") or []
        if plist:
            corr_csv = os.path.join(out_dir, "feature_norm_pearson_pairwise.csv")
            with open(corr_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["model_a", "model_b", "pearson_r"])
                w.writeheader()
                w.writerows(plist)

    # 可视化
    elev, azim = 20.0, -60.0
    rgb_img, depth_img, _ = (None, None, None)
    if args.dataset_root:
        rgb_img, depth_img, _ = _load_rgb_depth_for_figure(args.dataset_root, scene_id, ann_id, sel.camera)

    def panel_tuple(name: str, score_vec: np.ndarray) -> Tuple:
        return ("pointcloud", pc_common, score_vec, name)

    feat_norm_global_range = None
    if args.global_norm_for_featvis and feats_l2:
        feat_norm_global_range = global_feat_norm_range_from_models(feats_l2, MODEL_PANEL_ORDER)
    fn_scale_tag = " [global ||·||]" if feat_norm_global_range is not None else ""

    # 主图
    panels_main: List = []
    if rgb_img is not None:
        panels_main.append(("image", rgb_img, "RGB"))
    if depth_img is not None:
        panels_main.append(("image", depth_img, "Depth"))
    for m in MODEL_PANEL_ORDER:
        if m in feats_l2:
            panels_main.append(
                panel_tuple(
                    f"{_short_model_title(m)} (L2){fn_scale_tag}",
                    feats_l2[m],
                )
            )
    if len(panels_main) > 0:
        save_comparison_grid(
            panels_main,
            os.path.join(out_dir, "compare_all_models_l2.png"),
            elev=elev,
            azim=azim,
            ncols=4,
            feat_norm_global_range=feat_norm_global_range,
            suptitle="Per-Point Feature L2 Norms",
        )

    # Grasp 对齐：逐点距最近 GT 抓取平移；特征范数 turbo + 距离 < 阈值 红色覆盖
    grasp_dmin = (
        per_point_min_dist_to_gt_translations(pc_common, gt_resolved)
        if gt_resolved is not None
        else None
    )
    if grasp_dmin is not None and feats_l2:
        thr_ga = float(args.grasp_dist_threshold)
        cap_ga = (
            f"GT source: {gt_resolve_meta.get('source')}; "
            f"red: min dist to GT translation [:,13:16] < {thr_ga} m; "
            f"global_norm_for_featvis={feat_norm_global_range is not None}"
        )
        ga_panels: List = []
        if rgb_img is not None:
            ga_panels.append(("image", rgb_img, "RGB"))
        for m in MODEL_PANEL_ORDER:
            if m in feats_l2:
                ga_panels.append(
                    (
                        "pointcloud_grasp_align",
                        pc_common,
                        feats_l2[m],
                        grasp_dmin,
                        thr_ga,
                        _short_model_title(m),
                    )
                )
        if any(p[0] == "pointcloud_grasp_align" for p in ga_panels):
            save_comparison_grid(
                ga_panels,
                os.path.join(out_dir, "compare_grasp_aligned_featnorm.png"),
                elev=elev,
                azim=azim,
                layout="rgb_wide_first_row",
                feat_norm_global_range=feat_norm_global_range,
                suptitle="Grasp-Aligned Feature Norms",
                caption=cap_ga,
            )
        with open(os.path.join(out_dir, "grasp_aligned_featnorm_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "grasp_dist_threshold_m": thr_ga,
                    "num_gt_grasps": int(gt_resolved.shape[0]),
                    "gt_grasp_resolution": gt_resolve_meta,
                    "gt_translation_indices_in_grasp_row": "13:16",
                    "per_point_min_dist_to_nearest_gt_translation_m": {
                        "mean": float(np.mean(grasp_dmin)),
                        "std": float(np.std(grasp_dmin)),
                        "frac_points_below_threshold": float(np.mean(grasp_dmin < thr_ga)),
                    },
                    "feat_norm_color_scale": (
                        {
                            "mode": "global_minmax_across_models",
                            "min": feat_norm_global_range[0],
                            "max": feat_norm_global_range[1],
                        }
                        if feat_norm_global_range is not None
                        else {"mode": "per_panel_minmax"}
                    ),
                    "figure_caption": cap_ga,
                    "visualization": "turbo = per-point L2 feature norm; red = nearest GT translation distance < threshold",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    pca_panels: List = []
    if rgb_img is not None:
        pca_panels.append(("image", rgb_img, "RGB"))
    for m in MODEL_PANEL_ORDER:
        if m in feats_pca:
            pca_panels.append(panel_tuple(f"{_short_model_title(m)} (PCA-1)", feats_pca[m]))
    if pca_panels:
        save_comparison_grid(
            pca_panels,
            os.path.join(out_dir, "compare_all_models_pca1.png"),
            elev=elev,
            azim=azim,
            ncols=4,
            suptitle="Per-Point PCA-1 Feature Projection",
        )

    for gname, mlist in SUBGROUPS.items():
        sub: List = []
        if rgb_img is not None:
            sub.append(("image", rgb_img, "RGB"))
        for m in mlist:
            if m in feats_l2:
                sub.append(panel_tuple(_short_model_title(m) + fn_scale_tag, feats_l2[m]))
        if sub:
            save_comparison_grid(
                sub,
                os.path.join(out_dir, f"compare_{gname}_l2.png"),
                elev=elev,
                azim=azim,
                ncols=4,
                feat_norm_global_range=feat_norm_global_range,
                suptitle=f"Feature L2 Norms ({gname})",
            )

    # 表示一致：按族 RAW 参考的逐点 ||f_m - f_ref||_2（族间不混比）
    family_results, diff_report = representation_aware_family_diffs(feats_np)
    diff_report["representation_aware"] = True
    diff_by_family_json: Dict[str, Any] = {"families": {}, "global_report": diff_report}

    for df_key, (diff_maps, fam_meta) in family_results.items():
        entry: Dict[str, Any] = {
            "diff_family": fam_meta.get("diff_family"),
            "reference_model": fam_meta.get("reference_model"),
            "skipped": fam_meta.get("skipped", {}),
            "excluded_from_this_family": fam_meta.get("excluded_from_this_family", {}),
            "error": fam_meta.get("error"),
        }
        if fam_meta.get("note"):
            entry["note"] = fam_meta["note"]
        if fam_meta.get("members_present") is not None:
            entry["members_present"] = fam_meta["members_present"]

        if diff_maps and not fam_meta.get("error"):
            ref_name = str(fam_meta.get("reference_model") or "")
            rel_maps = relative_l2_diff_maps(
                feats_np, ref_name, diff_maps, epsilon=args.diff_rel_epsilon
            )
            mask_maps = top_fraction_binary_masks(
                diff_maps, ref_name, top_fraction=args.diff_top_fraction
            )
            entry["diff_rel_epsilon"] = float(args.diff_rel_epsilon)
            entry["diff_top_fraction"] = float(args.diff_top_fraction)
            if grasp_dmin is not None:
                entry["topk_diff_grasp_overlap_settings"] = {
                    "grasp_dist_threshold_m": float(args.grasp_dist_threshold),
                    "top_fraction_same_as_diff_mask": float(args.diff_top_fraction),
                    "definition": "(top-k by abs L2 diff to ref) ∩ (min_dist_to_gt_translation < τ) divided by k",
                    "improvement_vs_random": "overlap / mean_i 1[dist_to_gt(i)<τ]; baseline≈uniform random k points",
                }
            else:
                entry["topk_diff_grasp_overlap_note"] = "skipped: no GT grasps (dataset label or offline gt_grasp_group)"
            entry["per_model"] = {}
            for m, vec in diff_maps.items():
                v = np.asarray(vec, dtype=np.float64).ravel()
                pm: Dict[str, Any] = {
                    "mean_l2_diff": float(np.mean(v)),
                    "std_l2_diff": float(np.std(v)),
                    "max_l2_diff": float(np.max(v)),
                }
                if m in rel_maps:
                    rv = np.asarray(rel_maps[m], dtype=np.float64).ravel()
                    pm["mean_rel_diff"] = float(np.mean(rv))
                    pm["std_rel_diff"] = float(np.std(rv))
                    pm["max_rel_diff"] = float(np.max(rv))
                if m in mask_maps:
                    km = mask_maps[m]
                    pm["top_diff_fraction"] = float(args.diff_top_fraction)
                    pm["top_diff_point_count"] = int(np.sum(km > 0.5))
                if grasp_dmin is not None and m != ref_name:
                    tov = topk_diff_grasp_region_overlap(
                        v,
                        grasp_dmin,
                        top_fraction=args.diff_top_fraction,
                        grasp_dist_threshold=args.grasp_dist_threshold,
                    )
                    if tov is not None:
                        pm["topk_diff_grasp_overlap"] = tov
                entry["per_model"][m] = pm

            diff_cap_abs = (
                f"abs L2 to ref={ref_name}; family={df_key}; "
                f"rel ε={args.diff_rel_epsilon}; top_mask={args.diff_top_fraction}"
            )
            entry["figure_caption_abs_l2"] = diff_cap_abs
            entry["figure_caption_rel"] = (
                f"relative ||Δf||/(||f_ref||+ε); ref={ref_name}; family={df_key}; ε={args.diff_rel_epsilon}"
            )
            pct_label = f"{int(round(args.diff_top_fraction * 100))}pct"
            entry["figure_caption_mask"] = (
                f"top-{pct_label} by abs L2 vs ref={ref_name}; family={df_key}"
            )

            diff_panels: List = []
            if rgb_img is not None:
                diff_panels.append(("image", rgb_img, "RGB"))
            for m in MODEL_PANEL_ORDER:
                if m in diff_maps:
                    diff_panels.append(
                        (
                            "pointcloud",
                            pc_common,
                            diff_maps[m],
                            _short_model_title(m),
                        )
                    )
            if len(diff_panels) > 1:
                safe_df = df_key.replace(os.sep, "_").replace("/", "_")
                save_comparison_grid(
                    diff_panels,
                    os.path.join(out_dir, f"compare_diff_family_{safe_df}_l2.png"),
                    elev=elev,
                    azim=azim,
                    layout="rgb_wide_first_row",
                    ncols=4,
                    suptitle=DIFF_FAMILY_SUPTITLE.get(df_key, f"Feature Difference ({df_key})"),
                    caption=diff_cap_abs,
                )

            rel_panels: List = []
            if rgb_img is not None:
                rel_panels.append(("image", rgb_img, "RGB"))
            for m in MODEL_PANEL_ORDER:
                if m in rel_maps:
                    rel_panels.append(
                        (
                            "pointcloud",
                            pc_common,
                            rel_maps[m],
                            _short_model_title(m),
                        )
                    )
            if len(rel_panels) > 1:
                safe_df = df_key.replace(os.sep, "_").replace("/", "_")
                save_comparison_grid(
                    rel_panels,
                    os.path.join(out_dir, f"compare_diff_family_{safe_df}_rel.png"),
                    elev=elev,
                    azim=azim,
                    layout="rgb_wide_first_row",
                    ncols=4,
                    suptitle=DIFF_FAMILY_SUPTITLE.get(df_key, f"Feature Difference ({df_key})"),
                    caption=entry["figure_caption_rel"],
                )

            mask_panels: List = []
            if rgb_img is not None:
                mask_panels.append(("image", rgb_img, "RGB"))
            for m in MODEL_PANEL_ORDER:
                if m in mask_maps:
                    mask_panels.append(
                        (
                            "pointcloud_binary",
                            pc_common,
                            mask_maps[m],
                            _short_model_title(m),
                        )
                    )
            if len(mask_panels) > 1:
                safe_df = df_key.replace(os.sep, "_").replace("/", "_")
                save_comparison_grid(
                    mask_panels,
                    os.path.join(out_dir, f"compare_diff_family_{safe_df}_mask_top{pct_label}_l2.png"),
                    elev=elev,
                    azim=azim,
                    layout="rgb_wide_first_row",
                    ncols=4,
                    suptitle=DIFF_FAMILY_SUPTITLE.get(df_key, f"Feature Difference ({df_key})"),
                    caption=entry["figure_caption_mask"],
                )
        elif fam_meta.get("error") and df_key != "graspnet":
            print(f"[WARN] 特征差分（{df_key}）跳过: {fam_meta.get('error')}")

        diff_by_family_json["families"][df_key] = entry

    # 可选：全局单参考（含 graspnet），非表示一致，仅用于对照
    if args.enable_cross_model_diff:
        legacy_maps, legacy_meta = per_point_l2_diff_to_reference(feats_np, args.diff_ref_model)
        legacy_block: Dict[str, Any] = {
            "representation_aware": False,
            "reference_model": args.diff_ref_model,
            "skipped": legacy_meta.get("skipped", {}),
            "error": legacy_meta.get("error"),
            "note": "legacy single-reference diff; not representation-space aligned across families",
        }
        if legacy_maps and not legacy_meta.get("error"):
            leg_ref = str(args.diff_ref_model)
            legacy_rel = relative_l2_diff_maps(
                feats_np, leg_ref, legacy_maps, epsilon=args.diff_rel_epsilon
            )
            legacy_masks = top_fraction_binary_masks(
                legacy_maps, leg_ref, top_fraction=args.diff_top_fraction
            )
            legacy_block["diff_rel_epsilon"] = float(args.diff_rel_epsilon)
            legacy_block["diff_top_fraction"] = float(args.diff_top_fraction)
            if grasp_dmin is not None:
                legacy_block["topk_diff_grasp_overlap_settings"] = {
                    "grasp_dist_threshold_m": float(args.grasp_dist_threshold),
                    "top_fraction_same_as_diff_mask": float(args.diff_top_fraction),
                    "improvement_vs_random": "overlap / mean_i 1[dist_to_gt(i)<τ]",
                }
            else:
                legacy_block["topk_diff_grasp_overlap_note"] = "skipped: no GT grasps"
            legacy_block["per_model"] = {}
            for m, vec in legacy_maps.items():
                v = np.asarray(vec, dtype=np.float64).ravel()
                pm: Dict[str, Any] = {
                    "mean_l2_diff": float(np.mean(v)),
                    "std_l2_diff": float(np.std(v)),
                    "max_l2_diff": float(np.max(v)),
                }
                if m in legacy_rel:
                    rv = np.asarray(legacy_rel[m], dtype=np.float64).ravel()
                    pm["mean_rel_diff"] = float(np.mean(rv))
                    pm["std_rel_diff"] = float(np.std(rv))
                    pm["max_rel_diff"] = float(np.max(rv))
                if m in legacy_masks:
                    km = legacy_masks[m]
                    pm["top_diff_fraction"] = float(args.diff_top_fraction)
                    pm["top_diff_point_count"] = int(np.sum(km > 0.5))
                if grasp_dmin is not None and m != leg_ref:
                    tov = topk_diff_grasp_region_overlap(
                        v,
                        grasp_dmin,
                        top_fraction=args.diff_top_fraction,
                        grasp_dist_threshold=args.grasp_dist_threshold,
                    )
                    if tov is not None:
                        pm["topk_diff_grasp_overlap"] = tov
                legacy_block["per_model"][m] = pm
            leg_panels: List = []
            if rgb_img is not None:
                leg_panels.append(("image", rgb_img, "RGB"))
            leg_cap = (
                f"LEGACY single ref={leg_ref}; not representation-aligned; "
                f"ε_rel={args.diff_rel_epsilon}; top={args.diff_top_fraction}"
            )
            legacy_block["figure_caption"] = leg_cap
            for m in MODEL_PANEL_ORDER:
                if m in legacy_maps:
                    leg_panels.append(
                        (
                            "pointcloud",
                            pc_common,
                            legacy_maps[m],
                            _short_model_title(m),
                        )
                    )
            if leg_panels:
                safe_ref = leg_ref.replace(os.sep, "_").replace("/", "_")
                save_comparison_grid(
                    leg_panels,
                    os.path.join(out_dir, f"compare_diff_legacy_to_{safe_ref}_l2.png"),
                    elev=elev,
                    azim=azim,
                    layout="rgb_wide_first_row" if rgb_img is not None else "uniform",
                    ncols=4,
                    suptitle="Feature Difference Analysis (Legacy Cross-Model)",
                    caption=leg_cap,
                )
            leg_rel_panels: List = []
            if rgb_img is not None:
                leg_rel_panels.append(("image", rgb_img, "RGB"))
            for m in MODEL_PANEL_ORDER:
                if m in legacy_rel:
                    leg_rel_panels.append(
                        (
                            "pointcloud",
                            pc_common,
                            legacy_rel[m],
                            _short_model_title(m),
                        )
                    )
            if len(leg_rel_panels) > 1:
                safe_ref = leg_ref.replace(os.sep, "_").replace("/", "_")
                save_comparison_grid(
                    leg_rel_panels,
                    os.path.join(out_dir, f"compare_diff_legacy_to_{safe_ref}_rel.png"),
                    elev=elev,
                    azim=azim,
                    layout="rgb_wide_first_row",
                    ncols=4,
                    suptitle="Feature Difference Analysis (Legacy Cross-Model)",
                    caption=leg_cap,
                )
            pct_l = f"{int(round(args.diff_top_fraction * 100))}pct"
            leg_mask_panels: List = []
            if rgb_img is not None:
                leg_mask_panels.append(("image", rgb_img, "RGB"))
            for m in MODEL_PANEL_ORDER:
                if m in legacy_masks:
                    leg_mask_panels.append(
                        (
                            "pointcloud_binary",
                            pc_common,
                            legacy_masks[m],
                            _short_model_title(m),
                        )
                    )
            if len(leg_mask_panels) > 1:
                safe_ref = leg_ref.replace(os.sep, "_").replace("/", "_")
                save_comparison_grid(
                    leg_mask_panels,
                    os.path.join(out_dir, f"compare_diff_legacy_to_{safe_ref}_mask_top{pct_l}_l2.png"),
                    elev=elev,
                    azim=azim,
                    layout="rgb_wide_first_row",
                    ncols=4,
                    suptitle="Feature Difference Analysis (Legacy Cross-Model)",
                    caption=leg_cap,
                )
        elif legacy_meta.get("error"):
            print(f"[WARN] legacy 特征差分图跳过: {legacy_meta.get('error')}")
        diff_by_family_json["legacy_cross_model_diff"] = legacy_block

    with open(os.path.join(out_dir, "feature_diff_by_family.json"), "w", encoding="utf-8") as f:
        json.dump(diff_by_family_json, f, indent=2, ensure_ascii=False)

    # CSV 简要导出
    rows = []
    for log in logs:
        rows.append(
            {
                "model": log.get("model"),
                "load_type": log.get("load_type"),
                "source": log.get("source"),
                "checkpoint": log.get("checkpoint"),
                "status": log.get("status"),
                "tensor_note": (log.get("meta") or {}).get("checkpoint_tensor", ""),
            }
        )
    csv_path = os.path.join(out_dir, "run_summary.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"[done] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
