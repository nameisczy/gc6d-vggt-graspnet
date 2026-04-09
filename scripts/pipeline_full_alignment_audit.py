#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC6D alignment 五模式全面审计：训练状态、checkpoint 加载、前向张量、与 pure GraspNet 对照、可视化。

用法:
  cd ~/gc6d_grasp_pipeline && conda activate gc6d
  python scripts/pipeline_full_alignment_audit.py \\
    --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \\
    --camera realsense-d415 \\
    --graspnet_ckpt ~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar \\
    --out_dir ~/gc6d_grasp_pipeline/alignment_audit_out \\
    --batch_size 2 --num_batches 4 --device cuda:0

输出:
  - audit_config.json
  - param_stats.json
  - checkpoint_load_audit.json
  - forward_diff_pure.json
  - tensor_stats_per_mode.json
  - benchmark_from_eval_rewrite.json（若存在 eval_out_rewrite/*/summary_rewrite.json）
  - figures/<mode>/*.png
  - AUDIT_REPORT.md（数据驱动摘要）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 按路径加载，避免与环境中可能存在的同名 pip 包 `training` 冲突
_optim_spec = importlib.util.spec_from_file_location(
    "_gc6d_training_optim",
    os.path.join(ROOT, "training", "optim.py"),
)
if _optim_spec is None or _optim_spec.loader is None:
    raise ImportError("Cannot load %s" % os.path.join(ROOT, "training", "optim.py"))
_gc6d_optim = importlib.util.module_from_spec(_optim_spec)
_optim_spec.loader.exec_module(_gc6d_optim)
apply_alignment_freeze = _gc6d_optim.apply_alignment_freeze

# 复用特征分析工具
from scripts.feature_alignment_analysis import (
    channel_cosine_matrix,
    global_stats,
    linear_map_mse_cosine,
    offline_normalize_compare,
    per_channel_stats,
    per_channel_std_ratio,
    per_seed_l2_array,
    plot_box_channel_metric,
    plot_box_per_seed_norm,
    plot_heatmap,
    plot_histogram_compare,
    plot_pca2d,
    try_tsne,
    _ensure_matplotlib,
)
from data import GC6DOfflineUnifiedDataset, collate_gc6d
from torch.utils.data import DataLoader
from utils.load_model import load_policy_from_checkpoint
from utils.batch_images import load_images_batch
from models.graspnet_adapter import load_graspnet_pretrained
# 默认 alignment v2 checkpoint（可按需覆盖）
DEFAULT_MODE_CHECKPOINTS: Dict[str, str] = {
    "pure_graspnet": "checkpoints/alignment_runs/exp_pure/exp_pure_pure_graspnet_20260328_163000.pt",
    "vggt_replacement": "checkpoints/alignment_runs/exp_vggt_rep/exp_vggt_rep_vggt_replacement_20260328_163036.pt",
    "lift3d_replacement_clip": "checkpoints/alignment_runs/exp_l3clip_rep/exp_l3clip_rep_lift3d_replacement_clip_20260328_164446.pt",
    "lift3d_replacement_dinov2": "checkpoints/alignment_runs/exp_l3dino_rep/exp_l3dino_rep_lift3d_replacement_dinov2_20260328_164455.pt",
    "vggt_fusion_normalized": "checkpoints/alignment_runs/exp_vggt_fuse_norm/exp_vggt_fuse_norm_vggt_fusion_normalized_20260328_163119.pt",
}


def _prefix_group(name: str) -> str:
    if name.startswith("encoder"):
        return "encoder"
    if name.startswith("grasp_net.view_estimator.backbone"):
        return "graspnet_backbone"
    if name.startswith("grasp_net.view_estimator.vpmodule"):
        return "graspnet_vpmodule"
    if name.startswith("grasp_net.grasp_generator"):
        return "graspnet_head"
    if name.startswith("replacement_projector"):
        return "replacement_projector"
    if name.startswith("fusion_mlp") or name.startswith("vggt_ln") or name.startswith("vggt_proj"):
        return "fusion_modules"
    if name.startswith("adapter"):
        return "adapter"
    return "other"


def audit_trainable_groups(model: nn.Module) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"trainable_numel": 0, "frozen_numel": 0, "trainable_params": 0, "frozen_params": 0}
    )
    for name, p in model.named_parameters():
        g = _prefix_group(name)
        if p.requires_grad:
            groups[g]["trainable_numel"] += p.numel()
            groups[g]["trainable_params"] += 1
        else:
            groups[g]["frozen_numel"] += p.numel()
            groups[g]["frozen_params"] += 1
    return {k: dict(v) for k, v in sorted(groups.items())}


def load_ckpt_raw(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def audit_state_dict_load(
    model: nn.Module, state: Dict[str, torch.Tensor], strict: bool
) -> Dict[str, Any]:
    incomp = model.load_state_dict(state, strict=strict)
    if hasattr(incomp, "missing_keys"):
        missing, unexpected = incomp.missing_keys, incomp.unexpected_keys
    else:
        missing, unexpected = incomp  # type: ignore[misc]
    return {
        "strict": strict,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
    }


def tensor_summary(t: Optional[torch.Tensor]) -> Optional[Dict[str, float]]:
    if t is None:
        return None
    x = t.detach().float().cpu().numpy()
    return global_stats(x)


def diff_tensors(
    a: torch.Tensor, b: torch.Tensor, name: str
) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {"name": name, "error": "shape_mismatch", "a": list(a.shape), "b": list(b.shape)}
    d = (a.detach().float() - b.detach().float()).abs()
    return {
        "name": name,
        "mean_abs_diff": float(d.mean()),
        "max_abs_diff": float(d.max()),
        "rmse": float(torch.sqrt((d**2).mean())),
    }


def forward_pure_reference(grasp_net: nn.Module, pc: torch.Tensor) -> Dict[str, torch.Tensor]:
    """与 PureGraspNetPipeline.forward 一致，但返回中间量。"""
    end_points: Dict[str, Any] = {"point_clouds": pc}
    ve = grasp_net.view_estimator
    seed_features, seed_xyz, end_points = ve.backbone(pc, end_points)
    end_points = ve.vpmodule(seed_xyz, seed_features, end_points)
    end_points = grasp_net.grasp_generator(end_points)
    return {
        "seed_features": seed_features,
        "seed_xyz": seed_xyz,
        "objectness_score": end_points.get("objectness_score"),
        "grasp_score": end_points.get("grasp_score"),
        "grasp_width": end_points.get("grasp_width"),
        "grasp_angle_cls": end_points.get("grasp_angle_cls_scores"),
        "end_points": end_points,
    }


def collect_eval_rewrite_summaries(root: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    eroot = os.path.join(root, "eval_out_rewrite")
    if not os.path.isdir(eroot):
        return out
    for name in sorted(os.listdir(eroot)):
        sp = os.path.join(eroot, name, "summary_rewrite.json")
        if not os.path.isfile(sp):
            continue
        try:
            with open(sp, "r", encoding="utf-8") as f:
                j = json.load(f)
            j["_folder_name"] = name
            out.append(j)
        except Exception:
            continue
    return out


def _np_seed_features_bcs(sf: torch.Tensor) -> np.ndarray:
    """(B,256,S) -> numpy"""
    return sf.detach().float().cpu().numpy()


def _register_vpmodule_seed_capture(model: nn.Module) -> Tuple[Any, Dict[str, List[torch.Tensor]]]:
    """在 grasp_net.view_estimator.vpmodule 上挂 forward_pre_hook，捕获 (seed_xyz, seed_features, end_points)。"""
    store: Dict[str, List[torch.Tensor]] = {"seed_features": [], "seed_xyz": []}
    ve = getattr(model.grasp_net, "view_estimator", None)
    if ve is None or not hasattr(ve, "vpmodule"):
        return None, store

    def _pre_hook(_mod, inputs):
        if len(inputs) >= 2:
            store["seed_xyz"].append(inputs[0])
            store["seed_features"].append(inputs[1])

    h = ve.vpmodule.register_forward_pre_hook(_pre_hook)
    return h, store


def _model_requires_images(model: nn.Module) -> bool:
    """与 training.losses.model_requires_images 一致；避免 import training.*（与 pip 包名冲突）。"""
    enc = getattr(model, "encoder", None)
    if enc is not None and type(enc).__name__ == "VGGTEncoder":
        return True
    return bool(getattr(model, "requires_images", False))


@torch.no_grad()
def run_mode_forward(
    model: nn.Module,
    pc: torch.Tensor,
    metas: List[dict],
    data_dir: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    images = None
    if _model_requires_images(model):
        images = load_images_batch(metas, data_dir, device)
    hook_handle, cap = _register_vpmodule_seed_capture(model)
    try:
        ep = model(point_cloud=pc, images=images)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    extras: Dict[str, torch.Tensor] = {}
    if cap["seed_features"]:
        extras["vpmodule_in_seed_features"] = cap["seed_features"][-1]
    if cap["seed_xyz"]:
        extras["vpmodule_in_seed_xyz"] = cap["seed_xyz"][-1]
    try:
        ve = model.grasp_net.view_estimator
        ep0: Dict[str, Any] = {"point_clouds": pc}
        sf_bb, sxyz, _ = ve.backbone(pc, ep0)
        extras["backbone_seed_features"] = sf_bb
        extras["backbone_seed_xyz"] = sxyz
    except Exception:
        pass
    return ep, extras


def main() -> None:
    p = argparse.ArgumentParser(description="GC6D alignment 五模式全面审计")
    p.add_argument("--data_dir", type=str, default="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified")
    p.add_argument("--camera", type=str, default="realsense-d415")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=4)
    p.add_argument("--graspnet_ckpt", type=str, default=os.path.expanduser("~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"))
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "alignment_audit_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_pure", type=str, default=None)
    p.add_argument("--ckpt_vggt_rep", type=str, default=None)
    p.add_argument("--ckpt_l3clip", type=str, default=None)
    p.add_argument("--ckpt_l3dino", type=str, default=None)
    p.add_argument("--ckpt_vggt_fuse", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig_root = os.path.join(out_dir, "figures")
    os.makedirs(fig_root, exist_ok=True)

    device = torch.device(args.device)
    ckpt_map = {
        "pure_graspnet": args.ckpt_pure or os.path.join(ROOT, DEFAULT_MODE_CHECKPOINTS["pure_graspnet"]),
        "vggt_replacement": args.ckpt_vggt_rep or os.path.join(ROOT, DEFAULT_MODE_CHECKPOINTS["vggt_replacement"]),
        "lift3d_replacement_clip": args.ckpt_l3clip or os.path.join(ROOT, DEFAULT_MODE_CHECKPOINTS["lift3d_replacement_clip"]),
        "lift3d_replacement_dinov2": args.ckpt_l3dino or os.path.join(ROOT, DEFAULT_MODE_CHECKPOINTS["lift3d_replacement_dinov2"]),
        "vggt_fusion_normalized": args.ckpt_vggt_fuse or os.path.join(ROOT, DEFAULT_MODE_CHECKPOINTS["vggt_fusion_normalized"]),
    }

    with open(os.path.join(out_dir, "audit_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "resolved_checkpoints": ckpt_map}, f, indent=2, ensure_ascii=False)

    # 数据
    ds = GC6DOfflineUnifiedDataset(
        data_dir=args.data_dir,
        split="val",
        camera=args.camera,
        max_samples=args.batch_size * args.num_batches,
        load_gt_multi=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_gc6d, num_workers=0)
    batches: List[Tuple] = []
    for i, b in enumerate(loader):
        batches.append(b)
        if i + 1 >= args.num_batches:
            break
    if not batches:
        raise RuntimeError("无 batch，请检查 data_dir / camera / index")

    # 参考：仅 GraspNet 预训练（未经过 alignment 训练）
    ref_net = load_graspnet_pretrained(args.graspnet_ckpt, device, args.graspnet_root, is_training=False)
    ref_net.eval()

    param_stats: Dict[str, Any] = {}
    load_audits: Dict[str, Any] = {}
    forward_pure_diff: Dict[str, Any] = {}
    tensor_stats_modes: Dict[str, Any] = {}

    plt = _ensure_matplotlib()

    # --- pure: 参考 vs checkpoint pipeline ---
    pure_path = ckpt_map["pure_graspnet"]
    pure_ck = load_ckpt_raw(pure_path)
    pure_model = load_policy_from_checkpoint(
        pure_path, device=str(device), graspnet_ckpt=pure_ck.get("graspnet_ckpt") or args.graspnet_ckpt
    )
    apply_alignment_freeze(pure_model, pure_ck.get("model_mode", "pure_graspnet"))
    pure_model.eval()

    state = pure_ck.get("model", {})
    load_audits["pure_graspnet"] = {
        "path": pure_path,
        "model_class_name": type(pure_model).__name__,
        "ckpt_keys_top_level": [k for k in pure_ck.keys() if k != "model"][:40],
        "state_dict_num_keys": len(state),
        "load_strict_false": audit_state_dict_load(pure_model, state, strict=False),
    }
    # 与预训练文件直接加载的 grasp_net 权重差异：分 backbone / head（alignment 只应改变 head）
    ref_sd = ref_net.state_dict()
    trained_sd = pure_model.grasp_net.state_dict()

    def _mean_abs_group(prefix: str) -> Dict[str, Any]:
        diffs = []
        for k in sorted(set(ref_sd.keys()) & set(trained_sd.keys())):
            if not k.startswith(prefix):
                continue
            a, b = ref_sd[k].float().cpu(), trained_sd[k].float().cpu()
            if a.shape != b.shape:
                continue
            diffs.append(float((a - b).abs().mean().item()))
        return {
            "prefix": prefix,
            "n_tensors": len(diffs),
            "mean_of_mean_abs_diff": float(np.mean(diffs)) if diffs else None,
            "max_of_mean_abs_diff": float(np.max(diffs)) if diffs else None,
        }

    forward_pure_diff["grasp_net_weight_diff_vs_fresh_pretrained"] = {
        "backbone_view_estimator_backbone": _mean_abs_group("view_estimator.backbone"),
        "vpmodule": _mean_abs_group("view_estimator.vpmodule"),
        "grasp_generator": _mean_abs_group("grasp_generator"),
        "all_grasp_net_keys": _mean_abs_group(""),  # 空前缀：上面已覆盖；改为全 key
    }
    # 修正 all：遍历全部共有 key
    all_diffs = []
    for k in sorted(set(ref_sd.keys()) & set(trained_sd.keys())):
        a, b = ref_sd[k].float().cpu(), trained_sd[k].float().cpu()
        if a.shape != b.shape:
            continue
        all_diffs.append(float((a - b).abs().mean().item()))
    forward_pure_diff["grasp_net_weight_diff_vs_fresh_pretrained"]["all_tensors_summary"] = {
        "n_tensors": len(all_diffs),
        "mean_of_mean_abs_diff": float(np.mean(all_diffs)) if all_diffs else None,
    }
    del forward_pure_diff["grasp_net_weight_diff_vs_fresh_pretrained"]["all_grasp_net_keys"]

    diffs_accum: Dict[str, List[float]] = defaultdict(list)

    for batch in batches:
        pcs, _, _, metas = batch
        pcs = pcs.to(device)
        out_ref = forward_pure_reference(ref_net, pcs)
        out_pure = forward_pure_reference(pure_model.grasp_net, pcs)
        for key in ("seed_features", "seed_xyz"):
            d = diff_tensors(out_ref[key], out_pure[key], key)
            diffs_accum[key].append(d["mean_abs_diff"])
            diffs_accum[key + "_max"].append(d["max_abs_diff"])
        for key in ("objectness_score", "grasp_score"):
            if out_ref[key] is not None and out_pure[key] is not None:
                d = diff_tensors(out_ref[key], out_pure[key], key)
                diffs_accum[key + "_mad"].append(d["mean_abs_diff"])

    forward_pure_diff["per_batch_mean_abs_diff_seed_features"] = diffs_accum["seed_features"]
    forward_pure_diff["note"] = (
        "alignment 训练后 pure_graspnet 仅训 head（vpmodule+grasp_generator），backbone 冻结；"
        "seed_features 应对齐预训练 backbone；若 seed_features 差异大则说明 load 或 forward 异常。"
    )
    if diffs_accum["seed_features"]:
        forward_pure_diff["seed_features_mean_abs_diff_mean"] = float(np.mean(diffs_accum["seed_features"]))
        forward_pure_diff["seed_features_max_abs_diff_mean"] = float(np.mean(diffs_accum["seed_features_max"]))

    with open(os.path.join(out_dir, "forward_diff_pure.json"), "w", encoding="utf-8") as f:
        json.dump(forward_pure_diff, f, indent=2, ensure_ascii=False)

    # --- 各模式：参数、加载、张量统计 ---
    modes_order = [
        "pure_graspnet",
        "vggt_replacement",
        "lift3d_replacement_clip",
        "lift3d_replacement_dinov2",
        "vggt_fusion_normalized",
    ]

    for mode in modes_order:
        path = ckpt_map[mode]
        if not os.path.isfile(path):
            tensor_stats_modes[mode] = {"error": "checkpoint_missing", "path": path}
            continue
        ck = load_ckpt_raw(path)
        model = load_policy_from_checkpoint(path, device=str(device), graspnet_ckpt=ck.get("graspnet_ckpt") or args.graspnet_ckpt)
        mm = ck.get("model_mode") or mode
        apply_alignment_freeze(model, mm)
        model.eval()
        param_stats[mode] = audit_trainable_groups(model)
        st = ck.get("model", {})
        load_audits[mode] = {
            "path": path,
            "model_class_name": type(model).__name__,
            "ckpt_model_mode": ck.get("model_mode"),
            "ckpt_encoder_type": ck.get("encoder_type"),
            "state_dict_num_keys": len(st),
            "load_strict_false": audit_state_dict_load(model, st, strict=False),
        }

        # 聚合多 batch 的 seed 特征统计 + 与 ref backbone 对比
        ref_sf_list: List[np.ndarray] = []
        mode_sf_list: List[np.ndarray] = []
        mode_dir = os.path.join(fig_root, mode.replace("/", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        for batch in batches:
            pcs, _, _, metas = batch
            pcs = pcs.to(device)
            ep_ref = forward_pure_reference(ref_net, pcs)
            ref_sf_list.append(_np_seed_features_bcs(ep_ref["seed_features"]))
            ep_m, extras = run_mode_forward(model, pcs, list(metas), args.data_dir, device)
            # 进入 vpmodule 前的 seed（replacement/fusion 已与 backbone 不同）
            vin = extras.get("vpmodule_in_seed_features")
            if vin is not None:
                mode_sf_list.append(_np_seed_features_bcs(vin))

        if ref_sf_list and mode_sf_list:
            # 每个元素 (B,C,S)，沿 batch 维拼接而非 stack（避免 (n_batches,B,C,S) 四维）
            ref_stack = np.concatenate(ref_sf_list, axis=0)
            mode_stack = np.concatenate(mode_sf_list, axis=0)
            if ref_stack.shape[2] != mode_stack.shape[2]:
                tensor_stats_modes[mode] = {
                    "error": "seed_count_mismatch",
                    "ref_S": ref_stack.shape[2],
                    "mode_S": mode_stack.shape[2],
                }
            else:
                R = ref_stack.transpose(0, 2, 1).reshape(-1, ref_stack.shape[1])
                M = mode_stack.transpose(0, 2, 1).reshape(-1, mode_stack.shape[1])
                n = min(R.shape[0], M.shape[0], 8192)
                ridge = linear_map_mse_cosine(R[:n], M[:n])
                cos_mat = channel_cosine_matrix(R[: min(n, 2048)], M[: min(n, 2048)])
                pcr = per_channel_std_ratio(ref_stack, mode_stack)
                tensor_stats_modes[mode] = {
                    "ridge_graspnet_backbone_seed_vs_vpmodule_input_seed": ridge,
                    "per_channel_std_ratio_vs_graspnet_backbone": pcr,
                    "ref_seed_global_stats": global_stats(ref_stack.ravel()),
                    "mode_vpmodule_in_global_stats": global_stats(mode_stack.ravel()),
                }
                hm_title = (
                    "channel cosine: GraspNet backbone seed vs vpmodule input (%s)" % mode
                )
                plot_heatmap(plt, cos_mat, hm_title, os.path.join(mode_dir, "heatmap_ch_cosine_vs_graspnet.png"))
                # 直方图：值分布
                plot_histogram_compare(
                    plt,
                    {"graspnet_backbone": ref_stack.ravel(), "vpmodule_in": mode_stack.ravel()},
                    "seed feature values " + mode,
                    os.path.join(mode_dir, "hist_seed_values.png"),
                )
                plot_pca2d(
                    plt,
                    {"graspnet_backbone": ref_stack, "vpmodule_in": mode_stack},
                    os.path.join(mode_dir, "pca2d_seed.png"),
                )
                try_tsne(plt, {"graspnet_backbone": ref_stack, "vpmodule_in": mode_stack}, os.path.join(mode_dir, "tsne2d_seed.png"))
                # per-seed L2
                ps_ref = per_seed_l2_array(ref_stack)
                ps_m = per_seed_l2_array(mode_stack)
                plot_box_per_seed_norm(
                    plt,
                    {"graspnet_backbone": ps_ref, "vpmodule_in": ps_m},
                    os.path.join(mode_dir, "box_per_seed_l2.png"),
                )
        if mode == "pure_graspnet" and mode in tensor_stats_modes:
            tensor_stats_modes[mode]["train_meta_expected"] = (
                "freeze backbone, train vpmodule+head; vpmodule_in 应等于 ref backbone seed"
            )

        # 可视化：ref vs 当前模式 end_points 主分数
        try:
            pcs = batches[0][0].to(device)
            metas0 = list(batches[0][3])
            ep_r = forward_pure_reference(ref_net, pcs)
            ep_m, _ = run_mode_forward(model, pcs, metas0, args.data_dir, device)
            hs = {}
            if ep_r["objectness_score"] is not None:
                hs["ref_objectness"] = ep_r["objectness_score"].detach().float().cpu().numpy().ravel()
            if ep_m.get("objectness_score") is not None:
                hs["mode_objectness"] = ep_m["objectness_score"].detach().float().cpu().numpy().ravel()
            if hs:
                plot_histogram_compare(plt, hs, "objectness " + mode, os.path.join(mode_dir, "hist_objectness.png"))
        except Exception as ex:
            tensor_stats_modes.setdefault(mode, {})["viz_error"] = str(ex)

    with open(os.path.join(out_dir, "param_stats.json"), "w", encoding="utf-8") as f:
        json.dump(param_stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "checkpoint_load_audit.json"), "w", encoding="utf-8") as f:
        json.dump(load_audits, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "tensor_stats_per_mode.json"), "w", encoding="utf-8") as f:
        json.dump(tensor_stats_modes, f, indent=2, ensure_ascii=False)

    summ = collect_eval_rewrite_summaries(ROOT)
    with open(os.path.join(out_dir, "benchmark_from_eval_rewrite.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    # --- REPORT ---
    lines = [
        "# Pipeline alignment 审计报告（自动生成）",
        "",
        "## 1. Checkpoint 路径",
        "",
        json.dumps(ckpt_map, indent=2, ensure_ascii=False),
        "",
        "## 2. 参数组（trainable numel）",
        "",
    ]
    for m, g in param_stats.items():
        lines.append("### " + m)
        lines.append("```json")
        lines.append(json.dumps(g, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    lines.append("## 3. 加载 missing keys（strict=False）摘要")
    for m, la in load_audits.items():
        lf = la.get("load_strict_false", {})
        lines.append(
            "- **%s**: missing=%d unexpected=%d"
            % (m, lf.get("n_missing", 0), lf.get("n_unexpected", 0))
        )
    lines.append("")
    lines.append("## 4. pure 与预训练 GraspNet 权重差异（alignment 训练后 head 应变）")
    lines.append("```json")
    lines.append(json.dumps(forward_pure_diff.get("grasp_net_mean_abs_diff_vs_fresh_pretrained"), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 5. pure 前向 seed_features 与预训练参考差异（backbone 应对齐）")
    lines.append("```json")
    lines.append(json.dumps({k: forward_pure_diff.get(k) for k in forward_pure_diff if "seed" in k}, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 6. eval_out_rewrite 汇总（若存在）")
    lines.append("```json")
    lines.append(json.dumps([{k: s[k] for k in ("_folder_name", "AP", "pipeline_checkpoint", "eval_mode") if k in s} for s in summ], indent=2))
    lines.append("```")

    with open(os.path.join(out_dir, "AUDIT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Done. Output:", out_dir, flush=True)


if __name__ == "__main__":
    main()
