#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四项强制检查：pretrained 加载、与 baseline 中间量对齐、pure 无 cond 残留、eval 口径对照。
用法:
  conda activate gc6d
  cd ~/gc6d_grasp_pipeline
  python scripts/diagnose_pure_graspnet_checks.py [--graspnet_ckpt PATH] [--graspnet_root PATH]
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from models.graspnet_adapter import load_graspnet_pretrained, pred_decode_17d
from models.pure_graspnet import PureGraspNetPipeline, build_pure_graspnet_pipeline


def _tensor_stats(name: str, t: torch.Tensor) -> str:
    t = t.detach().float()
    return (
        f"{name}: shape={tuple(t.shape)} mean={float(t.mean()):.6g} std={float(t.std()):.6g} "
        f"norm(L2)={float(torch.norm(t.reshape(-1))):.6g}"
    )


def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--graspnet_ckpt",
        type=str,
        default=os.path.expanduser("~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"),
    )
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--npoints", type=int, default=1024)
    args = p.parse_args()

    ckpt_path = os.path.abspath(os.path.expanduser(args.graspnet_ckpt))
    graspnet_root = os.path.abspath(os.path.expanduser(args.graspnet_root))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # 检查 1：checkpoint 加载与 strict 报告
    # -------------------------------------------------------------------------
    _section("检查 1：pretrained checkpoint 加载")
    print("checkpoint 绝对路径:", ckpt_path)
    print("exists:", os.path.isfile(ckpt_path))
    print("graspnet_root:", graspnet_root)

    if not os.path.isfile(ckpt_path):
        print("ERROR: checkpoint 文件不存在，终止。")
        return

    ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt_raw, dict) and "model_state_dict" in ckpt_raw:
        state = ckpt_raw["model_state_dict"]
        print("ckpt 顶层 keys (节选):", list(ckpt_raw.keys())[:12])
    elif isinstance(ckpt_raw, dict) and "model" in ckpt_raw:
        state = ckpt_raw["model"]
        print("使用 ckpt['model'] 作为 state_dict")
    else:
        state = ckpt_raw
        print("将整包视为 state_dict（或顶层即权重）")

    print("state_dict 参数条目数:", len(state))
    sample_keys = list(state.keys())[:5]
    print("state_dict 前 5 个 key:", sample_keys)

    # 延迟 import baseline GraspNet 以构建「加载前」网络
    from models.graspnet_adapter import _load_baseline_graspnet_module

    mod = _load_baseline_graspnet_module(graspnet_root)
    GraspNet = mod.GraspNet
    pred_decode_baseline = mod.pred_decode

    net_fresh = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(device)

    # 取各子模块「第一个参数」在 load 前后的统计（便于核对权重是否被覆盖）
    def _first_param_stat(net: torch.nn.Module, prefix: str) -> tuple[str, str]:
        for name, p in net.named_parameters():
            if name.startswith(prefix):
                return name, _tensor_stats(name + " [BEFORE load]", p)
        return "(none)", "(no param matched prefix %s)" % prefix

    bb_n, bb_s = _first_param_stat(net_fresh, "view_estimator.backbone.")
    vp_n, vp_s = _first_param_stat(net_fresh, "view_estimator.vpmodule.")
    gg_n, gg_s = _first_param_stat(net_fresh, "grasp_generator.")

    print("\n--- 随机初始化 net：关键子模块第一个参数 [load 前] ---")
    print(bb_s)
    print(vp_s)
    print(gg_s)

    inc = net_fresh.load_state_dict(state, strict=False)
    print("\n--- load_state_dict(strict=False) ---")
    print("missing_keys (共 %d 个):" % len(inc.missing_keys))
    for k in inc.missing_keys:
        print("  ", k)
    print("unexpected_keys (共 %d 个):" % len(inc.unexpected_keys))
    for k in inc.unexpected_keys:
        print("  ", k)

    if len(inc.missing_keys) == 0 and len(inc.unexpected_keys) == 0:
        print("(missing/unexpected 均为空：与 checkpoint 键名完全对齐)")
    else:
        print(
            "\n说明：strict=False 下上述即为全部「未加载 / 未使用」的键；"
            "若存在 missing，则对应层仍为初始化权重。"
        )

    # strict=True 试探
    net_strict = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(device)
    strict_ok = True
    try:
        net_strict.load_state_dict(state, strict=True)
    except Exception as e:
        strict_ok = False
        print("\n--- load_state_dict(strict=True) 失败 ---")
        print(repr(e))

    if strict_ok:
        print("\n--- load_state_dict(strict=True): 成功（与预训练权重完全一致）---")

    print("\n--- 同上三个参数名在 load 后（应被 checkpoint 覆盖，mean/std 与随机初始化应明显不同）---")

    def _stat_by_name(net: torch.nn.Module, target: str) -> str:
        for name, p in net.named_parameters():
            if name == target:
                return _tensor_stats(name + " [AFTER load]", p)
        return "%s [AFTER load]: NOT FOUND" % target

    if bb_n != "(none)":
        print(_stat_by_name(net_fresh, bb_n))
    if vp_n != "(none)":
        print(_stat_by_name(net_fresh, vp_n))
    if gg_n != "(none)":
        print(_stat_by_name(net_fresh, gg_n))

    # -------------------------------------------------------------------------
    # 检查 2：baseline 完整 forward vs PureGraspNetPipeline（同权重、同输入）
    # -------------------------------------------------------------------------
    _section("检查 2：baseline GraspNet.forward 与 PureGraspNetPipeline 中间输出对比")

    net_baseline = load_graspnet_pretrained(ckpt_path, device, graspnet_root, is_training=False)
    net_baseline.eval()
    pure = build_pure_graspnet_pipeline(
        graspnet_ckpt=ckpt_path, graspnet_root=graspnet_root, device=device
    )
    pure.eval()

    pc = torch.randn(args.batch, args.npoints, 3, device=device, dtype=torch.float32) * 0.1

    end_in = {"point_clouds": pc}
    with torch.no_grad():
        # 中间量：与 Stage1 一致
        ve = net_baseline.view_estimator
        seed_b, xyz_b, ep_b = ve.backbone(pc, dict(end_in))
        ep_b = ve.vpmodule(xyz_b, seed_b, ep_b)

        ve2 = pure.grasp_net.view_estimator
        seed_p, xyz_p, ep_p = ve2.backbone(pc, dict(end_in))
        ep_p = ve2.vpmodule(xyz_p, seed_p, ep_p)

        out_full = net_baseline(dict(end_in))
        out_pure = pure(pc)

    def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    print("seed_features max|diff|:", _max_diff(seed_b, seed_p))
    print("seed_xyz max|diff|:", _max_diff(xyz_b, xyz_p))
    for key in ("objectness_score", "grasp_score_pred", "grasp_width_pred", "grasp_angle_cls_pred"):
        if key in out_full and key in out_pure:
            print(
                "%s max|diff|:" % key,
                _max_diff(out_full[key], out_pure[key]),
            )
            print("  baseline " + _tensor_stats("mean/std", out_full[key].flatten()[:4096]))
            print("  pure     " + _tensor_stats("mean/std", out_pure[key].flatten()[:4096]))
        else:
            print("missing key in outputs:", key, key in out_full, key in out_pure)

    print("\n--- 完整输出字典 key 集合是否一致 ---")
    k1, k2 = set(out_full.keys()), set(out_pure.keys())
    print("only in baseline:", sorted(k1 - k2))
    print("only in pure:", sorted(k2 - k1))

    # decode：baseline pred_decode vs pipeline pred_decode_17d
    with torch.no_grad():
        dec_list = pred_decode_baseline(out_full)
        dec17 = pred_decode_17d(out_full, device, max_grasps=4096)
    n0 = dec_list[0].shape[0] if dec_list and dec_list[0] is not None else 0
    row0 = dec_list[0] if n0 else None
    mask17 = dec17[0, :, 0] > 1e-8
    n17 = int(mask17.sum().item())
    print("\n--- decode 后 batch0 ---")
    print("baseline pred_decode 行数 (objectness 后):", n0)
    print("pipeline pred_decode_17d 非零 score 行数:", n17)
    if row0 is not None and row0.numel():
        s = row0[:, 0].cpu().numpy()
        print("baseline score min/max/mean:", float(s.min()), float(s.max()), float(s.mean()))
    s17 = dec17[0, :n17, 0].cpu().numpy() if n17 else np.array([])
    if s17.size:
        print("pipeline top non-pad score min/max/mean:", float(s17.min()), float(s17.max()), float(s17.mean()))

    # -------------------------------------------------------------------------
    # 检查 3：pure 结构无 encoder/adapter/cond
    # -------------------------------------------------------------------------
    _section("检查 3：pure_graspnet 无 encoder / adapter / cond 残留")

    m = pure
    print("type(model):", type(m).__name__)
    print("model.encoder is None:", m.encoder is None)
    print("model.adapter is None:", m.adapter is None)
    print("model.use_adapter:", getattr(m, "use_adapter", None))
    print("model.encoder_type:", getattr(m, "encoder_type", None))

    src = inspect.getsource(PureGraspNetPipeline.forward)
    print("\nPureGraspNetPipeline.forward 源码（是否含 cond/adapter/encoder 调用）:")
    print(src)
    assert "self.encoder" not in src and "self.adapter" not in src

    # 运行一次 forward 并确认 seed 仅经 backbone+vpmodule（手动与 net 内一致）
    with torch.no_grad():
        ep_manual = {"point_clouds": pc}
        sf_m, xyz_m, ep_m = pure.grasp_net.view_estimator.backbone(pc, ep_manual)
        ep_m = pure.grasp_net.view_estimator.vpmodule(xyz_m, sf_m, ep_m)
    print("手动 Stage1 与 pure.forward 中 vpmodule 后 fp2_xyz max|diff|:", _max_diff(ep_m["fp2_xyz"], out_pure["fp2_xyz"]))

    # -------------------------------------------------------------------------
    # 检查 4：eval_benchmark 默认口径（静态，与 repro baseline_infer 对照）
    # -------------------------------------------------------------------------
    _section("检查 4：current pipeline eval 与 graspnet-baseline / repro 口径对照")

    print(
        """eval_benchmark.py 默认（摘自 argparse）:
  split: 默认 val（choices train/val/test）；你跑 test 时需显式 --split test
  camera: 默认 realsense-d415
  top_k: 默认 50（API TOP_K）
  background_filter: 默认 True（除非 --no_background_filter）
  pre_dump_collision_filter: 默认 True；collision_thresh 默认 0.01；collision_voxel_size 默认 0.01
  max_dump_grasps: 默认 4096
  pred_decode_17d: 按 score 降序后取前 max_grasps，再写入 dump（与 baseline pred_decode 排序逻辑一致）

graspnet-baseline / repro baseline_infer（gc6d_graspnet_repro）:
  pred_decode(end_points) 得到 grasp 列表；collision_thresh>0 时对 GraspGroup 做 ModelFreeCollisionDetector
  pipeline eval 在 dump 前可选同样 pre_dump 碰撞过滤（默认开）

说明：若 split/camera/top_k/预碰撞 与 baseline 单帧脚本不一致，AP 不可直接横向对比。"""
    )


if __name__ == "__main__":
    main()
