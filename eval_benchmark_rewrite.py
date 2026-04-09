#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC6D benchmark 评估（重写版，不依赖 eval_benchmark.py）。

对齐 ~/gc6d_graspnet_repro 的路径与口径：
  - 数据：原始 GC6D 根目录 + load_gc6d_frame（depth→点云、workspace、20000 点采样）
  - 相机默认 realsense-d435、13 个 eval imgId（与 repro infer_gc6d_pretrained.yaml 一致）
  - 推理：PureGraspNetPipeline 或 pipeline checkpoint + graspnet-baseline 的 pred_decode
  - 碰撞：graspnet-baseline ModelFreeCollisionDetector（与 baseline_infer 一致）
  - dump：不注入 GT object_id，保持 pred_decode 中 obj_id=-1（与 baseline_infer 一致）
  - 评估：GraspClutter6DEval.eval_all / eval_scene，与 repro eval_gc6d.py 一致

第一阶段：pure GraspNet；若 ``--pipeline_checkpoint`` 为 VGGT（vggt_base/vggt_ft），
会从 ``load_gc6d_frame`` 的 ``rgb`` 构造 ``(1,3,224,224)`` 传入模型（VGGT 不能仅用点云）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ensure_repro_on_path(repro_root: str) -> str:
    r = os.path.abspath(os.path.expanduser(repro_root))
    if r not in sys.path:
        sys.path.insert(0, r)
    return r


def _import_repro_loaders(repro_root: str):
    _ensure_repro_on_path(repro_root)
    try:
        from src.data.gc6d_dataset_wrapper import load_gc6d_frame
    except ImportError as e:
        wpath = os.path.join(repro_root, "src", "data", "gc6d_dataset_wrapper.py")
        hint = ""
        try:
            if os.path.isfile(wpath) and os.path.getsize(wpath) == 0:
                hint = "（该文件在磁盘上为 0 字节，请从备份或 Cursor Local History 恢复）"
        except OSError:
            pass
        raise ImportError(
            "无法从 gc6d_graspnet_repro 导入 load_gc6d_frame：%s。请确认 %s 非空且含函数 load_gc6d_frame%s"
            % (e, wpath, hint)
        ) from e
    from src.data.gc6d_layout import get_eval_img_ids, load_test_scene_ids, scene_id_6d

    return load_gc6d_frame, get_eval_img_ids, load_test_scene_ids, scene_id_6d


def _baseline_pred_decode_and_collision(
    end_points: dict,
    cloud_np: np.ndarray,
    graspnet_baseline_root: str,
    collision_thresh: float,
    voxel_size: float,
) -> np.ndarray:
    """与 baseline_infer 一致：graspnet-baseline pred_decode + ModelFreeCollisionDetector（GraspGroup 回退逻辑与 graspnet_adapter 一致）。"""
    from models.graspnet_adapter import _load_baseline_graspnet_module, apply_model_free_collision_filter

    mod = _load_baseline_graspnet_module(graspnet_baseline_root)
    grasp_preds = mod.pred_decode(end_points)
    g0 = grasp_preds[0]
    if g0 is None or g0.numel() == 0:
        arr = np.zeros((0, 17), dtype=np.float32)
    else:
        arr = g0.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    if collision_thresh is not None and float(collision_thresh) > 0 and arr.shape[0] > 0:
        arr = apply_model_free_collision_filter(
            arr,
            cloud_np,
            collision_thresh=float(collision_thresh),
            voxel_size=float(voxel_size),
            graspnet_baseline_root=graspnet_baseline_root,
        )
    return np.asarray(arr, dtype=np.float32)


def _save_grasp_npy(arr: np.ndarray, path: str) -> None:
    """与 repro prediction_io 一致：(N,17) float32。"""
    if arr.size == 0:
        arr = np.zeros((0, 17), dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[1] == 17
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, arr.astype(np.float32))


def _load_eval_model(
    *,
    device: torch.device,
    graspnet_ckpt: str,
    graspnet_root: str,
    pipeline_checkpoint: Optional[str],
) -> torch.nn.Module:
    """pure 用 graspnet_ckpt；LIFT3D/VGGT+adapter 用 pipeline_checkpoint + graspnet_ckpt。"""
    if pipeline_checkpoint and os.path.isfile(pipeline_checkpoint):
        from utils.load_model import load_policy_from_checkpoint

        return load_policy_from_checkpoint(
            pipeline_checkpoint,
            device=str(device),
            graspnet_ckpt=graspnet_ckpt,
            graspnet_root=graspnet_root,
        )
    from models.pure_graspnet import build_pure_graspnet_pipeline

    return build_pure_graspnet_pipeline(
        graspnet_ckpt=graspnet_ckpt,
        graspnet_root=graspnet_root,
        device=device,
    )


def _encoder_requires_images(model: torch.nn.Module) -> bool:
    """VGGT / 图像分支必须提供 RGB，不能把点云误当图像喂给 backbone。"""
    if getattr(model, "requires_images", False):
        return True
    enc = getattr(model, "encoder", None)
    if enc is None:
        return False
    if type(enc).__name__ == "VGGTEncoder":
        return True
    if getattr(model, "encoder_img", None) is not None:
        return True
    return False


def _prepare_rgb_tensor(frame: dict, device: torch.device, image_size: int = 224) -> torch.Tensor:
    """与 GC6DLIFT3DFormatDataset 一致：Resize 224 + ImageNet normalize，(1,3,H,W)。"""
    rgb = frame.get("rgb")
    if rgb is None:
        raise ValueError("当前模型需要 RGB：load_gc6d_frame 应返回 rgb (H,W,3)，请检查数据与 repro 加载逻辑。")
    from PIL import Image
    from torchvision import transforms as T

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("rgb 形状应为 (H,W,3)，实际: %s" % (arr.shape,))
    img = Image.fromarray(arr).convert("RGB")
    tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return tf(img).unsqueeze(0).to(device)


def _forward_model(
    model: torch.nn.Module, frame: dict, pc_xyz: np.ndarray, device: torch.device
) -> dict:
    """pc_xyz: (N,3)；VGGT+adapter 时从 frame['rgb'] 构造 images。"""
    t = torch.from_numpy(pc_xyz.astype(np.float32)).to(device).unsqueeze(0)
    model.eval()
    images = None
    if _encoder_requires_images(model):
        images = _prepare_rgb_tensor(frame, device)
    with torch.no_grad():
        if hasattr(model, "grasp_net"):
            ep = model(point_cloud=t, images=images)
        else:
            if images is not None:
                ep = model(point_cloud=t, images=images)
            else:
                ep = model({"point_clouds": t})
    return ep


def _agg_extra_from_lists(
    grasp_ll: list, score_ll: list, collision_ll: list, top_k: int
) -> Tuple[list[float], list[float]]:
    """单次 eval_scene(return_list=True) 内，按视角聚合到 rates/fc 列表。"""
    rates: list[float] = []
    fcs: list[float] = []
    for glist, slist, clist in zip(grasp_ll, score_ll, collision_ll):
        if not len(slist):
            continue
        slist = np.asarray(slist).ravel()
        clist = np.asarray(clist).ravel()
        n = min(len(slist), top_k)
        if n == 0:
            continue
        rates.append(1.0 - np.mean(clist[:n].astype(np.float64)))
        fcs.append(float(np.sum(slist[:n] > 0)))
    return rates, fcs


def main() -> None:
    p = argparse.ArgumentParser(description="GC6D benchmark（重写，对齐 gc6d_graspnet_repro）")
    p.add_argument("--repro_root", type=str, default=os.path.expanduser("~/gc6d_graspnet_repro"))
    p.add_argument("--gc6d_root", type=str, default=os.environ.get("GC6D_ROOT", "/mnt/ssd/ziyaochen/GraspClutter6D"))
    p.add_argument("--gc6d_api_repo", type=str, default=os.path.expanduser("~/graspclutter6dAPI"))
    p.add_argument("--graspnet_root", type=str, default=os.path.expanduser("~/graspnet-baseline"))
    p.add_argument(
        "--graspnet_ckpt",
        type=str,
        default=os.path.expanduser("~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"),
        help="预训练 GraspNet（pure_pretrained 或 pipeline 加载 backbone 用）",
    )
    p.add_argument(
        "--pipeline_checkpoint",
        type=str,
        default=None,
        help="可选：current pipeline 保存的 .pt（pure_graspnet）；不设则仅用 --graspnet_ckpt 构建 pure 模型",
    )
    p.add_argument("--camera", type=str, default="realsense-d435", help="与 repro infer_gc6d_pretrained 默认一致")
    p.add_argument("--num_point", type=int, default=20000)
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--num_eval_views", type=int, default=13)
    p.add_argument("--collision_thresh", type=float, default=0.01)
    p.add_argument("--voxel_size", type=float, default=0.01)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--background_filter", action="store_true", default=True)
    p.add_argument("--no_background_filter", dest="background_filter", action="store_false")
    p.add_argument("--proc", type=int, default=2, help="eval_all 并行进程数")
    p.add_argument("--max_scenes", type=int, default=0, help="0=全部 test 场景；>0 只跑前 N 个（调试）")
    p.add_argument("--dump_dir", type=str, default=None, help="dump 根目录；默认 ROOT/eval_out_rewrite/<tag>")
    p.add_argument("--tag", type=str, default="pure_graspnet_repro_aligned")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dump_only", action="store_true", help="只写 dump，不跑 API")
    p.add_argument("--eval_only", action="store_true", help="只对已有 dump 跑评估（需 --dump_dir）")
    p.add_argument("--extra_stats", action="store_true", help="逐 scene eval_scene(return_list=True) 聚合 top_k 碰撞率与 FC")
    p.add_argument("--skip_existing", action="store_true", help="若 .npy 已存在则跳过推理")
    args = p.parse_args()

    gc6d_root = os.path.abspath(os.path.expanduser(args.gc6d_root))
    repro_root = os.path.abspath(os.path.expanduser(args.repro_root))
    dump_dir = args.dump_dir or os.path.join(ROOT, "eval_out_rewrite", args.tag)
    dump_dir = os.path.abspath(dump_dir)
    device = torch.device(args.device)

    # graspclutter6dAPI
    api_parent = os.path.abspath(os.path.expanduser(args.gc6d_api_repo))
    if api_parent not in sys.path:
        sys.path.insert(0, api_parent)
    from graspclutter6dAPI.graspclutter6d_eval import GraspClutter6DEval

    load_gc6d_frame, get_eval_img_ids, load_test_scene_ids, scene_id_6d = _import_repro_loaders(repro_root)

    if args.eval_only and not os.path.isdir(dump_dir):
        raise FileNotFoundError("--eval_only 需要已存在的 --dump_dir: %s" % dump_dir)

    if not args.eval_only:
        print("[rewrite] 加载模型（pure 或 pipeline checkpoint）...", flush=True)
        model = _load_eval_model(
            device=device,
            graspnet_ckpt=args.graspnet_ckpt,
            graspnet_root=args.graspnet_root,
            pipeline_checkpoint=args.pipeline_checkpoint,
        )
        model.eval()
        model.to(device)
        print("[rewrite] 模型就绪。开始 dump（首帧会触发 CUDA/PointNet++ 初始化，可能数十秒无输出属正常）", flush=True)

        if args.num_eval_views != 13:
            print(
                "[warn] num_eval_views=%d（repro 默认为 13）。若与复现对比，请使用 13。"
                % args.num_eval_views,
                flush=True,
            )
        scene_ids = load_test_scene_ids(gc6d_root)
        if args.max_scenes and args.max_scenes > 0:
            print("[warn] max_scenes=%d：仅部分场景，AP 与全量 test 不可比。" % args.max_scenes, flush=True)
            scene_ids = scene_ids[: args.max_scenes]
        img_ids = get_eval_img_ids(args.camera, args.num_eval_views)
        os.makedirs(dump_dir, exist_ok=True)
        total_frames = len(scene_ids) * len(img_ids)
        print(
            "[rewrite] dump 共 %d 场景 × %d 视角 ≈ %d 帧，目录: %s"
            % (len(scene_ids), len(img_ids), total_frames, dump_dir),
            flush=True,
        )

        n_ok, n_skip, n_fail = 0, 0, 0
        for scene_id in scene_ids:
            sid6 = scene_id_6d(scene_id)
            out_scene_dir = os.path.join(dump_dir, sid6, args.camera)
            os.makedirs(out_scene_dir, exist_ok=True)
            print("[rewrite] scene %s: 开始写 %d 个视角..." % (sid6, len(img_ids)), flush=True)
            for img_id in img_ids:
                out_npy = os.path.join(out_scene_dir, "%06d.npy" % img_id)
                if args.skip_existing and os.path.isfile(out_npy):
                    n_skip += 1
                    continue
                try:
                    frame = load_gc6d_frame(
                        gc6d_root,
                        scene_id,
                        img_id,
                        depth_scale=args.depth_scale,
                        use_workspace=True,
                        num_point=args.num_point,
                    )
                except Exception as e:
                    print("[skip] scene %s img %s: %s" % (scene_id, img_id, e))
                    n_fail += 1
                    continue
                pc = frame["point_clouds"]
                assert pc.shape == (args.num_point, 3)
                try:
                    end_points = _forward_model(model, frame, pc, device)
                    arr = _baseline_pred_decode_and_collision(
                        end_points,
                        pc,
                        args.graspnet_root,
                        args.collision_thresh,
                        args.voxel_size,
                    )
                    _save_grasp_npy(arr, out_npy)
                    n_ok += 1
                    print(
                        "[rewrite] ok scene=%s img=%06d (%d/%d)"
                        % (sid6, img_id, n_ok, total_frames),
                        flush=True,
                    )
                except Exception as e:
                    print("[fail] scene %s img %s: %s" % (scene_id, img_id, e))
                    n_fail += 1
                    continue
            print("[dump] scene %s done" % scene_id, flush=True)
        print(
            "[dump summary] ok=%d skip=%d fail=%d dump_dir=%s"
            % (n_ok, n_skip, n_fail, dump_dir),
            flush=True,
        )

    if args.dump_only:
        print("--dump_only: 跳过 API 评估")
        return

    # ---------- API 评估（对齐 repro eval_gc6d：全量 test 优先 eval_all；否则逐 scene）----------
    with open(os.path.join(gc6d_root, "split_info", "grasp_test_scene_ids.json")) as f:
        all_test_scenes = [int(x) for x in json.load(f)]
    eval_scenes = all_test_scenes
    if args.max_scenes and args.max_scenes > 0:
        eval_scenes = all_test_scenes[: args.max_scenes]

    ge = GraspClutter6DEval(root=gc6d_root, camera=args.camera, split="test")

    collision_rem: Optional[float] = None
    fc_mean: Optional[float] = None

    use_eval_all = (
        (not args.extra_stats)
        and (len(eval_scenes) == len(all_test_scenes))
        and (args.proc is not None and int(args.proc) > 1)
    )

    if use_eval_all:
        _, ap_frac = ge.eval_all(dump_dir, proc=int(args.proc))
        ap_mean = 100.0 * float(ap_frac[0])
        ap_04 = 100.0 * float(ap_frac[1])
        ap_08 = 100.0 * float(ap_frac[2])
        eval_mode: str = "eval_all"
    else:
        acc_list = []
        all_rates: list[float] = []
        all_fcs: list[float] = []
        for sid in eval_scenes:
            acc = ge.eval_scene(
                sid,
                dump_dir,
                TOP_K=args.top_k,
                return_list=bool(args.extra_stats),
                background_filter=args.background_filter,
            )
            if args.extra_stats:
                scene_acc, grasp_ll, score_ll, collision_ll = acc
                acc_list.append(scene_acc)
                r, fc = _agg_extra_from_lists(grasp_ll, score_ll, collision_ll, args.top_k)
                all_rates.extend(r)
                all_fcs.extend(fc)
            else:
                acc_list.append(acc)
        res = np.array([[np.asarray(ann) for ann in scene_acc] for scene_acc in acc_list])
        _res = res.copy().transpose(3, 0, 1, 2).reshape(6, -1)
        _res = np.mean(_res, axis=1)
        ap_mean = 100.0 * float(np.mean(res))
        ap_04 = 100.0 * float(_res[1])
        ap_08 = 100.0 * float(_res[3])
        eval_mode = "eval_scene_per_scene" + ("_extra_stats" if args.extra_stats else "")
        if args.extra_stats and all_rates and all_fcs:
            collision_rem = float(np.mean(all_rates))
            fc_mean = float(np.mean(all_fcs))

    summary = {
        "eval_mode": eval_mode,
        "dump_dir": dump_dir,
        "gc6d_root": gc6d_root,
        "camera": args.camera,
        "top_k": args.top_k,
        "background_filter": args.background_filter,
        "collision_thresh": args.collision_thresh,
        "voxel_size": args.voxel_size,
        "num_point": args.num_point,
        "depth_scale": args.depth_scale,
        "repro_aligned": True,
        "AP": round(ap_mean, 4),
        "AP0.4": round(ap_04, 4),
        "AP0.8": round(ap_08, 4),
        "graspnet_ckpt": args.graspnet_ckpt,
        "pipeline_checkpoint": args.pipeline_checkpoint,
    }
    if collision_rem is not None:
        summary["top50_collision_remaining_rate"] = round(collision_rem, 4)
    if fc_mean is not None:
        summary["top50_force_closure_success_count_mean"] = round(fc_mean, 2)

    os.makedirs(dump_dir, exist_ok=True)
    out_path = os.path.join(dump_dir, "summary_rewrite.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "\n[Rewrite Eval] AP=%.2f  AP0.4=%.2f  AP0.8=%.2f  [%s]"
        % (ap_mean, ap_04, ap_08, eval_mode),
        flush=True,
    )
    if collision_rem is not None and fc_mean is not None:
        print(
            "  top50_collision_remaining_rate=%.4f  top50_force_closure_success_count_mean=%.2f"
            % (collision_rem, fc_mean),
            flush=True,
        )
    print("Summary: %s" % out_path)


if __name__ == "__main__":
    main()
