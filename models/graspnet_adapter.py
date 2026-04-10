# -*- coding: utf-8 -*-
"""
Encoder + Adapter + 预训练 GraspNet（graspnet-baseline）head。
- 输入：LIFT3D 用 point_cloud；VGGT 用 images + point_cloud（GraspNet 需点云）。
-  conditioning：encoder 输出 -> adapter(256) -> 加到 GraspNet backbone 的 seed_features 上。
- 训练用 17D matching loss（GC6D 数据），不依赖 GraspNet 的 process_grasp_labels。
"""

import importlib.util
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 baseline loss_utils 一致，可微 17D 解码用
GRASP_MAX_WIDTH = 0.1
GRASP_MAX_TOLERANCE = 0.05
NUM_ANGLE = 12
NUM_DEPTH = 4


def _batch_viewpoint_params_to_matrix(batch_towards: torch.Tensor, batch_angle: torch.Tensor) -> torch.Tensor:
    """可微：approach 向量 + 转角 -> 旋转矩阵 (N,3,3)。与 baseline loss_utils 一致。"""
    axis_x = batch_towards
    axis_x = axis_x / (axis_x.norm(dim=-1, keepdim=True) + 1e-8)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = (axis_y.norm(dim=-1) == 0)
    axis_y[mask_y, 1] = 1
    axis_y = axis_y / (axis_y.norm(dim=-1, keepdim=True) + 1e-8)
    axis_z = torch.linalg.cross(axis_x, axis_y, dim=-1)
    axis_z = axis_z / (axis_z.norm(dim=-1, keepdim=True) + 1e-8)
    sin_a = torch.sin(batch_angle)
    cos_a = torch.cos(batch_angle)
    ones = torch.ones_like(sin_a)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos_a, -sin_a, zeros, sin_a, cos_a], dim=-1).reshape(-1, 3, 3)
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    return torch.matmul(R2, R1)

# graspnet-baseline 路径（默认与 gc6d_grasp_pipeline 并列）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRASPNET_BASELINE = os.path.abspath(os.path.join(ROOT, "..", "graspnet-baseline"))

_baseline_graspnet_module = None


def _load_baseline_graspnet_module(root: str):
    """用 importlib 从 baseline/models/graspnet.py 直接加载，不经过 'models' 包名。"""
    global _baseline_graspnet_module
    if _baseline_graspnet_module is not None:
        return _baseline_graspnet_module
    root = os.path.abspath(os.path.expanduser(root))
    models_dir = os.path.join(root, "models")
    graspnet_py = os.path.join(models_dir, "graspnet.py")
    if not os.path.isfile(graspnet_py):
        raise FileNotFoundError(f"graspnet-baseline graspnet.py not found: {graspnet_py}")
    # models_dir 放最前，这样 graspnet.py 里 "from backbone import" 等能直接找到 models/backbone
    path_add = [models_dir, root]
    for sub in ("utils", "pointnet2", "knn"):
        p = os.path.join(root, sub)
        if os.path.isdir(p):
            path_add.append(p)
    saved_path = list(sys.path)
    for p in path_add:
        if p in sys.path:
            sys.path.remove(p)
    for i, p in enumerate(path_add):
        sys.path.insert(i, p)
    try:
        spec = importlib.util.spec_from_file_location("_graspnet_baseline_graspnet", graspnet_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {graspnet_py}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _baseline_graspnet_module = mod
        return mod
    finally:
        sys.path[:] = saved_path


def load_graspnet_pretrained(
    checkpoint_path: str,
    device: torch.device,
    graspnet_root: Optional[str] = None,
    is_training: bool = False,
) -> nn.Module:
    """加载 baseline 的 GraspNet，is_training=False 以便推理/17D 解码（不跑 process_grasp_labels）。"""
    root = (graspnet_root or os.environ.get("GRASPNET_BASELINE", GRASPNET_BASELINE))
    root = os.path.abspath(os.path.expanduser(root))
    mod = _load_baseline_graspnet_module(root)
    GraspNet = mod.GraspNet

    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=is_training,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    net.load_state_dict(state, strict=False)
    return net.to(device)


def raw_pred_decode_num_grasps(end_points) -> int:
    """pred_decode 后单 batch 的 grasp 条数（objectness 过滤后），用于 debug 对齐。"""
    root = os.environ.get("GRASPNET_BASELINE", GRASPNET_BASELINE)
    root = os.path.abspath(os.path.expanduser(root))
    mod = _load_baseline_graspnet_module(root)
    grasp_preds: List[torch.Tensor] = mod.pred_decode(end_points)
    if not grasp_preds or grasp_preds[0] is None:
        return 0
    g = grasp_preds[0]
    return int(g.shape[0]) if g.dim() > 0 else 0


def pred_decode_17d(end_points, device: torch.device, max_grasps: int = 128) -> torch.Tensor:
    """从 GraspNet end_points 解码为 (B, K, 17)，K 为 pad 后的数量（推理用，不可微）。

    与 graspnet-baseline 的 pred_decode 一致：先得到全部 objectness 过滤后的 grasp，
    **再按 score（第 0 列）降序排序后** 取前 max_grasps 条。若直接取 g[:k] 会按 seed 顺序截断，
    会丢掉高分抓取，与 baseline_infer / repro dump 流程不一致。
    """
    root = os.environ.get("GRASPNET_BASELINE", GRASPNET_BASELINE)
    root = os.path.abspath(os.path.expanduser(root))
    mod = _load_baseline_graspnet_module(root)
    grasp_preds: List[torch.Tensor] = mod.pred_decode(end_points)

    B = len(grasp_preds)
    out = torch.zeros(B, max_grasps, 17, dtype=torch.float32, device=device)
    for i in range(B):
        g = grasp_preds[i]
        if g is None or g.numel() == 0:
            continue
        if g.dim() == 1:
            g = g.unsqueeze(0)
        g = g.to(device)
        scores = g[:, 0]
        order = torch.argsort(scores, descending=True)
        g = g[order]
        k = min(g.shape[0], max_grasps)
        out[i, :k] = g[:k]
    return out


def apply_model_free_collision_filter(
    gg_array: np.ndarray,
    scene_points: np.ndarray,
    collision_thresh: float = 0.01,
    voxel_size: float = 0.01,
    approach_dist: float = 0.05,
    graspnet_baseline_root: Optional[str] = None,
) -> np.ndarray:
    """与 gc6d_graspnet_repro 的 baseline_infer 一致：dump 前用 GraspNet 点云碰撞检测过滤。

    依赖 graspnet-baseline 的 ``collision_detector.ModelFreeCollisionDetector``。
    Grasp 容器优先 ``graspnetAPI.GraspGroup``（与 baseline_infer 一致）；若未安装则回退到
    ``graspclutter6dAPI.grasp.GraspGroup``（17D 属性相同，碰撞逻辑可复用）。
    collision_thresh<=0 时跳过，原样返回。
    """
    if gg_array is None or gg_array.size == 0:
        return gg_array
    if collision_thresh is None or float(collision_thresh) <= 0:
        return gg_array

    root = graspnet_baseline_root or os.environ.get("GRASPNET_BASELINE", GRASPNET_BASELINE)
    root = os.path.abspath(os.path.expanduser(root))
    utils_dir = os.path.join(root, "utils")
    if not os.path.isdir(utils_dir):
        raise FileNotFoundError(f"graspnet-baseline utils not found: {utils_dir}")

    saved_path = list(sys.path)
    try:
        if root not in sys.path:
            sys.path.insert(0, root)
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        from collision_detector import ModelFreeCollisionDetector
        try:
            from graspnetAPI import GraspGroup as GraspGroupGN
        except ImportError:
            from graspclutter6dAPI.grasp import GraspGroup as GraspGroupGN
    finally:
        sys.path[:] = saved_path

    cloud = np.asarray(scene_points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"scene_points expected (N,3), got {cloud.shape}")
    arr = np.asarray(gg_array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 17:
        raise ValueError(f"gg_array expected (M,17), got {arr.shape}")

    gg = GraspGroupGN(arr)
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=float(voxel_size))
    collision_mask = mfcdetector.detect(gg, approach_dist=float(approach_dist), collision_thresh=float(collision_thresh))
    gg_f = gg[~collision_mask]
    return gg_f.grasp_group_array


def _compute_differentiable_seed_state(end_points: dict) -> dict:
    """
    与 pred_decode_17d_differentiable 内部一致的可微 per-seed 聚合（不含排序/截断）。
    返回 dict：score_val, width_val, tol_val, depth_val, R_flat, center, num_seed, dev。
    """
    obj_score = end_points["objectness_score"].float()
    grasp_score = end_points["grasp_score_pred"].float()
    angle_cls = end_points["grasp_angle_cls_pred"].float()
    width_pred = end_points["grasp_width_pred"].float()
    tolerance_pred = end_points["grasp_tolerance_pred"].float()
    center = end_points["fp2_xyz"].float()
    approach = -end_points["grasp_top_view_xyz"].float()

    B, num_angle, num_seed, num_depth = grasp_score.shape
    dev = grasp_score.device

    angle_cls_per = angle_cls.mean(dim=3)
    angle_weights = F.softmax(angle_cls_per, dim=1)
    angle_vals = (torch.arange(num_angle, device=dev, dtype=torch.float32) * (3.14159265 / num_angle)).view(1, num_angle, 1)
    angle_val = (angle_weights * angle_vals).sum(dim=1)

    score_per_depth = (angle_weights.unsqueeze(3) * grasp_score).sum(dim=1)
    depth_weights = F.softmax(score_per_depth, dim=2)
    depth_vals = (torch.arange(1, num_depth + 1, device=dev, dtype=torch.float32) * 0.01).view(1, 1, num_depth)
    depth_val = (depth_weights * depth_vals).sum(dim=2)

    score_w = (angle_weights.unsqueeze(3) * grasp_score).sum(dim=1)
    score_val = (depth_weights * score_w).sum(dim=2)
    tol_w = (angle_weights.unsqueeze(3) * tolerance_pred).sum(dim=1)
    tol_val = (depth_weights * tol_w).sum(dim=2)
    score_val = score_val * (tol_val / GRASP_MAX_TOLERANCE).clamp(0, 1)
    p_obj = F.softmax(obj_score, dim=1)[:, 1, :]
    score_val = score_val * (p_obj + 1e-8)

    width_w = (angle_weights.unsqueeze(3) * width_pred).sum(dim=1)
    width_val = (depth_weights * width_w).sum(dim=2)
    width_val = (1.2 * width_val).clamp(0, GRASP_MAX_WIDTH)

    approach_flat = approach.reshape(-1, 3)
    angle_flat = angle_val.reshape(-1)
    R = _batch_viewpoint_params_to_matrix(approach_flat, angle_flat).reshape(B, num_seed, 3, 3)
    R_flat = R.reshape(B, num_seed, 9)

    return {
        "score_val": score_val,
        "width_val": width_val,
        "tol_val": tol_val,
        "depth_val": depth_val,
        "R_flat": R_flat,
        "center": center,
        "num_seed": num_seed,
        "dev": dev,
    }


def per_seed_soft_scalars_from_end_points(end_points: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """与可微解码一致的 (score, width, tolerance)，形状均为 (B, num_seed)。供 reranker / ranking loss 使用。"""
    st = _compute_differentiable_seed_state(end_points)
    return st["score_val"], st["width_val"], st["tol_val"]


def pred_decode_17d_differentiable(
    end_points: dict, device: torch.device, max_grasps: int = 128, sort_and_truncate: bool = True
) -> torch.Tensor:
    """可微 17D 解码，用于训练 backward。用 softmax 替代 argmax，输出 (B, K, 17)。
    与 baseline 对齐：score 乘以 objectness 概率，使背景 seed 得分压低（baseline 只保留 objectness==1）。
    sort_and_truncate=False 时不做按 score 排序与截断（实验 1：训练时不排序，仅 eval 时排序）。"""
    st = _compute_differentiable_seed_state(end_points)
    score_val = st["score_val"]
    width_val = st["width_val"]
    depth_val = st["depth_val"]
    R_flat = st["R_flat"]
    center = st["center"]
    num_seed = st["num_seed"]
    dev = st["dev"]
    B = score_val.shape[0]

    height_val = 0.02
    out = torch.cat([
        score_val.unsqueeze(2),
        width_val.unsqueeze(2),
        torch.full_like(score_val.unsqueeze(2), height_val, device=dev),
        depth_val.unsqueeze(2),
        R_flat,
        center,
        torch.full_like(score_val.unsqueeze(2), -1.0, device=dev),
    ], dim=2)  # (B, num_seed, 17)

    # 与 benchmark 一致：按 score 降序排序后取 top max_grasps，使训练优化的「top-K」与评估用的 top-K 一致
    if sort_and_truncate:
        sort_idx = torch.argsort(out[:, :, 0], dim=1, descending=True)  # (B, num_seed)
        out = torch.gather(out, 1, sort_idx.unsqueeze(2).expand(-1, -1, 17))
        if num_seed >= max_grasps:
            out = out[:, :max_grasps]
        else:
            pad = out.new_zeros(B, max_grasps - num_seed, 17)
            out = torch.cat([out, pad], dim=1)
    else:
        if num_seed >= max_grasps:
            out = out[:, :max_grasps]
        else:
            pad = out.new_zeros(B, max_grasps - num_seed, 17)
            out = torch.cat([out, pad], dim=1)
    return out


class EncoderAdapterGraspNet(nn.Module):
    """
    Encoder (LIFT3D / VGGT) -> adapter -> cond(256)；
    point_cloud -> GraspNet backbone -> seed_features + cond -> ApproachNet -> Stage2。
    use_adapter=False 时不加 conditioning（head-only baseline）。
    adapter_cond_mode: additive | concat | gated | film。
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_feat_dim: int,
        grasp_net: nn.Module,
        adapter: Optional[nn.Module] = None,
        use_adapter: bool = True,
        adapter_cond_coeff: float = 2.0,
        adapter_cond_mode: str = "additive",
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_feat_dim = encoder_feat_dim
        self.grasp_net = grasp_net
        self.use_adapter = use_adapter
        self.adapter_cond_coeff = adapter_cond_coeff
        self.adapter_cond_mode = adapter_cond_mode
        if adapter is None and use_adapter:
            adapter = nn.Sequential(
                nn.Linear(encoder_feat_dim, 256),
                nn.ReLU(inplace=True),
            )
        self.adapter = adapter
        self._encoder_type = getattr(encoder, "encoder_type", "unknown")
        if use_adapter and adapter_cond_mode == "concat":
            # (B,512,S) -> (B,256,S)；与逐 seed 的 Linear 等价，用 1x1 Conv1d 在 seed 维混合通道
            self.concat_proj = nn.Conv1d(512, 256, kernel_size=1, bias=True)
        else:
            self.concat_proj = None
        if use_adapter and adapter_cond_mode == "gated":
            self.cond_gate = nn.Linear(256, 1)
        elif use_adapter and adapter_cond_mode == "film":
            self.film_proj = nn.Linear(256, 512)
        else:
            self.cond_gate = None
            self.film_proj = None

    def forward(
        self,
        point_cloud: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        point_cloud: (B, N, 3) 必选（GraspNet 需要）
        images: (B, 3, 224, 224) 仅 VGGT 时使用
        return: end_points（含 objectness_score, grasp_score_pred, fp2_xyz 等，供 pred_decode_17d）
        """
        B = point_cloud.shape[0]
        device = point_cloud.device

        if self.use_adapter:
            if images is not None:
                enc_feat = self.encoder(images)
            else:
                if getattr(self, "_encoder_type", None) in ("vggt_base", "vggt_ft"):
                    raise ValueError(
                        "VGGT encoder 需要 RGB images (B,3,224,224)，不能用 point_cloud 代替；"
                        "评估/训练前向请传入与点云对齐的 images。"
                    )
                enc_feat = self.encoder(point_cloud)
            cond = self.adapter(enc_feat.float())
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            cond = cond[:, :256]
            if cond.shape[1] < 256:
                cond = torch.cat([cond, cond.new_zeros(B, 256 - cond.shape[1])], dim=1)
        else:
            cond = point_cloud.new_zeros(B, 256)

        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        backbone = view_estimator.backbone
        seed_features, seed_xyz, end_points = backbone(point_cloud, end_points)
        # conditioning: additive | concat | gated | film
        c = self.adapter_cond_coeff * cond.unsqueeze(2)  # (B, 256, 1)
        if self.use_adapter and self.adapter_cond_mode == "concat" and self.concat_proj is not None:
            cond_expand = c.expand(-1, -1, seed_features.shape[2])  # (B,256,S)
            sf_cat = torch.cat([seed_features, cond_expand], dim=1)  # (B,512,S)
            seed_features = self.concat_proj(sf_cat)  # Conv1d: (B,C,L) -> (B,256,S)
        elif self.use_adapter and self.adapter_cond_mode == "gated" and self.cond_gate is not None:
            gate = torch.sigmoid(self.cond_gate(cond)).unsqueeze(2)  # (B, 1, 1)
            seed_features = seed_features + gate * c
        elif self.use_adapter and self.adapter_cond_mode == "film" and self.film_proj is not None:
            gb = self.film_proj(cond)
            half = gb.shape[1] // 2
            gamma, beta = gb[:, :half].unsqueeze(2), gb[:, half:].unsqueeze(2)
            if gamma.shape[1] == seed_features.shape[1]:
                seed_features = gamma * seed_features + beta
            else:
                seed_features = seed_features + c
        else:
            seed_features = seed_features + c
        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        # 返回 cond 供训练时打通到 adapter 的梯度（Stage1 仅训 adapter 时 grad 易为 0，见 run.log）
        end_points["_cond"] = cond
        return end_points


def build_encoder_adapter_graspnet(
    encoder_type: str,
    graspnet_ckpt: str,
    encoder_feat_dim: int = 256,
    graspnet_root: Optional[str] = None,
    lift3d_root: Optional[str] = None,
    lift3d_ckpt: Optional[str] = None,
    vggt_ckpt: Optional[str] = None,
    lora_r: int = 8,
    lora_scale: float = 1.0,
    lora_last_n_blocks: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
    use_adapter: bool = True,
    adapter_cond_coeff: float = 2.0,
    adapter_cond_mode: str = "additive",
) -> EncoderAdapterGraspNet:
    """构建 Encoder + Adapter + GraspNet。encoder_type: lift3d | lift3d_clip | vggt_base | vggt_ft。
    vggt_base：仅加载预训练权重，用于 benchmark 不微调；vggt_ft：加载微调 ckpt，用于继续训或评估。
    lora_last_n_blocks：仅对 backbone 最后 N 个 block 注入 LoRA（None=全注入），VGGT 建议 2~4。
    """
    if encoder_type == "lift3d":
        from .lift3d_encoder import LIFT3DEncoder
        lift3d_root = lift3d_root or os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
        encoder = LIFT3DEncoder(
            lift3d_root=lift3d_root,
            feat_dim=encoder_feat_dim,
            use_lora=True,
            lora_r=lora_r,
            lora_scale=lora_scale,
            lora_last_n_blocks=lora_last_n_blocks,
            ckpt_path=lift3d_ckpt,
        )
    elif encoder_type == "lift3d_clip":
        from .lift3d_clip_encoder import LIFT3DClipEncoder
        lift3d_root = lift3d_root or os.environ.get("LIFT3D_ROOT", os.path.expanduser("~/LIFT3D"))
        encoder = LIFT3DClipEncoder(
            lift3d_root=lift3d_root,
            feat_dim=encoder_feat_dim,
            freeze_backbone=True,
            normalize_pc=True,
            lora_r=lora_r,
            lora_scale=lora_scale,
        )
    elif encoder_type == "vggt_base":
        from .vggt_encoder import VGGTEncoder
        # base：预训练权重，不微调 encoder 时用于评估；LoRA 仍注入便于后续 ft 加载同一结构
        encoder = VGGTEncoder(
            feat_dim=encoder_feat_dim,
            freeze_backbone=True,
            ckpt_path=vggt_ckpt,
            lora_r=lora_r,
            lora_scale=lora_scale,
            lora_last_n_blocks=lora_last_n_blocks,
        )
    elif encoder_type == "vggt_ft":
        from .vggt_encoder import VGGTEncoder
        # ft：微调后的 encoder checkpoint，用于继续训练或评估
        encoder = VGGTEncoder(
            feat_dim=encoder_feat_dim,
            freeze_backbone=True,
            ckpt_path=vggt_ckpt,
            lora_r=lora_r,
            lora_scale=lora_scale,
            lora_last_n_blocks=lora_last_n_blocks,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    adapter = nn.Sequential(
        nn.Linear(encoder_feat_dim, 256),
        nn.ReLU(inplace=True),
    ) if use_adapter else None
    model = EncoderAdapterGraspNet(
        encoder=encoder,
        encoder_feat_dim=encoder_feat_dim,
        grasp_net=grasp_net,
        adapter=adapter,
        use_adapter=use_adapter,
        adapter_cond_coeff=adapter_cond_coeff,
        adapter_cond_mode=adapter_cond_mode,
    )
    model.encoder_type = encoder_type
    return model
