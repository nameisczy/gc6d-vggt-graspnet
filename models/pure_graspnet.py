# -*- coding: utf-8 -*-
"""
Pure GraspNet 模式（当前 pipeline 内）：
point cloud -> GraspNet backbone -> vpmodule -> grasp_generator

不包含 encoder / adapter / conditioning。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .graspnet_adapter import load_graspnet_pretrained


class PureGraspNetPipeline(nn.Module):
    """仅保留 GraspNet 主干与 head，不做任何 cond。"""

    def __init__(
        self,
        grasp_net: nn.Module,
        *,
        score_calibration_mode: str = "none",
        score_delta_scale: float = 0.1,
        score_calibration_hidden: int = 12,
        reranker: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.grasp_net = grasp_net
        self.encoder = None
        self.adapter = None
        self.use_adapter = False
        self.adapter_cond_coeff = 0.0
        self.adapter_cond_mode = "none"
        self.encoder_type = "pure_graspnet"
        self.model_mode = "pure_graspnet"
        self.requires_images = False
        self.score_calibration_mode = str(score_calibration_mode)
        self.score_delta_scale = float(score_delta_scale)
        self.score_calibration_hidden = int(score_calibration_hidden)
        if self.score_calibration_mode == "residual":
            h = int(score_calibration_hidden)
            self.score_calibration_head = nn.Sequential(
                nn.Conv1d(12, h, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(h, 12, kernel_size=1),
            )
        else:
            self.score_calibration_head = None
        # 训练时由外部 loss 使用；forward 不调用，保持与旧 checkpoint 兼容
        self.reranker = reranker
        self.reranker_extended_features = getattr(reranker, "in_dim", 0) >= 9 if reranker is not None else False

    def forward(self, point_cloud: torch.Tensor, images: Optional[torch.Tensor] = None) -> dict:
        if images is not None:
            raise ValueError("PureGraspNetPipeline 仅支持点云输入")
        end_points = {"point_clouds": point_cloud}
        view_estimator = self.grasp_net.view_estimator
        seed_features, seed_xyz, end_points = view_estimator.backbone(point_cloud, end_points)
        end_points = view_estimator.vpmodule(seed_xyz, seed_features, end_points)
        end_points = self.grasp_net.grasp_generator(end_points)
        if self.score_calibration_head is not None:
            score = end_points["grasp_score_pred"]  # (B, A, Ns, D)
            B, A, Ns, D = score.shape
            x = score.reshape(B, A, Ns * D)
            delta = self.score_calibration_head(x).reshape(B, A, Ns, D)
            end_points["grasp_score_pred_raw_pretrained"] = score
            end_points["grasp_score_delta"] = torch.tanh(delta) * self.score_delta_scale
            end_points["grasp_score_pred"] = score + end_points["grasp_score_delta"]
        return end_points


def build_pure_graspnet_pipeline(
    *,
    graspnet_ckpt: str,
    graspnet_root: Optional[str] = None,
    score_calibration_mode: str = "none",
    score_delta_scale: float = 0.1,
    score_calibration_hidden: int = 12,
    device: Optional[torch.device] = None,
    use_lora_head: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    inject_view_estimator_last: bool = False,
    reranker_enabled: bool = False,
    reranker_extended_features: bool = True,
) -> PureGraspNetPipeline:
    import torch as _t

    if device is None:
        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    reranker: Optional[nn.Module] = None
    if reranker_enabled:
        from .reranker import ResidualReranker

        if reranker_extended_features:
            reranker = ResidualReranker(in_dim=9, hidden1=128, hidden2=64)
        else:
            reranker = ResidualReranker(in_dim=3, hidden1=64, hidden2=32)
    model = PureGraspNetPipeline(
        grasp_net=grasp_net,
        score_calibration_mode=score_calibration_mode,
        score_delta_scale=score_delta_scale,
        score_calibration_hidden=score_calibration_hidden,
        reranker=reranker,
    )
    if use_lora_head:
        from .graspnet_head_lora import inject_lora_grasp_head, print_module_train_stats

        inject_lora_grasp_head(
            model.grasp_net,
            lora_r=lora_r,
            lora_alpha=float(lora_alpha),
            inject_view_estimator_last=inject_view_estimator_last,
        )
        print_module_train_stats("pure_graspnet.grasp_net", model.grasp_net)
    if reranker is not None:
        from .graspnet_head_lora import print_module_train_stats

        print_module_train_stats("pure_graspnet.reranker", reranker)
    from .graspnet_head_lora import parameter_train_stats

    tot, trn, pct = parameter_train_stats(model)
    print(f"[pure_graspnet pipeline] total_params={tot:,} trainable_params={trn:,} ({pct:.4f}%)")
    return model.to(device)

