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
) -> PureGraspNetPipeline:
    import torch as _t

    if device is None:
        device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    grasp_net = load_graspnet_pretrained(graspnet_ckpt, device, graspnet_root, is_training=False)
    model = PureGraspNetPipeline(
        grasp_net=grasp_net,
        score_calibration_mode=score_calibration_mode,
        score_delta_scale=score_delta_scale,
        score_calibration_hidden=score_calibration_hidden,
    )
    return model.to(device)

