# -*- coding: utf-8 -*-
"""Encoder → GraspNet head 对齐用的 1×1 Conv 投影（两阶段）。"""

from __future__ import annotations

import torch.nn as nn


def make_conv1d_projector(c_in: int, c_out: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv1d(c_out, c_out, kernel_size=1, bias=True),
    )


def make_fusion_mlp_concat512(c_out: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(512, c_out, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv1d(c_out, c_out, kernel_size=1, bias=True),
    )
