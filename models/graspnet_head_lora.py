# -*- coding: utf-8 -*-
"""
在 GraspNet 预训练 head（vpmodule / grasp_generator / 可选 view 末层）上注入 LoRA 旁路。
原 Conv1d/Conv2d/Linear 权重冻结，仅训练低秩适配器：y = W*x + scale * B(A(x))。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lora import LinearWithLoRA, inject_lora as inject_linear_lora

logger = logging.getLogger(__name__)


class Conv1dWithLoRA(nn.Module):
    """Conv1d + 并行 1x1 低秩旁路；原卷积冻结。"""

    def __init__(self, conv: nn.Conv1d, r: int, scale: float):
        super().__init__()
        self.base = conv
        for p in self.base.parameters():
            p.requires_grad_(False)
        ic, oc = conv.in_channels, conv.out_channels
        ks = conv.kernel_size[0]
        self.lora_down = nn.Conv1d(ic, r, kernel_size=ks, stride=conv.stride, padding=conv.padding, groups=1, bias=False)
        self.lora_up = nn.Conv1d(r, oc, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.scale * self.lora_up(self.lora_down(x))


class Conv2dWithLoRA(nn.Module):
    """Conv2d + 并行 1x1 低秩旁路；原卷积冻结。"""

    def __init__(self, conv: nn.Conv2d, r: int, scale: float):
        super().__init__()
        self.base = conv
        for p in self.base.parameters():
            p.requires_grad_(False)
        ic, oc = conv.in_channels, conv.out_channels
        kh, kw = conv.kernel_size
        self.lora_down = nn.Conv2d(
            ic, r, kernel_size=(kh, kw), stride=conv.stride, padding=conv.padding, bias=False
        )
        self.lora_up = nn.Conv2d(r, oc, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.scale * self.lora_up(self.lora_down(x))


def _lora_scale(lora_alpha: float, lora_r: int) -> float:
    return float(lora_alpha) / max(float(lora_r), 1.0)


def _replace_conv_lora(module: nn.Module, r: int, scale: float, stats: List[str]) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            wrapped = Conv1dWithLoRA(child, r=r, scale=scale)
            setattr(module, name, wrapped)
            stats.append(f"Conv1d LoRA: {name}")
        elif isinstance(child, nn.Conv2d):
            wrapped = Conv2dWithLoRA(child, r=r, scale=scale)
            setattr(module, name, wrapped)
            stats.append(f"Conv2d LoRA: {name}")
        else:
            _replace_conv_lora(child, r, scale, stats)


def inject_lora_grasp_head(
    grasp_net: nn.Module,
    *,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    target: str = "vpmodule_grasp_gen",
    inject_view_estimator_last: bool = False,
) -> Dict[str, Any]:
    """
    在 grasp_net 上注入 LoRA。
    target:
      - vpmodule_grasp_gen: view_estimator.vpmodule + grasp_generator
      - 若 inject_view_estimator_last：额外对 view_estimator 子树中 Linear 做 inject（与 models.lora.inject_lora 一致）
    返回注入信息 dict。
    """
    scale = _lora_scale(lora_alpha, lora_r)
    stats: List[str] = []
    ve = grasp_net.view_estimator

    if target in ("vpmodule_grasp_gen", "all"):
        _replace_conv_lora(ve.vpmodule, lora_r, scale, stats)
        _replace_conv_lora(grasp_net.grasp_generator, lora_r, scale, stats)
        # Linear 层（ApproachNet / 等）
        inject_linear_lora(ve.vpmodule, r=lora_r, scale=scale, log_injected=False)
        inject_linear_lora(grasp_net.grasp_generator, r=lora_r, scale=scale, log_injected=False)
        stats.append("Linear LoRA (vpmodule + grasp_generator)")

    if inject_view_estimator_last:
        # 仅 backbone 中带 blocks 的模块：最后若干 block 的 Linear
        bb = getattr(ve, "backbone", None)
        if bb is not None:
            inject_linear_lora(bb, r=lora_r, scale=scale, last_n_blocks=2, log_injected=True)
            stats.append("view_estimator.backbone Linear LoRA (last_n_blocks=2)")

    freeze_all_original_grasp_params(grasp_net)

    if logger.isEnabledFor(logging.INFO):
        logger.info("[graspnet_head_lora] injected: %s", "; ".join(stats[:20]) + ("..." if len(stats) > 20 else ""))
    return {"lora_r": lora_r, "lora_alpha": lora_alpha, "scale": scale, "stats": stats}


def freeze_all_original_grasp_params(grasp_net: nn.Module) -> None:
    """冻结所有非 LoRA 参数（含 backbone）；可训练 LoRA 旁路参数。"""
    for n, p in grasp_net.named_parameters():
        if "lora_" in n or "lora_down" in n or "lora_up" in n or "lora_A" in n or "lora_B" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def unfreeze_lora_only_trainable(grasp_net: nn.Module) -> None:
    """与 freeze_all_original_grasp_params 相同语义，显式调用。"""
    freeze_all_original_grasp_params(grasp_net)


def parameter_train_stats(module: nn.Module) -> Tuple[int, int, float]:
    """(total, trainable, pct_trainable)."""
    total = 0
    train = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            train += n
    pct = 100.0 * float(train) / max(total, 1)
    return total, train, pct


def print_module_train_stats(tag: str, module: nn.Module) -> None:
    t, tr, pct = parameter_train_stats(module)
    print(f"[{tag}] total_params={t:,} trainable_params={tr:,} ({pct:.4f}%)")
