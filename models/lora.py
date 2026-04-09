# -*- coding: utf-8 -*-
"""
简易 LoRA：对 nn.Linear 注入低秩旁路，仅训练 LoRA 参数，原 linear 权重冻结。
用于 LIFT3D / VGGT encoder 在 GC6D 上微调。
"""

import re
import logging
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class LinearWithLoRA(nn.Module):
    """Wraps nn.Linear with LoRA: out = Wx + scale * (x @ B^T @ A^T). 原 linear 权重冻结，只训 lora_A/lora_B。"""

    def __init__(self, linear: nn.Linear, r: int = 8, scale: float = 1.0):
        super().__init__()
        self.linear = linear
        self.scale = scale
        in_features = linear.in_features
        out_features = linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(out_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        # 标准 LoRA：冻结原权重，只训低秩增量
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = out + self.scale * (x @ self.lora_B.T @ self.lora_A.T)
        return out


def _collect_linear_full_paths(module: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Module, str, nn.Linear]]:
    """(full_path, parent_module, child_name, linear_module)."""
    out: List[Tuple[str, nn.Module, str, nn.Linear]] = []
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            out.append((full, module, name, child))
        else:
            out.extend(_collect_linear_full_paths(child, full))
    return out


def _block_index_from_path(full_path: str) -> Optional[int]:
    """从完整路径解析 block 索引，如 'blocks.3.xxx' -> 3。若无 blocks.(\d+) 则返回 None。"""
    m = re.search(r"blocks\.(\d+)", full_path)
    return int(m.group(1)) if m else None


def inject_lora(
    module: nn.Module,
    r: int = 8,
    scale: float = 1.0,
    target_module_names: Optional[List[str]] = None,
    last_n_blocks: Optional[int] = None,
    block_name_pattern: str = "blocks",
    log_injected: bool = True,
) -> List[Tuple[str, nn.Module]]:
    """
    对 module 中 nn.Linear 替换为 LinearWithLoRA；原 linear 在 LinearWithLoRA 内已冻结。
    - target_module_names: 若为 None 则按 path 过滤；否则只对「局部 name」在该列表中的注入（兼容旧逻辑）。
    - last_n_blocks: 若给定，只对「完整路径中 blocks.(\d+) 属于最后 last_n_blocks 个 block」的 Linear 注入。
    - log_injected: 是否打印被注入的层完整路径。
    返回被替换的 (full_path, lora_module) 列表。
    """
    linear_list = _collect_linear_full_paths(module)
    if not linear_list:
        return []

    block_indices = [_block_index_from_path(full) for full, _, _, _ in linear_list]
    max_block = max((b for b in block_indices if b is not None), default=None)
    thresh = None
    if last_n_blocks is not None and last_n_blocks > 0 and max_block is not None:
        thresh = max_block - last_n_blocks + 1

    replaced: List[Tuple[str, nn.Module]] = []
    for (full_path, parent, name, linear), bidx in zip(linear_list, block_indices):
        # 仅当存在 block 编号且指定了 last_n_blocks 时按 block 过滤；否则全部注入
        if thresh is not None:
            if bidx is None or bidx < thresh:
                continue
        if target_module_names is not None and name not in target_module_names:
            continue
        lora_linear = LinearWithLoRA(linear, r=r, scale=scale)
        setattr(parent, name, lora_linear)
        replaced.append((full_path, lora_linear))

    if log_injected and replaced:
        logger.info("[LoRA] 注入 %d 层 (last_n_blocks=%s): %s", len(replaced), last_n_blocks, [p for p, _ in replaced])
    return replaced


def get_lora_params(module: nn.Module) -> List[nn.Parameter]:
    """返回所有 LoRA 相关参数（lora_A, lora_B）。"""
    return [p for n, p in module.named_parameters() if "lora_" in n]


def get_lora_params_from_last_n_blocks(module: nn.Module, n_blocks: int) -> List[nn.Parameter]:
    """
    只返回「最后 n_blocks 个 block」内的 LoRA 参数，用于 Stage3 降强度（不要全开）。
    通过 param 名字中的 blocks.(\d+) 解析 block 索引，取索引最大的 n_blocks 个 block 的 lora 参数。
    """
    import re
    lora_with_names = [(n, p) for n, p in module.named_parameters() if "lora_" in n]
    if not lora_with_names or n_blocks <= 0:
        return []
    block_indices = {}
    for name, p in lora_with_names:
        m = re.search(r"blocks\.(\d+)", name)
        idx = int(m.group(1)) if m else 0
        block_indices[id(p)] = idx
    max_idx = max(block_indices.values())
    thresh = max_idx - n_blocks + 1
    return [p for n, p in lora_with_names if block_indices[id(p)] >= thresh]


def get_non_lora_params(module: nn.Module) -> List[nn.Parameter]:
    """返回所有非 LoRA 参数。"""
    return [p for n, p in module.named_parameters() if "lora_" not in n]
