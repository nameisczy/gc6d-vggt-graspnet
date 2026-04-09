# -*- coding: utf-8 -*-
"""
VGGT 图像 encoder 封装：RGB (B,3,224,224) -> (B,768) -> 适配器 -> (B, feat_dim)。
支持 LoRA 微调（与 LIFT3D 配置对齐：lora_r, lora_scale）；Stage1 仅训 adapter+head，Stage2 仅训 backbone LoRA，Stage3 联合。
"""

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn

from .lora import inject_lora, get_lora_params, get_lora_params_from_last_n_blocks, get_non_lora_params


def _ensure_vggt_on_path() -> None:
    """使 ``vggt.models.vggt`` / ``vggt`` 可导入：优先已安装包，否则把 VGGT_ROOT 加入 sys.path。"""
    try:
        import vggt.models.vggt  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import vggt  # noqa: F401
        return
    except ImportError:
        pass
    root = os.path.abspath(os.path.expanduser(os.environ.get("VGGT_ROOT", os.path.expanduser("~/vggt"))))
    pkg = os.path.join(root, "vggt", "__init__.py")
    if os.path.isfile(pkg) and root not in sys.path:
        sys.path.insert(0, root)


def _normalize_vggt_ckpt_path(ckpt_path: Optional[str]) -> Optional[str]:
    """CLI 常误传字面量 'None'；空串 / none / __NONE__ 视为使用预训练权重。"""
    if ckpt_path is None:
        return None
    s = str(ckpt_path).strip()
    if not s or s.lower() == "none" or s == "__NONE__":
        return None
    return s


def _load_vggt_backbone(ckpt_path: Optional[str] = None):
    """
    加载 VGGT backbone。预训练：``VGGT.from_pretrained("facebook/VGGT-1B")``。
    ckpt_path 非空且非 "__NONE__" 时再 load_state_dict（微调权重）。
    """
    ckpt_path = _normalize_vggt_ckpt_path(ckpt_path)
    _ensure_vggt_on_path()
    VGGT = None
    try:
        from vggt.models.vggt import VGGT as _VGGT  # 推荐：与仓库结构一致

        VGGT = _VGGT
    except ImportError:
        try:
            from vggt import VGGT as _VGGT2  # 兼容旧式包导出

            VGGT = _VGGT2
        except ImportError as e:
            raise ImportError(
                "无法导入 VGGT（需要 vggt.models.vggt.VGGT 或 vggt.VGGT）。请任选其一：\n"
                "  1) cd <VGGT 仓库> && pip install -e .\n"
                "  2) 设置 VGGT_ROOT 指向含 vggt/ 的仓库根。\n"
                f"  当前 VGGT_ROOT={os.environ.get('VGGT_ROOT', '(默认 ~/vggt)')}"
            ) from e
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    if ckpt_path:
        ckpt_path = os.path.expanduser(ckpt_path)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[VGGT] loaded {ckpt_path} missing={len(miss)} unexpected={len(unexp)}")
    return model


class VGGTEncoder(nn.Module):
    """
    VGGT 图像 encoder + LoRA（可选）+ 768->feat_dim 适配器。
    forward(images) -> (B, feat_dim)。images: (B, 3, 224, 224)。
    - Stage1: freeze_backbone=True，只训 adapter + head（LoRA 不训）。
    - Stage2: 仅训 backbone 内 LoRA（set_backbone_lora_trainable(True)）。
    - Stage3: 训 LoRA + adapter + head。
    """

    def __init__(
        self,
        feat_dim: int = 256,
        freeze_backbone: bool = True,
        ckpt_path: Optional[str] = None,
        sample_k: int = 1024,
        lora_r: int = 8,
        lora_scale: float = 1.0,
        lora_last_n_blocks: Optional[int] = None,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.sample_k = int(sample_k)
        self.adapter_dropout = float(adapter_dropout)
        self.backbone = _load_vggt_backbone(ckpt_path)
        inject_lora(self.backbone, r=lora_r, scale=lora_scale, last_n_blocks=lora_last_n_blocks)
        self.backbone.eval()
        # VGGT 输出 world_points 经 pt_mlp 后 mean -> 768
        self.pt_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 768),
        )
        self.adapter = nn.Sequential(
            nn.Linear(768, feat_dim),
            nn.ReLU(inplace=True),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.pt_mlp.parameters():
                p.requires_grad = False

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,224,224) -> (B,768)."""
        if images.dim() == 4:
            images_v = images.unsqueeze(1)
        else:
            images_v = images
        out = self.backbone(images_v)
        wp = out["world_points"]
        conf = out["world_points_conf"]
        if wp.ndim != 5:
            raise ValueError(
                "VGGT backbone 的 world_points 维数异常: shape=%s（期望 5 维 B×V×H×W×3）。"
                "通常是因为输入不是 RGB 图像 (B,3,224,224)，例如误把点云 (B,N,3) 传给了图像 encoder。"
                % (tuple(wp.shape),)
            )
        B, V, H, W, _ = wp.shape
        wp = wp.reshape(B, V * H * W, 3)
        conf = conf.reshape(B, V * H * W)
        k = min(self.sample_k, wp.shape[1])
        idx = torch.topk(conf, k=k, dim=1, largest=True).indices
        pts = torch.gather(wp, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        feat = self.pt_mlp(pts).mean(dim=1)
        return feat

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224)
        return: (B, feat_dim)
        """
        h = self._encode_images(images)
        out = self.adapter(h)
        if self.adapter_dropout > 0 and self.training:
            out = torch.nn.functional.dropout(out, p=self.adapter_dropout, training=True)
        return out

    def get_backbone_lora_params(self, last_n_blocks: Optional[int] = None) -> List[nn.Parameter]:
        """Backbone 内 LoRA 参数。last_n_blocks 非空时只返回最后 n 个 block 的 LoRA（Stage3 降强度）。"""
        if last_n_blocks is not None and last_n_blocks > 0:
            return get_lora_params_from_last_n_blocks(self.backbone, last_n_blocks)
        return get_lora_params(self.backbone)

    def set_backbone_trainable(self, trainable: bool):
        """全 backbone 可训（一般不用；Stage2 用 set_backbone_lora_trainable）。"""
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        for p in self.pt_mlp.parameters():
            p.requires_grad = trainable

    def set_backbone_lora_trainable(self, trainable: bool, last_n_blocks: Optional[int] = None):
        """仅 backbone 内 LoRA 可训。last_n_blocks 非空时只训最后 n 个 block（Stage3 降强度）。"""
        all_lora = get_lora_params(self.backbone)
        to_train = self.get_backbone_lora_params(last_n_blocks=last_n_blocks) if last_n_blocks else all_lora
        to_train_ids = {id(p) for p in to_train}
        for p in all_lora:
            p.requires_grad = trainable and (id(p) in to_train_ids)

    def set_adapter_trainable(self, trainable: bool):
        for p in self.adapter.parameters():
            p.requires_grad = trainable
