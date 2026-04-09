# -*- coding: utf-8 -*-
"""从 GC6D offline batch 的 meta 加载 RGB，与 eval / 训练一致（ImageNet normalize）。"""

from __future__ import annotations

import os
from typing import List

import torch


def load_images_batch(
    metas: List[dict],
    data_dir: str,
    device: torch.device,
    image_size: int = 224,
) -> torch.Tensor:
    """meta 含 rgb_path；缺失则用零张量。"""
    from PIL import Image
    from torchvision import transforms as T

    tfm = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    batch = []
    for m in metas:
        rp = m.get("rgb_path", "")
        if not rp:
            batch.append(torch.zeros(3, image_size, image_size))
            continue
        p = rp if os.path.isfile(rp) else os.path.join(data_dir, rp)
        if not os.path.isfile(p):
            batch.append(torch.zeros(3, image_size, image_size))
            continue
        img = Image.open(p).convert("RGB")
        batch.append(tfm(img))
    return torch.stack(batch, dim=0).to(device)
