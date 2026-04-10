# -*- coding: utf-8 -*-
"""
单例 GC6D 样本上的多模型特征图对比（同一 pc_common、可复现）。
"""

__all__ = ["select_representative_example", "build_pc_common"]


def select_representative_example(*args, **kwargs):
    from .scene_selection import select_representative_example as _fn

    return _fn(*args, **kwargs)


def build_pc_common(*args, **kwargs):
    from .point_cloud_common import build_pc_common as _fn

    return _fn(*args, **kwargs)
