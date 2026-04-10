# -*- coding: utf-8 -*-
"""
从 checkpoint 加载 policy：根据 ckpt 内 encoder_type 选择 Placeholder 或 LIFT3D。
"""

from typing import Any, Dict, List, Optional

import torch


def _strip_module_prefix(key: str) -> str:
    k = key
    while k.startswith("module."):
        k = k[7:]
    return k


def _state_key_names(state: Dict[str, Any]) -> List[str]:
    """DDP 保存的 ``module.xxx`` 与单卡 ``xxx`` 统一成后者，便于 ``startswith`` 推断。"""
    return [_strip_module_prefix(k) for k in state.keys()]


def _replacement_projector_in_channels(state: Dict[str, Any]) -> Optional[int]:
    """Conv1d projector 第一层的 in_channels（PointNext=512，CLIP/DINOv2=768）。"""
    for k, v in state.items():
        nk = _strip_module_prefix(k)
        if nk == "replacement_projector.0.weight" and isinstance(v, torch.Tensor) and v.dim() == 3:
            return int(v.shape[1])
    return None


# 训练脚本 / 模型类里保存的 model_mode 短名 → load 分支使用的规范名（须与 train_alignment_experiments 分支一致）
_MODEL_MODE_ALIASES: Dict[str, str] = {
    "lift3d_clip_progressive_replacement": "lift3d_progressive_replacement_clip",
    "dinov2_progressive_replacement": "lift3d_progressive_replacement_dinov2",
}


def _canonical_model_mode(mm: Optional[str]) -> Optional[str]:
    if mm is None:
        return None
    return _MODEL_MODE_ALIASES.get(mm, mm)


def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """从训练保存或 graspnet-baseline 原始 ckpt 中取出权重 dict。"""
    if not isinstance(ckpt, dict):
        return {}
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    return {}


def _is_raw_graspnet_baseline_weights(state: Dict[str, Any]) -> bool:
    """
    graspnet-baseline 保存的 GraspNet 权重：顶层为 view_estimator.* / grasp_generator.*，
    无 PureGraspNetPipeline 的 grasp_net. 前缀。
    """
    if not state:
        return False
    keys = list(state.keys())
    if any(k.startswith("grasp_net.") for k in keys):
        return False
    if any(k.startswith("view_estimator.") for k in keys):
        return True
    if any(k.startswith("grasp_generator.") for k in keys):
        return True
    return False

# 延迟 import 避免与 data/models 循环依赖
def _get_models():
    from models import (
        GC6DGraspPolicy,
        build_lift3d_policy,
        build_lift3d_clip_policy,
        build_vggt_base_policy,
        build_vggt_ft_policy,
    )
    return (
        GC6DGraspPolicy,
        build_lift3d_policy,
        build_lift3d_clip_policy,
        build_vggt_base_policy,
        build_vggt_ft_policy,
    )


def load_policy_from_checkpoint(
    ckpt_path: str,
    device: str = "cuda",
    lift3d_root: Optional[str] = None,
    graspnet_ckpt: Optional[str] = None,
    graspnet_root: Optional[str] = None,
) -> torch.nn.Module:
    """
    加载 checkpoint。encoder_type 取自上一步保存的 ckpt：
    - "lift3d_clip"：LIFT3D 官方 lift3d_clip + GC6D head
    - "lift3d"：LIFT3D PointNext + GC6D head（或 EncoderAdapterGraspNet，见下）
    - "vggt_base"：VGGT 原始 + GC6D head（输入 RGB）
    - "vggt_ft"：VGGT 微调 + GC6D head（输入 RGB）
    - 否则：Placeholder + GC6D head

    若 state_dict 中含有 grasp_net.* / adapter.*（即 train_adapter_graspnet 保存的 ckpt），
    则构建 EncoderAdapterGraspNet 并加载；此时需提供 graspnet_ckpt（可从 ckpt 内或参数传入）。
    """
    (
        GC6DGraspPolicy,
        build_lift3d_policy,
        build_lift3d_clip_policy,
        build_vggt_base_policy,
        build_vggt_ft_policy,
    ) = _get_models()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = _extract_state_dict(ckpt)
    if not state and isinstance(ckpt, dict):
        # 兼容极少数「整文件即 state_dict」的旧格式
        state = ckpt if any(isinstance(v, torch.Tensor) for v in ckpt.values()) else {}

    encoder_type = ckpt.get("encoder_type", "placeholder")
    model_mode = ckpt.get("model_mode")
    grasp_head_type = ckpt.get("grasp_head_type", "simple")
    num_proposals = ckpt.get("grasp_head_num_proposals", 4)

    def _kwargs_head():
        kw = {"grasp_head_type": grasp_head_type}
        if grasp_head_type == "graspnet":
            kw["num_proposals"] = num_proposals
        return kw

    # LIFT3D local fusion（seed NN + concat_proj / residual_proj）：先于此分支，避免仅有 grasp_net.* 时误判为 adapter 线
    is_lift3d_local_fusion = ckpt.get("fusion_mode") in ("concat_proj", "residual_proj") or any(
        k.startswith("lift3d_seed_proj.") for k in state.keys()
    )
    is_pure_graspnet = (
        model_mode == "pure_graspnet"
        or encoder_type == "pure_graspnet"
        or (
            any(k.startswith("grasp_net.") for k in state.keys())
            and not any(k.startswith("adapter.") for k in state.keys())
            and not any(k.startswith("encoder.") for k in state.keys())
            and not any(k.startswith("lift3d_seed_proj.") for k in state.keys())
        )
    )

    # 判断是否为 train_adapter_graspnet 保存的 EncoderAdapterGraspNet 结构
    is_adapter_graspnet = any(
        k.startswith("grasp_net.") or k.startswith("adapter.") for k in state.keys()
    )

    # LoRA 配置必须与 checkpoint 一致，否则 load_state_dict 会 shape mismatch（如 ckpt 为 lora_r=2，当前默认 8）
    lora_r = ckpt.get("lora_r")
    lora_scale = ckpt.get("lora_scale")
    if lora_r is None or lora_scale is None:
        # 旧 ckpt 未保存 lora_r/lora_scale：从 state 里任意 lora_B 的 shape 推断 r
        for k, v in state.items():
            if k.endswith(".lora_B") and hasattr(v, "shape") and len(v.shape) >= 1:
                lora_r = int(v.shape[0])
                break
        lora_r = lora_r if lora_r is not None else 8
        lora_scale = lora_scale if lora_scale is not None else 1.0

    # 原始 graspnet-baseline checkpoint（如 checkpoint-rs.tar）：直接走 PureGraspNetPipeline，
    # build_pure_graspnet_pipeline 已从同一路径加载权重，无需再 load_state_dict。
    if _is_raw_graspnet_baseline_weights(state):
        from models.pure_graspnet import build_pure_graspnet_pipeline

        _device = torch.device(device) if isinstance(device, str) else device
        model = build_pure_graspnet_pipeline(
            graspnet_ckpt=ckpt_path,
            graspnet_root=graspnet_root or (ckpt.get("graspnet_root") if isinstance(ckpt, dict) else None),
            device=_device,
        )
        return model.to(_device)

    if is_lift3d_local_fusion:
        import os
        from models.lift3d_local_fusion import build_lift3d_local_fusion_graspnet

        _graspnet_ckpt = graspnet_ckpt or ckpt.get("graspnet_ckpt") or os.environ.get("GRASPNET_CKPT")
        if not _graspnet_ckpt:
            _base = os.environ.get("GRASPNET_BASELINE", os.path.expanduser("~/graspnet-baseline"))
            _default = os.path.join(_base, "logs", "log_rs", "checkpoint-rs.tar")
            if os.path.isfile(_default):
                _graspnet_ckpt = _default
        if not _graspnet_ckpt:
            raise ValueError(
                "Checkpoint 为 Lift3DLocalFusionGraspNet，但未提供 graspnet 预训练路径。"
                "请传参 --graspnet_ckpt 或设置环境变量 GRASPNET_CKPT。"
            )
        fusion_mode = ckpt.get("fusion_mode")
        if fusion_mode not in ("concat_proj", "residual_proj"):
            fusion_mode = "concat_proj" if any(k.startswith("fusion_concat_proj.") for k in state.keys()) else "residual_proj"
        _ra = ckpt.get("residual_alpha")
        residual_alpha = 1.0 if _ra is None else float(_ra)
        _device = torch.device(device) if isinstance(device, str) else device
        model = build_lift3d_local_fusion_graspnet(
            fusion_mode=fusion_mode,
            graspnet_ckpt=_graspnet_ckpt,
            graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
            lift3d_root=lift3d_root or ckpt.get("lift3d_root"),
            lift3d_ckpt=ckpt.get("lift3d_ckpt"),
            residual_alpha=residual_alpha,
            encoder_feat_dim=256,
            lora_r=lora_r,
            lora_scale=lora_scale,
            lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
            device=_device,
        )
        model.load_state_dict(state, strict=False)
        return model.to(_device)

    # Alignment experiments（v2）：CLIP/DINOv2 replacement、VGGT replacement/fusion、旧 PointNext/LIFT3D fusion
    # 键名需去 module. 前缀，否则 DDP 存盘时匹配不到 replacement_projector. / encoder. / progressive_adapter.
    _sk = _state_key_names(state)
    is_alignment_exp = model_mode in (
        "lift3d_replacement_clip",
        "lift3d_replacement_dinov2",
        "lift3d_progressive_replacement_clip",
        "lift3d_progressive_replacement_dinov2",
        "lift3d_clip_progressive_replacement",
        "dinov2_progressive_replacement",
        "lift3d_replacement_distill_clip",
        "lift3d_replacement_distill_dinov2",
        "lift3d_clip_replacement_distill",
        "vggt_replacement",
        "vggt_progressive_replacement",
        "vggt_replacement_distill",
        "vggt_progressive_fusion",
        "vggt_fusion_distill",
        "vggt_fusion_normalized",
        "lift3d_replacement",
        "lift3d_fusion_normalized",
    ) or (
        any(k.startswith("replacement_projector.") for k in _sk)
        and any(k.startswith("encoder.") for k in _sk)
        and not any(k.startswith("adapter.") for k in _sk)
        and not any(k.startswith("lift3d_seed_proj.") for k in _sk)
    ) or (
        any(k.startswith("fusion_mlp.") for k in _sk)
        and (
            any(k.startswith("lift3d_ln.") for k in _sk)
            or any(k.startswith("vggt_ln.") for k in _sk)
        )
    ) or any(k.startswith("progressive_ln.") for k in _sk) or any(
        k.startswith("distill_ln.") for k in _sk
    ) or any(k.startswith("fusion_seed_ln.") for k in _sk) or any(
        k.startswith("fusion_distill_ln.") for k in _sk
    ) or any(k.startswith("progressive_adapter.") for k in _sk)

    if is_alignment_exp:
        import os

        _graspnet_ckpt = graspnet_ckpt or ckpt.get("graspnet_ckpt") or os.environ.get("GRASPNET_CKPT")
        if not _graspnet_ckpt:
            _base = os.environ.get("GRASPNET_BASELINE", os.path.expanduser("~/graspnet-baseline"))
            _default = os.path.join(_base, "logs", "log_rs", "checkpoint-rs.tar")
            if os.path.isfile(_default):
                _graspnet_ckpt = _default
        if not _graspnet_ckpt:
            raise ValueError("alignment checkpoint 需要 graspnet_ckpt")
        _device = torch.device(device) if isinstance(device, str) else device
        _mm = _canonical_model_mode(model_mode)
        if _mm is None:
            sk = _state_key_names(state)
            if any(k.startswith("fusion_mlp.") for k in sk) and any(k.startswith("vggt_ln.") for k in sk):
                _mm = "vggt_fusion_normalized"
            elif any(k.startswith("fusion_mlp.") for k in sk) and any(k.startswith("lift3d_ln.") for k in sk):
                _mm = "lift3d_fusion_normalized"
            elif any(k.startswith("fusion_seed_ln.") for k in sk):
                _mm = "vggt_progressive_fusion"
            elif any(k.startswith("fusion_distill_ln.") for k in sk):
                _mm = "vggt_fusion_distill"
            elif any(k.startswith("progressive_ln.") for k in sk) or any(
                k.startswith("progressive_adapter.") for k in sk
            ):
                # 注意：progressive_ln 为 elementwise_affine=False 时 state 里可能无 progressive_ln.*，
                # 但 progressive_adapter（Conv1d）总有参数；键名可能带 module. 前缀（见 _state_key_names）。
                if any(k.startswith("encoder.pt_mlp.") for k in sk):
                    _mm = "vggt_progressive_replacement"
                elif any(k.startswith("encoder.dinov2.") for k in sk):
                    _mm = "lift3d_progressive_replacement_dinov2"
                else:
                    _mm = "lift3d_progressive_replacement_clip"
            elif any(k.startswith("distill_ln.") for k in sk):
                if any(k.startswith("encoder.pt_mlp.") for k in sk):
                    _mm = "vggt_replacement_distill"
                elif any(k.startswith("encoder.dinov2.") for k in sk):
                    _mm = "lift3d_replacement_distill_dinov2"
                else:
                    _mm = "lift3d_replacement_distill_clip"
            elif any(k.startswith("encoder.dinov2.") for k in sk):
                _mm = "lift3d_replacement_dinov2"
            elif any(k.startswith("encoder.cls_token") for k in sk) or any(
                k.startswith("encoder.resblocks.") for k in sk
            ):
                _mm = "lift3d_replacement_clip"
            elif any(k.startswith("encoder.pt_mlp.") for k in sk):
                _mm = "vggt_replacement"
            elif any(k.startswith("replacement_projector.") for k in sk):
                # PointNext 替代为 512→256；CLIP/DINOv2 为 768→256，不可误判为 lift3d_replacement（512）
                rp_in = _replacement_projector_in_channels(state)
                if rp_in == 768:
                    if any(k.startswith("progressive_adapter.") for k in sk):
                        if any(k.startswith("encoder.dinov2.") for k in sk):
                            _mm = "lift3d_progressive_replacement_dinov2"
                        else:
                            _mm = "lift3d_progressive_replacement_clip"
                    elif any(k.startswith("encoder.dinov2.") for k in sk):
                        _mm = "lift3d_replacement_dinov2"
                    elif any(k.startswith("encoder.cls_token") for k in sk) or any(
                        k.startswith("encoder.resblocks.") for k in sk
                    ):
                        _mm = "lift3d_replacement_clip"
                    else:
                        # 冻结 encoder 未写入 ckpt 时：优先读保存的 model_mode
                        tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
                        mm2 = _canonical_model_mode(ckpt.get("model_mode") or tm.get("model_mode"))
                        if mm2 == "lift3d_progressive_replacement_dinov2":
                            _mm = "lift3d_progressive_replacement_dinov2"
                        else:
                            _mm = "lift3d_progressive_replacement_clip"
                else:
                    _mm = "lift3d_replacement"

        if _mm == "lift3d_fusion_normalized":
            from models.lift3d_fusion_normalized_pipeline import build_lift3d_fusion_normalized_graspnet

            model = build_lift3d_fusion_normalized_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                lift3d_root=lift3d_root or ckpt.get("lift3d_root"),
                lift3d_ckpt=ckpt.get("lift3d_ckpt"),
                encoder_feat_dim=256,
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                fuse_residual=bool(ckpt.get("fuse_residual", False)),
                fuse_alpha=float(ckpt.get("fuse_alpha", 0.1)),
                device=_device,
            )
        elif _mm == "vggt_fusion_normalized":
            from models.vggt_fusion_normalized_pipeline import build_vggt_fusion_normalized_graspnet

            model = build_vggt_fusion_normalized_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                vggt_ckpt=ckpt.get("vggt_ckpt"),
                feat_dim=256,
                sample_k=int(ckpt.get("vggt_sample_k", 1024)),
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                fuse_residual=bool(ckpt.get("fuse_residual", False)),
                fuse_alpha=float(ckpt.get("fuse_alpha", 0.1)),
                device=_device,
            )
        elif _mm == "vggt_replacement":
            from models.vggt_replacement_pipeline import build_vggt_replacement_graspnet

            model = build_vggt_replacement_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                vggt_ckpt=ckpt.get("vggt_ckpt"),
                feat_dim=256,
                sample_k=int(ckpt.get("vggt_sample_k", 1024)),
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                replacement_align_mode=ckpt.get("replacement_align_mode", "none"),
                replacement_affine_init_scale=float(ckpt.get("replacement_affine_init_scale", 1.0)),
                replacement_adapter_hidden=int(ckpt.get("replacement_adapter_hidden", 256)),
                replacement_adapter_depth=int(ckpt.get("replacement_adapter_depth", 2)),
                replacement_scale_mode=ckpt.get("replacement_scale_mode", "none"),
                replacement_fixed_alpha=float(ckpt.get("replacement_fixed_alpha", 1.0)),
                replacement_learnable_scale_init=float(ckpt.get("replacement_learnable_scale_init", 1.0)),
                device=_device,
            )
        elif _mm == "vggt_progressive_replacement":
            from models.vggt_progressive_replacement_pipeline import build_vggt_progressive_replacement_graspnet

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_vggt_progressive_replacement_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                vggt_ckpt=ckpt.get("vggt_ckpt"),
                feat_dim=256,
                sample_k=int(ckpt.get("vggt_sample_k", 1024)),
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                progressive_alpha=float(tm.get("progressive_alpha") or ckpt.get("progressive_alpha", 0.5)),
                score_calibration_mode=str(
                    tm.get("progressive_score_calibration_mode")
                    or ckpt.get("progressive_score_calibration_mode", "none")
                ),
                score_delta_scale=float(
                    tm.get("progressive_score_delta_scale") or ckpt.get("progressive_score_delta_scale", 0.1)
                ),
                score_calibration_hidden=int(
                    tm.get("progressive_score_calibration_hidden")
                    or ckpt.get("progressive_score_calibration_hidden", 12)
                ),
                device=_device,
            )
        elif _mm == "vggt_replacement_distill":
            from models.vggt_replacement_distill_pipeline import build_vggt_replacement_distill_graspnet

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_vggt_replacement_distill_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                vggt_ckpt=ckpt.get("vggt_ckpt"),
                feat_dim=256,
                sample_k=int(ckpt.get("vggt_sample_k", 1024)),
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                distill_loss_type=str(tm.get("distill_loss_type") or ckpt.get("distill_loss_type", "l2")),
                device=_device,
            )
        elif _mm == "vggt_progressive_fusion":
            from models.vggt_backbone_fusion_pipeline import build_vggt_fusion_progressive_graspnet

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_vggt_fusion_progressive_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                vggt_ckpt=ckpt.get("vggt_ckpt"),
                feat_dim=256,
                sample_k=int(ckpt.get("vggt_sample_k", 1024)),
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                progressive_alpha=float(tm.get("progressive_alpha") or ckpt.get("progressive_alpha", 0.5)),
                device=_device,
            )
        elif _mm == "vggt_fusion_distill":
            from models.vggt_backbone_fusion_pipeline import build_vggt_fusion_distill_graspnet

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_vggt_fusion_distill_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                vggt_ckpt=ckpt.get("vggt_ckpt"),
                feat_dim=256,
                sample_k=int(ckpt.get("vggt_sample_k", 1024)),
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                distill_loss_type=str(tm.get("distill_loss_type") or ckpt.get("distill_loss_type", "l2")),
                fusion_distill_alpha=float(
                    tm.get("fusion_distill_alpha") if tm.get("fusion_distill_alpha") is not None else ckpt.get("fusion_distill_alpha", 0.2)
                ),
                device=_device,
            )
        elif _mm == "lift3d_replacement_clip":
            from models.lift3d_clip_replacement_pipeline import build_lift3d_clip_replacement_graspnet

            model = build_lift3d_clip_replacement_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                lift3d_root=lift3d_root or ckpt.get("lift3d_root"),
                device=_device,
            )
        elif _mm == "lift3d_replacement_dinov2":
            from models.dinov2_replacement_pipeline import build_dinov2_replacement_graspnet

            model = build_dinov2_replacement_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                device=_device,
            )
        elif _mm == "lift3d_progressive_replacement_clip":
            from models.lift3d_progressive_replacement_pipeline import (
                build_lift3d_clip_progressive_replacement_graspnet,
            )

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_lift3d_clip_progressive_replacement_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                lift3d_root=lift3d_root or ckpt.get("lift3d_root"),
                progressive_alpha=float(tm.get("progressive_alpha") or ckpt.get("progressive_alpha", 0.5)),
                device=_device,
            )
        elif _mm == "lift3d_progressive_replacement_dinov2":
            from models.lift3d_progressive_replacement_pipeline import (
                build_dinov2_progressive_replacement_graspnet,
            )

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_dinov2_progressive_replacement_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                progressive_alpha=float(tm.get("progressive_alpha") or ckpt.get("progressive_alpha", 0.5)),
                device=_device,
            )
        elif _mm == "lift3d_replacement_distill_clip":
            from models.lift3d_replacement_distill_pipeline import build_lift3d_clip_replacement_distill_graspnet

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_lift3d_clip_replacement_distill_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                lift3d_root=lift3d_root or ckpt.get("lift3d_root"),
                distill_loss_type=str(tm.get("distill_loss_type") or ckpt.get("distill_loss_type", "l2")),
                device=_device,
            )
        elif _mm == "lift3d_replacement_distill_dinov2":
            from models.lift3d_replacement_distill_pipeline import build_dinov2_replacement_distill_graspnet

            tm = ckpt.get("train_meta") if isinstance(ckpt.get("train_meta"), dict) else {}
            model = build_dinov2_replacement_distill_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                distill_loss_type=str(tm.get("distill_loss_type") or ckpt.get("distill_loss_type", "l2")),
                device=_device,
            )
        else:
            from models.lift3d_replacement_pipeline import build_lift3d_replacement_graspnet

            model = build_lift3d_replacement_graspnet(
                graspnet_ckpt=_graspnet_ckpt,
                graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
                lift3d_root=lift3d_root or ckpt.get("lift3d_root"),
                lift3d_ckpt=ckpt.get("lift3d_ckpt"),
                encoder_feat_dim=256,
                lora_r=lora_r,
                lora_scale=lora_scale,
                lora_last_n_blocks=ckpt.get("lora_last_n_blocks"),
                device=_device,
            )
        model.load_state_dict(state, strict=False)
        return model.to(_device)

    if is_pure_graspnet:
        import os
        from models.pure_graspnet import build_pure_graspnet_pipeline

        _graspnet_ckpt = graspnet_ckpt or ckpt.get("graspnet_ckpt") or os.environ.get("GRASPNET_CKPT")
        if not _graspnet_ckpt:
            _base = os.environ.get("GRASPNET_BASELINE", os.path.expanduser("~/graspnet-baseline"))
            _default = os.path.join(_base, "logs", "log_rs", "checkpoint-rs.tar")
            if os.path.isfile(_default):
                _graspnet_ckpt = _default
        if not _graspnet_ckpt:
            raise ValueError(
                "Checkpoint 为 pure_graspnet，但未提供 graspnet 预训练路径。"
                "请传参 --graspnet_ckpt 或设置环境变量 GRASPNET_CKPT。"
            )
        _device = torch.device(device) if isinstance(device, str) else device
        model = build_pure_graspnet_pipeline(
            graspnet_ckpt=_graspnet_ckpt,
            graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
            score_calibration_mode=ckpt.get("pure_score_calibration_mode", "none"),
            score_delta_scale=float(ckpt.get("pure_score_delta_scale", 0.1)),
            score_calibration_hidden=int(ckpt.get("pure_score_calibration_hidden", 12)),
            device=_device,
        )
        model.load_state_dict(state, strict=False)
        return model.to(_device)

    if is_adapter_graspnet:
        import os
        from models.graspnet_adapter import build_encoder_adapter_graspnet
        _graspnet_ckpt = graspnet_ckpt or ckpt.get("graspnet_ckpt") or os.environ.get("GRASPNET_CKPT")
        if not _graspnet_ckpt:
            _base = os.environ.get("GRASPNET_BASELINE", os.path.expanduser("~/graspnet-baseline"))
            _default = os.path.join(_base, "logs", "log_rs", "checkpoint-rs.tar")
            if os.path.isfile(_default):
                _graspnet_ckpt = _default
        if not _graspnet_ckpt:
            raise ValueError(
                "Checkpoint 为 EncoderAdapterGraspNet，但未提供 graspnet 预训练路径。"
                "请传参 --graspnet_ckpt 或设置环境变量 GRASPNET_CKPT，或使用含 graspnet_ckpt 的 ckpt。"
            )
        _device = torch.device(device) if isinstance(device, str) else device
        model = build_encoder_adapter_graspnet(
            encoder_type=encoder_type,
            graspnet_ckpt=_graspnet_ckpt,
            encoder_feat_dim=256,
            graspnet_root=graspnet_root or ckpt.get("graspnet_root"),
            lift3d_root=lift3d_root,
            vggt_ckpt=None,
            lora_r=lora_r,
            lora_scale=lora_scale,
            device=_device,
            use_adapter=ckpt.get("use_adapter", True),
            adapter_cond_coeff=ckpt.get("adapter_cond_coeff", 2.0),
            adapter_cond_mode=ckpt.get("adapter_cond_mode", "additive"),
        )
        model.load_state_dict(state, strict=False)
        return model.to(_device)

    if encoder_type == "lift3d_clip":
        model = build_lift3d_clip_policy(
            encoder_feat_dim=256,
            width_min=0.01,
            width_max=0.12,
            lift3d_root=lift3d_root,
            freeze_backbone=True,
            normalize_pc=True,
            lora_r=lora_r,
            lora_scale=lora_scale,
            **_kwargs_head(),
        )
    elif encoder_type == "lift3d":
        model = build_lift3d_policy(
            encoder_feat_dim=256,
            width_min=0.01,
            width_max=0.12,
            lift3d_root=lift3d_root,
            use_lora=True,
            lora_r=lora_r,
            lora_scale=lora_scale,
            normalize_pc=True,
            **_kwargs_head(),
        )
    elif encoder_type == "vggt_base":
        model = build_vggt_base_policy(
            encoder_feat_dim=256,
            width_min=0.01,
            width_max=0.12,
            lora_r=lora_r,
            lora_scale=lora_scale,
            **_kwargs_head(),
        )
    elif encoder_type == "vggt_ft":
        model = build_vggt_ft_policy(
            encoder_feat_dim=256,
            width_min=0.01,
            width_max=0.12,
            ckpt_path=None,
            freeze_backbone=True,
            lora_r=lora_r,
            lora_scale=lora_scale,
            **_kwargs_head(),
        )
    else:
        model = GC6DGraspPolicy(encoder_feat_dim=256)

    model.load_state_dict(state, strict=True)
    return model.to(device)
