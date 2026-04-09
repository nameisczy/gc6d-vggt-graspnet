# -*- coding: utf-8 -*-
"""
LIFT3D baseline（Stage2 equivalent）：冻结 LIFT3D encoder，只训练 adapter + GraspNet head。
"""

from __future__ import annotations

from typing import List, Optional

import torch


def apply_lift3d_baseline_freeze(model: torch.nn.Module) -> None:
    """
    encoder 全部 requires_grad=False；
    adapter 与 grasp_net 可训。
    不修改 grasp_net 以外模块（如 film/gate 若存在）。
    """
    if not hasattr(model, "encoder"):
        raise ValueError("model 需有 encoder（Lift3DGraspPipeline / EncoderAdapterGraspNet）")
    for p in model.encoder.parameters():
        p.requires_grad = False
    if getattr(model, "adapter", None) is not None:
        for p in model.adapter.parameters():
            p.requires_grad = True
    for p in model.grasp_net.parameters():
        p.requires_grad = True
    # gated / film 附加层若存在
    for name in ("cond_gate", "film_proj", "concat_proj"):
        m = getattr(model, name, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True
    # LIFT3D local fusion（seed 对齐 + concat/residual 头）
    for name in ("lift3d_seed_proj", "fusion_concat_proj", "fusion_residual_proj"):
        m = getattr(model, name, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True


def apply_lift3d_replacement_freeze(model: torch.nn.Module, freeze_head: bool = False) -> None:
    """LIFT3D encoder + GraspNet backbone 冻结；默认 replacement_projector + vpmodule + grasp_generator 可训。"""
    if not hasattr(model, "encoder"):
        raise ValueError("lift3d_replacement 需要 model.encoder")
    for p in model.encoder.parameters():
        p.requires_grad = False
    if not hasattr(model, "grasp_net"):
        raise ValueError("需要 model.grasp_net")
    backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
    if backbone is None:
        raise ValueError("view_estimator.backbone 不存在")
    for p in backbone.parameters():
        p.requires_grad = False
    for p in model.grasp_net.view_estimator.vpmodule.parameters():
        p.requires_grad = not freeze_head
    for p in model.grasp_net.grasp_generator.parameters():
        p.requires_grad = not freeze_head
    rp = getattr(model, "replacement_projector", None)
    if rp is not None:
        for p in rp.parameters():
            p.requires_grad = True
    for name in (
        "replacement_ln",
        "replacement_adapter",
        "replacement_affine_gamma",
        "replacement_affine_beta",
        "replacement_learnable_scale",
    ):
        m = getattr(model, name, None)
        if hasattr(m, "parameters"):
            for p in m.parameters():
                p.requires_grad = True
        elif isinstance(m, torch.nn.Parameter):
            m.requires_grad = True


def apply_vggt_replacement_freeze(model: torch.nn.Module, freeze_head: bool = False) -> None:
    """VGGT encoder + GraspNet backbone 冻结；可选冻结整套 head，仅训 replacement 对齐模块。"""
    apply_lift3d_replacement_freeze(model, freeze_head=freeze_head)


def apply_vggt_progressive_replacement_freeze(
    model: torch.nn.Module,
    *,
    freeze_head: bool = True,
    train_encoder: bool = False,
    vggt_train_encoder_last_n_blocks: int = 4,
) -> None:
    """VGGT encoder + backbone 冻结；progressive 的 projector/LN/adapter 可训；head 是否可训由 freeze_head 决定。
    train_encoder=True 时仅解冻 VGGT backbone 内 LoRA（保守微调），encoder 其余仍冻结。
    """
    if not hasattr(model, "encoder"):
        raise ValueError("vggt_progressive_replacement 需要 model.encoder")
    for p in model.encoder.parameters():
        p.requires_grad = False
    if train_encoder and hasattr(model.encoder, "set_backbone_lora_trainable"):
        model.encoder.set_backbone_lora_trainable(True, last_n_blocks=vggt_train_encoder_last_n_blocks)
    backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
    if backbone is None:
        raise ValueError("view_estimator.backbone 不存在")
    for p in backbone.parameters():
        p.requires_grad = False
    for p in model.replacement_projector.parameters():
        p.requires_grad = True
    for p in model.progressive_ln.parameters():
        p.requires_grad = True
    for p in model.progressive_adapter.parameters():
        p.requires_grad = True
    for p in model.grasp_net.view_estimator.vpmodule.parameters():
        p.requires_grad = not freeze_head
    for p in model.grasp_net.grasp_generator.parameters():
        p.requires_grad = not freeze_head
    sch = getattr(model, "score_calibration_head", None)
    if sch is not None:
        for p in sch.parameters():
            p.requires_grad = not freeze_head


def apply_vggt_progressive_fusion_freeze(model: torch.nn.Module, *, freeze_head: bool = True) -> None:
    """Part2：VGGT encoder + backbone 冻结；仅训 fusion_to_seed_projector / fusion_seed_ln / fusion_seed_adapter；head 由 freeze_head 控制（默认冻结）。"""
    if not hasattr(model, "encoder"):
        raise ValueError("vggt_progressive_fusion 需要 model.encoder")
    for p in model.encoder.parameters():
        p.requires_grad = False
    backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
    if backbone is None:
        raise ValueError("view_estimator.backbone 不存在")
    for p in backbone.parameters():
        p.requires_grad = False
    for p in model.fusion_to_seed_projector.parameters():
        p.requires_grad = True
    for p in model.fusion_seed_ln.parameters():
        p.requires_grad = True
    for p in model.fusion_seed_adapter.parameters():
        p.requires_grad = True
    for p in model.grasp_net.view_estimator.vpmodule.parameters():
        p.requires_grad = not freeze_head
    for p in model.grasp_net.grasp_generator.parameters():
        p.requires_grad = not freeze_head


def apply_vggt_fusion_distill_freeze(model: torch.nn.Module) -> None:
    """Part2 distill：encoder+backbone+head 冻结；仅训 fusion_distill_projector / fusion_distill_ln / fusion_distill_adapter。"""
    if not hasattr(model, "encoder"):
        raise ValueError("vggt_fusion_distill 需要 model.encoder")
    for p in model.encoder.parameters():
        p.requires_grad = False
    backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
    if backbone is None:
        raise ValueError("view_estimator.backbone 不存在")
    for p in backbone.parameters():
        p.requires_grad = False
    for p in model.grasp_net.view_estimator.vpmodule.parameters():
        p.requires_grad = False
    for p in model.grasp_net.grasp_generator.parameters():
        p.requires_grad = False
    for p in model.fusion_distill_projector.parameters():
        p.requires_grad = True
    for p in model.fusion_distill_ln.parameters():
        p.requires_grad = True
    for p in model.fusion_distill_adapter.parameters():
        p.requires_grad = True


def apply_vggt_replacement_distill_freeze(
    model: torch.nn.Module,
    *,
    train_encoder: bool = False,
    vggt_train_encoder_last_n_blocks: int = 4,
) -> None:
    """Distill：encoder+backbone+head 冻结；仅训 distill projector/LN/adapter。
    train_encoder=True 时额外解冻 VGGT backbone LoRA。
    """
    if not hasattr(model, "encoder"):
        raise ValueError("vggt_replacement_distill 需要 model.encoder")
    for p in model.encoder.parameters():
        p.requires_grad = False
    if train_encoder and hasattr(model.encoder, "set_backbone_lora_trainable"):
        model.encoder.set_backbone_lora_trainable(True, last_n_blocks=vggt_train_encoder_last_n_blocks)
    backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
    if backbone is None:
        raise ValueError("view_estimator.backbone 不存在")
    for p in backbone.parameters():
        p.requires_grad = False
    for p in model.grasp_net.view_estimator.vpmodule.parameters():
        p.requires_grad = False
    for p in model.grasp_net.grasp_generator.parameters():
        p.requires_grad = False
    for p in model.distill_projector.parameters():
        p.requires_grad = True
    for p in model.distill_ln.parameters():
        p.requires_grad = True
    for p in model.distill_adapter.parameters():
        p.requires_grad = True


def apply_vggt_fusion_normalized_freeze(model: torch.nn.Module, freeze_head: bool = False) -> None:
    """VGGT encoder + GraspNet backbone 冻结；可选冻结整套 head，仅训 normalization / projector / fusion。"""
    if not hasattr(model, "encoder"):
        raise ValueError("vggt_fusion_normalized 需要 model.encoder")
    for p in model.encoder.parameters():
        p.requires_grad = False
    backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
    if backbone is None:
        raise ValueError("view_estimator.backbone 不存在")
    for p in backbone.parameters():
        p.requires_grad = False
    for p in model.grasp_net.view_estimator.vpmodule.parameters():
        p.requires_grad = not freeze_head
    for p in model.grasp_net.grasp_generator.parameters():
        p.requires_grad = not freeze_head
    for name in ("vggt_ln", "vggt_proj", "fusion_mlp"):
        m = getattr(model, name, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True


def apply_lift3d_fusion_normalized_freeze(model: torch.nn.Module) -> None:
    """LIFT3D encoder + GraspNet backbone 冻结；LN / proj / fusion_mlp + head 可训。"""
    if not hasattr(model, "encoder"):
        raise ValueError("lift3d_fusion_normalized 需要 model.encoder")
    for p in model.encoder.parameters():
        p.requires_grad = False
    backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
    if backbone is None:
        raise ValueError("view_estimator.backbone 不存在")
    for p in backbone.parameters():
        p.requires_grad = False
    for p in model.grasp_net.view_estimator.vpmodule.parameters():
        p.requires_grad = True
    for p in model.grasp_net.grasp_generator.parameters():
        p.requires_grad = True
    for name in ("lift3d_ln", "lift3d_proj", "fusion_mlp"):
        m = getattr(model, name, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True


def apply_alignment_freeze(
    model: torch.nn.Module,
    model_mode: str,
    *,
    pure_freeze_vpmodule: bool = False,
    pure_only_train_last_score_head: bool = False,
    pure_train_score_calibrator_only: bool = False,
    freeze_head: bool = False,
    vggt_train_encoder: bool = False,
    vggt_train_encoder_last_n_blocks: int = 4,
) -> None:
    """统一入口：五主实验 + 旧 PointNext replacement / lift3d_fusion_normalized（兼容）。"""
    if model_mode == "pure_graspnet":
        apply_pure_graspnet_freeze(
            model,
            freeze_backbone=True,
            freeze_vpmodule=pure_freeze_vpmodule,
            only_last_score_head=pure_only_train_last_score_head,
            train_score_calibrator_only=pure_train_score_calibrator_only,
        )
    elif model_mode in ("lift3d_replacement_clip", "lift3d_replacement_dinov2", "lift3d_replacement"):
        apply_lift3d_replacement_freeze(model, freeze_head=freeze_head)
    elif model_mode in ("lift3d_progressive_replacement_clip", "lift3d_progressive_replacement_dinov2"):
        apply_vggt_progressive_replacement_freeze(model, freeze_head=freeze_head)
    elif model_mode in ("lift3d_replacement_distill_clip", "lift3d_replacement_distill_dinov2"):
        apply_vggt_replacement_distill_freeze(model)
    elif model_mode == "vggt_replacement":
        apply_vggt_replacement_freeze(model, freeze_head=freeze_head)
    elif model_mode == "vggt_fusion_normalized":
        apply_vggt_fusion_normalized_freeze(model, freeze_head=freeze_head)
    elif model_mode == "vggt_progressive_replacement":
        apply_vggt_progressive_replacement_freeze(
            model,
            freeze_head=freeze_head,
            train_encoder=vggt_train_encoder,
            vggt_train_encoder_last_n_blocks=vggt_train_encoder_last_n_blocks,
        )
    elif model_mode == "vggt_replacement_distill":
        apply_vggt_replacement_distill_freeze(
            model,
            train_encoder=vggt_train_encoder,
            vggt_train_encoder_last_n_blocks=vggt_train_encoder_last_n_blocks,
        )
    elif model_mode == "vggt_progressive_fusion":
        apply_vggt_progressive_fusion_freeze(model, freeze_head=freeze_head)
    elif model_mode == "vggt_fusion_distill":
        apply_vggt_fusion_distill_freeze(model)
    elif model_mode == "lift3d_fusion_normalized":
        apply_lift3d_fusion_normalized_freeze(model)
    else:
        raise ValueError("未知 model_mode: %s" % model_mode)


def apply_pure_graspnet_freeze(
    model: torch.nn.Module,
    freeze_backbone: bool = True,
    freeze_vpmodule: bool = False,
    only_last_score_head: bool = False,
    train_score_calibrator_only: bool = False,
) -> None:
    """
    pure_graspnet 模式：
    - freeze_backbone=True: 冻结 grasp_net.view_estimator.backbone，仅训 head（vpmodule + grasp_generator）
    - freeze_vpmodule=True: 同时冻结 vpmodule，仅训 grasp_generator（对照实验）
    - freeze_backbone=False: grasp_net 全部可训
    """
    if not hasattr(model, "grasp_net"):
        raise ValueError("pure_graspnet 模式要求 model.grasp_net 存在")
    for p in model.grasp_net.parameters():
        p.requires_grad = True
    if freeze_backbone:
        backbone = getattr(model.grasp_net.view_estimator, "backbone", None)
        if backbone is None:
            raise ValueError("grasp_net.view_estimator.backbone 不存在，无法执行 freeze_backbone")
        for p in backbone.parameters():
            p.requires_grad = False
    if freeze_vpmodule:
        vp = getattr(model.grasp_net.view_estimator, "vpmodule", None)
        if vp is None:
            raise ValueError("grasp_net.view_estimator.vpmodule 不存在")
        for p in vp.parameters():
            p.requires_grad = False
    if only_last_score_head:
        for p in model.grasp_net.parameters():
            p.requires_grad = False
        op = getattr(model.grasp_net.grasp_generator, "operation", None)
        tol = getattr(model.grasp_net.grasp_generator, "tolerance", None)
        if op is None or tol is None:
            raise ValueError("pure_only_train_last_score_head 需要 operation / tolerance")
        for name in ("conv3",):
            m = getattr(op, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True
        for name in ("conv3",):
            m = getattr(tol, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True
    if train_score_calibrator_only:
        for p in model.grasp_net.parameters():
            p.requires_grad = False
        sch = getattr(model, "score_calibration_head", None)
        if sch is None:
            raise ValueError("train_score_calibrator_only 需要 score_calibration_head")
        for p in sch.parameters():
            p.requires_grad = True


def set_pure_graspnet_train_state(model: torch.nn.Module) -> None:
    """
    训练步中：整体 model.train()（dropout 等），但 **冻结的 backbone 必须 eval()**，
    否则 BatchNorm 等仍会在 train 模式下更新 running_mean/var（requires_grad=False 也无效）。
    """
    model.train()
    bb = getattr(model.grasp_net.view_estimator, "backbone", None)
    if bb is not None:
        bb.eval()


def set_alignment_train_state(
    model: torch.nn.Module,
    model_mode: str,
    *,
    pure_freeze_vpmodule: bool = False,
    freeze_head: bool = False,
) -> None:
    model.train()
    bb = getattr(getattr(model, "grasp_net", None), "view_estimator", None)
    if bb is not None:
        backbone = getattr(bb, "backbone", None)
        if backbone is not None:
            backbone.eval()
    if model_mode == "pure_graspnet":
        if pure_freeze_vpmodule:
            vp = getattr(model.grasp_net.view_estimator, "vpmodule", None)
            if vp is not None:
                vp.eval()
    if freeze_head:
        vp = getattr(model.grasp_net.view_estimator, "vpmodule", None)
        gg = getattr(model.grasp_net, "grasp_generator", None)
        if vp is not None:
            vp.eval()
        if gg is not None:
            gg.eval()
        enc = getattr(model, "encoder", None)
        if enc is not None:
            enc.eval()
    elif model_mode in (
        "vggt_replacement",
        "vggt_fusion_normalized",
        "vggt_progressive_replacement",
        "vggt_replacement_distill",
        "lift3d_progressive_replacement_clip",
        "lift3d_progressive_replacement_dinov2",
        "lift3d_replacement_distill_clip",
        "lift3d_replacement_distill_dinov2",
        "vggt_progressive_fusion",
        "vggt_fusion_distill",
    ):
        enc = getattr(model, "encoder", None)
        if enc is not None:
            enc.eval()


def trainable_parameter_groups(
    model: torch.nn.Module,
    lr: float,
    encoder_lr_scale: float = 0.1,
) -> List[dict]:
    """
    baseline 下 encoder 不应可训；若误开启，仍用较低 lr（与旧逻辑一致）。
    """
    encoder = getattr(model, "encoder", None)
    enc_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
    ad_ids = set()
    if getattr(model, "adapter", None) is not None:
        ad_ids = {id(p) for p in model.adapter.parameters()}
    extra_ids = set()
    for name in (
        "cond_gate",
        "film_proj",
        "concat_proj",
        "score_calibration_head",
        "lift3d_seed_proj",
        "fusion_concat_proj",
        "fusion_residual_proj",
        "replacement_projector",
        "replacement_ln",
        "replacement_adapter",
        "lift3d_proj",
        "fusion_mlp",
        "lift3d_ln",
        "vggt_ln",
        "vggt_proj",
        "progressive_ln",
        "progressive_adapter",
        "distill_projector",
        "distill_ln",
        "distill_adapter",
        "fusion_to_seed_projector",
        "fusion_seed_ln",
        "fusion_seed_adapter",
        "fusion_distill_projector",
        "fusion_distill_ln",
        "fusion_distill_adapter",
    ):
        m = getattr(model, name, None)
        if m is not None:
            extra_ids |= {id(p) for p in m.parameters()}

    enc_params: List[torch.nn.Parameter] = []
    ad_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in enc_ids:
            enc_params.append(p)
        elif pid in ad_ids or pid in extra_ids:
            ad_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if ad_params:
        groups.append({"params": ad_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if enc_params:
        groups.append({"params": enc_params, "lr": lr * encoder_lr_scale})
    return groups


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    encoder_lr_scale: float = 0.1,
) -> torch.optim.Optimizer:
    groups = trainable_parameter_groups(model, lr=lr, encoder_lr_scale=encoder_lr_scale)
    if not groups:
        raise RuntimeError("没有可训练参数，请检查 freeze 逻辑")
    return torch.optim.Adam(groups)
