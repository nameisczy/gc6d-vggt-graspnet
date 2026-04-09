# Pipeline alignment 审计报告（自动生成）

## 1. Checkpoint 路径

{
  "pure_graspnet": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_pure/exp_pure_pure_graspnet_20260328_163000.pt",
  "vggt_replacement": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/vggt_replacement_scale_x10/vggt_replacement_scale_x10_vggt_replacement_20260331_205825.pt",
  "lift3d_replacement_clip": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_l3clip_rep/exp_l3clip_rep_lift3d_replacement_clip_20260328_164446.pt",
  "lift3d_replacement_dinov2": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_l3dino_rep/exp_l3dino_rep_lift3d_replacement_dinov2_20260328_164455.pt",
  "vggt_fusion_normalized": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_vggt_fuse_norm/exp_vggt_fuse_norm_vggt_fusion_normalized_20260328_163119.pt"
}

## 2. 参数组（trainable numel）

### pure_graspnet
```json
{
  "graspnet_backbone": {
    "trainable_numel": 0,
    "frozen_numel": 641856,
    "trainable_params": 0,
    "frozen_params": 48
  },
  "graspnet_head": {
    "trainable_numel": 148080,
    "frozen_numel": 0,
    "trainable_params": 29,
    "frozen_params": 0
  },
  "graspnet_vpmodule": {
    "trainable_numel": 236028,
    "frozen_numel": 0,
    "trainable_params": 10,
    "frozen_params": 0
  }
}
```

### vggt_replacement
```json
{
  "encoder": {
    "trainable_numel": 0,
    "frozen_numel": 1268476724,
    "trainable_params": 0,
    "frozen_params": 2579
  },
  "graspnet_backbone": {
    "trainable_numel": 0,
    "frozen_numel": 641856,
    "trainable_params": 0,
    "frozen_params": 48
  },
  "graspnet_head": {
    "trainable_numel": 148080,
    "frozen_numel": 0,
    "trainable_params": 29,
    "frozen_params": 0
  },
  "graspnet_vpmodule": {
    "trainable_numel": 236028,
    "frozen_numel": 0,
    "trainable_params": 10,
    "frozen_params": 0
  },
  "replacement_projector": {
    "trainable_numel": 262656,
    "frozen_numel": 0,
    "trainable_params": 4,
    "frozen_params": 0
  }
}
```

### lift3d_replacement_clip
```json
{
  "encoder": {
    "trainable_numel": 0,
    "frozen_numel": 87846336,
    "trainable_params": 0,
    "frozen_params": 310
  },
  "graspnet_backbone": {
    "trainable_numel": 0,
    "frozen_numel": 641856,
    "trainable_params": 0,
    "frozen_params": 48
  },
  "graspnet_head": {
    "trainable_numel": 148080,
    "frozen_numel": 0,
    "trainable_params": 29,
    "frozen_params": 0
  },
  "graspnet_vpmodule": {
    "trainable_numel": 236028,
    "frozen_numel": 0,
    "trainable_params": 10,
    "frozen_params": 0
  },
  "replacement_projector": {
    "trainable_numel": 262656,
    "frozen_numel": 0,
    "trainable_params": 4,
    "frozen_params": 0
  }
}
```

### lift3d_replacement_dinov2
```json
{
  "encoder": {
    "trainable_numel": 0,
    "frozen_numel": 86580480,
    "trainable_params": 0,
    "frozen_params": 175
  },
  "graspnet_backbone": {
    "trainable_numel": 0,
    "frozen_numel": 641856,
    "trainable_params": 0,
    "frozen_params": 48
  },
  "graspnet_head": {
    "trainable_numel": 148080,
    "frozen_numel": 0,
    "trainable_params": 29,
    "frozen_params": 0
  },
  "graspnet_vpmodule": {
    "trainable_numel": 236028,
    "frozen_numel": 0,
    "trainable_params": 10,
    "frozen_params": 0
  },
  "replacement_projector": {
    "trainable_numel": 262656,
    "frozen_numel": 0,
    "trainable_params": 4,
    "frozen_params": 0
  }
}
```

### vggt_fusion_normalized
```json
{
  "encoder": {
    "trainable_numel": 0,
    "frozen_numel": 1268476724,
    "trainable_params": 0,
    "frozen_params": 2579
  },
  "fusion_modules": {
    "trainable_numel": 461312,
    "frozen_numel": 0,
    "trainable_params": 10,
    "frozen_params": 0
  },
  "graspnet_backbone": {
    "trainable_numel": 0,
    "frozen_numel": 641856,
    "trainable_params": 0,
    "frozen_params": 48
  },
  "graspnet_head": {
    "trainable_numel": 148080,
    "frozen_numel": 0,
    "trainable_params": 29,
    "frozen_params": 0
  },
  "graspnet_vpmodule": {
    "trainable_numel": 236028,
    "frozen_numel": 0,
    "trainable_params": 10,
    "frozen_params": 0
  }
}
```

## 3. 加载 missing keys（strict=False）摘要
- **pure_graspnet**: missing=0 unexpected=0
- **vggt_replacement**: missing=0 unexpected=0
- **lift3d_replacement_clip**: missing=0 unexpected=0
- **lift3d_replacement_dinov2**: missing=0 unexpected=0
- **vggt_fusion_normalized**: missing=0 unexpected=0

## 4. pure 与预训练 GraspNet 权重差异（alignment 训练后 head 应变）
```json
null
```

## 5. pure 前向 seed_features 与预训练参考差异（backbone 应对齐）
```json
{
  "per_batch_mean_abs_diff_seed_features": [
    0.2735481858253479,
    0.2698117792606354,
    0.25380784273147583,
    0.263675719499588
  ],
  "seed_features_mean_abs_diff_mean": 0.2652108818292618,
  "seed_features_max_abs_diff_mean": 5.77490496635437
}
```

## 6. eval_out_rewrite 汇总（若存在）
```json
[
  {
    "_folder_name": "exp_l3clip_164446",
    "AP": 0.9406,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_l3clip_rep/exp_l3clip_rep_lift3d_replacement_clip_20260328_164446.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "exp_l3dino_164455",
    "AP": 1.812,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_l3dino_rep/exp_l3dino_rep_lift3d_replacement_dinov2_20260328_164455.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "exp_pure_130150",
    "AP": 13.3484,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_pure/exp_pure_pure_graspnet_20260330_130150.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "exp_pure_163000",
    "AP": 3.747,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_pure/exp_pure_pure_graspnet_20260328_163000.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "exp_vggt_fuse_163119",
    "AP": 3.3434,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_vggt_fuse_norm/exp_vggt_fuse_norm_vggt_fusion_normalized_20260328_163119.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "exp_vggt_rep_163036",
    "AP": 0.6952,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/exp_vggt_rep/exp_vggt_rep_vggt_replacement_20260328_163036.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "pure_head_only_pure_graspnet_20260331_110110",
    "AP": 4.8064,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_experiments/pure_head_only_20260331_110110/pure_head_only_pure_graspnet_20260331_110110.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "pure_head_only_pure_graspnet_20260331_110110_before_train",
    "AP": 15.2961,
    "pipeline_checkpoint": null,
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "pure_pretrained_rewrite",
    "AP": 4.867,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/gc6d_vggt_ft_adapter_graspnet_s4.pt",
    "eval_mode": "eval_all"
  },
  {
    "_folder_name": "pure_vpmodule_head_pure_graspnet_20260331_110049",
    "AP": 4.7817,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_experiments/pure_vpmodule_head_20260331_110049/pure_vpmodule_head_pure_graspnet_20260331_110049.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "pure_vpmodule_head_pure_graspnet_20260331_110049_before_train",
    "AP": 15.2591,
    "pipeline_checkpoint": null,
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "smoke_rewrite",
    "AP": 0.0,
    "pipeline_checkpoint": null,
    "eval_mode": "eval_scene_per_scene"
  },
  {
    "_folder_name": "smoke_rewrite2",
    "AP": 3.2947,
    "pipeline_checkpoint": null,
    "eval_mode": "eval_scene_per_scene"
  },
  {
    "_folder_name": "vggt_replacement_baseline_vggt_replacement_20260331_172834",
    "AP": 0.5801,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/vggt_replacement_baseline/vggt_replacement_baseline_vggt_replacement_20260331_172834.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "vggt_replacement_scale_x2_vggt_replacement_20260331_205839",
    "AP": 0.4397,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/vggt_replacement_scale_x2/vggt_replacement_scale_x2_vggt_replacement_20260331_205839.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  },
  {
    "_folder_name": "vggt_replacement_scale_x5_vggt_replacement_20260331_111847",
    "AP": 0.6103,
    "pipeline_checkpoint": "/home/ziyaochen/gc6d_grasp_pipeline/checkpoints/alignment_runs/vggt_replacement_scale_x5/vggt_replacement_scale_x5_vggt_replacement_20260331_111847.pt",
    "eval_mode": "eval_scene_per_scene_extra_stats"
  }
]
```