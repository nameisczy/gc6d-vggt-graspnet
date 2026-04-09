#!/usr/bin/env bash
# LIFT3D baseline 两组默认实验（新 pipeline：train_lift3d_pipeline.py）
# 数据与仓库路径按你机器修改。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
export GC6D_ROOT="${GC6D_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
export GRASPNET_CKPT="${GRASPNET_CKPT:-$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar}"
export CAM="${CAM:-realsense-d415}"

EPOCHS="${EPOCHS:-10}"
BS="${BS:-4}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"

# 实验 1：collision_aux on（默认）
python train_lift3d_pipeline.py \
  --data_dir "$DATA" \
  --dataset_root "$GC6D_ROOT" \
  --camera "$CAM" \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr "$LR" \
  --seed "$SEED" \
  --graspnet_ckpt "$GRASPNET_CKPT" \
  --graspnet_root "$GRASPNET_BASELINE" \
  --lift3d_root "$LIFT3D_ROOT" \
  --collision_aux \
  --exp_name "exp1_lift3d_collision_on" \
  --run_eval_after \
  --eval_extra_stats

# 实验 2：collision_aux off
python train_lift3d_pipeline.py \
  --data_dir "$DATA" \
  --dataset_root "$GC6D_ROOT" \
  --camera "$CAM" \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr "$LR" \
  --seed "$SEED" \
  --graspnet_ckpt "$GRASPNET_CKPT" \
  --graspnet_root "$GRASPNET_BASELINE" \
  --lift3d_root "$LIFT3D_ROOT" \
  --no_collision_aux \
  --exp_name "exp2_lift3d_collision_off" \
  --run_eval_after \
  --eval_extra_stats

echo "训练与 eval 已依次执行。请根据 checkpoints/lift3d_pipeline/*_train_summary.json 与 eval_out/*/summary_*.json 填表。"
