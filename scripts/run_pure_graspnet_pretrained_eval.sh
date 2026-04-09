#!/usr/bin/env bash
# 实验 1：pure GraspNet（预训练权重），不在 GC6D 上训练，仅写出 checkpoint 并走 eval_benchmark.py
# 与当前 pipeline 一致：train_lift3d_pipeline.py + eval_benchmark.py

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
export GC6D_ROOT="${GC6D_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
export GRASPNET_CKPT="${GRASPNET_CKPT:-$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar}"
export CAM="${CAM:-realsense-d415}"
SEED="${SEED:-42}"

python train_lift3d_pipeline.py \
  --model_mode pure_graspnet \
  --pretrained_eval_only \
  --data_dir "$DATA" \
  --dataset_root "$GC6D_ROOT" \
  --camera "$CAM" \
  --epochs 0 \
  --batch_size 1 \
  --graspnet_ckpt "$GRASPNET_CKPT" \
  --graspnet_root "$GRASPNET_BASELINE" \
  --seed "$SEED" \
  --exp_name "pure_graspnet_pretrained" \
  --run_eval_after \
  --eval_split test \
  --eval_extra_stats

echo "完成。checkpoint 在 checkpoints/lift3d_pipeline/；summary 在 eval_out/lift3d/。"
