#!/usr/bin/env bash
# 实验 2：pure GraspNet + GC6D 训练（默认冻结 backbone，只训 vpmodule + grasp_generator，与 Stage2「冻结 encoder、训 head」同构）
# 可调：EPOCHS LR BS SEED；全量微调加 --no_freeze_graspnet_backbone

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
export GC6D_ROOT="${GC6D_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
export GRASPNET_CKPT="${GRASPNET_CKPT:-$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar}"
export CAM="${CAM:-realsense-d415}"

EPOCHS="${EPOCHS:-200}"
BS="${BS:-4}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"

python train_lift3d_pipeline.py \
  --model_mode pure_graspnet \
  --freeze_graspnet_backbone \
  --data_dir "$DATA" \
  --dataset_root "$GC6D_ROOT" \
  --camera "$CAM" \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr "$LR" \
  --seed "$SEED" \
  --graspnet_ckpt "$GRASPNET_CKPT" \
  --graspnet_root "$GRASPNET_BASELINE" \
  --collision_aux \
  --exp_name "pure_graspnet_gc6d_train" \
  --run_eval_after \
  --eval_split test \
  --eval_extra_stats

echo "完成。见 checkpoints 与 eval_out。"
