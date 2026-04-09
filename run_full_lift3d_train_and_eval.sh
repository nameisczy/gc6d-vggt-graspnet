#!/bin/bash
# LIFT3D 全量训练 + test 集评估。用法: ./run_full_lift3d_train_and_eval.sh
# 可覆盖: DATA, LIFT3D_ROOT, GRASPNET_BASELINE, DATASET_ROOT, CUDA_VISIBLE_DEVICES, BATCH_SIZE
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"
GRASPNET_CKPT="${GRASPNET_CKPT:-$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar}"

echo "========== 1) LIFT3D 全量训练 (Stage1->2->3->4) =========="
MODE=full bash run_train_adapter_lift3d.sh

echo ""
echo "========== 2) LIFT3D 评估 (test) =========="
CKPT="checkpoints/gc6d_lift3d_adapter_graspnet_s4.pt"
python eval_benchmark.py \
  --data_dir "$DATA" --checkpoint "$CKPT" --split test \
  --dataset_root "$DATASET_ROOT" \
  --lift3d_root "$LIFT3D_ROOT" \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE"

echo "Done. Eval 结果见 eval_out/lift3d/"
