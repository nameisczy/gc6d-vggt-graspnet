#!/bin/bash
# 对三个 encoder 的最终 ckpt 跑 test 集评估。可覆盖: DATA, DATASET_ROOT, GRASPNET_BASELINE, LIFT3D_ROOT
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
GRASPNET_CKPT="${GRASPNET_CKPT:-$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar}"

echo "========== 1) LIFT3D (s4) =========="
python eval_benchmark.py \
  --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_adapter_graspnet_s4.pt --split test \
  --dataset_root "$DATASET_ROOT" \
  --lift3d_root "$LIFT3D_ROOT" \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE"

echo ""
echo "========== 2) VGGT Base (s2) =========="
python eval_benchmark.py \
  --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_adapter_graspnet_s2.pt --split test \
  --dataset_root "$DATASET_ROOT" \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE"

echo ""
echo "========== 3) VGGT Ft (s4) =========="
python eval_benchmark.py \
  --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_adapter_graspnet_s4.pt --split test \
  --dataset_root "$DATASET_ROOT" \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE"

echo "Done. 结果见 eval_out/lift3d/ eval_out/vggt_base/ eval_out/vggt_ft/"
