#!/bin/bash
# One-sample pipeline: train -> eval -> visualize

set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

DATA_DIR="${DATA_DIR:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
SAVE_DIR="${SAVE_DIR:-${PWD}/checkpoints}"

echo "=== 1) Train (one sample, no reg) ==="
python train.py --data_dir "$DATA_DIR" --split train --camera realsense-d415 --max_samples 1 --max_steps 2000 --save_dir "$SAVE_DIR"

echo ""
echo "=== 2) Benchmark (need gc6d env) ==="
python eval_benchmark.py --data_dir "$DATA_DIR" --checkpoint "$SAVE_DIR/gc6d_grasp_policy_one_sample.pt" --max_samples 1 --dataset_root "${DATASET_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"

echo ""
echo "=== 3) Visualize ==="
python visualize_offline.py --data_dir "$DATA_DIR" --checkpoint "$SAVE_DIR/gc6d_grasp_policy_one_sample.pt" --max_samples 1 --num_grasps 5 --save_ply

echo "Done. Check vis_out/"
