#!/usr/bin/env bash
# LoRA + reranker + ranking
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_NAME="lora_reranker"
OUTPUT_DIR="./outputs/${EXP_NAME}"
LOG_DIR="./logs/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

GRASPNET_CKPT="${GRASPNET_CKPT:-/path/to/graspnet.pth}"
DATA_DIR="${DATA_DIR:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"

echo "Running experiment: ${EXP_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"

TRAIN_CMD=( python train_residual_rerank.py
  --graspnet_ckpt "${GRASPNET_CKPT}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUTPUT_DIR}"
  --experiment_mode lora_reranker
  --loss_type ranking
  --reranker_type ranking
  --reranker_fusion add
  --reranker_lambda 0.1
  --ranking_top_k 50
  --ranking_margin 0.1
  --use_lora_head
  --lora_r 8
  --lora_alpha 16
)

EVAL_CMD=( python eval_benchmark_rewrite.py
  --graspnet_ckpt "${GRASPNET_CKPT}"
  --pipeline_checkpoint "${OUTPUT_DIR}/last.pt"
  --dump_dir "${OUTPUT_DIR}/eval"
  --tag "${EXP_NAME}"
)

echo "Training command:"
printf ' %q' "${TRAIN_CMD[@]}"
echo
echo "Evaluation command:"
printf ' %q' "${EVAL_CMD[@]}"
echo

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
"${EVAL_CMD[@]}" 2>&1 | tee "${LOG_DIR}/eval.log"

echo "Done: ${EXP_NAME}"
