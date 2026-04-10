#!/usr/bin/env bash
# 顺序运行全部 residual rerank 实验脚本（训练 + eval_benchmark_rewrite）
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Master: running all experiments from ${SCRIPT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Set GRASPNET_CKPT and optionally DATA_DIR before running if needed."
echo

EXPS=(
  run_exp_baseline.sh
  run_exp_lora_only.sh
  run_exp_reranker_only_default.sh
  run_exp_lora_reranker.sh
  run_exp_reranker_3d_only.sh
  run_exp_reranker_unbounded.sh
  run_exp_reranker_no_center_norm.sh
  run_exp_reranker_mul.sh
)

for name in "${EXPS[@]}"; do
  echo "================================================================================"
  echo "Starting: ${name}"
  echo "================================================================================"
  bash "${SCRIPT_DIR}/${name}"
done

echo "All experiments finished."
