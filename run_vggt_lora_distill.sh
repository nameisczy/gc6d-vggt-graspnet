#!/usr/bin/env bash
# Train vggt_replacement_distill with VGGT encoder LoRA (--vggt_train_encoder), then eval_benchmark_rewrite.
set -euo pipefail

ROOT="${HOME}/gc6d_grasp_pipeline"
cd "$ROOT"

GRASPNET_CKPT="${GRASPNET_CKPT:-${HOME}/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}"
GRASPNET_ROOT="${GRASPNET_ROOT:-${HOME}/graspnet-baseline}"
DATA_DIR="${DATA_DIR:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
DATASET_ROOT="/mnt/ssd/ziyaochen/GraspClutter6D"
REPRO_ROOT="${REPRO_ROOT:-${HOME}/gc6d_graspnet_repro}"
CAMERA="${CAMERA:-realsense-d435}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP="vggt_lora_distill"
OUT_DIR="${ROOT}/checkpoints/alignment_runs/${EXP}_${TS}"
mkdir -p "$OUT_DIR"
LOG_TRAIN="${OUT_DIR}/train_${TS}.log"
LOG_EVAL="${OUT_DIR}/eval_${TS}.log"
EVAL_TAG="${EXP}_${TS}_eval"
EVAL_DUMP="${OUT_DIR}/eval_out_${TS}"

echo "[$(date -Iseconds)] Output dir: ${OUT_DIR}" | tee -a "${LOG_TRAIN}"
echo "[$(date -Iseconds)] START train_alignment_experiments (vggt_replacement_distill + LoRA)" | tee -a "${LOG_TRAIN}"

python train_alignment_experiments.py \
  --model_mode vggt_replacement_distill \
  --data_dir "${DATA_DIR}" \
  --dataset_root "${DATASET_ROOT}" \
  --camera "${CAMERA}" \
  --val_split test_seen \
  --graspnet_ckpt "${GRASPNET_CKPT}" \
  --graspnet_root "${GRASPNET_ROOT}" \
  --exp_name "${EXP}" \
  --out_dir "${OUT_DIR}" \
  --distill_task_weight 1.0 \
  --vggt_train_encoder \
  --vggt_train_encoder_last_n_blocks 4 \
  --encoder_lr_scale 0.1 \
  --batch_size 4 \
  --epochs 10 \
  --lr 0.001 \
  2>&1 | tee -a "${LOG_TRAIN}"

echo "[$(date -Iseconds)] Resolving checkpoint -> best.pth" | tee -a "${LOG_TRAIN}"
MAIN_PT=""
while IFS= read -r f; do
  [[ "$(basename "$f")" =~ ^epoch_ ]] && continue
  MAIN_PT="$f"
  break
done < <(ls -t "${OUT_DIR}"/*.pt 2>/dev/null || true)
if [[ -z "${MAIN_PT}" ]]; then
  echo "ERROR: no non-epoch *.pt in ${OUT_DIR}" | tee -a "${LOG_TRAIN}"
  exit 1
fi
cp -a "${MAIN_PT}" "${OUT_DIR}/best.pth"
echo "[$(date -Iseconds)] Saved ${OUT_DIR}/best.pth (from ${MAIN_PT})" | tee -a "${LOG_TRAIN}"

echo "[$(date -Iseconds)] START eval_benchmark_rewrite" | tee -a "${LOG_EVAL}"
python eval_benchmark_rewrite.py \
  --repro_root "${REPRO_ROOT}" \
  --gc6d_root "${DATASET_ROOT}" \
  --graspnet_root "${GRASPNET_ROOT}" \
  --graspnet_ckpt "${GRASPNET_CKPT}" \
  --pipeline_checkpoint "${OUT_DIR}/best.pth" \
  --camera "${CAMERA}" \
  --top_k 50 \
  --tag "${EVAL_TAG}" \
  --dump_dir "${EVAL_DUMP}" \
  --extra_stats \
  2>&1 | tee -a "${LOG_EVAL}"

echo "[$(date -Iseconds)] DONE. Train log: ${LOG_TRAIN}  Eval log: ${LOG_EVAL}" | tee -a "${LOG_EVAL}"
