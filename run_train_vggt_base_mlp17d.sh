#!/bin/bash
# VGGT base + MLP head 直接 17D（simple_17d）：全量数据，单阶段。
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LOG_DIR="${LOG_DIR:-logs/vggt_base_mlp17d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"

STEPS="${STEPS:-3000}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
BS="${BS:-32}"
VAL_EVERY="${VAL_EVERY:-500}"
LORA_R="${LORA_R:-8}"
LORA_SCALE="${LORA_SCALE:-1.0}"
LOSS_BEST_GT_WEIGHT="${LOSS_BEST_GT_WEIGHT:-0.3}"
PRED2GT_TOP_FRAC="${PRED2GT_TOP_FRAC:-1.0}"
MATCH_MODE="${MATCH_MODE:-bidir}"

echo "VGGT base MLP17D full-data | steps=$STEPS lr=$LR | best_gt_w=$LOSS_BEST_GT_WEIGHT pred2gt_frac=$PRED2GT_TOP_FRAC match=$MATCH_MODE | log=$LOG_DIR | val_every=$VAL_EVERY"
exec > >(tee -a "$RUN_LOG") 2>&1

python train_vggt_base.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS --lr $LR \
  --weight_decay $WEIGHT_DECAY --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE \
  --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE \
  --val_every $VAL_EVERY \
  --save_name gc6d_vggt_base_mlp17d.pt

echo "Done. Checkpoint: checkpoints/gc6d_vggt_base_mlp17d.pt | Logs: $LOG_DIR"
