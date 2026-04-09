#!/bin/bash
# 小样本 + MLP head 直接 17D（simple_17d）：VGGT base 单阶段。
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
N="${N:-100}"
BS="${BS:-32}"
VAL_EVERY="${VAL_EVERY:-100}"
LORA_R="${LORA_R:-2}"
LORA_SCALE="${LORA_SCALE:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-8e-2}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.3}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.2}"
STEPS_BASE="${STEPS_BASE:-1200}"
LR_VGGT_BASE="${LR_VGGT_BASE:-5e-4}"
LOSS_BEST_GT_WEIGHT="${LOSS_BEST_GT_WEIGHT:-0.3}"
PRED2GT_TOP_FRAC="${PRED2GT_TOP_FRAC:-1.0}"
MATCH_MODE="${MATCH_MODE:-bidir}"
LOG_DIR="${LOG_DIR:-logs/vggt_base_small_mlp17d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "VGGT base small MLP17D | N=$N steps=$STEPS_BASE best_gt_w=$LOSS_BEST_GT_WEIGHT | log=$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
python train_vggt_base.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS_BASE --lr $LR_VGGT_BASE \
  --weight_decay $WEIGHT_DECAY --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE \
  --save_name gc6d_vggt_small_base_mlp17d.pt
echo "Done. Checkpoint: checkpoints/gc6d_vggt_small_base_mlp17d.pt | Logs: $LOG_DIR"
