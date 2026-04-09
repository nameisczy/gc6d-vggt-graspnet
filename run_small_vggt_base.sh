#!/bin/bash
# 小样本：仅 VGGT base。步数 STEPS_BASE=S1+S3(1200)，与 LIFT3D/VGGT ft 对齐。
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
# VGGT base 步数 = S1+S3（与三阶段 S1+S3=1200 对齐）
STEPS_BASE="${STEPS_BASE:-1200}"
LR_VGGT_BASE="${LR_VGGT_BASE:-5e-4}"
LOG_DIR="${LOG_DIR:-logs/vggt_base_small_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "VGGT base small | N=$N steps=$STEPS_BASE (S1+S3) val_every=$VAL_EVERY | log=$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
python train_vggt_base.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS_BASE --lr $LR_VGGT_BASE \
  --weight_decay $WEIGHT_DECAY --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --save_name gc6d_vggt_small_base.pt
echo "Done. Checkpoint: checkpoints/gc6d_vggt_small_base.pt | Logs: $LOG_DIR"
