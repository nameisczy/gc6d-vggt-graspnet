#!/bin/bash
# VGGT 原始：全量数据，冻结 encoder 只训 head（单阶段），带 head 正则。
# 默认 3000 step 与 ft 的 stage1+stage3（1000+2000）对齐，并减轻轻微过拟合倾向；可覆盖 STEPS、LR、WEIGHT_DECAY。

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LOG_DIR="${LOG_DIR:-logs/vggt_base_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"

STEPS="${STEPS:-3000}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
BS="${BS:-32}"
VAL_EVERY="${VAL_EVERY:-500}"
LORA_R="${LORA_R:-8}"
LORA_SCALE="${LORA_SCALE:-1.0}"

# 若仍过拟合：STEPS=2000 或 WEIGHT_DECAY=2e-2、LR=5e-4
echo "VGGT base full-data | steps=$STEPS lr=$LR weight_decay=$WEIGHT_DECAY | lora_r=$LORA_R lora_scale=$LORA_SCALE | log=$LOG_DIR | val_every=$VAL_EVERY"
exec > >(tee -a "$RUN_LOG") 2>&1

python train_vggt_base.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS --lr $LR \
  --weight_decay $WEIGHT_DECAY --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE \
  --val_every $VAL_EVERY \
  --save_name gc6d_vggt_base.pt

echo "Done. Checkpoint: checkpoints/gc6d_vggt_base.pt | Logs: $LOG_DIR"
