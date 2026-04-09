#!/bin/bash
# 小样本：仅 VGGT 微调三阶段。S2 步数最多；S3=300+early stop 防 joint 爆。
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
N="${N:-100}"
BS="${BS:-32}"
VAL_EVERY="${VAL_EVERY:-100}"
LORA_R="${LORA_R:-2}"
LORA_SCALE="${LORA_SCALE:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
WEIGHT_DECAY_S2="${WEIGHT_DECAY_S2:-8e-2}"
WEIGHT_DECAY_S3="${WEIGHT_DECAY_S3:-1e-1}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.3}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.2}"
STEPS_BASE="${STEPS_BASE:-1200}"
STEPS1_VGGT="${STEPS1_VGGT:-800}"
STEPS2_VGGT="${STEPS2_VGGT:-1200}"
STEPS3_VGGT="${STEPS3_VGGT:-400}"
LR1_VGGT="${LR1_VGGT:-1e-3}"
LR2_VGGT="${LR2_VGGT:-1e-6}"
LR3_VGGT="${LR3_VGGT:-1e-6}"
LR_HEAD3_VGGT="${LR_HEAD3_VGGT:-1e-3}"
LORA_LAST_N_BLOCKS="${LORA_LAST_N_BLOCKS:-2}"
LOG_DIR="${LOG_DIR:-logs/vggt_ft_small_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "VGGT ft small | N=$N S1=$STEPS1_VGGT S2=$STEPS2_VGGT S3=$STEPS3_VGGT (S1+S3=$STEPS_BASE) wd_s2=$WEIGHT_DECAY_S2 wd_s3=$WEIGHT_DECAY_S3 | log=$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "========== VGGT ft Stage1 =========="
python train_vggt_ft_stage1.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS1_VGGT --lr $LR1_VGGT \
  --weight_decay $WEIGHT_DECAY --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --save_name gc6d_vggt_small_ft_stage1.pt
echo "========== VGGT ft Stage2 =========="
python train_vggt_ft_stage2.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS2_VGGT --lr $LR2_VGGT \
  --weight_decay $WEIGHT_DECAY_S2 --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --ckpt_stage1 checkpoints/gc6d_vggt_small_ft_stage1.pt \
  --log_dir "$LOG_DIR" \
  --save_name gc6d_vggt_small_ft_stage2.pt
echo "========== VGGT ft Stage3 (lora_last_n_blocks=$LORA_LAST_N_BLOCKS, early_stop_val_worse=${EARLY_STOP_VAL_WORSE:-0}) =========="
python train_vggt_ft_stage3.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS3_VGGT --lr $LR3_VGGT --lr_head $LR_HEAD3_VGGT \
  --weight_decay $WEIGHT_DECAY_S3 --lora_r $LORA_R --lora_scale $LORA_SCALE --lora_last_n_blocks $LORA_LAST_N_BLOCKS \
  --val_every $VAL_EVERY --early_stop_val_worse ${EARLY_STOP_VAL_WORSE:-0} \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --ckpt_stage2 checkpoints/gc6d_vggt_small_ft_stage2.pt \
  --log_dir "$LOG_DIR" \
  --save_name gc6d_vggt_small_ft_stage3.pt \
  --save_best_ckpt
echo "Done. Checkpoints: checkpoints/gc6d_vggt_small_ft_stage*.pt | Logs: $LOG_DIR"
