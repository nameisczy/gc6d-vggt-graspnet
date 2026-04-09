#!/bin/bash
# LIFT3D 微调：全量数据，三阶段。步数按实际收敛/过拟合重设：Stage1 早收敛少步，Stage2 易过拟合控制步数。
# 可覆盖：DATA, LIFT3D_ROOT, LOG_DIR, STEPS1, STEPS2, STEPS3, LR1, LR2, LR3, LR_HEAD3, WEIGHT_DECAY, WEIGHT_DECAY_S2, LORA_R_S2, WEIGHT_DECAY_S3, LORA_R_S3

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
LOG_DIR="${LOG_DIR:-logs/lift3d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"

# 三阶段 step：Stage1 早收敛 1000；Stage2 控步防过拟合 4000；Stage3 联合也易过拟合，默认 2000
STEPS1="${STEPS1:-1000}"
STEPS2="${STEPS2:-4000}"
STEPS3="${STEPS3:-2000}"
LR1="${LR1:-1e-3}"
LR2="${LR2:-1e-4}"
LR3="${LR3:-5e-5}"
LR_HEAD3="${LR_HEAD3:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
LORA_R="${LORA_R:-8}"
LORA_SCALE="${LORA_SCALE:-1.0}"
# Stage2 可单独加强正则/LoRA（默认与 Stage1 一致）
WEIGHT_DECAY_S2="${WEIGHT_DECAY_S2:-$WEIGHT_DECAY}"
LORA_R_S2="${LORA_R_S2:-$LORA_R}"
# Stage3 与 Stage2 一样加强正则、降学习率，可单独覆盖
WEIGHT_DECAY_S3="${WEIGHT_DECAY_S3:-2e-2}"
LORA_R_S3="${LORA_R_S3:-$LORA_R}"
BS="${BS:-32}"
VAL_EVERY="${VAL_EVERY:-500}"

# 若仍过拟合：STEPS2=3000、STEPS3=1500 或 LORA_R_S3=4、WEIGHT_DECAY_S3=3e-2

echo "LIFT3D full-data | S1=$STEPS1 S2=$STEPS2 S3=$STEPS3 | LR2=$LR2 wd_s2=$WEIGHT_DECAY_S2 lora_s2=$LORA_R_S2 | LR3=$LR3 wd_s3=$WEIGHT_DECAY_S3 lora_s3=$LORA_R_S3 | log=$LOG_DIR | val_every=$VAL_EVERY"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "========== LIFT3D Stage1 =========="
python train_stage1_freeze_encoder.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS1 --lr $LR1 \
  --weight_decay $WEIGHT_DECAY --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE \
  --val_every $VAL_EVERY \
  --save_name gc6d_lift3d_stage1.pt

echo "========== LIFT3D Stage2 =========="
python train_stage2_lora_encoder.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS2 --lr $LR2 \
  --weight_decay $WEIGHT_DECAY_S2 \
  --lora_r $LORA_R_S2 --lora_scale $LORA_SCALE \
  --ckpt_stage1 checkpoints/gc6d_lift3d_stage1.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --val_every $VAL_EVERY \
  --save_name gc6d_lift3d_stage2.pt

echo "========== LIFT3D Stage3 =========="
python train_stage3_joint.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS3 --lr $LR3 --lr_head $LR_HEAD3 \
  --weight_decay $WEIGHT_DECAY_S3 \
  --lora_r $LORA_R_S3 --lora_scale $LORA_SCALE \
  --ckpt_stage2 checkpoints/gc6d_lift3d_stage2.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --val_every $VAL_EVERY \
  --save_name gc6d_lift3d_stage3.pt

echo "Done. Checkpoints: checkpoints/gc6d_lift3d_stage*.pt | Logs: $LOG_DIR"
