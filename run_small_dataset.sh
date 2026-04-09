#!/bin/bash
# 小数据集上跑通三种 encoder（可分别跑：run_small_vggt_base.sh / run_small_lift3d.sh / run_small_vggt_ft.sh）。
# 步数：VGGT base = STEPS_BASE；LIFT3D/VGGT ft 满足 S1+S3=STEPS_BASE，便于观察对比。

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# ---------- 数据与公共（N=100 小样本：微调易过拟合→少步+强正则；base 给足步数让 loss 下去） ----------
DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
N="${N:-100}"
BS="${BS:-32}"
VAL_EVERY="${VAL_EVERY:-100}"
LORA_R="${LORA_R:-2}"
LORA_SCALE="${LORA_SCALE:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.3}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.2}"

# ---------- 步数基准：VGGT base 用 STEPS_BASE；三阶段 S1+S3=STEPS_BASE ----------
STEPS_BASE="${STEPS_BASE:-1200}"
STEPS1_LIFT="${STEPS1_LIFT:-600}"
STEPS2_LIFT="${STEPS2_LIFT:-400}"
STEPS3_LIFT="${STEPS3_LIFT:-600}"
LR1_LIFT="${LR1_LIFT:-1e-3}"
LR2_LIFT="${LR2_LIFT:-3e-5}"
LR3_LIFT="${LR3_LIFT:-3e-5}"
LR_HEAD3_LIFT="${LR_HEAD3_LIFT:-1e-3}"

# ---------- VGGT 原始（单阶段，步数=STEPS_BASE） ----------
STEPS_VGGT_BASE="${STEPS_VGGT_BASE:-$STEPS_BASE}"
LR_VGGT_BASE="${LR_VGGT_BASE:-5e-4}"

# ---------- VGGT 微调三阶段（S1+S3=STEPS_BASE） ----------
STEPS1_VGGT="${STEPS1_VGGT:-600}"
STEPS2_VGGT="${STEPS2_VGGT:-400}"
STEPS3_VGGT="${STEPS3_VGGT:-600}"
LR1_VGGT="${LR1_VGGT:-1e-3}"
LR2_VGGT="${LR2_VGGT:-2e-6}"
LR3_VGGT="${LR3_VGGT:-2e-6}"
LR_HEAD3_VGGT="${LR_HEAD3_VGGT:-1e-3}"

# ---------- 日志目录（所有 Python 的 --log_dir；若需整次运行日志请用: ./run_small_dataset.sh 2>&1 | tee logs/run.log） ----------
LOG_DIR="${LOG_DIR:-logs/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "LOG_DIR=$LOG_DIR N=$N BS=$BS val_every=$VAL_EVERY lora_r=$LORA_R head_dropout=$HEAD_DROPOUT adapter_dropout=$ADAPTER_DROPOUT | LIFT: S1=$STEPS1_LIFT S2=$STEPS2_LIFT S3=$STEPS3_LIFT | VGGT base: $STEPS_VGGT_BASE | VGGT ft: S1=$STEPS1_VGGT S2=$STEPS2_VGGT S3=$STEPS3_VGGT"

# ---------- 1. LIFT3D 微调（三阶段） ----------
echo "========== LIFT3D Stage1 (max_steps=$STEPS1_LIFT lr=$LR1_LIFT N=$N) =========="
python train_stage1_freeze_encoder.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS1_LIFT --lr $LR1_LIFT \
  --weight_decay $WEIGHT_DECAY --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --save_name gc6d_lift3d_small_stage1.pt

echo "========== LIFT3D Stage2 max_steps=$STEPS2_LIFT lr=$LR2_LIFT =========="
python train_stage2_lora_encoder.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS2_LIFT --lr $LR2_LIFT \
  --weight_decay $WEIGHT_DECAY --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --ckpt_stage1 checkpoints/gc6d_lift3d_small_stage1.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --save_name gc6d_lift3d_small_stage2.pt

echo "========== LIFT3D Stage3 max_steps=$STEPS3_LIFT lr=$LR3_LIFT lr_head=$LR_HEAD3_LIFT =========="
python train_stage3_joint.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS3_LIFT --lr $LR3_LIFT --lr_head $LR_HEAD3_LIFT \
  --weight_decay $WEIGHT_DECAY --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --ckpt_stage2 checkpoints/gc6d_lift3d_small_stage2.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --save_name gc6d_lift3d_small_stage3.pt

# ---------- 2. VGGT 原始（单阶段） ----------
echo "========== VGGT base max_steps=$STEPS_VGGT_BASE lr=$LR_VGGT_BASE =========="
python train_vggt_base.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS_VGGT_BASE --lr $LR_VGGT_BASE \
  --weight_decay $WEIGHT_DECAY --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --save_name gc6d_vggt_small_base.pt

# ---------- 3. VGGT 微调（三阶段） ----------
echo "========== VGGT ft Stage1 max_steps=$STEPS1_VGGT lr=$LR1_VGGT =========="
python train_vggt_ft_stage1.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS1_VGGT --lr $LR1_VGGT \
  --weight_decay $WEIGHT_DECAY --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --save_name gc6d_vggt_small_ft_stage1.pt

echo "========== VGGT ft Stage2 max_steps=$STEPS2_VGGT lr=$LR2_VGGT =========="
python train_vggt_ft_stage2.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS2_VGGT --lr $LR2_VGGT \
  --weight_decay $WEIGHT_DECAY --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --ckpt_stage1 checkpoints/gc6d_vggt_small_ft_stage1.pt \
  --log_dir "$LOG_DIR" \
  --save_name gc6d_vggt_small_ft_stage2.pt

echo "========== VGGT ft Stage3 (max_steps=$STEPS3_VGGT lr=$LR3_VGGT lr_head=$LR_HEAD3_VGGT) =========="
python train_vggt_ft_stage3.py \
  --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS3_VGGT --lr $LR3_VGGT --lr_head $LR_HEAD3_VGGT \
  --weight_decay $WEIGHT_DECAY --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT \
  --ckpt_stage2 checkpoints/gc6d_vggt_small_ft_stage2.pt \
  --log_dir "$LOG_DIR" \
  --save_name gc6d_vggt_small_ft_stage3.pt

echo "Done. Checkpoints: checkpoints/gc6d_*_small_*.pt"
echo "Logs: $LOG_DIR (run.log + per-script *.log)"
