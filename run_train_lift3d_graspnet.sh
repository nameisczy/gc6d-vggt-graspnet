#!/bin/bash
# LIFT3D + GraspNet head：全量数据，三阶段。与 run_train_lift3d.sh 同结构，head 为 graspnet，并加 loss_best_gt_weight。
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
LOG_DIR="${LOG_DIR:-logs/lift3d_graspnet_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"

STEPS1="${STEPS1:-1000}"
STEPS2="${STEPS2:-4000}"
STEPS3="${STEPS3:-2000}"
LR1="${LR1:-1e-3}"
# 改进前设定：LR2=1e-4, LR3=5e-5（曾试过 3e-4/1e-4，loss 略降但 test 成功率反而略低，故保持保守便于与小批量对比）
LR2="${LR2:-1e-4}"
LR3="${LR3:-5e-5}"
LR_HEAD3="${LR_HEAD3:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
LORA_R="${LORA_R:-8}"
LORA_SCALE="${LORA_SCALE:-1.0}"
WEIGHT_DECAY_S2="${WEIGHT_DECAY_S2:-$WEIGHT_DECAY}"
LORA_R_S2="${LORA_R_S2:-$LORA_R}"
WEIGHT_DECAY_S3="${WEIGHT_DECAY_S3:-2e-2}"
LORA_R_S3="${LORA_R_S3:-$LORA_R}"
BS="${BS:-32}"
VAL_EVERY="${VAL_EVERY:-500}"
NUM_PROPOSALS="${NUM_PROPOSALS:-4}"
LOSS_BEST_GT_WEIGHT="${LOSS_BEST_GT_WEIGHT:-0.3}"
PRED2GT_TOP_FRAC="${PRED2GT_TOP_FRAC:-1.0}"
MATCH_MODE="${MATCH_MODE:-bidir}"

echo "LIFT3D GraspNet full-data | S1=$STEPS1 S2=$STEPS2 S3=$STEPS3 | LR2=$LR2 LR3=$LR3 best_gt_w=$LOSS_BEST_GT_WEIGHT pred2gt_frac=$PRED2GT_TOP_FRAC match=$MATCH_MODE | log=$LOG_DIR | val_every=$VAL_EVERY"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "========== LIFT3D Stage1 (graspnet head) =========="
python train_stage1_freeze_encoder.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS1 --lr $LR1 \
  --weight_decay $WEIGHT_DECAY --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --lora_r $LORA_R --lora_scale $LORA_SCALE \
  --grasp_head_type graspnet --num_proposals $NUM_PROPOSALS --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT \
  --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE \
  --val_every $VAL_EVERY \
  --save_name gc6d_lift3d_graspnet_stage1.pt

echo "========== LIFT3D Stage2 =========="
python train_stage2_lora_encoder.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS2 --lr $LR2 \
  --weight_decay $WEIGHT_DECAY_S2 \
  --lora_r $LORA_R_S2 --lora_scale $LORA_SCALE \
  --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE \
  --ckpt_stage1 checkpoints/gc6d_lift3d_graspnet_stage1.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --val_every $VAL_EVERY \
  --save_name gc6d_lift3d_graspnet_stage2.pt

echo "========== LIFT3D Stage3 =========="
python train_stage3_joint.py \
  --data_dir "$DATA" --max_samples 0 --batch_size $BS --max_steps $STEPS3 --lr $LR3 --lr_head $LR_HEAD3 \
  --weight_decay $WEIGHT_DECAY_S3 \
  --lora_r $LORA_R_S3 --lora_scale $LORA_SCALE \
  --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE \
  --ckpt_stage2 checkpoints/gc6d_lift3d_graspnet_stage2.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" \
  --val_every $VAL_EVERY \
  --save_name gc6d_lift3d_graspnet_stage3.pt

echo "Done. Checkpoints: checkpoints/gc6d_lift3d_graspnet_stage*.pt | Logs: $LOG_DIR"
