#!/bin/bash
# 小样本 + MLP head 直接 17D（simple_17d）：LIFT3D 三阶段，与 run_small_lift3d_graspnet 同结构。
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
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
STEPS1_LIFT="${STEPS1_LIFT:-800}"
STEPS2_LIFT="${STEPS2_LIFT:-1200}"
STEPS3_LIFT="${STEPS3_LIFT:-400}"
LR1_LIFT="${LR1_LIFT:-1e-3}"
LR2_LIFT="${LR2_LIFT:-2e-5}"
LR3_LIFT="${LR3_LIFT:-2e-5}"
LR_HEAD3_LIFT="${LR_HEAD3_LIFT:-1e-3}"
LORA_LAST_N_BLOCKS="${LORA_LAST_N_BLOCKS:-2}"
LOSS_BEST_GT_WEIGHT="${LOSS_BEST_GT_WEIGHT:-0.3}"
PRED2GT_TOP_FRAC="${PRED2GT_TOP_FRAC:-1.0}"
MATCH_MODE="${MATCH_MODE:-bidir}"
LOG_DIR="${LOG_DIR:-logs/lift3d_small_mlp17d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "LIFT3D small MLP17D | N=$N S1=$STEPS1_LIFT S2=$STEPS2_LIFT S3=$STEPS3_LIFT best_gt_w=$LOSS_BEST_GT_WEIGHT | log=$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "========== LIFT3D Stage1 (simple_17d head, loss_17d) =========="
python train_stage1_freeze_encoder.py --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS1_LIFT --lr $LR1_LIFT --weight_decay $WEIGHT_DECAY --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE --save_name gc6d_lift3d_small_mlp17d_stage1.pt
echo "========== LIFT3D Stage2 =========="
python train_stage2_lora_encoder.py --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS2_LIFT --lr $LR2_LIFT --weight_decay $WEIGHT_DECAY_S2 --lora_r $LORA_R --lora_scale $LORA_SCALE --val_every $VAL_EVERY --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE --ckpt_stage1 checkpoints/gc6d_lift3d_small_mlp17d_stage1.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" --save_name gc6d_lift3d_small_mlp17d_stage2.pt
echo "========== LIFT3D Stage3 =========="
python train_stage3_joint.py --data_dir "$DATA" --max_samples $N --batch_size $BS --max_steps $STEPS3_LIFT --lr $LR3_LIFT --lr_head $LR_HEAD3_LIFT \
  --weight_decay $WEIGHT_DECAY_S3 --lora_r $LORA_R --lora_scale $LORA_SCALE --lora_last_n_blocks $LORA_LAST_N_BLOCKS \
  --val_every $VAL_EVERY --early_stop_val_worse ${EARLY_STOP_VAL_WORSE:-0} \
  --head_dropout $HEAD_DROPOUT --adapter_dropout $ADAPTER_DROPOUT --loss_17d --loss_best_gt_weight $LOSS_BEST_GT_WEIGHT --pred2gt_top_frac $PRED2GT_TOP_FRAC --match_mode $MATCH_MODE \
  --ckpt_stage2 checkpoints/gc6d_lift3d_small_mlp17d_stage2.pt --lift3d_root "$LIFT3D_ROOT" --log_dir "$LOG_DIR" --save_name gc6d_lift3d_small_mlp17d_stage3.pt
echo "Done. Checkpoints: checkpoints/gc6d_lift3d_small_mlp17d_stage*.pt | Logs: $LOG_DIR"
