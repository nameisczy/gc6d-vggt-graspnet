#!/bin/bash
# VGGT Base + Adapter + 预训练 GraspNet：只做 Stage1（adapter），再 Stage2（adapter+head）用 stage2+stage4 步数。
# 用法: MODE=1sample|small|full [GRASPNET_CKPT=path] ./run_train_adapter_vggt_base.sh
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
export PYTHONPATH="$GRASPNET_BASELINE:$GRASPNET_BASELINE/pointnet2:$GRASPNET_BASELINE/utils:$GRASPNET_BASELINE/knn:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
GRASPNET_PRETRAIN="${GRASPNET_PRETRAIN:-rs}"
if [ -z "$GRASPNET_CKPT" ]; then
  [ "$GRASPNET_PRETRAIN" = "kn" ] && GRASPNET_CKPT="$GRASPNET_BASELINE/logs/log_kn/checkpoint-kn.tar" || GRASPNET_CKPT="$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar"
fi

MODE="${MODE:-full}"
case "$MODE" in
  1sample) MAX_SAMPLES=1 ;;
  small)   MAX_SAMPLES=100 ;;
  full)    MAX_SAMPLES=0 ;;
  *)       echo "MODE must be 1sample|small|full"; exit 1 ;;
esac

LOG_DIR="${LOG_DIR:-logs/adapter_vggt_base_${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

SAVE_SUFFIX=""
[ "$MODE" = "1sample" ] && SAVE_SUFFIX="_1sample"
[ "$MODE" = "small" ]   && SAVE_SUFFIX="_small100"

LR="${LR:-1e-3}"
[ "$MODE" = "1sample" ] && LR="${LR:-3e-3}"
EXTRA_ARGS=""
[ "$MODE" = "1sample" ] && EXTRA_ARGS="--log_grad_norm"

echo "Adapter+VGGT_base+GraspNet | MODE=$MODE | lr=$LR | Stage1 then Stage2(steps=4k+2k)"

echo "========== Stage 1 (adapter only) =========="
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 1 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --max_samples $MAX_SAMPLES --batch_size 4 --lr "$LR" \
  --save_name "gc6d_vggt_base_adapter_graspnet_s1${SAVE_SUFFIX}" \
  $EXTRA_ARGS

echo "========== Stage 2 (adapter + grasp head, steps=4k+2k) =========="
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --max_samples $MAX_SAMPLES --batch_size 4 --lr "$LR" \
  --load_ckpt "checkpoints/gc6d_vggt_base_adapter_graspnet_s1${SAVE_SUFFIX}.pt" \
  --save_name "gc6d_vggt_base_adapter_graspnet_s2${SAVE_SUFFIX}" \
  $EXTRA_ARGS

echo "Done. Final ckpt: checkpoints/gc6d_vggt_base_adapter_graspnet_s2${SAVE_SUFFIX}.pt"
