#!/bin/bash
# VGGT Ft + Adapter + 预训练 GraspNet：Stage1->2->3->4 全流程。
# 用法: MODE=1sample|small|full [GRASPNET_CKPT=path] ./run_train_adapter_vggt_ft.sh
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

LOG_DIR="${LOG_DIR:-logs/adapter_vggt_ft_${MODE}_$(date +%Y%m%d_%H%M%S)}"
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

for STAGE in 1 2 3 4; do
  LOAD=""
  [ "$STAGE" -gt 1 ] && LOAD="--load_ckpt checkpoints/gc6d_vggt_ft_adapter_graspnet_s$((STAGE-1))${SAVE_SUFFIX}.pt"
  echo "========== Stage $STAGE =========="
  python train_adapter_graspnet.py \
    --data_dir "$DATA" --encoder vggt_ft --stage $STAGE \
    --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
    --max_samples $MAX_SAMPLES --batch_size 4 --lr "$LR" \
    --save_name "gc6d_vggt_ft_adapter_graspnet_s${STAGE}${SAVE_SUFFIX}" \
    $EXTRA_ARGS $LOAD
done
echo "Done. Final: checkpoints/gc6d_vggt_ft_adapter_graspnet_s4${SAVE_SUFFIX}.pt"
