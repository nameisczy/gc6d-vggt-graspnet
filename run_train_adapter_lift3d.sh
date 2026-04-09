#!/bin/bash
# LIFT3D + Adapter + 预训练 GraspNet：Stage1 训 adapter，Stage2 训 head，Stage3 训 encoder，Stage4 联合。
# 用法: MODE=1sample|small|full [ENCODER=lift3d|lift3d_clip] [GRASPNET_CKPT=path] ./run_train_adapter_lift3d.sh
# ENCODER=lift3d_clip 时使用 LIFT3D 官方 lift3d_clip_base() 预训练（CLIP+MAE），见 https://github.com/PKU-HMI-Lab/LIFT3D
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
# LIFT3D PointNext 预训练（可选）：设 LIFT3D_CKPT 后 encoder 会加载该权重，否则随机初始化
LIFT3D_CKPT="${LIFT3D_CKPT:-}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
# 与 LIFT3D_ROOT/VGGT_ROOT 一致：把 graspnet-baseline 加入 PYTHONPATH，便于加载 pointnet2/_ext 等
export PYTHONPATH="$GRASPNET_BASELINE:$GRASPNET_BASELINE/pointnet2:$GRASPNET_BASELINE/utils:$GRASPNET_BASELINE/knn:${PYTHONPATH:-}"
# pointnet2._ext 等加载需要 PyTorch 的 lib（libc10.so）
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
# 预训练二选一：rs=realsense（默认，与 GC6D realsense-d415 一致），kn=kinect
GRASPNET_PRETRAIN="${GRASPNET_PRETRAIN:-rs}"
if [ -z "$GRASPNET_CKPT" ]; then
  if [ "$GRASPNET_PRETRAIN" = "kn" ]; then
    GRASPNET_CKPT="$GRASPNET_BASELINE/logs/log_kn/checkpoint-kn.tar"
  else  
    GRASPNET_CKPT="$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar"
  fi
fi
if [ ! -f "$GRASPNET_CKPT" ]; then
  echo "WARN: GraspNet checkpoint not found: $GRASPNET_CKPT (set GRASPNET_CKPT or GRASPNET_PRETRAIN=rs|kn)"
fi

ENCODER="${ENCODER:-lift3d}"
[ "$ENCODER" != "lift3d" ] && [ "$ENCODER" != "lift3d_clip" ] && echo "ENCODER must be lift3d or lift3d_clip" && exit 1

MODE="${MODE:-full}"
case "$MODE" in
  1sample) MAX_SAMPLES=1 ;;
  small)   MAX_SAMPLES=100 ;;
  full)    MAX_SAMPLES=0 ;;
  *)       echo "MODE must be 1sample|small|full"; exit 1 ;;
esac

LOG_DIR="${LOG_DIR:-logs/adapter_${ENCODER}_${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
TRAIN_LOG="$LOG_DIR/train.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "Logs: 全部输出已写入 $RUN_LOG，训练/val/debug 同时写入 $TRAIN_LOG"

SAVE_SUFFIX=""
[ "$MODE" = "1sample" ] && SAVE_SUFFIX="_1sample"
[ "$MODE" = "small" ]   && SAVE_SUFFIX="_small100"
# 保存名带 encoder 区分，避免 lift3d 与 lift3d_clip 互相覆盖
SAVE_PREFIX="gc6d_${ENCODER}_adapter_graspnet"

# 单样本用更大 lr 并打梯度范数便于排查
LR="${LR:-1e-3}"
[ "$MODE" = "1sample" ] && LR="${LR:-3e-3}"
EXTRA_ARGS="--log_file $TRAIN_LOG"
[ "$MODE" = "1sample" ] && EXTRA_ARGS="$EXTRA_ARGS --log_grad_norm"
# 训练目标 vs benchmark 诊断：每 N 步打 rank_corr + debug_AP（需 GC6D 根目录）
[ -n "${DEBUG_BENCHMARK_EVERY:-}" ] && [ "${DEBUG_BENCHMARK_EVERY:-0}" -gt 0 ] && EXTRA_ARGS="$EXTRA_ARGS --debug_benchmark_every $DEBUG_BENCHMARK_EVERY --debug_benchmark_n ${DEBUG_BENCHMARK_N:-64}"
[ -n "${DEBUG_DATASET_ROOT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --debug_dataset_root $DEBUG_DATASET_ROOT"

echo "Adapter+LIFT3D+GraspNet | ENCODER=$ENCODER MODE=$MODE max_samples=$MAX_SAMPLES | lr=$LR | pretrain=$GRASPNET_PRETRAIN ckpt=$GRASPNET_CKPT"

for STAGE in 1 2 3 4; do
  LOAD=""
  if [ "$STAGE" -gt 1 ]; then
    PREV=$((STAGE-1))
    LOAD="--load_ckpt checkpoints/${SAVE_PREFIX}_s${PREV}${SAVE_SUFFIX}.pt"
  fi
  echo "========== Stage $STAGE =========="
  python train_adapter_graspnet.py \
    --data_dir "$DATA" --encoder "$ENCODER" --stage $STAGE \
    --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" --lift3d_root "$LIFT3D_ROOT" \
    $([ -n "$LIFT3D_CKPT" ] && [ "$ENCODER" = "lift3d" ] && echo "--lift3d_ckpt $LIFT3D_CKPT") \
    --max_samples $MAX_SAMPLES --batch_size 4 --lr "$LR" \
    --save_name "${SAVE_PREFIX}_s${STAGE}${SAVE_SUFFIX}" \
    $EXTRA_ARGS $LOAD
done
echo "Done. Checkpoints: checkpoints/${SAVE_PREFIX}_s*${SAVE_SUFFIX}.pt"
