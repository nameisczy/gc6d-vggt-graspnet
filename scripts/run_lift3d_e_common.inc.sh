# shellcheck shell=bash
# 由 run_lift3d_e{1,2,3}.sh source；勿直接执行。
# 并行训练时在各自终端设置不同 CUDA_VISIBLE_DEVICES（若多卡）。

: "${DATA:=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
: "${GC6D_ROOT:=/mnt/ssd/ziyaochen/GraspClutter6D}"
: "${LIFT3D_ROOT:=$HOME/LIFT3D}"
: "${GRASPNET_BASELINE:=$HOME/graspnet-baseline}"
: "${GRASPNET_CKPT:=$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar}"
: "${CAM:=realsense-d415}"

# E1/E2/E3 默认 200 epoch，可用环境变量覆盖
: "${EPOCHS:=200}"
: "${BS:=4}"
: "${LR:=1e-3}"
: "${SEED:=42}"

_lift3d_e_build_args() {
  _LIFT3D_E_ARGS=(
    train_lift3d_pipeline.py
    --data_dir "$DATA"
    --dataset_root "$GC6D_ROOT"
    --camera "$CAM"
    --epochs "$EPOCHS"
    --batch_size "$BS"
    --lr "$LR"
    --seed "$SEED"
    --graspnet_ckpt "$GRASPNET_CKPT"
    --graspnet_root "$GRASPNET_BASELINE"
    --lift3d_root "$LIFT3D_ROOT"
    --collision_aux
    --run_eval_after
    --eval_extra_stats
  )
  if [[ -n "${LIFT3D_CKPT:-}" ]]; then
    _LIFT3D_E_ARGS+=(--lift3d_ckpt "$LIFT3D_CKPT")
  fi
}
