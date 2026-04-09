#!/bin/bash
# 在 overfit 使用的同一单样本上跑官方 benchmark（eval_grasp），验证 pipeline 是否跑通。
# 需 graspclutter6dAPI 与 GraspClutter6D 数据集根目录；与 run_overfit_test_lift3d.sh 使用相同 data_dir/split。
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
CKPT="${CKPT:-checkpoints/gc6d_lift3d_overfit_test.pt}"
OUT_DIR="${OUT_DIR:-eval_out}"
EXPORT_PLY="${EXPORT_PLY:-}"
USE_GT="${USE_GT:-}"
USE_GT_17D="${USE_GT_17D:-}"
R_FLATTEN="${R_FLATTEN:-row}"
R_PERMUTE_ALL="${R_PERMUTE_ALL:-}"
ROUNDTRIP_GT17D="${ROUNDTRIP_GT17D:-}"
OBJECT_ID_TEST="${OBJECT_ID_TEST:-}"
DEPTH="${DEPTH:-0.04}"
WIDTH_MAX="${WIDTH_MAX:-0.12}"
echo "Benchmark one sample (same as overfit: train split, 1 sample) | ckpt=$CKPT"
EXTRA=""
[ -n "$USE_GT" ] && EXTRA="$EXTRA --use_gt_as_pred"
[ -n "$USE_GT_17D" ] && EXTRA="$EXTRA --use_gt_17d"
[ -n "$EXPORT_PLY" ] && EXTRA="$EXTRA --export_ply $EXPORT_PLY"
[ -n "$R_PERMUTE_ALL" ] && EXTRA="$EXTRA --R_permute_all"
[ -n "$ROUNDTRIP_GT17D" ] && EXTRA="$EXTRA --roundtrip_gt17d"
[ -n "$OBJECT_ID_TEST" ] && EXTRA="$EXTRA --object_id_test"
EXTRA="$EXTRA --R_flatten $R_FLATTEN --depth $DEPTH --width_max $WIDTH_MAX"
python eval_benchmark.py \
  --data_dir "$DATA" \
  --checkpoint "$CKPT" \
  --split train \
  --max_samples 1 \
  --dataset_root "$DATASET_ROOT" \
  --lift3d_root "$LIFT3D_ROOT" \
  --out_dir "$OUT_DIR" \
  --top_k 50 \
  --debug_first_grasp \
  $EXTRA
echo "Done. Summary: $OUT_DIR/lift3d_clip/summary.json"
echo "Optional: USE_GT=1 | USE_GT_17D=1 | R_PERMUTE_ALL=1 | ROUNDTRIP_GT17D=1 | OBJECT_ID_TEST=1 | DEPTH=0.02 WIDTH_MAX=0.5 | EXPORT_PLY=path.ply"
