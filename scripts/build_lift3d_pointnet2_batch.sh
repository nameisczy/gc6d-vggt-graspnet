#!/bin/bash
# 在当前 conda 环境下编译 LIFT3D 内嵌的 OpenPoints pointnet2_batch（解决 No module named 'pointnet2_batch_cuda'）。
# 若你统一在 gc6d 环境跑 LIFT3D adapter 训练，请先 conda activate gc6d，再执行本脚本。
set -e

LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
DIR="$LIFT3D_ROOT/lift3d/models/point_next/openpoints/cpp/pointnet2_batch"
if [ ! -d "$DIR" ]; then
  echo "LIFT3D pointnet2_batch dir not found: $DIR (set LIFT3D_ROOT if needed)"
  exit 1
fi

echo "Building LIFT3D pointnet2_batch in: $DIR"
echo "Python: $(which python)"
cd "$DIR"
# 用 build_ext --inplace 避免 pip 隔离环境里没有 torch 导致 build 失败
python setup.py build_ext --inplace
echo "Done. pointnet2_batch_cuda should now be importable in this environment (from $DIR)."
