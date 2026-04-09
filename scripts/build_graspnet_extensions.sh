#!/bin/bash
# 在当前 conda 环境下编译 graspnet-baseline 的 pointnet2 与 knn 扩展（解决 no module named 'pointnet2._ext' / knn 等）。
# 在 gc6d 根目录或任意处执行均可。
# 要求：本机 CUDA 与当前 PyTorch 编译所用 CUDA 一致。否则会报：
#   RuntimeError: The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.8)
# 解决二选一：(1) 安装与系统 CUDA 匹配的 PyTorch，如 pip install torch --index-url https://download.pytorch.org/whl/cu118
#             (2) 安装 CUDA 12 并让 nvcc 指向 12（或 conda 里用 cuda-toolkit），使与 PyTorch 12.x 一致
# 若报错 Unknown CUDA arch (8.9)，可先：export TORCH_CUDA_ARCH_LIST="8.0;8.6"
set -e

GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
if [ ! -d "$GRASPNET_BASELINE" ]; then
  echo "GRASPNET_BASELINE not found: $GRASPNET_BASELINE"
  exit 1
fi

echo "Building graspnet-baseline extensions in: $GRASPNET_BASELINE"
echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '?')"
if command -v nvcc >/dev/null 2>&1; then echo "nvcc: $(nvcc --version | grep release)"; fi
# PyTorch 12.x 但系统/默认 nvcc 是 11.x 会报 mismatch。优先用 conda 里的 CUDA 12（需先 conda install）
PT_CUDA=$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
if [ -n "$CONDA_PREFIX" ] && [ -n "$PT_CUDA" ] && [[ "$PT_CUDA" == 12.* ]]; then
  if [ -f "$CONDA_PREFIX/bin/nvcc" ] || [ -d "$CONDA_PREFIX/lib/libcudart"* ]; then
    export CUDA_HOME="$CONDA_PREFIX"
    echo "Using CUDA_HOME=$CUDA_HOME (conda) to match PyTorch $PT_CUDA"
  fi
  # cusparse.h 等在 conda 里可能在 targets/x86_64-linux/include，加入 include 路径
  if [ -f "$CONDA_PREFIX/targets/x86_64-linux/include/cusparse.h" ]; then
    export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
    echo "Added conda targets include for cusparse.h etc."
  fi
fi
# 若仍会检测到 11.8，需在 conda 里装 CUDA 12 再编译。
# 若报 cusparse.h: No such file or directory，说明还缺 CUDA 开发头，请装完整 toolkit：
#   conda install -c conda-forge cuda-toolkit-dev   # 或 cuda-cusparse-dev + 其它 dev
#   conda install -c nvidia cuda-toolkit

# pointnet2：setup 只有 ext_modules，无 packages，pip install -e 会报错，改用原地编译
if [ -d "$GRASPNET_BASELINE/pointnet2" ]; then
  echo "--- pointnet2 (build_ext --inplace) ---"
  cd "$GRASPNET_BASELINE/pointnet2"
  mkdir -p pointnet2
  python setup.py build_ext --inplace
  # 编译产物可能被复制到 pointnet2/_ext.*.so，若在子目录则移到当前目录以便 import pointnet2._ext
  for f in pointnet2/_ext*.so; do
    [ -f "$f" ] && mv "$f" . && echo "Moved $f to ." && break
  done
  cd - >/dev/null
else
  echo "Skip pointnet2 (dir not found)"
fi

# knn：先创建目标目录再 pip install，避免 copy 报 No such file or directory
if [ -d "$GRASPNET_BASELINE/knn" ]; then
  echo "--- knn ---"
  cd "$GRASPNET_BASELINE/knn"
  mkdir -p knn_pytorch
  if ! pip install . 2>/dev/null; then
    echo "pip install . failed, trying build_ext --inplace"
    python setup.py build_ext --inplace
  fi
  cd - >/dev/null
else
  echo "Skip knn (dir not found)"
fi

# pointnet2._ext 等加载时需要 PyTorch 的 lib（libc10.so 等），需设 LD_LIBRARY_PATH
TORCH_LIB=$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
  export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
  echo "Set LD_LIBRARY_PATH for PyTorch lib (libc10.so etc.)"
fi
echo "Done. Verify: python -c 'import pointnet2._ext; from knn_modules import knn; print(\"OK\")'"
echo "若报 libc10.so not found，请先执行: export LD_LIBRARY_PATH=\$(python -c \"import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))\"):\$LD_LIBRARY_PATH"
