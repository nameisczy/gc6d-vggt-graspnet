#!/usr/bin/env bash
# E1：additive 全局 cond + adapter（默认 200 epoch，环境变量 EPOCHS 可改）
# 与其它 E 脚本并行：建议不同终端 + 不同 CUDA_VISIBLE_DEVICES

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_lift3d_e_common.inc.sh
source "$SCRIPT_DIR/run_lift3d_e_common.inc.sh"

_lift3d_e_build_args
python "${_LIFT3D_E_ARGS[@]}" \
  --fusion_mode additive \
  --exp_name "e1_additive_global_cond"
