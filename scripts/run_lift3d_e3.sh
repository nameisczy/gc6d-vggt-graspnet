#!/usr/bin/env bash
# E3：local residual_proj（默认 200 epoch；RESIDUAL_ALPHA 可覆盖，默认 1.0）

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_lift3d_e_common.inc.sh
source "$SCRIPT_DIR/run_lift3d_e_common.inc.sh"

: "${RESIDUAL_ALPHA:=1.0}"

_lift3d_e_build_args
python "${_LIFT3D_E_ARGS[@]}" \
  --fusion_mode residual_proj \
  --residual_alpha "$RESIDUAL_ALPHA" \
  --exp_name "e3_residual_proj"
