#!/usr/bin/env bash
# E2：local concat_proj（默认 200 epoch）

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_lift3d_e_common.inc.sh
source "$SCRIPT_DIR/run_lift3d_e_common.inc.sh"

_lift3d_e_build_args
python "${_LIFT3D_E_ARGS[@]}" \
  --fusion_mode concat_proj \
  --exp_name "e2_concat_proj"
