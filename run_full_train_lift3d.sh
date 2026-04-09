#!/bin/bash
# LIFT3D 全量训练（Stage1->2->3->4）。可覆盖: DATA, LIFT3D_ROOT, GRASPNET_BASELINE, BATCH_SIZE
set -e
cd "$(dirname "$0")"
export MODE=full
exec bash run_train_adapter_lift3d.sh
