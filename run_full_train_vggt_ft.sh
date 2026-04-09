#!/bin/bash
# VGGT Ft 全量训练（Stage1->2->3->4）。默认 BATCH_SIZE=2 防 OOM；可覆盖: DATA, GRASPNET_BASELINE, BATCH_SIZE
set -e
cd "$(dirname "$0")"
export MODE=full
export BATCH_SIZE="${BATCH_SIZE:-2}"
exec bash run_train_adapter_vggt_ft.sh
