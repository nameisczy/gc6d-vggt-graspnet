#!/bin/bash
# VGGT Base 全量训练（Stage1->Stage2）。可覆盖: DATA, GRASPNET_BASELINE, BATCH_SIZE
set -e
cd "$(dirname "$0")"
export MODE=full
exec bash run_train_adapter_vggt_base.sh
