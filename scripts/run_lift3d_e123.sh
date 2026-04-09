#!/usr/bin/env bash
# E1/E2/E3 已拆成独立脚本，便于多终端并行训练（默认各 200 epoch）。
# 本脚本仅打印用法；若仍需串行跑完三个实验，请手动依次执行 e1→e2→e3。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "并行训练（推荐，各 200 epoch，可在 3 个 tmux pane / 终端同时跑）："
echo "  cd $ROOT"
echo "  # 多卡示例："
echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/run_lift3d_e1.sh"
echo "  CUDA_VISIBLE_DEVICES=1 bash scripts/run_lift3d_e2.sh"
echo "  CUDA_VISIBLE_DEVICES=2 bash scripts/run_lift3d_e3.sh"
echo ""
echo "单卡请错开时间或改小 BS，避免 OOM。"
echo "覆盖 epoch：EPOCHS=200 bash scripts/run_lift3d_e1.sh  （默认已是 200）"
echo ""
echo "若坚持串行一次性跑完："
echo "  bash scripts/run_lift3d_e1.sh && bash scripts/run_lift3d_e2.sh && bash scripts/run_lift3d_e3.sh"
