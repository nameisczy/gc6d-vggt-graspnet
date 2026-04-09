#!/bin/bash
# 单样本 overfit 测试：1 样本、K=20 proposals、只训练 head（Stage1）、关闭正则、2000 step、LIFT3D 点云
# 集合多峰用 K=20 更易学；--loss_components_log 看 t/rot6d/width 谁卡住；可选 --action_weights 1,0.2,0.5
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DATA="${DATA:-/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified}"
LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
LOG_DIR="${LOG_DIR:-logs/overfit_test_lift3d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "LIFT3D overfit test | 1 sample, K=20, head only, wd=0, 2000 steps | debug_grad + loss_components | log=$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
python train_stage1_freeze_encoder.py \
  --data_dir "$DATA" --max_samples 1 --batch_size 1 --max_steps 2000 \
  --lr 1e-3 --weight_decay 0 \
  --lift3d_root "$LIFT3D_ROOT" --lora_r 8 --lora_scale 1.0 \
  --head_dropout 0 --adapter_dropout 0 \
  --grasp_head_type graspnet --num_proposals 20 \
  --val_every 0 \
  --debug_grad --loss_components_log \
  --log_dir "$LOG_DIR" --save_name gc6d_lift3d_overfit_test.pt
echo "Done. Checkpoint: checkpoints/gc6d_lift3d_overfit_test.pt | Logs: $LOG_DIR"
echo "Optional: add --action_weights 1,0.2,0.5 to downweight rot6d; --use_smooth_l1; --pred2gt_top_frac 0.25"
echo "Same-sample benchmark: ./run_benchmark_one_sample.sh"
