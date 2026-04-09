# 小数据集跑通三种 Encoder（三阶段 max_steps 分开 + 超参可改 + log）

为公平对比 **VGGT 原始 / VGGT 微调 / LIFT3D 微调**，使用同一小数据集；**三阶段 max_steps 与 lr 等超参可分别修改**，并写入 **log**。

## 1. 一键运行（默认超参 + 自动 log）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
./run_small_dataset.sh
```

- 输出：`checkpoints/gc6d_*_small_*.pt`
- 日志：`logs/run_YYYYMMDD_HHMMSS/` 下  
  - `run.log`：脚本整次运行的 stdout/stderr（tee）  
  - `gc6d_lift3d_small_stage1.log`、`stage2.log`、`stage3.log`、`gc6d_vggt_small_base.log`、`gc6d_vggt_small_ft_stage1.log` 等：各 Python 训练脚本的 log

## 2. 可调环境变量（三阶段 max_steps 与超参分开）

在运行前导出即可，再执行 `./run_small_dataset.sh`。

### 数据与公共

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATA` | `/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified` | 数据目录 |
| `LIFT3D_ROOT` | `$HOME/LIFT3D` | LIFT3D 根目录 |
| `N` | 200 | 使用前 N 条样本 |
| `BS` | 32 | batch_size |
| `LOG_DIR` | `logs/run_YYYYMMDD_HHMMSS` | 本次运行 log 目录（不设则按时间戳生成） |

### LIFT3D 三阶段（各自 max_steps / lr）

| 变量 | 默认 | 说明 |
|------|------|------|
| `STEPS1_LIFT` | 3000 | Stage1 max_steps |
| `STEPS2_LIFT` | 2000 | Stage2 max_steps |
| `STEPS3_LIFT` | 2000 | Stage3 max_steps |
| `LR1_LIFT` | 1e-3 | Stage1 lr |
| `LR2_LIFT` | 1e-4 | Stage2 lr |
| `LR3_LIFT` | 1e-4 | Stage3 encoder lr |
| `LR_HEAD3_LIFT` | 1e-3 | Stage3 head lr |

### VGGT 原始（单阶段）

| 变量 | 默认 | 说明 |
|------|------|------|
| `STEPS_VGGT_BASE` | 3000 | max_steps |
| `LR_VGGT_BASE` | 1e-3 | lr |

### VGGT 微调三阶段

| 变量 | 默认 | 说明 |
|------|------|------|
| `STEPS1_VGGT` | 3000 | Stage1 max_steps |
| `STEPS2_VGGT` | 2000 | Stage2 max_steps |
| `STEPS3_VGGT` | 2000 | Stage3 max_steps |
| `LR1_VGGT` | 1e-3 | Stage1 lr |
| `LR2_VGGT` | 1e-5 | Stage2 lr |
| `LR3_VGGT` | 1e-5 | Stage3 encoder lr |
| `LR_HEAD3_VGGT` | 1e-3 | Stage3 head lr |

### 示例：只改 LIFT3D 三阶段步数

```bash
export STEPS1_LIFT=5000
export STEPS2_LIFT=3000
export STEPS3_LIFT=3000
./run_small_dataset.sh
```

### 示例：指定 log 目录

```bash
export LOG_DIR=logs/my_run_001
./run_small_dataset.sh
```

## 3. 单脚本单独跑（同样支持分阶段超参 + log）

每个训练脚本都支持 `--max_steps`、`--lr`、`--lr_head`（如有）、`--log_dir` 等，例如：

```bash
# LIFT3D Stage1：单独设 max_steps 与 lr，并写 log
python train_stage1_freeze_encoder.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 200 --batch_size 32 --max_steps 5000 --lr 1e-3 \
  --lift3d_root ~/LIFT3D --log_dir logs/my_lift3d_s1 \
  --save_name gc6d_lift3d_small_stage1.pt
```

- 指定 `--log_dir` 时，会在该目录下生成 `<save_name 不含扩展名>.log`（如 `gc6d_lift3d_small_stage1.log`），内容与控制台一致。

## 4. 评估与可视化

用对应 ckpt 即可，例如：

```bash
python eval_benchmark.py --checkpoint checkpoints/gc6d_lift3d_small_stage3.pt --lift3d_root ~/LIFT3D
python eval_benchmark.py --checkpoint checkpoints/gc6d_vggt_small_base.pt
python eval_benchmark.py --checkpoint checkpoints/gc6d_vggt_small_ft_stage3.pt
```
