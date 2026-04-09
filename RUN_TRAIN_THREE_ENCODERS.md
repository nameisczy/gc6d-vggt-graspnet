# 三种 Encoder 分开训练（全量数据 + head 正则 + step 侧重 encoder）

对比 **LIFT3D 微调 / VGGT 原始 / VGGT 微调** 的 3D 表示时，用**全量数据**、**grasp head 加正则**，且**三阶段 step 侧重 encoder 训练**。

## 1. Grasp head 正则

- 所有训练脚本对 **head 与 adapter** 使用 **AdamW + weight_decay**（默认 `1e-2`），作为 L2 正则。
- 可通过 `--weight_decay`（或脚本内 `WEIGHT_DECAY`）调节。

## 2. 三阶段 step 分配思路

目标是对比 **encoder 的 3D 表示**，因此：

- **Stage1**：冻结 encoder，只训 head。让 head 适应冻结特征即可，步数不必多。**默认 2000**。
- **Stage2**：冻结 head，只训 encoder（LoRA 或全量）。重点练 3D 表示，步数给足。**默认 12000**。
- **Stage3**：encoder + head 联合微调，避免过拟合、步数适中。**默认 4000**。

即：**少 → 多 → 中**（2000 / 12000 / 4000）。若数据量或显存差异大，可按比例缩放或只改 Stage2。

## 3. 全量数据

- 脚本中统一使用 **`--max_samples 0`** 表示使用 **index 中全部样本**（不截断）。
- `--batch_size` 默认 32；可通过环境变量 `BS` 覆盖。

## 4. 三种 encoder 分开训练指令

三条命令互不依赖，可单独、按任意顺序运行。

### LIFT3D 微调（点云，三阶段）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
./run_train_lift3d.sh
```

- 输出 ckpt：`checkpoints/gc6d_lift3d_stage1.pt`、`stage2.pt`、`stage3.pt`
- 日志：`logs/lift3d_YYYYMMDD_HHMMSS/`（含 `run.log` 与各阶段脚本的 log）
- 可覆盖：`DATA`、`LIFT3D_ROOT`、`LOG_DIR`、`STEPS1`、`STEPS2`、`STEPS3`、`LR1`、`LR2`、`LR3`、`LR_HEAD3`、`WEIGHT_DECAY`、`BS`

### VGGT 原始（RGB，单阶段）

```bash
./run_train_vggt_base.sh
```

- 输出 ckpt：`checkpoints/gc6d_vggt_base.pt`
- 日志：`logs/vggt_base_YYYYMMDD_HHMMSS/`
- 可覆盖：`DATA`、`LOG_DIR`、`STEPS`、`LR`、`WEIGHT_DECAY`、`BS`

### VGGT 微调（RGB，三阶段）

```bash
./run_train_vggt_ft.sh
```

- 输出 ckpt：`checkpoints/gc6d_vggt_ft_stage1.pt`、`stage2.pt`、`stage3.pt`
- 日志：`logs/vggt_ft_YYYYMMDD_HHMMSS/`
- 可覆盖：`DATA`、`LOG_DIR`、`STEPS1`、`STEPS2`、`STEPS3`、`LR1`、`LR2`、`LR3`、`LR_HEAD3`、`WEIGHT_DECAY`、`BS`

## 5. 只改 step 或数据量的例子

```bash
# LIFT3D：加大 Stage2
STEPS2=20000 ./run_train_lift3d.sh

# 指定数据目录和 log 目录
DATA=/path/to/offline_unified LOG_DIR=logs/exp1 ./run_train_lift3d.sh
```

## 6. 默认 step 汇总

| 流程 | Stage1 | Stage2 | Stage3 |
|------|--------|--------|--------|
| LIFT3D 微调 | 2000 | **12000** | 4000 |
| VGGT 原始 | — | — | 6000（单阶段） |
| VGGT 微调 | 2000 | **12000** | 4000 |
