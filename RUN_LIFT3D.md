# LIFT3D Encoder + GC6D Grasp Head 三阶段训练与评估

## 环境与依赖

- 已跑通占位 encoder 的 GC6D pipeline（单数据点训练 / 评估 / 可视化）。
- 需要 **LIFT3D** 仓库（含 PointNext）：设置 `LIFT3D_ROOT` 或通过 `--lift3d_root` 指定，默认 `~/LIFT3D`。
- LIFT3D 内 PointNext 依赖 openpoints（在 `lift3d/models/point_next/` 下），无需额外安装。

## 数据格式（LIFT3D）

离线数据：深度用 GC6D 内置函数转点云，grasp 转 action 直接拼接，state=0。  
LIFT3D 需求格式：`(images, point_clouds, robot_states, raw_states, actions, texts)`。  
见 `GC6DLIFT3DFormatDataset` 与 `collate_lift3d`（`data/dataset.py`）。

## 三阶段流程概览

| 阶段 | 内容 | 可训练参数 | 输出 ckpt |
|------|------|------------|-----------|
| Stage 1 | **不接我们的 head**；LIFT3D 格式数据单独训 encoder（仅 LoRA） | 仅 LoRA | `gc6d_lift3d_stage1.pt`（仅存 encoder_backbone） |
| Stage 2 | 冻结 encoder，接我们的 GC6D head，只训 head | 仅 grasp head | `gc6d_lift3d_stage2.pt` |
| Stage 3 | encoder 与 head 一起训练 | 全部或仅 LoRA（见下） | `gc6d_lift3d_stage3.pt` |

## 1. Stage 1：不接 head，单独 LoRA 微调 encoder（单数据点）

使用 **LIFT3D 格式数据**（images, point_clouds, robot_states=0, raw_states=0, actions, texts）。  
模型为：PointNext + LoRA（无 adapter、无我们的 GC6D head）+ **LIFT3D 自带的 GraspHead** 仅用于算 loss（该 head 冻结）。只更新 **LoRA**，实现“单独训练 encoder”。

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline

# 指定 LIFT3D 根目录（若未设置 LIFT3D_ROOT）
export LIFT3D_ROOT=/path/to/LIFT3D

python train_lift3d.py \
  --stage 1 \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-3 \
  --lr_lora 1e-4 \
  --lift3d_root "$LIFT3D_ROOT"
```

输出：`checkpoints/gc6d_lift3d_stage1.pt`。

## 2. Stage 2：冻结 encoder，接 head，只训 head

加载 Stage 1 的 ckpt（仅含 `encoder_backbone`），将其载入当前 policy 的 `encoder.backbone`，**冻结整个 encoder**，只训练我们的 GC6D grasp head。

```bash
python train_lift3d.py \
  --stage 2 \
  --ckpt checkpoints/gc6d_lift3d_stage1.pt \
  --max_samples 1 \
  --max_steps 2000 \
  --lift3d_root "$LIFT3D_ROOT"
```

输出：`checkpoints/gc6d_lift3d_stage2.pt`。

## 3. Stage 3：encoder + grasp head 一起训练

在 Stage 2 的 ckpt 上继续。

- **默认**：解冻 encoder 与 grasp head，全部一起训练。
- **仅解冻 LoRA**：加 `--stage3_unfreeze_lora_only`，只训练 LoRA + grasp head。

```bash
# 全部解冻
python train_lift3d.py \
  --stage 3 \
  --ckpt checkpoints/gc6d_lift3d_stage2.pt \
  --max_samples 1 \
  --max_steps 2000 \
  --lift3d_root "$LIFT3D_ROOT"

# 仅解冻 LoRA
python train_lift3d.py \
  --stage 3 \
  --ckpt checkpoints/gc6d_lift3d_stage2.pt \
  --stage3_unfreeze_lora_only \
  --max_samples 1 \
  --max_steps 2000 \
  --lift3d_root "$LIFT3D_ROOT"
```

输出：`checkpoints/gc6d_lift3d_stage3.pt`。

## 4. 评估与可视化（使用 LIFT3D ckpt）

评估、离线可视化、动画脚本会根据 ckpt 里的 `encoder_type` 自动选用 **LIFT3D** 或 **Placeholder**。使用 LIFT3D 的 ckpt 时需传 `--lift3d_root`（或设置 `LIFT3D_ROOT`）。

**Benchmark 评估：**

```bash
python eval_benchmark.py \
  --checkpoint checkpoints/gc6d_lift3d_stage1.pt \
  --lift3d_root "$LIFT3D_ROOT"
```

**离线可视化（点云 + 预测/GT 抓取）：**

```bash
python visualize_offline.py \
  --checkpoint checkpoints/gc6d_lift3d_stage1.pt \
  --lift3d_root "$LIFT3D_ROOT" \
  --num_grasps 5
```

**抓取动画 GIF：**

```bash
python visualize_grasp_animation.py \
  --checkpoint checkpoints/gc6d_lift3d_stage1.pt \
  --lift3d_root "$LIFT3D_ROOT" \
  --use_official_gripper
```

## 5. 参数摘要

| 参数 | 含义 | 默认 |
|------|------|------|
| `--data_dir` | offline_unified 数据目录 | `/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified` |
| `--max_samples` | 训练样本数（先 1 跑通） | 1 |
| `--max_steps` | 训练步数 | 2000 |
| `--lr` | 主学习率 | 1e-3 |
| `--lr_lora` | Stage1 LoRA 学习率 | 1e-4 |
| `--lift3d_root` | LIFT3D 仓库根目录 | `$LIFT3D_ROOT` 或 `~/LIFT3D` |
| `--lora_r` | LoRA 秩 | 8 |
| `--lora_scale` | LoRA 缩放 | 1.0 |
| `--ckpt` | Stage2/3 加载的上一阶段 ckpt | - |
| `--stage3_unfreeze_lora_only` | Stage3 只解冻 LoRA | False |

## 6. 代码结构（本次新增/修改）

- `models/lora.py`：LoRA 注入与参数筛选。
- `models/lift3d_encoder.py`：LIFT3D PointNext 封装 + LoRA + 512→256 适配器。
- `models/policy.py`：`GC6DGraspPolicyLIFT3D`、`build_lift3d_policy`。
- `train_lift3d.py`：三阶段训练入口。
- `utils/load_model.py`：按 ckpt 的 `encoder_type` 加载 Placeholder 或 LIFT3D policy。
- `eval_benchmark.py`、`visualize_offline.py`、`visualize_grasp_animation.py`：统一用 `load_policy_from_checkpoint(..., lift3d_root=...)` 加载模型。
