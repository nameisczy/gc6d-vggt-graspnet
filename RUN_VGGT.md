# VGGT 原始 / VGGT 微调：单数据点跑通与对比

目标：对比三种 encoder（**VGGT 原始**、**VGGT 微调**、**LIFT3D 微调**）。  
VGGT 使用 **RGB 图像** 训练，图像路径由 index 中的 `rgb_path` 指定（支持相对 `data_dir`）。

## 环境与数据

- 数据：`/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified`，index 中需有 `rgb_path`（绝对或相对 data_dir）。
- 依赖：`vggt`（`VGGT.from_pretrained("facebook/VGGT-1B")`）。
- 在 `gc6d_grasp_pipeline` 目录下执行。

---

## 一、VGGT 原始（冻结 encoder，只训 head）

单阶段：冻结 VGGT encoder，只训练 adapter + grasp head。

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline

python train_vggt_base.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-3
```

输出：`checkpoints/gc6d_vggt_base.pt`（`encoder_type=vggt_base`）。

---

## 二、VGGT 微调（三阶段，与 LIFT3D 流程对齐）

### Stage1：冻结 encoder，只训 adapter + head

```bash
python train_vggt_ft_stage1.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-3
```

输出：`checkpoints/gc6d_vggt_ft_stage1.pt`。

### Stage2：冻结 head + adapter，只训 encoder

```bash
python train_vggt_ft_stage2.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-5 \
  --ckpt_stage1 checkpoints/gc6d_vggt_ft_stage1.pt
```

输出：`checkpoints/gc6d_vggt_ft_stage2.pt`。

### Stage3：encoder + adapter + head 一起训

```bash
python train_vggt_ft_stage3.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-5 \
  --lr_head 1e-3 \
  --ckpt_stage2 checkpoints/gc6d_vggt_ft_stage2.pt
```

输出：`checkpoints/gc6d_vggt_ft_stage3.pt`（`encoder_type=vggt_ft`）。

---

## 三、评估与可视化（VGGT 自动用 RGB）

评估与可视化会根据 ckpt 内 `encoder_type` 自动选用 **LIFT3D 格式数据集**（含 RGB），并执行 `model(images)`。

```bash
# VGGT 原始
python eval_benchmark.py --checkpoint checkpoints/gc6d_vggt_base.pt
python visualize_offline.py --checkpoint checkpoints/gc6d_vggt_base.pt

# VGGT 微调（任选 stage）
python eval_benchmark.py --checkpoint checkpoints/gc6d_vggt_ft_stage3.pt
python visualize_offline.py --checkpoint checkpoints/gc6d_vggt_ft_stage3.pt
```

---

## 四、三种 encoder 对比小结

| 模型           | 输入   | 训练流程                         | 输出 ckpt / encoder_type   |
|----------------|--------|----------------------------------|----------------------------|
| VGGT 原始      | RGB    | 冻结 encoder → 只训 head         | `gc6d_vggt_base.pt` / vggt_base |
| VGGT 微调      | RGB    | Stage1 训 head → Stage2 训 encoder → Stage3 联合 | `gc6d_vggt_ft_stage*.pt` / vggt_ft |
| LIFT3D 微调    | 点云   | Stage1 训 head → Stage2 LoRA encoder → Stage3 联合 | `gc6d_lift3d_stage*.pt` / lift3d_clip |

扩展多样本：将 `--max_samples 1` 改为更大或去掉，并酌情调整 `--max_steps`、`--lr`。
