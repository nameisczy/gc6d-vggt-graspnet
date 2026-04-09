# LIFT3D Encoder 三阶段训练与评估

从「一个数据点跑通」开始：先冻结 LIFT3D encoder 训 grasp head，再冻结 head 用 LoRA 微调 encoder，最后联合训练。

## 环境

- `LIFT3D` 仓库路径：默认 `~/LIFT3D`，或设置 `LIFT3D_ROOT` / `--lift3d_root`。
- 数据：`/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified`（或 `--data_dir`）。
- 在 `gc6d_grasp_pipeline` 目录下执行下列命令。

## 一、Stage1：冻结 LIFT3D encoder，只训 adapter + grasp head

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline

# 单数据点（先跑通）
python train_stage1_freeze_encoder.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-3 \
  --lift3d_root ~/LIFT3D

# 输出：checkpoints/gc6d_lift3d_stage1.pt
```

## 二、Stage2：冻结 head 与 adapter，只训 backbone 内 LoRA

```bash
python train_stage2_lora_encoder.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-4 \
  --ckpt_stage1 checkpoints/gc6d_lift3d_stage1.pt \
  --lift3d_root ~/LIFT3D

# 输出：checkpoints/gc6d_lift3d_stage2.pt
```

## 三、Stage3：encoder（LoRA）+ adapter + head 一起训

```bash
python train_stage3_joint.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --max_samples 1 \
  --max_steps 2000 \
  --lr 1e-4 \
  --lr_head 1e-3 \
  --ckpt_stage2 checkpoints/gc6d_lift3d_stage2.pt \
  --lift3d_root ~/LIFT3D

# 输出：checkpoints/gc6d_lift3d_stage3.pt
```

## 四、评估与可视化（任选 Stage1/2/3 的 ckpt）

评估脚本会从 ckpt 里读 `encoder_type=lift3d_clip` 并自动用 `build_lift3d_clip_policy` 加载。

```bash
# 评估（官方 eval_grasp）
python eval_benchmark.py \
  --checkpoint checkpoints/gc6d_lift3d_stage1.pt \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --lift3d_root ~/LIFT3D

# 离线可视化（红=预测，绿=GT）
python visualize_offline.py \
  --checkpoint checkpoints/gc6d_lift3d_stage1.pt \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --lift3d_root ~/LIFT3D
```

## 五、扩展到更多样本

把 `--max_samples 1` 改为更大（如 `256` 或去掉以用全量 train），并视情况调 `--max_steps`、`--lr`。

## 六、代码结构摘要

| 文件 | 作用 |
|------|------|
| `models/lift3d_clip_encoder.py` | LIFT3D 官方 lift3d_clip_base 封装，768→feat_dim 适配器，freeze/LoRA 控制 |
| `models/policy.py` | `build_lift3d_clip_policy()`，`GC6DGraspPolicyLIFT3D` |
| `train_stage1_freeze_encoder.py` | 冻结 backbone，训 adapter + head |
| `train_stage2_lora_encoder.py` | 冻结 head+adapter，只训 backbone LoRA |
| `train_stage3_joint.py` | LoRA + adapter + head 联合训 |
| `utils/load_model.py` | `encoder_type=lift3d_clip` 时用 `build_lift3d_clip_policy` 加载 |
