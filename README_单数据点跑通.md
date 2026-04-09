# GC6D Grasp Pipeline — 单数据点跑通

三个 encoder（LIFT3D LoRA / VGGT 原始 / VGGT LoRA）接 **GC6D 专用 grasp head** 的完整流程，从**一个数据点**跑通：占位 encoder + 自写 grasp head，无正则，训练 loss 应收敛到 0。

## 目录结构

```
gc6d_grasp_pipeline/
├── data/
│   ├── dataset.py          # 新划分 offline_unified 加载
│   └── __init__.py
├── models/
│   ├── placeholder_encoder.py  # 占位 encoder
│   ├── gc6d_grasp_head.py      # GC6D 专用 10D grasp head
│   ├── policy.py               # encoder + head
│   └── __init__.py
├── utils/
│   ├── action2grasp.py     # 10D action -> GraspGroup
│   └── __init__.py
├── train.py                # 单样本训练，无正则
├── eval_benchmark.py       # GC6D 6-DOF benchmark 评估
├── visualize_offline.py    # 离线渲染
├── run_one_sample.sh       # 一键：训练 -> 评估 -> 可视化
└── README_单数据点跑通.md
```

## 环境

- **训练 / 可视化**：任意有 `torch` 的环境。
- **Benchmark 评估**：需要 **gc6d** 环境及 `graspclutter6dAPI`（`eval_grasp` 等）。

```bash
# 训练与可视化
conda activate your_torch_env
pip install torch open3d

# 评估（可选，需 gc6d）
conda activate gc6d
# 确保 PYTHONPATH 含 graspclutter6dAPI
```

## 数据

使用**新划分**的离线数据：

- 路径：`/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified`
- `index_train_realsense-d415.jsonl` / `index_val_realsense-d415.jsonl`
- 每个 npz：`point_cloud`, `action`(10D), `gt_grasp_group`, `sceneId`, `annId`

## 单数据点跑通

### 1. 训练（单样本，无正则，loss → 0）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python train.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --split train \
  --camera realsense-d415 \
  --max_samples 1 \
  --max_steps 2000
```

- 默认保存：`checkpoints/gc6d_grasp_policy_one_sample.pt`
- 单样本过拟合，无正则，MSE loss 应收敛到 < 1e-6

### 2. Benchmark 评估（需 gc6d 环境）

```bash
conda activate gc6d
export PYTHONPATH="/home/ziyaochen/graspclutter6dAPI:${PYTHONPATH}"

python eval_benchmark.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --checkpoint checkpoints/gc6d_grasp_policy_one_sample.pt \
  --max_samples 1 \
  --dataset_root /mnt/ssd/ziyaochen/GraspClutter6D
```

### 3. 离线可视化

```bash
python visualize_offline.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --checkpoint checkpoints/gc6d_grasp_policy_one_sample.pt \
  --max_samples 1 \
  --num_grasps 5 \
  --save_ply
```

- 输出在 `vis_out/`：点云 ply、预测 grasp 的渲染图（若支持离屏）。

### 一键执行

```bash
./run_one_sample.sh
```

- 步骤 2 若未在 gc6d 环境，可跳过或单独在 gc6d 下执行。

## 模型说明

- **PlaceholderEncoder**：点云 (B,N,3) → mean pool → MLP → (B, 256)，仅用于跑通。
- **GC6DGraspHead**：256 → 10D [t(3), R6d(6), width(1)]，width 用 sigmoid 映射到 [0.01, 0.12]。
- **GC6DGraspPolicy**：encoder + head，输入点云，输出 10D action。

未使用现有 LIFT3D grasp head 与训练脚本，全部为新写流程。
