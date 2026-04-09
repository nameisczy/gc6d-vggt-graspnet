# 过拟合抑制与训练效果增强

小样本（如 N=100）时易过拟合（train loss 降、val loss 升），且整体效果一般。下面列出可用手段，并标明当前实现情况。

---

## 一、过拟合抑制

### 1. 已实现 / 可直接用

| 方法 | 说明 | 使用方式 |
|------|------|----------|
| **少步数** | 小样本下大幅减少 max_steps | `run_small_dataset.sh` 默认 S1=150 S2=200 S3=100；可再减 |
| **强 weight_decay** | 加大 L2 正则 | 脚本里 `WEIGHT_DECAY=3e-2`；可试 `5e-2` |
| **小 LoRA 秩** | 降低可训参数量 | `LORA_R=2`（已默认）；可试 `1` |
| **降学习率** | Stage2/3 用更小 lr | 脚本里 LR2/LR3 已调低 |
| **Head/Adapter Dropout** | 结构上在 head 前、adapter 输出加 dropout | 见下「Dropout」小节 |

### 2. Grasp Head 类型（已实现，可对比）

- **simple**（默认）：单层 Linear 输出 10D（`GC6DGraspHead`）。
- **mature / lift3d_action**：LIFT3D **action head** 风格（MLP + LayerNorm + 分离 t/rotation/width 头），输出 10D。保留用于与 GraspNet 等 head 对比。
- **graspnet**：GraspNet 风格 proposal head（Contact-GraspNet / 6DoF-GraspNet 等），输出多组 (R, t, width, score)，再经 softmax(score) 加权聚合为单条 10D，与现有 pipeline 一致。可选 `--num_proposals 4`（默认 4）。
- 使用：训练时加 `--grasp_head_type mature` 或 `--grasp_head_type graspnet`（Stage1/Base 指定；Stage2/Stage3 从 ckpt 自动读取类型与 `num_proposals`）。加载 ckpt 时 `load_policy_from_checkpoint` 自动按 ckpt 内 `grasp_head_type`、`grasp_head_num_proposals` 恢复。

### 3. Dropout（已加，需在训练时传参）

- **GC6DGraspHead / MatureGraspHead**：`dropout_p`，在 head 输入特征上做 dropout，默认 0。
- **LIFT3DClipEncoder / VGGTEncoder**：`adapter_dropout`，在 adapter 输出上做 dropout，默认 0。

小样本建议：`--head_dropout 0.2 --adapter_dropout 0.1`（或 0.2）。  
若训练脚本已支持 `--head_dropout` / `--adapter_dropout`，在 `run_small_dataset.sh` 里对每个 `python train_*` 加上这两项即可；否则需在对应 train_*.py 的 parser 和 `build_*_policy` 调用处增加这两个参数并传入。

### 4. 可进一步实现或手动做的

| 方法 | 说明 |
|------|------|
| **Early stopping** | 按 val_loss 保存最佳 ckpt，或 val 连续 N 次不降则停训；需改 train 循环 |
| **同分布 val** | 从同一批样本中划分 80/20 做 train/val，避免 train 用 100、val 用 533 的分布差异 |
| **数据增强** | 点云/图像：随机平移、缩放、旋转、jitter；需在 Dataset `__getitem__` 或 DataLoader 里加 |
| **Label smoothing** | 对回归可考虑对 target 加小幅噪声（慎用） |
| **冻结更多层** | 只训 head、少训 adapter 或 LoRA，进一步减容量 |
| **减小 head 容量** | 如 head 改为单层更窄 MLP（需改 `GC6DGraspHead`） |

---

## 二、训练效果增强（loss 下去、泛化更好）

| 方法 | 说明 |
|------|------|
| **学习率 schedule** | warmup + 按 step 或 val 衰减（如 cosine）；需在 optimizer 外接 scheduler |
| **梯度裁剪** | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` 防爆 |
| **Batch size** | 小样本时 BS 不宜过大，避免每 epoch 步数过少；当前 32 可试 16 |
| **更多/更均衡数据** | 增加 N 或平衡场景/物体；从 100 提到 500/1000 通常最有效 |
| **Loss 形式** | 对旋转用 geodesic/余弦等；对 width 用 BCE 等；需改 loss 与数据格式 |
| **初始化** | 检查 head/adapter 初始化是否合理（当前多为默认） |

---

## 三、小样本脚本推荐用法

- 默认已：少步数、强 weight_decay、小 LoRA、低 lr、`VAL_EVERY=100`。
- 建议同时启用 **dropout**：在 `run_small_dataset.sh` 里为各阶段训练命令增加：
  - `--head_dropout 0.2 --adapter_dropout 0.1`
- 若仍过拟合：再减步数（如 S2=150 S3=80）、或 `WEIGHT_DECAY=5e-2`、或 `LORA_R=1`。
- VGGT base 若 loss 下不去：适当增加 `STEPS_VGGT_BASE` 或略提 `LR_VGGT_BASE`，并观察 val 是否一起降（防过拟合）。

---

## 四、训练脚本中 dropout 的接入方式

在以下脚本中：

- `train_stage1_freeze_encoder.py`, `train_stage2_lora_encoder.py`, `train_stage3_joint.py`
- `train_vggt_base.py`, `train_vggt_ft_stage1.py`, `train_vggt_ft_stage2.py`, `train_vggt_ft_stage3.py`

1. 在 `parser.add_argument` 部分增加：
   - `--head_dropout`, type=float, default=0.0
   - `--adapter_dropout`, type=float, default=0.0
2. 在调用 `build_lift3d_clip_policy` / `build_vggt_base_policy` / `build_vggt_ft_policy` 时传入：
   - `head_dropout=args.head_dropout`, `adapter_dropout=args.adapter_dropout`

评估/加载 ckpt 时用默认 0 即可（eval 不启用 dropout）。
