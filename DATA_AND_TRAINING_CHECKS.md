# 数据与训练一致性检查

## 运行检查（必须执行）

在 **gc6d_grasp_pipeline 根目录** 下执行，会检查 train/val 一致性、action 统计、width 单位、点云与 action 尺度，并：

- 在 **logs/** 下生成带时间戳的 log：`logs/check_data_consistency_YYYYMMDD_HHMMSS.log`
- 写出 **checkpoints/action_stats_train.npz** 供后续归一化用

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
python scripts/check_data_consistency.py --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified --camera realsense-d415 --max_samples 500
```

若数据在其他路径，改 `--data_dir`；指定 log 目录可加 `--log_dir /path/to/logs`。终端与 log 文件内容一致；log 中 `[OK]`/`[WARN]`/`[?]` 标出是否一致、单位是否合理、action 与点云是否同尺度。

---

## 一、检查项说明（脚本已覆盖或需上游确认）

### 1. Train / Val：image_size、crop、normalize 是否完全一致

**当前 pipeline 内：**
- **LIFT3D（点云）**：train/val 均用 `GC6DOfflineUnifiedDataset`，无 image；点云在 encoder 内 `normalize_pc`（center + scale），train/val 同一套。
- **VGGT（图像）**：train/val 均用 `GC6DLIFT3DFormatDataset(image_size=224)`，transform 为 `Resize((224,224)) + ToTensor() + Normalize(ImageNet)`，**train/val 同一套尺度**。  
  - **Train 增强**（仅 train）：`ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02)` + `adjust_gamma(0.7~1.3)`，可选 `RandomResizedCrop`；**几何一致**（仅图像增强，无 pc 变换）。
  - Val 无增强，仅 Resize+ToTensor+Normalize。

**待确认**：离线 npz 生成时，train/val 的 resize/crop 是否一致。

---

### 2. depth → point_cloud 时用的 K 是否为「变换后的 K」

点云来自 npz 的 `point_cloud`，**本 repo 不生成点云**。需在 **GC6D 数据生成 / 打包脚本** 中确认：
- depth 转点云时用的内参 **K** 是否与当前图像（或 crop）对应（例如 crop 后 K 是否随之平移/缩放）。
- 若图像被 resize/crop，K 必须用变换后的内参，否则点云与图像不对齐。

---

### 3. action / gt_grasps 坐标系是否与点云一致（Tcw / Twc）

npz 中 `action` 为 10D，来自「grasp 转 action 直接拼接」。需在 **数据生成侧** 确认：
- action 的 (t, R) 是在 **相机系（camera）** 还是 **世界系（world）**；
- 点云是在 **相机系** 还是 **世界系**；
- 二者必须一致（例如均为相机系），否则 head 学到的 t/R 与点云不对应。

---

### 4. action 是否做 normalization？val 是否用同一套参数？

**当前**：**未做**。训练与验证均直接使用 npz 中的原始 `action`，loss 为 `MSE(pred, action)`，无均值/方差归一化。

**若要做**：
- 用 **仅 train** 统计各维 mean/std（或 robust scale），对 action 的 t(3)、R6d(6)、width(1) 分别归一化；
- 训练时对 target 归一化，预测头输出「归一化空间」再反归一化得到 10D，或 head 直接预测 10D 且 loss 在归一化空间计算；
- **val 必须使用 train 的 mean/std**，不能重新统计。

---

### 5. width 单位一致？（米 vs 毫米）

**当前**：head 与 utils 中 `width_min=0.01`, `width_max=0.12`，按 **米** 理解（1cm～12cm）。  
若 npz 中 `action` 第 10 维是 **毫米**，需在数据生成或 Dataset 中统一转为米，再喂给当前 pipeline。

---

## 二、Stage3「一 joint 就爆」已做 / 建议

| 项 | 已做 / 修改 |
|----|-------------|
| **encoder lr = head lr 的 0.03× 或 0.1×** | Stage3 默认 `lr_head * 0.03`；VGGT ft Stage3 同理；early stop 默认 val 第一次变差即停。 |
| **Stage3 步数 200～400** | `train_stage3_joint.py` / `train_vggt_ft_stage3.py` 默认 `--max_steps 400`。 |
| **early stop：val 第一次明显变差就停** | `--early_stop_val_worse 1`（默认）。 |
| **只放开最后 1～2 个 block 的 LoRA** | 已实现：`--lora_last_n_blocks 2`（默认）；`get_lora_params_from_last_n_blocks(backbone, n)` 按 param 名 `blocks.(\d+)` 取最后 n 个 block 的 LoRA。 |

---

## 三、多 GT / loss 方式（已实现）

- **多 gt_grasps**：`GC6DLIFT3DFormatDataset(load_gt_multi=True)` 时从 npz 的 `gt_grasp_group` 转成多条 10D action 放入 `meta["actions_multi"]`；`utils/loss.py` 提供 `action_loss_multi_gt(pred, gt_primary, gt_multi)`（min over gt）与 `pad_actions_multi(metas, gt_primary, device)`。VGGT Stage3 加 `--use_multi_gt` 即启用。
- **预测 topK**：GraspNet head 已可出多 proposal；当前 loss 仍为单输出 10D 与 min-over-gt。

---

## 四、Train augmentation（已实现）

- **图像**：train 时 `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02)` + `adjust_gamma(0.7~1.3)`，可选 `RandomResizedCrop(224)`；val 仅 Resize+ToTensor+Normalize。**几何一致**：仅对图像做增强，点云与 action 不变。
- **点云**：无增强；若将来加点云 augment，需对 action 的 t 做同变换。
