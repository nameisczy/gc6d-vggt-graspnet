# GC6D Grasp Pipeline 结构说明

## 一、数据与划分

### 1. 数据集划分（index 文件）

- **谁在做划分**：本仓库**不生成** train/val/test 划分，只**读取**已有 index。
- **约定路径**：在 `--data_dir`（如 `offline_unified`）下要有：
  - `index_train_{camera}.jsonl`（训练）
  - `index_val_{camera}.jsonl`（验证）
  - `index_test_{camera}.jsonl`（最终测 AP 时用）
- **谁读这些 index**：`data/dataset.py` 里的 `GC6DOfflineUnifiedDataset` / `GC6DLIFT3DFormatDataset`，通过参数 `split="train"|"val"|"test"` 拼出文件名 `index_{split}_{camera}.jsonl` 并加载。
- **生成 index 的脚本**：若你有单独的数据预处理/划分脚本（例如在 GraspClutter6D 或别的 repo），生成上述 jsonl 后放到 `data_dir` 即可；本仓库仅用 `scripts/check_data_consistency.py` 做 train/val 一致性和数据检查，不写 index。

---

## 二、模型结构（当前主流程：Adapter + GraspNet）

### 2. 整体结构（Encoder → Adapter → GraspNet）

定义在 **`models/graspnet_adapter.py`**：

- **Encoder**：点云或图像 → 特征  
  - LIFT3D：`models/lift3d_encoder.py`（点云，可选 LoRA）  
  - VGGT：`models/vggt_encoder.py`（图像）
- **Adapter**：`nn.Sequential(Linear(256,256), ReLU)`，把 encoder 特征压成 256 维 conditioning。
- **GraspNet**：`grasp_net`（backbone + view_estimator + grasp_generator），来自 graspnet-baseline，输出 17D 抓取；输入点云 + 上面的 conditioning。

所以：
- **Encoder 微调**：在 Stage3/4 解冻并训练 encoder（LoRA + adapter/pt_mlp 等）。
- **Head**：这里没有单独的「head 脚本」，**GraspNet 整体相当于 head**（view_estimator + grasp_generator）；Stage2/4 会训 grasp_net。
- **Adapter**：Stage1 只训 adapter；后面 stage 一起训。

---

## 三、训练脚本分工

### 3. 主推流程：Adapter + GraspNet（LIFT3D / VGGT）

| 文件 | 作用 |
|------|------|
| **`train_adapter_graspnet.py`** | **主训练入口**。Stage1→2→3→4：adapter → head(grasp_net) → encoder 微调 → 联合。读 `index_train` / `index_val`，不读 test；最终 AP 用 `eval_benchmark.py`。 |
| **`run_train_adapter_lift3d.sh`** | 调上面脚本，LIFT3D + Adapter + GraspNet，MODE=1sample/small/full。 |
| **`run_train_adapter_vggt_base.sh`** | 同上，VGGT base。 |
| **`run_train_adapter_vggt_ft.sh`** | 同上，VGGT 微调版。 |

- **Encoder 微调**：在 `train_adapter_graspnet.py` 的 **Stage3、Stage4** 里完成（解冻 encoder LoRA + adapter/pt_mlp，Stage4 再解冻 grasp_net 联合训）。
- **Head（GraspNet）**：在 **Stage2、Stage4** 里训练（Stage2 只训 adapter+grasp_net，Stage4 全模型一起训）。

### 4. 其他训练脚本（旧/备用）

| 文件 | 作用 |
|------|------|
| `train_stage1_freeze_encoder.py` | 旧：冻结 encoder，训 head（非 GraspNet 的 policy head）。 |
| `train_stage2_lora_encoder.py` | 旧：Stage2，加 LoRA 训 encoder。 |
| `train_stage3_joint.py` | 旧：联合训 encoder + head。 |
| `train_lift3d.py` | 旧：LIFT3D + 简单 head 一条龙。 |
| `train_vggt_base.py` | 旧：VGGT base + head。 |
| `train_vggt_ft_stage1/2/3.py` | 旧：VGGT 微调分 stage。 |
| `train.py` | 通用小入口，placeholder encoder + head。 |

当前**主流程**以 **`train_adapter_graspnet.py` + 三个 `run_train_adapter_*.sh`** 为准；上面这些是旧/备用，不参与「哪个文档是 encoder 微调、哪个是 head」的主线说明。

---

## 四、评估与工具

| 文件 | 作用 |
|------|------|
| **`eval_benchmark.py`** | 用训好的 ckpt 在指定 `--split`（train/val/test）上 dump 预测并调 GraspClutter6D API 算 AP；**最终 test AP 用 `--split test`**。 |
| **`utils/load_model.py`** | 根据 ckpt 里 `encoder_type` 等构建模型（含 EncoderAdapterGraspNet），供评估/可视化加载。 |
| **`scripts/check_data_consistency.py`** | 检查 train/val 数据一致性、action 统计等，**不划分数据集**。 |
| **`visualize_offline.py`** / **`visualize_grasp_animation.py`** | 离线/动画可视化预测与 GT。 |

---

## 五、数据与配置约定（简要）

- **数据目录**：`--data_dir` 指向 offline_unified（内有 `index_*_{camera}.jsonl` 和 npz）。
- **GC6D 原始根目录**：`--dataset_root` 指向 GraspClutter6D（含 `split_info/`, `scenes/`, `models_m/` 等），仅评估时用。
- **相机**：默认 `realsense-d415`，与 index 文件名里的 `{camera}` 一致。

---

## 六、一句话对应表

| 你想做的事 | 用到的文件/概念 |
|------------|------------------|
| 分割/生成 train/val/test 数据集 | 本仓库不做；用你已有的 index_*_{camera}.jsonl，由 `data/dataset.py` 按 `split` 读取。 |
| Encoder 微调 | `train_adapter_graspnet.py` 的 **Stage3、Stage4**（LIFT3D/VGGT 的 LoRA + adapter/pt_mlp 等）。 |
| Head（抓取预测） | 当前主流程里 **GraspNet 即 head**，在 **Stage2、Stage4** 训练；逻辑在 `models/graspnet_adapter.py`。 |
| Adapter | 同文件 **Stage1** 起训，之后各 stage 一起训；结构在 `graspnet_adapter.py` 的 `EncoderAdapterGraspNet`。 |
| 最终 test AP | `eval_benchmark.py --split test --checkpoint ... --dataset_root ...`。 |
