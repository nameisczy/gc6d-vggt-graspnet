# 三种 Encoder 完整训练指令

**对比设定**：**LIFT3D = 点云 encoder 仅**（不读图像），**VGGT = 图像 encoder 仅**（不读点云）。二者为不同模态的 encoder 对比，默认不做点云+图像融合；可选 `--use_images` 为 LIFT3D 加图像分支（会引入 VGGT 图像 encoder，仅用于融合实验）。

**前置**：数据已打包到 `offline_unified`（含 `index_train_*.jsonl`、`index_val_*.jsonl`）。  
**Validation**：默认每 **500** 步在验证集上算一次 loss 并打日志；`VAL_EVERY=0` 可关闭。  
**LoRA**：VGGT base、VGGT ft、LIFT3D ft 均使用 LoRA 微调，默认 **LORA_R=8、LORA_SCALE=1.0**，便于公平对比。  
**Loss**：三种 encoder 均用 **预测 top-K grasps + 多 GT**，默认 **双向最近邻**：`loss = α×(预测→GT) + (1−α)×(GT→预测)`，推荐 `--match_mode bidir --alpha 0.7`。可选 `--match_mode min`（仅 GT→预测）或 `--match_mode hungarian`（一对一匹配）；数据侧 `load_gt_multi=True` 读取多 GT（`gt_grasp_group`）。  
**17D 直接**：run 脚本默认 **--loss_17d**。**GraspNet head** 输出 11D、**MLP head 直接 17D**（`simple_17d` / `mature_17d`）输出 17D，均可与 GT 17D 直接比较；原 MLP（`simple` / `mature`）输出 10D，训练时在 loss 内可微转 17D。

**性能对比（GraspNet vs MLP 17D）**：为公平对比两种 head，**GraspNet 与 MLP 17D 的 run 脚本在 loss 与训练超参上完全一致**：同一 encoder（LIFT3D / VGGT base / VGGT ft）、同小批量或同全量下，仅 `--grasp_head_type`（`graspnet` vs `simple_17d`）和 GraspNet 的 `--num_proposals` 不同；其余 **loss_17d、loss_best_gt_weight、pred2gt_top_frac、match_mode** 以及 **步数、学习率、weight_decay、LoRA、batch_size、val_every** 等均相同，可直接对比同一 encoder 下两种 head 的 val/test 表现。

---

## 三种 Encoder 小样本训练指令（当前试验用）

**环境**：为避免依赖冲突，LIFT3D 相关在 **lift3d** 环境，VGGT 相关在 **vggt** 环境；评估时用与 checkpoint 对应的环境即可（eval 脚本在 pipeline 内，与训练同环境）。

**公共变量**（按需修改）：
- `DATA`：统一数据目录，默认 `/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified`
- `LIFT3D_ROOT`：LIFT3D 仓库根路径，仅 LIFT3D 训练需要，默认 `$HOME/LIFT3D`
- `N`：小样本条数，默认 **100**
- `VAL_EVERY`：验证间隔步数，默认 **100**；`0` 关闭
- `CUDA_VISIBLE_DEVICES`：GPU，如 `1`

三种 encoder 均接 **GraspNet head**（或 MLP head 见下），在 gc6d 小样本上训练；**默认 --loss_17d**。以下在 **gc6d_grasp_pipeline 根目录** 执行。

### 1. LIFT3D 在 gc6d 上 LoRA 微调（点云 encoder，三阶段）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export CUDA_VISIBLE_DEVICES=1

conda activate lift3d
./run_small_lift3d_graspnet.sh
```

- **可选**：`N=50 VAL_EVERY=50 ./run_small_lift3d_graspnet.sh` 或调 `STEPS1_LIFT`/`STEPS2_LIFT`/`STEPS3_LIFT`、`LORA_R`、`WEIGHT_DECAY_S2`/`WEIGHT_DECAY_S3`
- **产出**：`checkpoints/gc6d_lift3d_small_graspnet_stage1.pt` → `stage2.pt` → `stage3.pt`（最终用 stage3）；日志 `logs/lift3d_small_graspnet_<时间戳>/`

### 2. VGGT 原始（冻结 encoder，只训 head，单阶段）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

conda activate vggt
./run_small_vggt_base_graspnet.sh
```

- **可选**：`N=50 VAL_EVERY=50 ./run_small_vggt_base_graspnet.sh` 或调 `STEPS_BASE`、`LR_VGGT_BASE`、`LORA_R`
- **产出**：`checkpoints/gc6d_vggt_small_base_graspnet.pt`；日志 `logs/vggt_base_small_graspnet_<时间戳>/`

### 3. VGGT 在 gc6d 上 LoRA 微调（三阶段）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

conda activate vggt
./run_small_vggt_ft_graspnet.sh
```

- **可选**：`N=50 VAL_EVERY=50 ./run_small_vggt_ft_graspnet.sh` 或调 `STEPS1_VGGT`/`STEPS2_VGGT`/`STEPS3_VGGT`、`LORA_LAST_N_BLOCKS`、`EARLY_STOP_VAL_WORSE=1`
- **产出**：`checkpoints/gc6d_vggt_small_ft_graspnet_stage1.pt` → `stage2.pt` → `stage3.pt`（最终用 stage3）；日志 `logs/vggt_ft_small_graspnet_<时间戳>/`

### 小样本训练后：benchmark 评估与可视化

评估（用与 checkpoint 对应的环境：LIFT3D 用 lift3d，VGGT 用 vggt）：

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified

# LIFT3D（在 lift3d 环境下）
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_graspnet_stage3.pt --split val --max_samples 20

# VGGT base（在 vggt 环境下）
conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_base_graspnet.pt --split val --max_samples 20

# VGGT ft（在 vggt 环境下）
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_ft_graspnet_stage3.pt --split val --max_samples 20

# 成功率偏低时可加 --use_proposals，用 K 个 proposal 作为多候选（见「提高 benchmark 成功率的改进措施」）
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_graspnet_stage3.pt --split val --max_samples 20 --use_proposals
```

可视化抓取（同样先激活对应环境）：

```bash
python visualize_offline.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_graspnet_stage3.pt --split val --max_samples 1
# 无头环境可加 --no_render 只保存 PLY/图
```

---

## 三种 Encoder 小批量 / 全量训练指令汇总

| 类型 | Encoder | 命令（在 pipeline 根目录执行） | 产出 checkpoint |
|------|---------|--------------------------------|-----------------|
| **小批量** | LIFT3D | `conda activate lift3d` 后 `./run_small_lift3d_graspnet.sh` | `gc6d_lift3d_small_graspnet_stage3.pt` |
| **小批量** | VGGT Base | `conda activate vggt` 后 `./run_small_vggt_base_graspnet.sh` | `gc6d_vggt_small_base_graspnet.pt` |
| **小批量** | VGGT Ft | `conda activate vggt` 后 `./run_small_vggt_ft_graspnet.sh` | `gc6d_vggt_small_ft_graspnet_stage3.pt` |
| **全量** | LIFT3D | `conda activate lift3d` 后 `./run_train_lift3d_graspnet.sh` | `gc6d_lift3d_graspnet_stage3.pt` |
| **全量** | VGGT Base | `conda activate vggt` 后 `./run_train_vggt_base_graspnet.sh` | `gc6d_vggt_base_graspnet.pt` |
| **全量** | VGGT Ft | `conda activate vggt` 后 `./run_train_vggt_ft_graspnet.sh` | `gc6d_vggt_ft_graspnet_stage3.pt` |

以上脚本默认 **--loss_17d**、**GraspNet head**（`--grasp_head_type graspnet`）。**所有 run 脚本已加可执行权限**（`chmod +x run_*.sh`），在 pipeline 根目录直接 `./run_*.sh` 即可。

**改用 MLP 头直接 17D**：用下面「三种 Encoder 小批量/全量指令（MLP 17D）」中的 6 个 run 脚本或对应 python 命令。

### GraspNet head 小批量、全量（复制即用）

**小批量 — LIFT3D（GraspNet）**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
conda activate lift3d
./run_small_lift3d_graspnet.sh
```

**小批量 — VGGT Base（GraspNet）**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
conda activate vggt
./run_small_vggt_base_graspnet.sh
```

**小批量 — VGGT Ft（GraspNet）**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
conda activate vggt
./run_small_vggt_ft_graspnet.sh
```

**全量 — LIFT3D（GraspNet）**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
conda activate lift3d
./run_train_lift3d_graspnet.sh
```

**全量 — VGGT Base（GraspNet）**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
conda activate vggt
./run_train_vggt_base_graspnet.sh
```

**全量 — VGGT Ft（GraspNet）**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
conda activate vggt
./run_train_vggt_ft_graspnet.sh
```

---

## Adapter + 预训练 GraspNet（graspnet-baseline）

**说明**：Encoder（LIFT3D/VGGT）→ Adapter → 预训练 GraspNet head；Stage1 只训 adapter，Stage2 训 head，Stage3 训 encoder，Stage4 联合。VGGT base 仅 Stage1 + Stage2（步数=4k+2k）。Loss 为 17D matching（与 GC6D 数据一致）。**不复用**现有 train_stage* 逻辑，见 `train_adapter_graspnet.py`。

**依赖**：需 graspnet-baseline 预训练 checkpoint，且需 **先编译** pointnet2 / knn 扩展（仅改 PYTHONPATH 不够，必须在本机用当前 conda 环境 build 出 `.so`）。  
- **LIFT3D encoder**（`run_train_adapter_lift3d.sh`）：还需编译 LIFT3D 内嵌的 OpenPoints 的 pointnet2_batch，否则会报 `No module named 'pointnet2_batch_cuda'`。在 **你实际跑训练的环境**（若统一用 gc6d 则用 gc6d）下执行一次：  
  `cd $LIFT3D_ROOT/lift3d/models/point_next/openpoints/cpp/pointnet2_batch && pip install .`（或 `python setup.py build_ext --inplace`）。也可用 pipeline 提供的脚本：`./scripts/build_lift3d_pointnet2_batch.sh`（会读 `LIFT3D_ROOT`，当前 shell 的 Python 即编译环境）。

1. **一次性编译扩展**（在 gc6d 用的那个 conda 环境下执行）。**CUDA 版本要对齐**：系统/`nvcc` 的 CUDA 必须与当前 PyTorch 编译用的 CUDA 一致，否则会报 `The detected CUDA version (11.8) mismatches ... compile PyTorch (12.8)`。  
   - 方案 A：装与系统 CUDA 一致的 PyTorch，例如系统是 11.8 则：`pip install torch --index-url https://download.pytorch.org/whl/cu118`  
   - 方案 B：装 CUDA 12 并让 `nvcc` 指向 12（conda 安装 `cuda-nvcc` 后若报 **cusparse.h: No such file or directory**，再装 `cuda-toolkit` 或 `cuda-toolkit-dev` 提供头文件）  
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export GRASPNET_BASELINE=~/graspnet-baseline
# 若报错 Unknown CUDA arch (8.9)，先：export TORCH_CUDA_ARCH_LIST="8.0;8.6"
./scripts/build_graspnet_extensions.sh
```
2. **与 LIFT3D/VGGT 一致，export 并加 PYTHONPATH**（run 脚本里已自动加）；**导入 pointnet2._ext 需能找到 libc10.so**，run 脚本已自动加 PyTorch lib 到 `LD_LIBRARY_PATH`，手跑时请一并执行：
```bash
export GRASPNET_BASELINE=~/graspnet-baseline
export PYTHONPATH=$GRASPNET_BASELINE:$GRASPNET_BASELINE/pointnet2:$GRASPNET_BASELINE/utils:$GRASPNET_BASELINE/knn:$PYTHONPATH
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH
```

预训练有两种：
- **rs**（realsense，默认）：`~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar`，与 GC6D 的 realsense-d415 一致；
- **kn**（kinect）：`~/graspnet-baseline/logs/log_kn/checkpoint-kn.tar`。  
通过 `GRASPNET_PRETRAIN=rs|kn` 选择，或直接设 `GRASPNET_CKPT=/path/to/checkpoint.tar`。

### 三种 encoder：1 条 / 小批量 / 全量

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
# 预训练默认 rs；用 kinect 预训练则：export GRASPNET_PRETRAIN=kn
export GRASPNET_PRETRAIN=rs

# LIFT3D：1 条 / 小批量 100 / 全量
conda activate lift3d
MODE=1sample ./run_train_adapter_lift3d.sh
MODE=small  ./run_train_adapter_lift3d.sh
MODE=full   ./run_train_adapter_lift3d.sh

# VGGT Base：1 条 / 小批量 / 全量（仅 Stage1 + Stage2）
conda activate vggt
MODE=1sample ./run_train_adapter_vggt_base.sh
MODE=small  ./run_train_adapter_vggt_base.sh
MODE=full   ./run_train_adapter_vggt_base.sh

# VGGT Ft：1 条 / 小批量 / 全量（Stage1→2→3→4）
MODE=1sample ./run_train_adapter_vggt_ft.sh
MODE=small  ./run_train_adapter_vggt_ft.sh
MODE=full   ./run_train_adapter_vggt_ft.sh
```

产出：`checkpoints/gc6d_{lift3d|vggt_base|vggt_ft}_adapter_graspnet_s*.pt`（s4 为 LIFT3D/VGGT_ft 最终，s2 为 VGGT_base 最终）。

---

## 所有评估指令

**前置**：在 **gc6d_grasp_pipeline 根目录** 执行；先设 `DATA`，按 checkpoint 类型选 **lift3d** 或 **vggt** 环境。  
**参数**：`--split val` 验证集，`--split test` 测试集；`--max_samples 20` 快速试跑，`--max_samples 0` 或省略为全量；GraspNet 模型可加 `--use_proposals` 用 K 个 proposal 作多候选（通常提高成功率）。

### 公共环境与 DATA

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
```

### GraspNet head — 小批量 checkpoint（验证集快速 20 条）

```bash
# LIFT3D 小批量（lift3d 环境）
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_graspnet_stage3.pt --split val --max_samples 20

# VGGT Base 小批量（vggt 环境）
conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_base_graspnet.pt --split val --max_samples 20

# VGGT Ft 小批量（vggt 环境）
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_ft_graspnet_stage3.pt --split val --max_samples 20
```

### GraspNet head — 小批量 + use_proposals（推荐，成功率更高）

```bash
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_graspnet_stage3.pt --split val --max_samples 20 --use_proposals

conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_base_graspnet.pt --split val --max_samples 20 --use_proposals
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_ft_graspnet_stage3.pt --split val --max_samples 20 --use_proposals
```

### GraspNet head — 全量 checkpoint（验证集 / 测试集全量）

```bash
# 验证集全量
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split val

conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_graspnet.pt --split val
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_graspnet_stage3.pt --split val

# 测试集全量（官方 benchmark）
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split test

conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_graspnet.pt --split test
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_graspnet_stage3.pt --split test
```

### GraspNet head — 全量 + use_proposals

```bash
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split test --use_proposals

conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_graspnet.pt --split test --use_proposals
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_graspnet_stage3.pt --split test --use_proposals
```

### MLP 17D head — 小批量 checkpoint

```bash
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_mlp17d_stage3.pt --split val --max_samples 20

conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_base_mlp17d.pt --split val --max_samples 20
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_small_ft_mlp17d_stage3.pt --split val --max_samples 20
```

### MLP 17D head — 全量 checkpoint

```bash
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_mlp17d_stage3.pt --split val
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_mlp17d_stage3.pt --split test

conda activate vggt
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_mlp17d.pt --split val
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_mlp17d.pt --split test
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_mlp17d_stage3.pt --split val
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_mlp17d_stage3.pt --split test
```

### 可视化（任选一 checkpoint）

```bash
# 示例：LIFT3D GraspNet 小批量，验证集 1 条
conda activate lift3d
python visualize_offline.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_graspnet_stage3.pt --split val --max_samples 1
# 无头环境加 --no_render 只保存 PLY/图
```

---

## 三种 Encoder 小批量 / 全量指令（MLP 头直接 17D）

使用 **MLP 头直接输出 17D**（`simple_17d` 单层、`mature_17d` 多层）时，**不要**传 `--num_proposals`，必须 **--loss_17d**。在 **gc6d_grasp_pipeline 根目录** 执行，先设 `DATA`、`LIFT3D_ROOT`（仅 LIFT3D）、`CUDA_VISIBLE_DEVICES`。

**推荐**：直接用下面 6 个 run 脚本，与 GraspNet 版结构一致，仅 head 为 `simple_17d`、无 `num_proposals`。

| 类型 | Encoder | 命令（在 pipeline 根目录执行） | 产出 checkpoint |
|------|---------|--------------------------------|-----------------|
| **小批量** | LIFT3D | `conda activate lift3d` 后 `./run_small_lift3d_mlp17d.sh` | `gc6d_lift3d_small_mlp17d_stage3.pt` |
| **小批量** | VGGT Base | `conda activate vggt` 后 `./run_small_vggt_base_mlp17d.sh` | `gc6d_vggt_small_base_mlp17d.pt` |
| **小批量** | VGGT Ft | `conda activate vggt` 后 `./run_small_vggt_ft_mlp17d.sh` | `gc6d_vggt_small_ft_mlp17d_stage3.pt` |
| **全量** | LIFT3D | `conda activate lift3d` 后 `./run_train_lift3d_mlp17d.sh` | `gc6d_lift3d_mlp17d_stage3.pt` |
| **全量** | VGGT Base | `conda activate vggt` 后 `./run_train_vggt_base_mlp17d.sh` | `gc6d_vggt_base_mlp17d.pt` |
| **全量** | VGGT Ft | `conda activate vggt` 后 `./run_train_vggt_ft_mlp17d.sh` | `gc6d_vggt_ft_mlp17d_stage3.pt` |

**小批量 — LIFT3D（MLP 17D）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export N=${N:-100}
conda activate lift3d

python train_stage1_freeze_encoder.py --data_dir "$DATA" --max_samples $N --batch_size 32 --max_steps 800 --lr 1e-3 --weight_decay 5e-2 --lift3d_root "$LIFT3D_ROOT" --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --save_name gc6d_lift3d_small_mlp17d_stage1.pt
python train_stage2_lora_encoder.py --data_dir "$DATA" --max_samples $N --batch_size 32 --max_steps 1200 --lr 2e-5 --weight_decay 8e-2 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage1 checkpoints/gc6d_lift3d_small_mlp17d_stage1.pt --lift3d_root "$LIFT3D_ROOT" --save_name gc6d_lift3d_small_mlp17d_stage2.pt
python train_stage3_joint.py --data_dir "$DATA" --max_samples $N --batch_size 32 --max_steps 400 --lr 2e-5 --lr_head 1e-3 --weight_decay 1e-1 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage2 checkpoints/gc6d_lift3d_small_mlp17d_stage2.pt --lift3d_root "$LIFT3D_ROOT" --save_name gc6d_lift3d_small_mlp17d_stage3.pt
```

**小批量 — VGGT Base（MLP 17D）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
conda activate vggt

python train_vggt_base.py --data_dir "$DATA" --max_samples ${N:-100} --batch_size 32 --max_steps 1200 --lr 5e-4 --weight_decay 8e-2 --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --save_name gc6d_vggt_small_base_mlp17d.pt
```

**小批量 — VGGT Ft（MLP 17D）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
conda activate vggt

python train_vggt_ft_stage1.py --data_dir "$DATA" --max_samples ${N:-100} --batch_size 32 --max_steps 800 --lr 1e-3 --weight_decay 5e-2 --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --save_name gc6d_vggt_small_ft_mlp17d_stage1.pt
python train_vggt_ft_stage2.py --data_dir "$DATA" --max_samples ${N:-100} --batch_size 32 --max_steps 1200 --lr 1e-6 --weight_decay 8e-2 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage1 checkpoints/gc6d_vggt_small_ft_mlp17d_stage1.pt --save_name gc6d_vggt_small_ft_mlp17d_stage2.pt
python train_vggt_ft_stage3.py --data_dir "$DATA" --max_samples ${N:-100} --batch_size 32 --max_steps 400 --lr 1e-6 --lr_head 1e-3 --weight_decay 1e-1 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage2 checkpoints/gc6d_vggt_small_ft_mlp17d_stage2.pt --save_name gc6d_vggt_small_ft_mlp17d_stage3.pt
```

**全量 — LIFT3D（MLP 17D）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
conda activate lift3d

python train_stage1_freeze_encoder.py --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 1000 --lr 1e-3 --weight_decay 1e-2 --lift3d_root "$LIFT3D_ROOT" --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --val_every 500 --save_name gc6d_lift3d_mlp17d_stage1.pt
python train_stage2_lora_encoder.py --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 4000 --lr 1e-4 --weight_decay 1e-2 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage1 checkpoints/gc6d_lift3d_mlp17d_stage1.pt --lift3d_root "$LIFT3D_ROOT" --val_every 500 --save_name gc6d_lift3d_mlp17d_stage2.pt
python train_stage3_joint.py --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 2000 --lr 5e-5 --lr_head 1e-3 --weight_decay 2e-2 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage2 checkpoints/gc6d_lift3d_mlp17d_stage2.pt --lift3d_root "$LIFT3D_ROOT" --val_every 500 --save_name gc6d_lift3d_mlp17d_stage3.pt
```

**全量 — VGGT Base（MLP 17D）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
conda activate vggt

python train_vggt_base.py --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 3000 --lr 1e-3 --weight_decay 1e-2 --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight 0.3 --val_every 500 --save_name gc6d_vggt_base_mlp17d.pt
```

**全量 — VGGT Ft（MLP 17D）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
conda activate vggt

python train_vggt_ft_stage1.py --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 1000 --lr 1e-3 --weight_decay 1e-2 --grasp_head_type simple_17d --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --val_every 500 --save_name gc6d_vggt_ft_mlp17d_stage1.pt
python train_vggt_ft_stage2.py --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 4000 --lr 1e-4 --weight_decay 1e-2 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage1 checkpoints/gc6d_vggt_ft_mlp17d_stage1.pt --val_every 500 --save_name gc6d_vggt_ft_mlp17d_stage2.pt
python train_vggt_ft_stage3.py --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 2000 --lr 5e-5 --lr_head 1e-3 --weight_decay 2e-2 --loss_17d --loss_best_gt_weight 0.3 --match_mode bidir --ckpt_stage2 checkpoints/gc6d_vggt_ft_mlp17d_stage2.pt --val_every 500 --save_name gc6d_vggt_ft_mlp17d_stage3.pt
```

评估 17D 头：`python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/<上述产出>.pt --split val --use_proposals`（单输出 K=1，直接 17D 送 benchmark）。若用 `mature_17d`，把上述命令里的 `simple_17d` 换成 `mature_17d` 即可。

---

## 三种 Encoder 全量训练指令（GraspNet，推荐提升 benchmark 成功率）

小样本测试集成功率较低（如 1.24%）时，建议用**全量数据**训练后再在测试集上评估。三种 encoder 均接 GraspNet head，默认 `loss_best_gt_weight=0.3`。在 **gc6d_grasp_pipeline 根目录** 执行，环境与数据路径同前。

### 1. LIFT3D 全量（点云 + LoRA 三阶段 + GraspNet）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export CUDA_VISIBLE_DEVICES=1

conda activate lift3d
./run_train_lift3d_graspnet.sh
```

- 默认：S1=1000, S2=4000, S3=2000，**LR2=1e-4、LR3=5e-5**（改进前设定，与 small-batch 可对比），pred2gt_frac=1.0，match=bidir，best_gt_w=0.3
- 产出：`checkpoints/gc6d_lift3d_graspnet_stage1.pt` → `stage2.pt` → `stage3.pt`（评估用 stage3）
- 可选：`STEPS1=1500 STEPS2=5000 STEPS3=2500 ./run_train_lift3d_graspnet.sh` 或 `LOSS_BEST_GT_WEIGHT=0.4`

**Stage2/Stage3 loss 与成功率**：Stage2/3 用 1e-4/5e-5 时 val_loss 会卡在 ~0.042，曾试过 LR2=3e-4、LR3=1e-4，loss 略降但 **test 成功率反而略低**（1.15% vs 1.18%），故全量保持改进前保守 LR，便于与小批量公平对比。

**match / pred2gt_frac**：小批量实测 **bidir + pred2gt_frac=1.0** 比 hungarian+0.25 成功率更高，故默认改回 bidir、1.0；需要时可试 `MATCH_MODE=hungarian PRED2GT_TOP_FRAC=0.25`。

### 2. VGGT Base 全量（冻结 encoder + GraspNet，单阶段）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

conda activate vggt
./run_train_vggt_base_graspnet.sh
```

- 默认：steps=3000，val_every=500，best_gt_w=0.3
- 产出：`checkpoints/gc6d_vggt_base_graspnet.pt`
- 可选：`STEPS=4000 LOSS_BEST_GT_WEIGHT=0.4 ./run_train_vggt_base_graspnet.sh`

### 3. VGGT Ft 全量（LoRA 三阶段 + GraspNet）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

conda activate vggt
./run_train_vggt_ft_graspnet.sh
```

- 默认：S1=1000, S2=4000, S3=2000，val_every=500，best_gt_w=0.3
- 产出：`checkpoints/gc6d_vggt_ft_graspnet_stage1.pt` → `stage2.pt` → `stage3.pt`（评估用 stage3）
- 可选：`STEPS1=1500 STEPS2=5000 STEPS3=2500 ./run_train_vggt_ft_graspnet.sh`

全量训练后在测试集评估（示例 LIFT3D）：
```bash
conda activate lift3d
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split test --max_samples 0
```

---

## 训练性能完善建议（加强训练、小批量试验）

在现有 loss（bidir + best_gt_weight）基础上，可从以下方向加强训练与泛化，**先在小批量上试再上全量**。

| 方向 | 说明 | 小批量试验方式 |
|------|------|----------------|
| **best_gt_weight** | 默认 0.3；可试 0.25 或 0.4 | `LOSS_BEST_GT_WEIGHT=0.4 ./run_small_lift3d_graspnet.sh` |
| **pred2gt_top_frac / match_mode** | 默认 bidir + pred2gt_frac=1.0（实测比 hungarian+0.25 成功率更高） | 可试 `MATCH_MODE=hungarian PRED2GT_TOP_FRAC=0.25` |
| **学习率 / 步数** | 全量可适当加步数、小批量可减 lr 防震荡 | `LR1_LIFT=5e-4 STEPS1_LIFT=1200` 或全量 `STEPS2=5000 STEPS3=2500` |
| **正则** | 过拟合时加大 weight_decay 或减小 LoRA r | `WEIGHT_DECAY_S3=1.5e-1 LORA_R=4` |
| **数据增强** | 点云/图像增强已部分存在；可加强 color/gamma（VGGT） | 检查 dataset 的 `train_color_augment`、随机 crop 等是否打开 |
| **多 GT 匹配** | 已用 load_gt_multi；可试 match_mode=hungarian 一对一 | 训练脚本加 `--match_mode hungarian` |
| **Early stop** | val_loss 连续变差时提前停 Stage3 | `EARLY_STOP_VAL_WORSE=1 ./run_small_lift3d_graspnet.sh` |

小批量快速验证流程建议：同一套超参（如 N=100）跑完三阶段 → 看 val_loss 是否下降、是否过拟合 → 再在全量上复现。

---

## 提高 benchmark 成功率的改进措施（成功率偏低时必看）

当前默认评估方式：**单条 10D 预测 + 噪声采样** 得到 `top_k` 个相似抓取送 benchmark，实际等价于「一次只送一个预测的多个副本」，成功率约 1.x%。可从以下方面改进。

### 1. 评估：用 K 个 proposal 作为多候选（推荐优先尝试）

GraspNet head 输出 **K 个不同 proposal**，原先只把 softmax 加权后的**单条 10D** 送 benchmark。改为把 **K 个 proposal 各自转 17D 后一起送 benchmark**，让 benchmark 在 K 个候选中选最优，往往能明显提高成功率。

**用法**：评估时加 `--use_proposals`（需模型带 `forward_proposals`，即 GraspNet head 的 LIFT3D/VGGT 均支持）：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split test --max_samples 0 --use_proposals
```
若需恢复「单条 10D + 噪声」的旧行为，可加 `--no_use_proposals`。

**直接 17D（推荐与 --use_proposals 同用）**：GraspNet head 原始输出为 11D（t, R6d, w, score），与 GC6D 的 17D 兼容。若先转 10D 再转 17D 可能引入轴序歧义（R_permute）。加 `--direct_17d` 则用 **11D 直接转 17D**，不经 10D，与 benchmark 格式一致：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split test --max_samples 0 --use_proposals --direct_17d
```

### 2. 排查旋转轴语义（R_permute）

若预测位姿与场景坐标系不一致，可能是 10D→17D 时旋转轴顺序错误。对**同一条预测**用 6 种轴排列各跑一次 benchmark，看哪种成功率最高：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split val --max_samples 1 --R_permute_all
```
若某一排列明显优于默认 `012`，可在训练/评估中固定该排列：`--R_permute 021`（或 102/120/201/210）。

### 3. 用 GT 验证上界与诊断 AP≈0

- **--eval_gt_dump**（推荐）：用 **GT 17D** 按与预测相同的目录布局 dump，再跑同一套 GraspClutter6D API。若 **GT 的 AP 很高**（如 >30%），说明评估逻辑与 17D 格式正确，低 AP 来自训练；若 **GT 的 AP 也接近 0**，则需检查评估/坐标系/路径等。  
  `python eval_benchmark.py --data_dir "$DATA" --dataset_root /你的/GraspClutter6D根目录 --split val --eval_gt_dump`  
  或设置 `export GC6D_ROOT=/你的/GraspClutter6D根目录` 后可省略 `--dataset_root`。  
  （无需 checkpoint；需 data_dir 下 index 与 npz 中含 gt_grasp_group）

### 4. 训练侧可调项

| 措施 | 说明 | 示例 |
|------|------|------|
| **best_gt_weight** | 多 GT 下让至少一个 pred 逼近主 GT，默认 0.3 | `LOSS_BEST_GT_WEIGHT=0.4 ./run_small_lift3d_graspnet.sh` |
| **num_proposals** | 增加 head 的 K（训练与评估都需一致），多候选多机会 | 训练脚本中 `NUM_PROPOSALS=8`，评估时 `--use_proposals` |
| **更长训练** | 全量多训几步，或适当提高 Stage2/3 学习率再对比 | `STEPS2=5000 STEPS3=2500` 或按「抗过拟合」小节微调 LR |
| **action 权重** | 若失败多为旋转错误，可加大旋转项在 loss 中的权重 | 在 loss 中为 rotation 分量单独设权重（需改代码） |

### 5. 其它

- **object_id**：benchmark 期望的物体 id 若与 dataset 不一致会失败，可用 `--object_id_test` 对同一条 GT 试 object_id=0,1,2,3,69 看哪个成功。
- **depth/width**：17D 中 `depth`、`width_max` 影响碰撞检测，可试 `--depth 0.02` 或 `--width_max 0.12`（见 `eval_benchmark.py` 参数）。

**训练也走 17D（与 eval 一致）**：在 17D 空间算 loss，pred 11D 可微转 17D 与 GT 17D 在 w/R/t 上做 matching，避免 10D 转换歧义。需 **GraspNet head** 且数据带 `gt_grasp_group`（`load_gt_multi=True` 时已有）。各阶段训练脚本均支持：
```bash
# 示例：LIFT3D 全量 Stage1 用 17D loss
python train_stage1_freeze_encoder.py --data_dir "$DATA" ... --grasp_head_type graspnet --num_proposals 4 --loss_17d --loss_best_gt_weight 0.3 --save_name gc6d_lift3d_graspnet_stage1.pt
```
小样本/全量脚本若需 17D，在调用 `train_*.py` 时加上 `--loss_17d` 即可（与 `--loss_best_gt_weight` 可同时使用）。

**建议顺序**：先开 `--use_proposals` 评估 → 用 `--eval_gt_dump` 区分「评估问题」与「训练问题」→ 再试 `--R_permute_all` 排查轴序 → 再调训练（**--loss_17d**、best_gt_weight、步数、num_proposals）。

---

## 小批量测试指令汇总（三种 encoder）

用于快速跑通与超参试验，默认 N=100、GraspNet head、best_gt_weight=0.3。按需改 `N`、`VAL_EVERY`、步数等。

| Encoder | 命令 | 产出 ckpt |
|---------|------|-----------|
| LIFT3D | `conda activate lift3d` 后 `./run_small_lift3d_graspnet.sh` | `gc6d_lift3d_small_graspnet_stage3.pt` |
| VGGT Base | `conda activate vggt` 后 `./run_small_vggt_base_graspnet.sh` | `gc6d_vggt_small_base_graspnet.pt` |
| VGGT Ft | `conda activate vggt` 后 `./run_small_vggt_ft_graspnet.sh` | `gc6d_vggt_small_ft_graspnet_stage3.pt` |

可选覆盖示例：
```bash
N=50 VAL_EVERY=50 ./run_small_lift3d_graspnet.sh
LOSS_BEST_GT_WEIGHT=0.4 STEPS3_LIFT=600 ./run_small_lift3d_graspnet.sh
EARLY_STOP_VAL_WORSE=1 ./run_small_vggt_ft_graspnet.sh
```

小批量训练后快速评估（val 20 条）：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_small_graspnet_stage3.pt --split val --max_samples 20
```

---

## 1. LIFT3D（点云 + LoRA 三阶段，simple head）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline

export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="$HOME/LIFT3D"   # 或你的 LIFT3D 路径
export CUDA_VISIBLE_DEVICES=1

# 默认：S1=1000, S2=4000, S3=2000（Stage1/2/3 均控步防过拟合），val_every=500
./run_train_lift3d.sh
```

**自定义步数 / 关闭 validation：**
```bash
STEPS1=1000 STEPS2=4000 STEPS3=2000 VAL_EVERY=500 ./run_train_lift3d.sh
VAL_EVERY=0 ./run_train_lift3d.sh   # 不做 validation
```

**产出**：`checkpoints/gc6d_lift3d_stage1.pt` → `gc6d_lift3d_stage2.pt` → `gc6d_lift3d_stage3.pt`（最终用 stage3）

---

## 2. VGGT Base（冻结 encoder，只训 head，单阶段）

**Simple head（默认）：**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

./run_train_vggt_base.sh
```

**GraspNet head：**
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

./run_train_vggt_base_graspnet.sh
```

**自定义：**
```bash
STEPS=4000 VAL_EVERY=500 ./run_train_vggt_base.sh
STEPS=4000 NUM_PROPOSALS=4 ./run_train_vggt_base_graspnet.sh
VAL_EVERY=0 ./run_train_vggt_base.sh
```

**产出**：`checkpoints/gc6d_vggt_base.pt`（simple）或 `checkpoints/gc6d_vggt_base_graspnet.pt`（GraspNet）

---

## 3. VGGT Ft（三阶段：adapter+head → encoder → 联合）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline

export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

# 默认：S1=1000, S2=4000, S3=2000（Stage1/2/3 均控步防过拟合），val_every=500
./run_train_vggt_ft.sh
```

**自定义：**
```bash
STEPS1=1000 STEPS2=4000 STEPS3=2000 VAL_EVERY=500 ./run_train_vggt_ft.sh
VAL_EVERY=0 ./run_train_vggt_ft.sh
```

**产出**：`checkpoints/gc6d_vggt_ft_stage1.pt` → `gc6d_vggt_ft_stage2.pt` → `gc6d_vggt_ft_stage3.pt`（最终用 stage3）

---

## 三种 Encoder 接 GraspNet Head 训练指令

使用 **GraspNet 风格 proposal head**（`--grasp_head_type graspnet`）时，Stage1/Base 指定 head 类型与 `--num_proposals`，Stage2/Stage3 会从上一阶段 ckpt 自动继承。为避免与默认 simple/mature head 的 ckpt 冲突，下面用单独的文件名（如 `*_graspnet_*`）。

### 1. LIFT3D + GraspNet head

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export CUDA_VISIBLE_DEVICES=1

# Stage1：指定 graspnet head 与 proposal 数（默认 4）
python train_stage1_freeze_encoder.py \
  --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 1000 --lr 1e-3 \
  --weight_decay 1e-2 --lift3d_root "$LIFT3D_ROOT" --lora_r 8 --lora_scale 1.0 \
  --grasp_head_type graspnet --num_proposals 4 \
  --val_every 500 --save_name gc6d_lift3d_graspnet_stage1.pt

# Stage2：从 Stage1 ckpt 自动读取 grasp_head_type / num_proposals
python train_stage2_lora_encoder.py \
  --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 4000 --lr 1e-4 \
  --weight_decay 1e-2 --lora_r 8 --lora_scale 1.0 \
  --ckpt_stage1 checkpoints/gc6d_lift3d_graspnet_stage1.pt --lift3d_root "$LIFT3D_ROOT" \
  --val_every 500 --save_name gc6d_lift3d_graspnet_stage2.pt

# Stage3
python train_stage3_joint.py \
  --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 2000 --lr 5e-5 --lr_head 1e-3 \
  --weight_decay 2e-2 --lora_r 8 --lora_scale 1.0 \
  --ckpt_stage2 checkpoints/gc6d_lift3d_graspnet_stage2.pt --lift3d_root "$LIFT3D_ROOT" \
  --val_every 500 --save_name gc6d_lift3d_graspnet_stage3.pt
```

**产出**：`checkpoints/gc6d_lift3d_graspnet_stage1.pt` → `gc6d_lift3d_graspnet_stage2.pt` → `gc6d_lift3d_graspnet_stage3.pt`（最终用 stage3）

### 2. VGGT Base + GraspNet head

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

python train_vggt_base.py \
  --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 3000 --lr 1e-3 \
  --weight_decay 1e-2 --lora_r 8 --lora_scale 1.0 \
  --grasp_head_type graspnet --num_proposals 4 \
  --val_every 500 --save_name gc6d_vggt_base_graspnet.pt
```

**产出**：`checkpoints/gc6d_vggt_base_graspnet.pt`

### 3. VGGT Ft + GraspNet head

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

# Stage1
python train_vggt_ft_stage1.py \
  --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 1000 --lr 1e-3 \
  --weight_decay 1e-2 --lora_r 8 --lora_scale 1.0 \
  --grasp_head_type graspnet --num_proposals 4 \
  --val_every 500 --save_name gc6d_vggt_ft_graspnet_stage1.pt

# Stage2
python train_vggt_ft_stage2.py \
  --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 4000 --lr 1e-5 \
  --weight_decay 1e-2 --lora_r 8 --lora_scale 1.0 \
  --ckpt_stage1 checkpoints/gc6d_vggt_ft_graspnet_stage1.pt \
  --val_every 500 --save_name gc6d_vggt_ft_graspnet_stage2.pt

# Stage3
python train_vggt_ft_stage3.py \
  --data_dir "$DATA" --max_samples 0 --batch_size 32 --max_steps 2000 --lr 5e-6 --lr_head 1e-3 \
  --weight_decay 2e-2 --lora_r 8 --lora_scale 1.0 \
  --ckpt_stage2 checkpoints/gc6d_vggt_ft_graspnet_stage2.pt \
  --val_every 500 --save_name gc6d_vggt_ft_graspnet_stage3.pt
```

**产出**：`checkpoints/gc6d_vggt_ft_graspnet_stage1.pt` → `gc6d_vggt_ft_graspnet_stage2.pt` → `gc6d_vggt_ft_graspnet_stage3.pt`（最终用 stage3）

评估时指定对应 ckpt 即可，例如：  
`python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_graspnet_stage3.pt --split val`

---

## 小样本运行指令（Stage3：lora_last_n_blocks=2，默认不 early stop）

以下脚本已统一：Stage3 默认 **--lora_last_n_blocks 2**、**--early_stop_val_worse 0**（跑满步数）；需要「val 第一次变差就停」时可设 `EARLY_STOP_VAL_WORSE=1`。VGGT 小样本带 ColorJitter+gamma 增强。在 **gc6d_grasp_pipeline 根目录** 执行。

### LIFT3D 小样本（点云，GraspNet head）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export CUDA_VISIBLE_DEVICES=1

./run_small_lift3d_graspnet.sh
```

- 可选：`N=50 STEPS3_LIFT=200 VAL_EVERY=50 ./run_small_lift3d_graspnet.sh`  
- 产出：`checkpoints/gc6d_lift3d_small_graspnet_stage1.pt` → `stage2` → `stage3`，日志在 `logs/lift3d_small_graspnet_*`

### VGGT 微调小样本（图像，GraspNet head）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

./run_small_vggt_ft_graspnet.sh
```

- 可选：`N=50 STEPS3_VGGT=200 LORA_LAST_N_BLOCKS=1 VAL_EVERY=50 ./run_small_vggt_ft_graspnet.sh`；匹配方式：`--match_mode min` 或 `--match_mode hungarian`（默认）。  
- 产出：`checkpoints/gc6d_vggt_small_ft_graspnet_stage1.pt` → `stage2` → `stage3`，日志在 `logs/vggt_ft_small_graspnet_*`

### VGGT 微调小样本（无 GraspNet，simple head）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

./run_small_vggt_ft.sh
```

- 产出：`checkpoints/gc6d_vggt_small_ft_stage1.pt` → `stage2` → `stage3`

### LIFT3D 小样本（无 GraspNet，simple head）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export CUDA_VISIBLE_DEVICES=1

./run_small_lift3d.sh
```

### VGGT Base 小样本（单阶段，GraspNet head）

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export CUDA_VISIBLE_DEVICES=1

./run_small_vggt_base_graspnet.sh
```

- 产出：`checkpoints/gc6d_vggt_small_base_graspnet.pt`

---

## 抗过拟合：学习率、正则、LoRA r（公平对比仅调这些）

当 **val_loss 随 step 上升**（train loss 继续降）时，只允许从 **学习率、weight_decay、LoRA r** 入手缓解，其它参数不变。  
三者均使用 LoRA；抗过拟合时可适当减小 **LORA_R**（如 4）或加大 **WEIGHT_DECAY**、降低 lr。

**VGGT 微调与 LIFT3D 微调使用同一套环境变量**，仅默认学习率不同（见下）。

### Stage2 覆盖（两脚本通用）

| 变量 | 含义 | 默认 |
|------|------|------|
| `LR2` | Stage2 学习率 | LIFT3D 1e-4，VGGT 1e-5 |
| `WEIGHT_DECAY_S2` | Stage2 weight_decay | 与 `WEIGHT_DECAY` 相同（1e-2） |
| `LORA_R_S2` | Stage2 LoRA rank | 与 `LORA_R` 相同（8） |

示例（只加强 Stage2 正则、不动 Stage1）：  
`WEIGHT_DECAY_S2=2e-2 LORA_R_S2=4 ./run_train_lift3d.sh`  
`LR2=5e-6 WEIGHT_DECAY_S2=2e-2 LORA_R_S2=4 ./run_train_vggt_ft.sh`

### Stage3 覆盖（两脚本通用）

| 变量 | 含义 | 默认 |
|------|------|------|
| `LR3` | Stage3 学习率 | LIFT3D 5e-5，VGGT 5e-6 |
| `WEIGHT_DECAY_S3` | Stage3 weight_decay | 2e-2 |
| `LORA_R_S3` | Stage3 LoRA rank | 与 `LORA_R` 相同（8） |

示例：  
`WEIGHT_DECAY_S3=3e-2 LORA_R_S3=4 ./run_train_lift3d.sh`  
`LR3=3e-6 WEIGHT_DECAY_S3=3e-2 LORA_R_S3=4 ./run_train_vggt_ft.sh`

### VGGT 微调（LoRA + lr + weight_decay）

默认 **STEPS1=1000, STEPS2=4000, STEPS3=2000**；**LR2=1e-5**，**LR3=5e-6**；Stage3 **WEIGHT_DECAY_S3=2e-2**。若仍过拟合：用上表调 `STEPS2/STEPS3`、`WEIGHT_DECAY_S2/S3`、`LORA_R_S2/S3`。

### LIFT3D ft（与 VGGT 配置对齐）

默认 **STEPS1=1000, STEPS2=4000, STEPS3=2000**；**LR2=1e-4**，**LR3=5e-5**；Stage3 **WEIGHT_DECAY_S3=2e-2**。覆盖方式同 VGGT，用同一套变量即可。

---

## Val 频率建议

- **500 步**：默认，兼顾曲线分辨率和时间（约 18k 步里 ~36 次 val，6k 步里 ~12 次）。
- **1000 步**：少跑几次 val，适合机器慢或 val 集很大时。
- **250 步**：更密，方便精细看 overfit。

所有脚本均支持环境变量 `VAL_EVERY`（默认 500）；设为 `0` 即不做 validation。

---

## 评估与可视化（含测试集）

**前置**：测试集已用默认指令打包（存在 `index_test_*.jsonl`）。评估/可视化代码已支持 `--split test`，**无需改代码**。

### 评估（官方 benchmark 指标）

在 **验证集** 上跑（默认）：
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified

python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_stage3.pt --split val
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base.pt        --split val
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_stage3.pt  --split val
```

在 **测试集** 上跑（官方 benchmark）：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_stage3.pt --split test
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base.pt        --split test
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_stage3.pt  --split test
```

- 结果按 encoder 写入不同目录：`eval_out/lift3d_clip/`、`eval_out/vggt_base/`、`eval_out/vggt_ft/`，互不覆盖。
- 每次会生成 `summary.json`、`results.json`。快速试跑可加 `--max_samples 20`。

### 可视化（看预测抓取 vs GT）

对 **某一个样本** 可视化（默认取该 split 下第一个样本）：
```bash
# 验证集上一个样本
python visualize_offline.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_stage3.pt --split val --max_samples 1

# 测试集上一个样本
python visualize_offline.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_ft_stage3.pt --split test --max_samples 1
```

无头环境可加 `--no_render` 只保存 PLY/图不弹窗。  
动画版：`python visualize_grasp_animation.py ...`，同样支持 `--split test`。
