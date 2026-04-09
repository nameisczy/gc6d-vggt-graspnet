# LIFT3D Baseline 新训练流水线

与旧 `train_adapter_graspnet.py` **独立**：入口为 `train_lift3d_pipeline.py`，模块拆分为 `training/losses.py`、`training/optim.py`、`models/lift3d_grasp_pipeline.py`。

## 默认行为（Stage2 equivalent）

- **LIFT3D PointNext encoder：冻结**
- **只训练**：`adapter` + 预训练 **GraspNet**（head / 全 grasp_net）
- **adapter**：`additive`，`adapter_cond_coeff=1.0`
- **无** rank_weighted、无旧 Stage1/3/4 主线
- **collision_aux**：默认可开（清障代理 loss，见 `training/losses.py`）
- **quality_aux**：默认关（接口在 `quality_aux_stub`）

## 数据与环境

- 原始 GC6D：`/mnt/ssd/ziyaochen/GraspClutter6D`（`--dataset_root`，仅 eval 用）
- 离线：`/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified`（`--data_dir`）
- `LIFT3D_ROOT`、`GRASPNET_BASELINE`、`GRASPNET_CKPT` 与此前一致

## Pure GraspNet baseline（`model_mode=pure_graspnet`）

用于对照：**不加 LIFT3D / 无 adapter / 无 cond**，仅 `point cloud -> GraspNet backbone -> vpmodule -> grasp_generator`。实现：`models/pure_graspnet.py`；`eval_benchmark.py` 与 `load_policy_from_checkpoint` 已支持。

| 实验 | 说明 | 脚本 / 命令要点 |
|------|------|------------------|
| **1. 预训练直评** | 不在 GC6D 上训练，仅保存「纯预训练」checkpoint 并 benchmark | `--model_mode pure_graspnet --pretrained_eval_only`，配 `--run_eval_after` |
| **2. GC6D 微调** | **默认冻结 backbone**，只训 **vpmodule + grasp_generator**（与 Stage2「冻结 encoder、训 GraspNet head」同构）；全量微调用 `--no_freeze_graspnet_backbone` | `--model_mode pure_graspnet --freeze_graspnet_backbone`（默认已开） |

```bash
bash scripts/run_pure_graspnet_pretrained_eval.sh
bash scripts/run_pure_graspnet_gc6d_train.sh
```

## 四类结果对照表（请自行填数）

| exp | model_mode | samples | epochs/steps | lr | seed | final val loss | AP | AP0.4 | AP0.8 | top50 collision 后剩余率 | top50 force-closure 成功数 |
|-----|------------|---------|--------------|-----|------|----------------|-----|-------|-------|---------------------------|----------------------------|
| pure GraspNet pretrained | pure_graspnet | | 0（不训练） | — | | （可选） | | | | | |
| pure GraspNet + GC6D training | pure_graspnet | | | | | | | | | | |
| best LIFT3D baseline | lift3d | | | | | | | | | | |
| best LIFT3D + aux/fusion | lift3d / local fusion | | | | | | | | | | |

## 训练

```bash
python train_lift3d_pipeline.py \
  --data_dir "$DATA" \
  --camera realsense-d415 \
  --epochs 10 \
  --graspnet_ckpt "$GRASPNET_CKPT" \
  --graspnet_root "$GRASPNET_BASELINE" \
  --lift3d_root "$LIFT3D_ROOT" \
  --collision_aux \
  --exp_name my_run
```

关闭 collision 辅助：`--no_collision_aux`

训练结束仅写出 **`final_val_loss`**（17D matching）到 `*_train_summary.json`。**AP** 需用 **`eval_benchmark.py`**（本仓库保留不变）；可加 `--run_eval_after` 自动调用。

## 局部融合对照 E1 / E2 / E3（仅 LIFT3D 线）

| 实验 | `--fusion_mode` | 说明 |
|------|-----------------|------|
| **E1** | `additive`（默认） | 全局 cond + additive，冻结 PointNext，训 `adapter` + GraspNet（与上节「默认行为」一致） |
| **E2** | `concat_proj` | `seed_features` 与 LIFT3D 最近邻 seed 特征 **concat** 后经 `Conv1d(512→256)` |
| **E3** | `residual_proj` | `seed_features + α * Conv1d(lift3d_seed)`，默认 `--residual_alpha 1.0` |

实现：`models/lift3d_local_fusion.py`（PointNext `forward_seg_feat` 最后一层 + 与 `seed_xyz` 同归一化空间的 NN gather）。  
Checkpoint 含 `fusion_mode` / `residual_alpha` / `lift3d_ckpt`；`utils/load_policy_from_checkpoint` 会据此构建 `Lift3DLocalFusionGraspNet`。

```bash
# E1
python train_lift3d_pipeline.py --fusion_mode additive ... --exp_name e1_additive

# E2
python train_lift3d_pipeline.py --fusion_mode concat_proj ... --exp_name e2_concat_proj

# E3
python train_lift3d_pipeline.py --fusion_mode residual_proj --residual_alpha 1.0 ... --exp_name e3_residual_proj
```

**并行训练（推荐）**：E1/E2/E3 已拆成三个脚本，**默认各 200 epoch**，可在不同终端 / tmux 同时跑（多卡时配不同 `CUDA_VISIBLE_DEVICES`）：

- `bash scripts/run_lift3d_e1.sh`
- `bash scripts/run_lift3d_e2.sh`
- `bash scripts/run_lift3d_e3.sh`

公共环境见 `scripts/run_lift3d_e_common.inc.sh`（`DATA`、`EPOCHS`、`BS`、`LIFT3D_CKPT` 等均可覆盖）。  
`bash scripts/run_lift3d_e123.sh` 仅打印上述用法；串行可手动 `e1 && e2 && e3`。

## 两组推荐实验与结果表（旧 baseline 对比）

| exp | samples | epochs/steps | lr | seed | adapter | coeff | collision_aux | quality_aux | final val loss | AP | AP0.4 | AP0.8 | top50 coll rem | top50 FC |
|-----|-----------|---------------|-----|------|---------|-------|---------------|-------------|----------------|-----|-------|-------|----------------|----------|
| 1 baseline + coll | | | | | additive | 1.0 | on | off | | | | | | |
| 2 no coll aux | | | | | additive | 1.0 | off | off | | | | | | |

填表：训练 summary 取 `checkpoints/lift3d_pipeline/*_train_summary.json`；AP 与 top-50 统计取 `eval_out/lift3d/<ckpt>/summary_test.json`（需 `--extra_stats`）。

## Shell

```bash
bash scripts/run_lift3d_baseline.sh
# E1/E2/E3：开三个终端分别跑（默认各 200 epoch）；多卡示例：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_lift3d_e1.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_lift3d_e2.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_lift3d_e3.sh
```

环境变量可覆盖：`DATA`、`GC6D_ROOT`、`CAM`、`EPOCHS`（E 脚本默认 200）、`GRASPNET_CKPT`、`LIFT3D_CKPT` 等。
