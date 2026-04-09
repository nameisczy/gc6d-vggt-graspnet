# GC6D Grasp Pipeline：训练目标与梯度断裂 — 修改建议

## 一、问题总结

### 1. 训练目标与 Benchmark 目标不一致（目标错位）

**现象（来自 train.log）：**
- 验证集 17D matching loss 从约 0.05 降至 0.035，但 rank_corr(train_decode vs bench_decode) 始终接近 0 甚至为负，最终诊断存在目标错位风险；debug_AP 约 1.0~1.6 且随训练无明显上升。

**根因：**
- **训练**用 `pred_decode_17d_differentiable`：输出 (B, K, 17) 按 **seed 下标顺序**（slot i = 第 i 个 seed），score 为 softmax 加权且已乘 objectness 概率。
- **Benchmark** 用 baseline `pred_decode`：先 **argmax 过滤** 只保留 objectness==1 的 seed，再按 **object-seed 顺序** pad 成 (B, K, 17)；评估时再按 score 降序取 top-K。
- 因此「同一 slot 下标」在两套 decode 中对应不同物理 grasp，matching loss 优化的是「前 num_seed 个 seed 的 17D」，而 AP 看的是「按 score 排序后的 top-K」。两套排序不一致，导致优化 loss 不能提升 AP。

### 2. 梯度断裂（Stage1 仅训 adapter）

**现象：** loss 无梯度 (grad_fn=None)，主路径断。

**根因：** cond 加在 seed_features 上，经冻结的 vpmodule/grasp_generator 反传后，在 cond 方向上的梯度分量接近 0，adapter 收不到有效梯度。

---

## 二、修改建议（按优先级）

### 建议 1：可微 decode 与评估一致（已实现）

1. **按 score 降序排序后取 top-K**（本次已改）：在 `pred_decode_17d_differentiable` 中，得到 (B, num_seed, 17) 后先按 `out[:,:,0]` 降序排序，再截断/pad 到 max_grasps。这样训练优化的就是「按 score 排名的 top-K」，与 eval_benchmark 中按 score 取 top-K 一致。
2. 可微 decode 中 score 已乘 objectness 概率（此前已做），背景 seed 得分压低。
3. rank_align_weight 默认 0.2，可按需调 0.3~0.5。
4. debug_benchmark_every 时：rank_corr 改为对 **两套 decode 各自按 score 排序后的同秩 score** 做 Pearson，更反映「同秩分数是否对齐」。

### 建议 2：解决 Stage1 梯度断裂（已实现选项）

- **推荐**：训练流程加 `--skip_stage1`，直接从 Stage2 训 adapter + grasp_net，梯度自然畅通。
- 或：Stage1 时使用 `--stage1_aux_cond_weight 1e-4`（或更大如 5e-4），始终加 cond^2 辅助项；或仅在检测到 loss 无梯度时自动加 1e-6（原有逻辑保留）。

### 建议 3：训练与评估 17D 一致

- loss 中 17D 与 baseline pred_decode 的 17D 格式一致（height/depth/R 行优先等）；eval_benchmark 已用同一套 pred_decode_17d。

### 建议 4：诊断评估流程

- 用 `eval_benchmark.py --eval_gt_dump` 以 GT 17D 跑 API：若 GT 的 AP 高则评估逻辑正确，低 AP 来自训练/目标错位。

### 建议 5：encoder 冻结时关闭 rank_align（VGGT Base 变差原因）

**现象（修改后实测）：** LIFT3D 微调、VGGT Ft 微调 AP 提升；**VGGT Base（encoder 不微调）AP 明显下降**（如 3.43→1.66）。

**可能原因：** rank_align_loss 让可微 decode 的 score 排序逼近 baseline decode 的排序。Encoder 微调时两者可一起往对 AP 有利的方向调；**encoder 冻结时** baseline 排序由冻结 backbone 决定，未必适合 GC6D，强行对齐可能损害 AP。

**建议：** 对仅训 adapter+head、encoder 冻结的配置（如 VGGT Base），训练时加 **`--rank_align_weight 0`**。脚本 `run_train_adapter_vggt_base.sh` 已默认传该参数。

---

## 三、已做的代码级修改

1. **models/graspnet_adapter.py**
   - 可微 decode 中 score_val 乘以 objectness 概率（此前已做）。
   - **新增**：得到 (B, num_seed, 17) 后按 score 降序排序再截断/pad 到 max_grasps，使训练目标与「评估时按 score 取 top-K」一致。

2. **train_adapter_graspnet.py**
   - rank_align_weight 默认 0.2。
   - **新增** `--skip_stage1`：跳过 Stage1（仅 adapter），可从 Stage2 开始训。
   - **新增** `--stage1_aux_cond_weight`：Stage1 时 cond^2 辅助项系数；若未设则仅在 loss 无梯度时用 1e-6；若 >0 则在 Stage1 始终加该系数。
   - **run_debug_rank_correlation**：对 train/bench 的 score 各自按 score 降序排序后，对「同秩」的 score 做 Pearson 相关，使诊断与「同秩比较」一致。

---

## 四、推荐运行方式

- **若此前 Stage1 梯度全断**：用 `--skip_stage1` 从 Stage2 开始；或修改 `run_train_adapter_lift3d.sh` 等脚本，将 stage=1 的调用改为从 stage=2 开始（不跑 stage 1）。
- **若仍跑 Stage1**：在 stage=1 的调用中加 `--stage1_aux_cond_weight 5e-4`（或更大），并保持 `--debug_benchmark_every 200` 观察 rank_corr 与 debug_AP。
- 训练结束后用 `eval_benchmark.py --checkpoint ... --split test` 得到正式 AP；用 `--eval_gt_dump` 校验评估流程。

---

## 五、验证目标是否对齐（修改后可选，快速检查用）

**说明**：下面是小规模、少步数的快速验证（约 200 样本、400 step），用来确认 rank_corr 和 debug_AP 是否随 loss 变好。**全量训练请用第六节的指令**（`--max_samples 0`、不传 `--steps`，用默认几千步）。

在项目根目录 `gc6d_grasp_pipeline` 下执行。按需替换 `GRASPNET_CKPT`、`LIFT3D_ROOT`。

**方式 A：LIFT3D 小规模 + 开 debug（仅验证对齐，非全量）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline

# 每 200 步打 rank_corr + debug_AP
export DEBUG_BENCHMARK_EVERY=200
export DEBUG_DATASET_ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
export MAX_SAMPLES=200

# 只跑 Stage2 约 400 步，看 loss↓ 时 rank_corr / debug_AP 是否一起变好
python train_adapter_graspnet.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --encoder lift3d --stage 2 \
  --graspnet_ckpt "${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}" \
  --graspnet_root "${GRASPNET_BASELINE:-$HOME/graspnet-baseline}" \
  --lift3d_root "${LIFT3D_ROOT:-$HOME/LIFT3D}" \
  --max_samples $MAX_SAMPLES --steps 400 --batch_size 4 \
  --debug_benchmark_every 200 --debug_benchmark_n 64 \
  --debug_dataset_root /mnt/ssd/ziyaochen/GraspClutter6D \
  --save_name gc6d_lift3d_s2_align_check
```

**方式 B：用 run_train_adapter_lift3d.sh（开 debug 环境变量）**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline

export DEBUG_BENCHMARK_EVERY=200
export DEBUG_DATASET_ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
export MODE=small

./run_train_adapter_lift3d.sh
```

**如何判断「目标已对齐」**

- 看 `train.log` 里每 200 步一行的 **debug**：
  - **rank_corr**：修改后为「同秩 score 的 Pearson」。随 step 增加应明显 >0（例如 0.3~0.8），不再长期接近 0 或为负。
  - **debug_AP**：随 **val loss 下降**应整体有上升趋势（允许波动），而不是 loss↓ 但 AP 不变或反而降。
- 若 rank_corr 持续 >0.3 且 loss↓ 时 debug_AP 有上升，可认为目标已对齐；再跑完整 Stage2/3/4 和 `eval_benchmark.py --split test` 看正式 AP。

---

## 六、LIFT3D / VGGT Base / VGGT Ft 训练与评估指令

**梯度断裂与目标对齐的修复在代码里对所有 encoder 生效**：`train_adapter_graspnet.py` 和 `models/graspnet_adapter.py` 是 LIFT3D / VGGT Base / VGGT Ft 共用的，因此用下面任一指令训练时，都会自动用到「可微 decode 按 score 取 top-K」和「Stage1 无梯度时 cond² 辅助 / 可选 skip_stage1」等修复，无需在脚本里再改。

以下均在 `gc6d_grasp_pipeline` 目录下执行；按需设置 `DATA`、`GRASPNET_CKPT`、`LIFT3D_ROOT`、`VGGT_CKPT` 等。

### 1. LIFT3D（全量训练 → benchmark 评估）

**训练（Stage2→3→4，跳过 Stage1 推荐）**  
全量：`--max_samples 0`，不传 `--steps` 时用默认全量步数（Stage2=4000，Stage3=5000，Stage4=4000，共 13000 step；encoder 阶段 3/4 多给步数便于对比）。

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export DEBUG_BENCHMARK_EVERY=200
export DEBUG_DATASET_ROOT=/mnt/ssd/ziyaochen/GraspClutter6D

# Stage 2（不 load，默认 4000 steps）
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder lift3d --stage 2 \
  --graspnet_ckpt "${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}" \
  --graspnet_root "${GRASPNET_BASELINE:-$HOME/graspnet-baseline}" \
  --lift3d_root "${LIFT3D_ROOT:-$HOME/LIFT3D}" \
  --max_samples 0 --batch_size 4 \
  --save_name gc6d_lift3d_adapter_graspnet_s2 \
  --debug_benchmark_every ${DEBUG_BENCHMARK_EVERY:-200} --debug_dataset_root "${DEBUG_DATASET_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"

# Stage 3（默认 5000 steps）、Stage 4（默认 4000 steps），接上一 stage ckpt
for STAGE in 3 4; do
  python train_adapter_graspnet.py \
    --data_dir "$DATA" --encoder lift3d --stage $STAGE \
    --graspnet_ckpt "${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}" \
    --graspnet_root "${GRASPNET_BASELINE:-$HOME/graspnet-baseline}" \
    --lift3d_root "${LIFT3D_ROOT:-$HOME/LIFT3D}" \
    --max_samples 0 --batch_size 4 \
    --load_ckpt checkpoints/gc6d_lift3d_adapter_graspnet_s$((STAGE-1)).pt \
    --save_name gc6d_lift3d_adapter_graspnet_s${STAGE} \
    --debug_benchmark_every ${DEBUG_BENCHMARK_EVERY:-200} --debug_dataset_root "${DEBUG_DATASET_ROOT:-/mnt/ssd/ziyaochen/GraspClutter6D}"
done
```

**评估**

```bash
python eval_benchmark.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --checkpoint checkpoints/gc6d_lift3d_adapter_graspnet_s4.pt \
  --dataset_root /mnt/ssd/ziyaochen/GraspClutter6D \
  --split test --camera realsense-d415 --top_k 50
```

---

### 2. VGGT Base（原始 VGGT + Adapter + GraspNet）

仅 Stage1（adapter）→ Stage2（adapter+head），无 Stage3/4。全量：`--max_samples 0`，不传 `--steps` 时默认 Stage1=1000、Stage2=6000（4k+2k）；且默认 `--rank_align_weight 0`（encoder 冻结时不对齐 baseline 排序）。可选 `--vggt_ckpt` 指定 VGGT 预训练权重，不传则用 vggt 包内默认。

**训练**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export MODE=full
# 可选：VGGT 预训练 ckpt（不设则用 vggt 默认）
# export VGGT_CKPT=/path/to/vggt_pretrained.pt

# Stage 1
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 1 \
  --graspnet_ckpt "${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}" \
  --graspnet_root "${GRASPNET_BASELINE:-$HOME/graspnet-baseline}" \
  --vggt_ckpt "${VGGT_CKPT:-}" \
  --max_samples 0 --batch_size 4 \
  --save_name gc6d_vggt_base_adapter_graspnet_s1

# Stage 2（脚本里 Stage2 步数=4k+2k）
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}" \
  --graspnet_root "${GRASPNET_BASELINE:-$HOME/graspnet-baseline}" \
  --vggt_ckpt "${VGGT_CKPT:-}" \
  --max_samples 0 --batch_size 4 \
  --load_ckpt checkpoints/gc6d_vggt_base_adapter_graspnet_s1.pt \
  --save_name gc6d_vggt_base_adapter_graspnet_s2
```

**评估**

```bash
python eval_benchmark.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --checkpoint checkpoints/gc6d_vggt_base_adapter_graspnet_s2.pt \
  --dataset_root /mnt/ssd/ziyaochen/GraspClutter6D \
  --split test --camera realsense-d415 --top_k 50
```

---

### 3. VGGT Ft（从 VGGT Base 开始，在 pipeline 里微调 encoder）

从 **VGGT Base（原始预训练）** 起步，Stage1→2→3→4 全流程，Stage3/4 会微调 encoder；**不需要**事先在 GC6D 上微调好的单独 ckpt。全量：`--max_samples 0`，不传 `--steps` 时默认 Stage1=1000、Stage2=4000、Stage3=5000、Stage4=4000（encoder 阶段多 step 便于对比）。`--vggt_ckpt` 可选，与 vggt_base 一样：不传则用 vggt 包默认，传则用指定 VGGT 预训练权重。

**训练**

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export MODE=full
# 可选：与 vggt_base 同，传 VGGT 预训练权重；不设则用 vggt 默认
# export VGGT_CKPT=/path/to/vggt_pretrained.pt

for STAGE in 1 2 3 4; do
  LOAD=""
  [ "$STAGE" -gt 1 ] && LOAD="--load_ckpt checkpoints/gc6d_vggt_ft_adapter_graspnet_s$((STAGE-1)).pt"
  python train_adapter_graspnet.py \
    --data_dir "$DATA" --encoder vggt_ft --stage $STAGE \
    --graspnet_ckpt "${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}" \
    --graspnet_root "${GRASPNET_BASELINE:-$HOME/graspnet-baseline}" \
    --vggt_ckpt "${VGGT_CKPT:-}" \
    --max_samples 0 --batch_size 4 \
    --save_name gc6d_vggt_ft_adapter_graspnet_s${STAGE} \
    $LOAD
done
```

**评估**

```bash
python eval_benchmark.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --checkpoint checkpoints/gc6d_vggt_ft_adapter_graspnet_s4.pt \
  --dataset_root /mnt/ssd/ziyaochen/GraspClutter6D \
  --split test --camera realsense-d415 --top_k 50
```

---

### 用脚本一键跑：三个 encoder 全量训练（修复后）

在 `gc6d_grasp_pipeline` 下执行；梯度/目标修复已在代码里，直接用脚本即可。

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
export DATA
```

**1. LIFT3D 全量（Stage1→2→3→4，默认步数 1k+4k+5k+4k）**

```bash
MODE=full DEBUG_BENCHMARK_EVERY=200 DEBUG_DATASET_ROOT=$ROOT ./run_train_adapter_lift3d.sh
# 结束后评估：python eval_benchmark.py --data_dir $DATA --checkpoint checkpoints/gc6d_lift3d_adapter_graspnet_s4.pt --dataset_root $ROOT --split test --camera realsense-d415 --top_k 50
```

**2. VGGT Base 全量（Stage1→2，默认步数 1k+6k，rank_align=0）**

```bash
MODE=full ./run_train_adapter_vggt_base.sh
# 结束后评估：python eval_benchmark.py --data_dir $DATA --checkpoint checkpoints/gc6d_vggt_base_adapter_graspnet_s2.pt --dataset_root $ROOT --split test --camera realsense-d415 --top_k 50
```

**3. VGGT Ft 全量（从 VGGT Base 开始，Stage1→2→3→4，Stage3/4 微调 encoder）**

```bash
MODE=full ./run_train_adapter_vggt_ft.sh
# 可选：VGGT_CKPT=预训练权重路径（与 vggt_base 同，不设则用 vggt 默认）
# 结束后评估：python eval_benchmark.py --data_dir $DATA --checkpoint checkpoints/gc6d_vggt_ft_adapter_graspnet_s4.pt --dataset_root $ROOT --split test --camera realsense-d415 --top_k 50
```

三个 encoder 可分别在不同终端或先后执行；训练日志在 `logs/adapter_*_full_*/`，ckpt 在 `checkpoints/`。
