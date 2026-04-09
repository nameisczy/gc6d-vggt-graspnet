# 实验与 Ablation 指令

以下实验用于区分：score 排序、head、adapter、encoder 微调 对 AP 的影响。每个实验单独跑并保存 log。

**公共环境**（在 `gc6d_grasp_pipeline` 下）：
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
export GRASPNET_CKPT="${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
```

### 6 个实验运行指令与 Log 一览

| 实验 | 训练 Log | 测试集 eval Log（eval_benchmark.py） |
|------|----------|--------------------------------------|
| 1 只改 eval 不排序 | `logs/exp1_no_sort_train.log` | `logs/eval_exp1_no_sort_test.log` |
| 2 rank-weighted | `logs/exp_rank_weighted.log` | `logs/eval_exp2_rank_weighted_test.log` |
| 3A head-only / 3B adapter+head | `logs/exp3a_head_only.log`、`logs/exp3b_adapter_head.log` | `logs/eval_exp3a_head_only_test.log`、`logs/eval_exp3b_adapter_head_test.log` |
| 4 系数 sweep | `logs/exp4_coeff_0.25.log` 等 | `logs/eval_exp4_coeff_0.25_test.log` 等 |
| 5 模式 additive/gated/film | `logs/exp5_mode_additive.log` 等 | `logs/eval_exp5_additive_test.log` 等 |
| 6A Stage2 only / 6B S2→S4 | `logs/exp6a_lift3d_s2_only.log`、`logs/exp6b_lift3d_s2_then_s4.log` | `logs/eval_exp6a_lift3d_s2_test.log`、`logs/eval_exp6b_lift3d_s2_s4_test.log` |

各实验训练命令见下方对应小节；**测试集 eval_benchmark.py 的完整指令与 log 见 [EXPERIMENT_RESULTS_ANALYSIS.md](./EXPERIMENT_RESULTS_ANALYSIS.md) 中「六个实验的 eval_benchmark 指令与 log」**。

---

## 实验 1：只改 eval 不改 train（判断是否“训练时硬排序介入太早”）

**目的**：若训练仍用旧版 soft decode（不按 score 排序截断），仅评估时按 score 排序，看 VGGT base 是否不会大幅下降。

**运行指令（带 log）**：
```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --no_sort_in_train_decode --rank_align_weight 0 \
  --max_samples 0 --batch_size 4 \
  --save_name gc6d_vggt_base_s2_no_sort_train \
  --log_file logs/exp1_no_sort_train.log
```

**做法**：使用 `--no_sort_in_train_decode`，训练时可微 decode 不排序、不截断，仅 eval/benchmark 时用排序后的 decode。比较 train/val loss 与 benchmark AP。

**评估时**：`load_policy_from_checkpoint` 已从 ckpt 读取 `use_adapter`、`adapter_cond_coeff`、`adapter_cond_mode`，与训练时一致；`strict=False` 以兼容 no-adapter 或不同 mode 的 ckpt。

---

## 修改 1：rank-weighted loss（已实现）

训练时用「排序+权重」代替「硬 top-K 截断」：前排权重大，多 grasp 参与 loss。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --max_samples 0 --batch_size 4 \
  --use_rank_weighted_loss --rank_weight_front 16 --rank_weight_mid 48 --rank_weight_rest_w 0.25 \
  --rank_align_weight 0 \
  --save_name gc6d_vggt_base_s2_rank_weighted \
  --log_file logs/exp_rank_weighted.log
```

**看点**：VGGT base 是否恢复、ft 是否不坏。

---

## 实验 2：训练时排序 + rank-weighted，不硬 top-K（同上）

与修改 1 一致，已用 `--use_rank_weighted_loss`。

---

## 实验 3：no-adapter（encoder + head only）vs 有 adapter

**3A：freeze encoder，只训 head，不加 adapter**
```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --no_adapter --train_head_only \
  --max_samples 0 --batch_size 4 \
  --save_name gc6d_vggt_base_head_only \
  --log_file logs/exp3a_head_only.log
```

**3B：freeze encoder，训 adapter + head（当前默认）**
```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --rank_align_weight 0 --max_samples 0 --batch_size 4 \
  --save_name gc6d_vggt_base_adapter_head \
  --log_file logs/exp3b_adapter_head.log
```

**看点**：若 3A AP > 3B，说明 adapter 在添乱。

---

## 实验 4：adapter 系数 sweep

```bash
for coeff in 0.25 0.5 1.0 2.0; do
  python train_adapter_graspnet.py \
    --data_dir "$DATA" --encoder vggt_base --stage 2 \
    --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
    --adapter_cond_coeff $coeff --rank_align_weight 0 \
    --max_samples 200 --steps 400 \
    --save_name gc6d_vggt_base_s2_coeff${coeff} \
    --log_file logs/exp4_coeff_${coeff}.log
done
```

**看点**：若结果对系数极度敏感，说明当前 adapter 注入方式脆弱。

---

## 实验 5：adapter 注入方式（additive / gated / film）

```bash
for mode in additive gated film; do
  python train_adapter_graspnet.py \
    --data_dir "$DATA" --encoder vggt_base --stage 2 \
    --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
    --adapter_cond_mode $mode --rank_align_weight 0 \
    --max_samples 200 --steps 400 \
    --save_name gc6d_vggt_base_s2_${mode} \
    --log_file logs/exp5_mode_${mode}.log
done
```

---

## 实验 6：两阶段简化（Stage2 only 或 Stage2→Stage4）

**6A：仅 Stage2（freeze encoder，训 adapter+head）**
```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder lift3d --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --lift3d_root "${LIFT3D_ROOT:-$HOME/LIFT3D}" \
  --max_samples 0 --batch_size 4 \
  --save_name gc6d_lift3d_s2_only \
  --log_file logs/exp6a_lift3d_s2_only.log
```

**6B：Stage2 训完再 Stage4 小 lr 联合**
```bash
# 先跑 6A，得到 checkpoints/gc6d_lift3d_s2_only.pt
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder lift3d --stage 4 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --lift3d_root "${LIFT3D_ROOT:-$HOME/LIFT3D}" \
  --load_ckpt checkpoints/gc6d_lift3d_s2_only.pt \
  --max_samples 0 --batch_size 4 --lr 3e-4 \
  --save_name gc6d_lift3d_s2_then_s4 \
  --log_file logs/exp6b_lift3d_s2_then_s4.log
```

**看点**：若 6A/6B 比完整 1→2→3→4 更稳，说明 Stage1/Stage3 可能有害或可省。

---

## 评估（每个实验结束后）

对测试集跑 AP 并写 log（`2>&1 | tee logs/xxx.log`）。**完整命令与各实验 log 路径见 [EXPERIMENT_RESULTS_ANALYSIS.md](./EXPERIMENT_RESULTS_ANALYSIS.md) 中「六个实验的 eval_benchmark 指令与 log」**。

通用形式（LIFT3D 实验 6A/6B 需加 `--lift3d_root "$LIFT3D_ROOT"`）：
```bash
python eval_benchmark.py \
  --data_dir "$DATA" --checkpoint checkpoints/<save_name>.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 \
  2>&1 | tee logs/eval_<实验名>_test.log
```

将 `<save_name>` 换成各实验的 `--save_name`（如 `gc6d_vggt_base_s2_no_sort_train`）。

---

## 已做代码修改摘要

| 修改 | 说明 |
|------|------|
| **rank-weighted loss** | `--use_rank_weighted_loss`：pred2gt 按 rank 加权（front 16=1.0, mid 48=0.5, rest=0.25），不硬 top-K |
| **no-adapter** | `--no_adapter`：EncoderAdapterGraspNet 不加 conditioning |
| **adapter 可配置** | `--adapter_cond_coeff`、`--adapter_cond_mode`（additive / gated / film） |
| **train_head_only** | `--train_head_only`：freeze encoder 与 adapter，只训 grasp_net |
| **optimizer 三组** | adapter lr、grasp_net lr、encoder lr 分组 |
| **debug 日志** | 每 debug 步输出 rank_corr、debug_AP、score mean/std、width mean/std |
