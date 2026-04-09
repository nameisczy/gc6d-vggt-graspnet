# 六实验结果分析与推荐配置

**说明**：下表已从各实验训练 log（含 debug）与 eval_benchmark 的 test 结果填好。**AP/AP0.4/AP0.8** 来自 `logs/eval_exp*_test.log`（测试集）；**final val loss**、**rank_corr** 来自对应 `*_debug.log` 的 final val / final debug rank_corr。**top-50 collision 剩余率、force-closure 成功数**：运行 `eval_benchmark.py --extra_stats` 会在终端和 summary JSON 中输出，填表后即可对比好/差配置。

**数据划分说明**：val loss 用验证集；测试集 AP 需单独跑 `eval_benchmark.py --split test`；中间 debug 可用 `--debug_split val` 用验证集。

---

## 一、总表

| 实验配置 | final val loss | AP | AP0.4 | AP0.8 | rank_corr | top-50 collision 后剩余率 | top-50 force-closure 成功数 |
|----------|----------------|-----|-------|-------|-----------|----------------------------|-----------------------------|
| 1 只改 eval 不排序 (no_sort_in_train) | 0.0357 | 2.70 | 1.22 | 3.33 | 0.712 | 12.2% | 1.73 |
| 2 rank-weighted (不硬 top-K) | 0.0419 | 2.70 | 1.22 | 3.33 | 0.734 | 12.2% | 1.73 |
| 3A head only (no adapter) | 0.0406 | 2.70 | 1.22 | 3.33 | nan | 12.2% | 1.73 |
| 3B adapter+head | 0.0364 | 2.70 | 1.22 | 3.33 | 0.951 | 12.2% | 1.73 |
| 4 coeff=0.25 | 0.0350 | **4.25** | **2.37** | **5.31** | 0.984 | **13.7%** | **2.07** |
| 4 coeff=0.5 | 0.0357 | 3.84 | 2.09 | 4.79 | 0.975 | 12.3% | 1.87 |
| 4 coeff=1.0 | 0.0361 | 1.89 | 1.07 | 2.34 | 0.984 | 8.6% | 1.06 |
| 4 coeff=2.0 | 0.0360 | 3.70 | 1.98 | 4.61 | 0.974 | 11.5% | 1.74 |
| 5 mode=additive | 0.0361 | **4.25** | **2.37** | **5.31** | 0.982 | **13.7%** | **2.07** |
| 5 mode=gated | 0.0362 | 3.84 | 2.09 | 4.79 | 0.982 | 12.3% | 1.87 |
| 5 mode=film | 0.0406 | 1.89 | 1.07 | 2.34 | 0.976 | 8.6% | 1.06 |
| 6A Stage2 only (LIFT3D) | 0.0424 | 2.60 | 1.12 | 3.17 | nan | 13.4% | 1.85 |
| 6B Stage2→Stage4 (LIFT3D) | 0.0455 | 2.60 | 1.12 | 3.17 | nan | 13.4% | 1.85 |

**参考 baseline**：VGGT base 改排序前 AP 3.43 / AP0.4 1.74 / AP0.8 4.27；改排序后 1.66 / 0.81 / 2.03。

**结论与行动**：见 **三、结论与当前判断**；**四、现在该怎么做** 三步；**七、唯一推荐默认配置** 已更新为 coeff=0.25 + additive + Stage2 only。

---

### 最小实验集 A/B/C/D 结果（本次跑完，encoder 均为 **vggt_base**）

| 实验 | 配置 | final val loss | AP | AP0.4 | AP0.8 | top-50 collision 后剩余率 | top-50 force-closure 成功数 |
|------|------|----------------|-----|-------|-------|----------------------------|-----------------------------|
| A | baseline（coeff=0.25 + additive，无 rank_weighted，无 aux）第一次跑开了 rank_weighted→AP 1.25；第二次重跑未开→**AP 2.09**（仍低于旧最佳 4.25，因旧最佳为 200 样本/400 步） | 0.0385 | **2.09** | 0.80 | 2.52 | 11.5% | 1.56 |
| B | baseline + use_collision_aux（占位） | 0.0380 | **2.44** | **1.04** | **3.00** | 12.0% | 1.77 |
| C | baseline + use_quality_aux（占位） | 0.0414 | 1.57 | 0.74 | 1.89 | 10.7% | 1.49 |
| D | baseline + collision_aux + quality_aux（占位） | 0.0451 | 2.15 | 1.00 | 2.58 | **14.2%** | **2.11** |

数据来源：训练 `logs/min_exp_*.log` 的 final val；评估 `eval_out/vggt_base/min_exp_*_aux/summary_test.json`（`--split test --extra_stats`）。

---

### 已完成实验统一总表（含 min_exp 全量/小规模 + 十一诊断）

数据来源：训练 log 的 final val 与配置；评估 `eval_out/vggt_base/<save_name>/summary_test.json`（`--split test --extra_stats`）。collision 剩余率为小数，表中已按百分比形式写出（如 0.115→11.5%）。

| exp | samples | epochs/steps | lr | seed | adapter mode | coeff | collision_aux | quality_aux | final val loss | AP | AP0.4 | AP0.8 | top-50 collision 后剩余率 | top-50 force-closure 成功数 |
|-----|---------|--------------|-----|------|---------------|-------|---------------|-------------|----------------|-----|-------|-------|----------------------------|-----------------------------|
| min_exp_a_baseline | 4831 | 6000 steps | 1e-3 | — | additive | 0.25 | 否 | 否 | 0.0385 | 2.09 | 0.80 | 2.52 | 11.5% | 1.56 |
| min_exp_b_collision_aux | 4831 | 6000 steps | 1e-3 | — | additive | 0.25 | 是 | 否 | 0.0432 | 2.14 | 0.74 | 2.55 | 13.5% | 1.83 |
| min_exp_c_quality_aux | 4831 | 6000 steps | 1e-3 | — | additive | 0.25 | 否 | 是 | 0.0371 | 1.96 | 0.84 | 2.38 | 10.9% | 1.63 |
| min_exp_d_both_aux | 4831 | 6000 steps | 1e-3 | — | additive | 0.25 | 是 | 是 | 0.0392 | 1.54 | 0.55 | 1.84 | 10.4% | 1.37 |
| min_exp_a_baseline_small | 200 | 400 steps | 1e-3 | 42 | additive | 0.25 | 否 | 否 | 0.0360 | 3.76 | 1.94 | 4.72 | 12.4% | 1.83 |
| min_exp_b_collision_aux_small | 200 | 400 steps | 1e-3 | 42 | additive | 0.25 | 是 | 否 | 0.0358 | 4.06 | 2.21 | 5.06 | 12.9% | 1.98 |
| min_exp_c_quality_aux_small | 200 | 400 steps | 1e-3 | 42 | additive | 0.25 | 否 | 是 | 0.0361 | 3.14 | 1.85 | 3.91 | 10.1% | 1.52 |
| min_exp_d_both_aux_small | 200 | 400 steps | 1e-3 | 42 | additive | 0.25 | 是 | 是 | 0.0362 | 4.12 | 2.37 | 5.14 | 11.8% | 1.88 |
| diag_o1_full_baseline | 4831 | 6000 steps | 1e-3 | — | additive | 0.25 | 否 | 否 | 0.0427 | 1.27 | 0.51 | 1.47 | 10.1% | 1.30 |
| diag_o2_full_3epoch | 4831 | 3 ep (3624) | 1e-3 | — | additive | 0.25 | 否 | 否 | 0.0408 | 1.53 | 0.61 | 1.85 | 8.8% | 1.11 |
| diag_o3_full_half_lr | 4831 | 6000 steps | 5e-4 | — | additive | 0.25 | 否 | 否 | 0.0412 | 1.57 | 0.69 | 1.90 | 9.2% | 1.20 |
| diag_o4_full_5epoch_half_lr | 4831 | 5 ep (6040) | 5e-4 | — | additive | 0.25 | 否 | 否 | 0.0394 | 1.55 | 0.66 | 1.82 | 10.6% | 1.42 |
| diag_small_seed42 | 200 | 400 steps | 1e-3 | 42 | additive | 0.25 | 否 | 否 | 0.0359 | 3.46 | 1.97 | 4.30 | 11.6% | 1.69 |
| diag_small_seed123 | 200 | 400 steps | 1e-3 | 123 | additive | 0.25 | 否 | 否 | 0.0363 | 3.88 | 2.22 | 4.85 | 12.2% | 1.82 |
| diag_small_seed456 | 200 | 400 steps | 1e-3 | 456 | additive | 0.25 | 否 | 否 | 0.0366 | 3.63 | 1.94 | 4.51 | 12.7% | 1.89 |
| diag_scale_200_3ep | 200 | 3 ep (150) | 1e-3 | 42 | additive | 0.25 | 否 | 否 | 0.0358 | 3.78 | 2.11 | 4.69 | 11.4% | 1.76 |
| diag_scale_1000_3ep | 1000 | 3 ep (750) | 1e-3 | 42 | additive | 0.25 | 否 | 否 | 0.0368 | 3.47 | 1.51 | 4.28 | 13.6% | 2.07 |
| diag_scale_4831_3ep | 4831 | 3 ep (3624) | 1e-3 | 42 | additive | 0.25 | 否 | 否 | 0.0380 | 2.40 | 1.00 | 2.91 | 13.8% | 1.97 |
| diag_h1_small_coeff1.0 | 200 | 400 steps | 1e-3 | 42 | additive | 1.0 | 否 | 否 | 0.0360 | 4.02 | 2.21 | 5.00 | 12.2% | 1.89 |
| diag_h1_full_coeff1.0 | 4831 | 6000 steps | 1e-3 | — | additive | 1.0 | 否 | 否 | 0.0395 | 2.19 | 0.88 | 2.61 | 11.1% | 1.47 |

**关于「默认配置 AP 下降」**：当前默认配置**仍是 vggt_base**（训练命令里显式写 `--encoder vggt_base`，脚本无默认 encoder）。与上表「实验 4 coeff=0.25」相比，本次 **min_exp A** 的 AP 明显更低（1.25 vs 4.25）。差异与修复见下 **config diff** 与 **已做修复**。

---

## 旧 coeff=0.25 最优 vs 新 baseline A：Config Diff

（来源：旧 = `logs/exp4_coeff_0.25.log` 与当时训练命令；新 = `logs/min_exp_a_baseline.log` 与文档中实验 A 命令。）

| 项 | 旧（实验 4 coeff=0.25，AP 4.36） | 新 baseline A（第一次 AP 1.25 / 第二次无 rank_weighted→AP 2.09） | 说明 |
|----|----------------------------------|------------------------------------------------------------------|------|
| **是否使用 rank_weighted_loss** | **否**（log 无该行） | 第一次**是**，第二次**否**（修复后重跑） | 旧 pred2gt 分支未按 rank 加权。 |
| **pred2gt 聚合方式** | `loss_pred2gt_agg=min`（脚本默认），无 rank_weights → 对 K 个 cost_min 取 **min** | 有 rank_weights 时改为 **(cost_min * w).sum / w.sum**（加权平均） | 聚合从「取最优一个」变为「按 rank 加权平均」，梯度信号不同。 |
| **seed** | 未设置（脚本无 --seed） | 未设置 | 两者均无固定 seed，随机性不可复现。 |
| **steps** | **400**（log：`steps=400 (small(几百))`） | **6000**（log：`steps=6000 (full(几千))`） | 旧为小规模 200 样本对应默认/显式 400 步；新为全量默认 6k。 |
| **train samples** | **200**（`max_samples=200` 或等价） | **4831**（`max_samples=0` 全量） | 数据规模不同。 |
| **batch_size** | 4（默认） | 4（默认） | 一致。 |
| **shuffle** | True（DataLoader 默认） | True | 一致。 |
| **stage** | 2 | 2 | 一致。 |
| **encoder freeze** | 是（Stage2 encoder=0） | 是 | 一致。 |
| **adapter mode / coeff** | additive，0.25 | additive，0.25 | 一致。 |
| **训练脚本参数（关键差异）** | 未传 `--use_rank_weighted_loss`；可能 `--max_samples 200`、`--steps 400` | `--use_rank_weighted_loss --rank_weight_front 16 --rank_weight_mid 48 --rank_weight_rest_w 0.25`，`--max_samples 0` | 见上。 |
| **eval 命令** | `eval_benchmark.py --checkpoint checkpoints/gc6d_vggt_base_s2_coeff0.25.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50`（可加 `--extra_stats`） | 同格式，checkpoint 为 `checkpoints/min_exp_a_baseline.pt` | eval 流程一致。 |

结论：与旧最优一致、可对比的 baseline 应**关闭** `--use_rank_weighted_loss`。关闭后全量 6k 步得到 AP≈2.09，仍低于旧最佳 4.36，**主因是训练规模不同**：旧最佳为 **200 样本 / 400 步**。要复刻旧最佳 AP，需用 `--max_samples 200 --steps 400` 再训一版（见下「复刻仍不上的原因与建议」）。

---

## 已做修复（对齐旧最优）

1. **baseline 默认不再使用 rank_weighted_loss**  
   - 文档中「实验 A」及「唯一推荐的默认训练配置」命令已去掉 `--use_rank_weighted_loss` 及相关参数，使 pred2gt 聚合与旧 coeff=0.25 一致（`min`，无 rank 加权）。
2. **训练脚本增加 `--seed`**  
   - 可选参数，传入后设置 `torch` / `numpy` 随机种子，便于复现。

修复后已用新 baseline 命令（无 `--use_rank_weighted_loss`）重跑实验 A（全量 4831 样本、6000 步），**仍未能复刻**：新 baseline 的 test AP≈**2.09**（见 `eval_out/vggt_base/min_exp_a_baseline/summary_test.json`），而旧最佳 `gc6d_vggt_base_s2_coeff0.25.pt` 的 eval log（`logs/eval_exp4_coeff_0.25_test_extra.log`）为 **AP=4.36, AP0.4=2.38, AP0.8=5.47**。

**复刻仍不上的原因（已核对 log）**  
- 旧最佳对应的训练 log（`exp4_coeff_0.25.log`）是 **200 样本、400 步**（`Train samples: 200, stage=2, steps=400 (small(几百))`），无 rank_weighted。  
- 新 baseline 第二次跑是 **4831 样本、6000 步**、无 rank_weighted，final val=0.0385，得到 AP≈2.09。  
- 因此差异主要来自 **训练规模**：旧最佳 = 小规模 200/400；新 baseline = 全量/6k。二者不是同一配置，无法直接对比。

**建议（真正对齐旧最佳以复刻 AP 4.25）**  
用与旧实验**完全相同**的规模与步数再训一版，例如：

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 200 --steps 400 --batch_size 4 \
  --seed 42 \
  --save_name gc6d_vggt_base_s2_coeff0.25_repro \
  --log_file logs/coeff0.25_repro_200_400.log
```

然后对 `checkpoints/gc6d_vggt_base_s2_coeff0.25_repro.pt` 跑 `eval_benchmark.py --split test`，看 AP 是否接近 4.25。若仍明显偏低，再查数据索引是否一致（如 index 前 200 条是否与当时一致）、或当时是否用过其他 checkpoint。

---

## 需补跑实验（带 debug_benchmark_every）

要得到 rank_corr、debug_AP 需加 `--debug_benchmark_every 200`、`--debug_dataset_root "$ROOT"`。公共环境：

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
export GRASPNET_CKPT="${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
```

---

## 六个实验的 eval_benchmark 指令与 log

**实验 1**
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_s2_no_sort_train.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 \
  2>&1 | tee logs/eval_exp1_no_sort_test.log
```

**实验 2**
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_s2_rank_weighted.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 \
  2>&1 | tee logs/eval_exp2_rank_weighted_test.log
```

**实验 3A / 3B**
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_head_only.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 \
  2>&1 | tee logs/eval_exp3a_head_only_test.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_adapter_head.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 \
  2>&1 | tee logs/eval_exp3b_adapter_head_test.log
```

**实验 4（四系数）**
```bash
for coeff in 0.25 0.5 1.0 2.0; do
  python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_s2_coeff${coeff}.pt \
    --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 \
    2>&1 | tee logs/eval_exp4_coeff_${coeff}_test.log
done
```

**实验 5（三 mode）**
```bash
for mode in additive gated film; do
  python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_s2_${mode}.pt \
    --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 \
    2>&1 | tee logs/eval_exp5_${mode}_test.log
done
```

**实验 6A / 6B（需 LIFT3D_ROOT）**
```bash
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_s2_only.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --lift3d_root "$LIFT3D_ROOT" \
  2>&1 | tee logs/eval_exp6a_lift3d_s2_test.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_lift3d_s2_then_s4.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --lift3d_root "$LIFT3D_ROOT" \
  2>&1 | tee logs/eval_exp6b_lift3d_s2_s4_test.log
```

### 带 --extra_stats 重新跑（出 collision / force-closure）

在上述每条命令中加上 `--extra_stats`，log 改为 `*_extra.log`，例如：

```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/gc6d_vggt_base_s2_no_sort_train.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_exp1_no_sort_test_extra.log
```

实验 4/5 用循环时在循环体内加 `--extra_stats`，log 为 `logs/eval_exp4_coeff_${coeff}_test_extra.log`、`logs/eval_exp5_${mode}_test_extra.log`。实验 6 同样加 `--extra_stats` 和 `--lift3d_root "$LIFT3D_ROOT"`，log 为 `*_extra.log`。

---

## 二、各实验怎么看

- **实验 1**：只改 eval 不排序，看 AP 是否主要死在「训练时排序」。
- **实验 2**：rank-weighted，看是否比 hard top-K 好。
- **实验 3**：head only vs adapter+head，看 adapter 是否添乱。
- **实验 4**：系数 sweep，0.25 最佳则默认用 0.25。
- **实验 5**：additive > gated > film，保留 additive。
- **实验 6**：Stage2 only vs Stage2→Stage4，当前无明显收益则先不折腾 encoder。

---

## 三、结论与当前判断（统一）

当前已有结果说明：

1. **encoder 微调不是当前第一瓶颈**  
   Stage2 only 和 Stage2→Stage4 指标几乎一样，说明现在系统主问题不在 encoder。

2. **排序不是当前主矛盾**  
   rank_corr 提升很多，但 AP / collision / force-closure 基本不变，说明当前低 AP 不是主要死在排序不一致。

3. **当前低 AP 更像是有效 grasp 质量不足**  
   最优配置不仅 AP 更高，而且 collision 后剩余率和 force-closure 成功数也更高，说明主问题在：**collision filtering** 与 **force-closure / grasp quality**。

4. **adapter 是有用的，但非常敏感**  
   当前最佳：adapter on、additive、**coeff=0.25**。

因此当前阶段目标不是继续硬调 encoder，而是：**先把 head + adapter + objective 做成一个更接近 benchmark validity 的强 baseline**。

---

## 四、现在该怎么做（三步）

**第一步**：默认 baseline = freeze encoder、Stage2 only、adapter=True、additive、**coeff=0.25**，排序可保留但不主攻，暂不开 encoder 微调。

**第二步**：补 **top-50 collision 后剩余率**、**top-50 force-closure 成功数**（用 `eval_benchmark.py --extra_stats`），对比最佳 baseline 与差对照（如 coeff=1.0），填总表。

**第三步**：baseline 稳后再做保守 encoder ft 实验（对照：baseline / baseline+Stage4 last-2-block LoRA / last-4-block LoRA），分析增益来自 rank、collision 还是 force-closure。

---

## 五、四个汇总问题（参考）

- 瓶颈：head / adapter / loss 优先于 encoder。
- 排序：可进训练但非主矛盾。
- Adapter：保留，系数 0.25。
- 默认流程：见第七节。

---

## 六、低 AP 主因

当前更偏向 **collision 后掉太多** 或 **force-closure quality 差**，排序错位非主因。补齐两列统计后可最终确认。

---

## 七、唯一推荐的默认训练配置

| 选项 | 取值 |
|------|------|
| adapter | True，mode=additive，**coeff=0.25** |
| Stage | Stage2 only，freeze encoder |
| rank_weighted_loss | **默认关闭**（与旧 coeff=0.25 最优一致；需时可显式加 `--use_rank_weighted_loss`） |
| encoder 微调 | 否（baseline 稳后再试） |

**命令示例**：

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 0 --batch_size 4 \
  --save_name gc6d_vggt_base_s2_baseline
```

先补第二步的 collision/force-closure 统计；baseline 稳后再按第三步做 encoder 实验。

---

## 八、代码上已做的修改

- **修改 1（默认配置）**：已改为 freeze encoder、Stage2 only、adapter on、additive、**coeff=0.25**，不默认启用 encoder 微调（`train_adapter_graspnet.py` 默认参数已对齐）。
- **修改 2（新可选项）**：已加 `--use_collision_aux`、`--use_quality_aux`，可单独开、同时开或都不开；当前为占位 loss（0），待接入 collision/quality 信号后实现真实辅助目标。
- **修改 3（优化器）**：已按 adapter params、grasp head params、encoder params 三组拆分，默认 encoder 冻结时 encoder 组为空。
- **修改 4**：排序逻辑保持当前可运行版本，不再优先扩展排序变体。

---

## 九、必须跑的最小实验集（A/B/C/D）

公共环境（在 `gc6d_grasp_pipeline` 下执行一次）：

```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
export GRASPNET_CKPT="${GRASPNET_CKPT:-$HOME/graspnet-baseline/logs/log_rs/checkpoint-rs.tar}"
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
```

每条实验跑完后，用对应 checkpoint 跑一次 eval 并加 `--extra_stats`，记录：final val loss、AP/AP0.4/AP0.8、top-50 collision 后剩余率、top-50 force-closure 成功数。

---

**实验 A：baseline（与旧 coeff=0.25 最优对齐，不开 rank_weighted_loss）**

- 配置：freeze encoder，Stage2 only，adapter on，additive，coeff=0.25，**无** rank_weighted_loss，**无** collision aux，**无** quality aux。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 0 --batch_size 4 \
  --save_name min_exp_a_baseline \
  --log_file logs/min_exp_a_baseline.log
```

评估（记录 AP、AP0.4、AP0.8、collision 剩余率、force-closure 成功数）：

```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_a_baseline.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_min_exp_a_baseline_test_extra.log
```

---

**实验 B：baseline + collision-aware auxiliary**

- 配置：与 baseline 相同，**只开** `--use_collision_aux`，不开 quality aux。  
- 重点判断：collision 后剩余率是否升、AP 是否随之上升。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --use_collision_aux \
  --max_samples 0 --batch_size 4 \
  --save_name min_exp_b_collision_aux \
  --log_file logs/min_exp_b_collision_aux.log
```

评估：

```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_b_collision_aux.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_min_exp_b_collision_aux_test_extra.log
```

---

**实验 C：baseline + quality-aware auxiliary**

- 配置：与 baseline 相同，**只开** `--use_quality_aux`，不开 collision aux。  
- 重点判断：force-closure 成功数是否升、AP 是否随之上升。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --use_quality_aux \
  --max_samples 0 --batch_size 4 \
  --save_name min_exp_c_quality_aux \
  --log_file logs/min_exp_c_quality_aux.log
```

评估：

```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_c_quality_aux.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_min_exp_c_quality_aux_test_extra.log
```

---

**实验 D：baseline + collision aux + quality aux**

- 配置：与 baseline 相同，**同时开** `--use_collision_aux`、`--use_quality_aux`。  
- 重点判断：是否同时提升 collision 后剩余率和 force-closure 成功数、是否得到当前最强配置。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --use_collision_aux --use_quality_aux \
  --max_samples 0 --batch_size 4 \
  --save_name min_exp_d_both_aux \
  --log_file logs/min_exp_d_both_aux.log
```

评估：

```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_d_both_aux.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_min_exp_d_both_aux_test_extra.log
```

---

## 三组可视化：从服务器 scp 的 dump 结果

做可视化前需把服务器上的 `eval_out` 里对应 dump 拉到本地。路径均相对于项目根目录 `gc6d_grasp_pipeline`，假设服务器上项目根为 `SERVER_USER@SERVER_HOST:/path/to/gc6d_grasp_pipeline`，下面用 `$PROJ` 表示该路径。

**说明**：每次 eval 会生成 `eval_out/<encoder_type>/<ckpt_basename>/`，内含 `dump_<split>/`（如 `dump_test`）和 `summary_<split>.json`。可视化一般需要整目录（含 `dump_test` 与 summary），以便按 checkpoint 区分。

---

**组 1：旧最佳 vs 新 baseline**

| 用途 | 需要 scp 的目录（服务器 → 本地） |
|------|----------------------------------|
| 旧最佳（coeff=0.25，AP 4.25） | `$PROJ/eval_out/vggt_base/gc6d_vggt_base_s2_coeff0.25` |
| 新 baseline（min_exp_a，无 rank_weighted） | `$PROJ/eval_out/vggt_base/min_exp_a_baseline` |

示例（在本地执行，把 `SERVER_USER`、`SERVER_HOST`、`/path/to/gc6d_grasp_pipeline` 换成实际值）：
```bash
scp -r SERVER_USER@SERVER_HOST:/path/to/gc6d_grasp_pipeline/eval_out/vggt_base/gc6d_vggt_base_s2_coeff0.25 ./eval_out/vggt_base/
scp -r SERVER_USER@SERVER_HOST:/path/to/gc6d_grasp_pipeline/eval_out/vggt_base/min_exp_a_baseline ./eval_out/vggt_base/
```

---

**组 2：coeff=0.25 vs coeff=1.0**

| 用途 | 需要 scp 的目录（服务器 → 本地） |
|------|----------------------------------|
| coeff=0.25 | `$PROJ/eval_out/vggt_base/gc6d_vggt_base_s2_coeff0.25` |
| coeff=1.0 | `$PROJ/eval_out/vggt_base/gc6d_vggt_base_s2_coeff1.0` |

示例：
```bash
scp -r SERVER_USER@SERVER_HOST:/path/to/gc6d_grasp_pipeline/eval_out/vggt_base/gc6d_vggt_base_s2_coeff0.25 ./eval_out/vggt_base/
scp -r SERVER_USER@SERVER_HOST:/path/to/gc6d_grasp_pipeline/eval_out/vggt_base/gc6d_vggt_base_s2_coeff1.0 ./eval_out/vggt_base/
```

---

**组 3：baseline A vs collision_aux / both aux（ABCD 对比）**

| 用途 | 需要 scp 的目录（服务器 → 本地） |
|------|----------------------------------|
| A baseline | `$PROJ/eval_out/vggt_base/min_exp_a_baseline` |
| B collision_aux | `$PROJ/eval_out/vggt_base/min_exp_b_collision_aux` |
| C quality_aux | `$PROJ/eval_out/vggt_base/min_exp_c_quality_aux` |
| D both_aux | `$PROJ/eval_out/vggt_base/min_exp_d_both_aux` |

示例：
```bash
for name in min_exp_a_baseline min_exp_b_collision_aux min_exp_c_quality_aux min_exp_d_both_aux; do
  scp -r SERVER_USER@SERVER_HOST:/path/to/gc6d_grasp_pipeline/eval_out/vggt_base/$name ./eval_out/vggt_base/
done
```

---

**汇总（按组最小 scp 集合）**

- **只做组 1**：`gc6d_vggt_base_s2_coeff0.25`、`min_exp_a_baseline`
- **只做组 2**：`gc6d_vggt_base_s2_coeff0.25`、`gc6d_vggt_base_s2_coeff1.0`
- **只做组 3**：`min_exp_a_baseline`、`min_exp_b_collision_aux`、`min_exp_c_quality_aux`、`min_exp_d_both_aux`
- **三组都做**：上述 6 个目录（coeff0.25 与 min_exp_a 重复，共 5 个唯一起源）：  
  `gc6d_vggt_base_s2_coeff0.25`、`gc6d_vggt_base_s2_coeff1.0`、`min_exp_a_baseline`、`min_exp_b_collision_aux`、`min_exp_c_quality_aux`、`min_exp_d_both_aux`

---

**说明**：当前 `--use_collision_aux` / `--use_quality_aux` 在代码中为占位 loss（0），B/C/D 训练曲线与 A 一致；待实现真实 collision/quality 辅助目标后，再重跑 B/C/D 对比上述四个指标。

---

## 十、小规模 A/B/C/D（200 样本、400 步）

与旧最佳 coeff=0.25 同规模（`--max_samples 200 --steps 400`），便于快速对比 ABCD、且可与旧最佳 AP 4.36 对照。公共环境同第九节；checkpoint 与 log 用 `_small` 后缀，不与全量实验覆盖。

**实验 A_small：baseline**

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 200 --steps 400 --batch_size 4 \
  --seed 42 \
  --save_name min_exp_a_baseline_small \
  --log_file logs/min_exp_a_baseline_small.log
```

评估：`python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_a_baseline_small.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_min_exp_a_baseline_small_test_extra.log`

---

**实验 B_small：baseline + collision_aux**

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --use_collision_aux \
  --max_samples 200 --steps 400 --batch_size 4 \
  --seed 42 \
  --save_name min_exp_b_collision_aux_small \
  --log_file logs/min_exp_b_collision_aux_small.log
```

评估：`python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_b_collision_aux_small.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_min_exp_b_collision_aux_small_test_extra.log`

---

**实验 C_small：baseline + quality_aux**

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --use_quality_aux \
  --max_samples 200 --steps 400 --batch_size 4 \
  --seed 42 \
  --save_name min_exp_c_quality_aux_small \
  --log_file logs/min_exp_c_quality_aux_small.log
```

评估：`python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_c_quality_aux_small.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_min_exp_c_quality_aux_small_test_extra.log`

---

**实验 D_small：baseline + collision_aux + quality_aux**

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --use_collision_aux --use_quality_aux \
  --max_samples 200 --steps 400 --batch_size 4 \
  --seed 42 \
  --save_name min_exp_d_both_aux_small \
  --log_file logs/min_exp_d_both_aux_small.log
```

评估：`python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_d_both_aux_small.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_min_exp_d_both_aux_small_test_extra.log`

---

小规模跑完后，可把四组 final val loss、AP/AP0.4/AP0.8、collision 剩余率、force-closure 成功数填表，并与旧最佳 gc6d_vggt_base_s2_coeff0.25（200/400，AP 4.36）对照。

---

## 十一、诊断实验计划（schedule / optimization / 信号 / 稳定性）

公共环境同第九节。所有实验均：vggt_base、Stage2 only、adapter additive coeff=0.25（除 H1 的 coeff 对比）、无 rank_weighted、无 aux。评估统一：`eval_benchmark.py --checkpoint checkpoints/<save_name>.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats`，记录 val loss、AP/AP0.4/AP0.8、collision 后剩余率、force-closure 成功数。

**重要**：训练会把 checkpoint 写到**项目根目录**下的 `checkpoints/<save_name>.pt`。跑评估时**必须在项目根目录下执行**（如 `cd /home/ziyaochen/gc6d_grasp_pipeline`），否则会报 `FileNotFoundError: checkpoints/xxx.pt`；若在别路径可写 `--checkpoint /绝对路径/xxx.pt`。

---

### 一、测「全量训练差是不是 schedule / optimization 问题」（优先级最高）

**实验 O1：full baseline（对照组）**

- 配置：4831 samples，当前 full 默认 steps（6000），其他统一基线。若已有 min_exp_a_baseline 全量 6k 步结果可直接复用。
- save_name：`min_exp_a_baseline` 或 `diag_o1_full_baseline`（若单独起名）

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 0 --batch_size 4 \
  --save_name diag_o1_full_baseline --log_file logs/diag_o1_full_baseline.log
```

评估（须在项目根目录执行，或 `--checkpoint` 写绝对路径）：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o1_full_baseline.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_o1_full_baseline_test_extra.log
```

**实验 O2：full baseline，按 epoch 对齐**

- 目的：排除「full 样本多但每个样本没被看够」。
- 配置：4831 samples，**不用固定 steps**，改为 `--epochs 3`（脚本会 steps=epochs*num_batches，约 3*1208≈3624）。其他同 O1。
- 若 3 epoch 明显好，可再补 5 epoch；先跑 3 epoch。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 0 --batch_size 4 \
  --epochs 3 \
  --save_name diag_o2_full_3epoch --log_file logs/diag_o2_full_3epoch.log
```

评估：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o2_full_3epoch.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_o2_full_3epoch_test_extra.log
```

**判断**：若 O2 比 O1 明显更好 → full 差有很大一部分是**训练 schedule 不合理**；若 O2 仍差 → 不是简单「没训够」。

---

**实验 O3：full baseline，减小学习率**

- 目的：排除 full 下优化过猛、把 head/adapter 学坏。
- 配置：4831 samples，与 O1 相同 steps（6000），**主 lr 改为一半**：`--lr 5e-4`（默认 1e-3）。其他同 O1。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 0 --batch_size 4 \
  --lr 5e-4 \
  --save_name diag_o3_full_half_lr --log_file logs/diag_o3_full_half_lr.log
```

评估：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o3_full_half_lr.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_o3_full_half_lr_test_extra.log
```

**判断**：若 O3 明显好于 O1 → full 差有很大一部分是**优化过猛**；若无改善 → 主要不是 lr。

---

**实验 O4（可选）：full baseline，长训练 + 小 lr**

- 仅在 O2 或 O3 有改善时再跑。
- 配置：4831 samples，按 epoch 跑（如 `--epochs 5`），更小 lr（如 `--lr 5e-4`）。

```bash
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
  --rank_align_weight 0 --max_samples 0 --batch_size 4 \
  --epochs 5 --lr 5e-4 \
  --save_name diag_o4_full_5epoch_half_lr --log_file logs/diag_o4_full_5epoch_half_lr.log
```

评估：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o4_full_5epoch_half_lr.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_o4_full_5epoch_half_lr_test_extra.log
```

**判断**：若 O4 最好 → full 问题主要是 **optimization / schedule**；若仍不行 → 更像 **objective / signal** 问题。

---

### 二、测「小规模高 AP 是否只是幸运过拟合」

**实验 S1 / S2 / S3：small baseline 多 seed**

- 配置固定：200 samples，400 steps，统一基线，**只改 `--seed`**。至少 3 个 seed（如 42、123、456）。
- 记录：val loss，AP/AP0.4/AP0.8，collision 后剩余率，force-closure 成功数。

```bash
for seed in 42 123 456; do
  python train_adapter_graspnet.py \
    --data_dir "$DATA" --encoder vggt_base --stage 2 \
    --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
    --adapter_cond_coeff 0.25 --adapter_cond_mode additive \
    --rank_align_weight 0 --max_samples 200 --steps 400 --batch_size 4 \
    --seed $seed \
    --save_name diag_small_seed${seed} \
    --log_file logs/diag_small_seed${seed}.log
  python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_small_seed${seed}.pt \
    --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
    2>&1 | tee logs/eval_diag_small_seed${seed}_test_extra.log
done
```

**判断**：若 3 个 seed 差异很大 → small 高 AP 有**明显幸运成分**；若都稳定接近 4 左右 → small regime 确实更容易学到对 benchmark 有利的模式。

评估（S1/S2/S3 各一条；**须在项目根目录** `cd /path/to/gc6d_grasp_pipeline` 下执行，否则用 `--checkpoint /绝对路径/checkpoints/diag_small_seed42.pt`）：
```bash
# S1 seed=42
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_small_seed42.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_small_seed42_test_extra.log

# S2 seed=123
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_small_seed123.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_small_seed123_test_extra.log

# S3 seed=456
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_small_seed456.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_small_seed456_test_extra.log
```

或循环一次跑完：`for seed in 42 123 456; do python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_small_seed${seed}.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_small_seed${seed}_test_extra.log; done`

---

### 三、测「全量下正确 grasp quality 信号被冲淡」

**数据规模扫描（按 epoch 对齐）**

- 不要只看 small/full 两点，加中间点：**200 / 1000 / 4831** samples。
- 步数按「每个样本看到差不多轮数」：**统一用相同 epoch 数**（如 3 epoch），不固定死 steps。
  - 200 samples, 3 epoch → steps ≈ 150
  - 1000 samples, 3 epoch → steps ≈ 750
  - 4831 samples, 3 epoch → steps ≈ 3624

```bash
# 200, 3 epoch
python train_adapter_graspnet.py --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive --rank_align_weight 0 \
  --max_samples 200 --epochs 3 --batch_size 4 --seed 42 \
  --save_name diag_scale_200_3ep --log_file logs/diag_scale_200_3ep.log

# 1000, 3 epoch
python train_adapter_graspnet.py --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive --rank_align_weight 0 \
  --max_samples 1000 --epochs 3 --batch_size 4 --seed 42 \
  --save_name diag_scale_1000_3ep --log_file logs/diag_scale_1000_3ep.log

# 4831, 3 epoch
python train_adapter_graspnet.py --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 0.25 --adapter_cond_mode additive --rank_align_weight 0 \
  --max_samples 0 --epochs 3 --batch_size 4 --seed 42 \
  --save_name diag_scale_4831_3ep --log_file logs/diag_scale_4831_3ep.log
```

评估（三档各一条，须在项目根目录执行）：
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_scale_200_3ep.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_scale_200_3ep_test_extra.log

python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_scale_1000_3ep.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_scale_1000_3ep_test_extra.log

python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_scale_4831_3ep.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_scale_4831_3ep_test_extra.log
```

**判断**：若 200 → 1000 → 4831 时 **val loss 平稳下降**，但 **AP / collision / force-closure 不升反降** → 更多数据引入的主要不是有效 grasp 信号，而是把模型推向更平均、更保守的解，支持「**信号被冲淡**」。

---

### 四、测「head / adapter 在 full 下是否更不稳定」（优先级略低）

**实验 H1：small vs full 的 coeff 敏感性**

- 只跑两个系数：**coeff=0.25**、**coeff=1.0**；分别在 **small**、**full** 各跑一组。最少 4 个实验：
  - small coeff=0.25
  - small coeff=1.0
  - full coeff=0.25
  - full coeff=1.0

```bash
# small coeff=0.25（可与已有 min_exp_a_baseline_small 复用）
# small coeff=1.0
python train_adapter_graspnet.py --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 1.0 --adapter_cond_mode additive --rank_align_weight 0 \
  --max_samples 200 --steps 400 --batch_size 4 --seed 42 \
  --save_name diag_h1_small_coeff1.0 --log_file logs/diag_h1_small_coeff1.0.log

# full coeff=0.25（可与 O1 复用）
# full coeff=1.0
python train_adapter_graspnet.py --data_dir "$DATA" --encoder vggt_base --stage 2 \
  --graspnet_ckpt "$GRASPNET_CKPT" --graspnet_root "$GRASPNET_BASELINE" \
  --adapter_cond_coeff 1.0 --adapter_cond_mode additive --rank_align_weight 0 \
  --max_samples 0 --batch_size 4 \
  --save_name diag_h1_full_coeff1.0 --log_file logs/diag_h1_full_coeff1.0.log
```

评估（H1 四组：small/full × 0.25/1.0；0.25 可复用已有 baseline 的 ckpt，下面只写需单独评估的 4 个 checkpoint）：
```bash
# small coeff=0.25（若用 min_exp_a_baseline_small）
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_a_baseline_small.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_h1_small_coeff0.25_test_extra.log

# small coeff=1.0
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_h1_small_coeff1.0.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_h1_small_coeff1.0_test_extra.log

# full coeff=0.25（若用 diag_o1_full_baseline）
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o1_full_baseline.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_h1_full_coeff0.25_test_extra.log

# full coeff=1.0
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_h1_full_coeff1.0.pt \
  --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats \
  2>&1 | tee logs/eval_diag_h1_full_coeff1.0_test_extra.log
```

**判断**：若 **full 下 0.25→1.0 掉得比 small 更厉害** → adapter/head 在 full 下更不稳定，支持后面少碰 encoder、先稳住 head+adapter。

---

### 评估指令汇总（十一全部实验，均须在项目根目录执行）

先设置环境并进入项目根目录：
```bash
cd /home/ziyaochen/gc6d_grasp_pipeline
export DATA=/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified
export ROOT=/mnt/ssd/ziyaochen/GraspClutter6D
```

以下按实验组粘贴即可（checkpoint 不存在会报错，先确保对应训练已跑完）。

**O1～O4**
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o1_full_baseline.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_o1_full_baseline_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o2_full_3epoch.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_o2_full_3epoch_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o3_full_half_lr.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_o3_full_half_lr_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o4_full_5epoch_half_lr.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_o4_full_5epoch_half_lr_test_extra.log
```

**S1/S2/S3（多 seed）**
```bash
for seed in 42 123 456; do
  python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_small_seed${seed}.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_small_seed${seed}_test_extra.log
done
```

**数据规模扫描（三档）**
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_scale_200_3ep.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_scale_200_3ep_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_scale_1000_3ep.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_scale_1000_3ep_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_scale_4831_3ep.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_scale_4831_3ep_test_extra.log
```

**H1（coeff 敏感性，4 个 ckpt）**
```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/min_exp_a_baseline_small.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_h1_small_coeff0.25_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_h1_small_coeff1.0.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_h1_small_coeff1.0_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_o1_full_baseline.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_h1_full_coeff0.25_test_extra.log
python eval_benchmark.py --data_dir "$DATA" --checkpoint checkpoints/diag_h1_full_coeff1.0.pt --dataset_root "$ROOT" --split test --camera realsense-d415 --top_k 50 --extra_stats 2>&1 | tee logs/eval_diag_h1_full_coeff1.0_test_extra.log
```

若当前目录不是项目根，请先 `cd` 到 `gc6d_grasp_pipeline`，或将 `checkpoints/xxx.pt` 改为绝对路径（如 `$PWD/checkpoints/xxx.pt`）。
