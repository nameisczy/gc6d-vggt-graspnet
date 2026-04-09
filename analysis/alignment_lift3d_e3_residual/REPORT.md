# Encoder vs GraspNet seed_features 对齐分析


## 一、数值统计表（完整见 `stats.json`）

- `global`：各张量 mean/std/min/max/median/q25/q75/l2/l1/abs_mean

- `per_channel_summary`：256 通道 mean/std/min/max/abs_mean/norm 及汇总

- `per_seed`：per-seed L2/L1 分位数

- `per_channel_distributions`：通道 mean/std/norm 的直方图计数

- `ratios`：相对 `graspnet_seed_features` 的全局比值

- `per_channel_std_ratio_vs_graspnet`：逐通道 std 比值摘要

- `batch_stability_l2` / `scene_stability` / `cross_scene_l2_range`：batch 与 scene 稳定性

- `offline_normalize`：LayerNorm 与通道 z-score 后的全局统计；`offline_normalize_l2_ratio_vs_graspnet_ln`


## 二、可视化图集（`figures/`）

- `hist_values.png` / `hist_abs.png`：值与绝对值密度

- `box_per_channel_std.png`：各通道 std 的箱线图

- `heatmap_ch_cosine_*.png`：通道余弦矩阵（cond_expand / lift3d_seed）

- `pca2d.png` / `umap2d.png` / `tsne2d.png`：流形可视化

- `violin_per_seed_l2.png` / `box_per_seed_l2.png`：per-seed L2

- `box_batch_l2.png`：各张量按 batch 的全局 L2


## 三、线性对齐探针（`linear_probe.json`）

{
  "lift3d_seed_to_graspnet_seed_ridge": {
    "mse": 8.45885214210126e+112,
    "cosine_global": 0.07259957334006144,
    "fro_W": 4.5258678397002384e+57
  },
  "lift3d_seed_to_graspnet_seed_ridge_normalized": {
    "mse": 2.8625907142939615e+99,
    "cosine_global": 0.0019106217270794307,
    "fro_W": 9.526670551272945e+50
  }
}

## 四、结论（根据本次运行自动摘要，需结合图核对）

### Scale mismatch

- **l2 比值** `lift3d_local_fusion__seed_features_vs_graspnet_seed_features`: 0.9783（encoder 相对 graspnet）
- **l2 比值** `lift3d_local_fusion__lift3d_raw_vs_graspnet_seed_features`: 5.4049（encoder 相对 graspnet）
- **l2 比值** `lift3d_local_fusion__lift3d_seed_vs_graspnet_seed_features`: 2.1592（encoder 相对 graspnet）
- **l2 比值** `lift3d_local_fusion__seed_xyz_vs_graspnet_seed_features`: 0.0966（encoder 相对 graspnet）
- **l2 比值** `lift3d_local_fusion__residual_delta_vs_graspnet_seed_features`: 1.2952（encoder 相对 graspnet）
- **l2 比值** `lift3d_local_fusion__seed_after_fusion_vs_graspnet_seed_features`: 1.6560（encoder 相对 graspnet）
- **逐通道 std 中位比** `lift3d_local_fusion__seed_features`: median=2.3786, 通道数>2x: 144, <0.5x: 2
- **逐通道 std 中位比** `lift3d_local_fusion__lift3d_seed`: median=1.8955, 通道数>2x: 126, <0.5x: 13
- **逐通道 std 中位比** `lift3d_local_fusion__residual_delta`: median=1.1398, 通道数>2x: 94, <0.5x: 56
- **逐通道 std 中位比** `lift3d_local_fusion__seed_after_fusion`: median=2.6755, 通道数>2x: 151, <0.5x: 1

### Distribution mismatch

- LayerNorm 后 L2 与 graspnet 之比 `lift3d_local_fusion__seed_features`: 1.0000
- LayerNorm 后 L2 与 graspnet 之比 `lift3d_local_fusion__lift3d_seed`: 1.0000
- LayerNorm 后 L2 与 graspnet 之比 `lift3d_local_fusion__residual_delta`: 1.0000
- LayerNorm 后 L2 与 graspnet 之比 `lift3d_local_fusion__seed_after_fusion`: 1.0000

### Channel semantic mismatch（线性可对齐性 + 热图）

- `lift3d_seed_to_graspnet_seed_ridge`: MSE=84588521421012599747895418984776162987938545895582394254919915818196053928871997265280215268042668630970644561920.000000, cosine=0.0726, ||W||_F=4525867839700238442062304906199071639789765946600109113344.00
- `lift3d_seed_to_graspnet_seed_ridge_normalized`: MSE=2862590714293961493922469813587311826250842161796753480398854506703423693202404269778671312819978240.000000, cosine=0.0019, ||W||_F=952667055127294516082073929667437790656631006035968.00
- 通道余弦热图：若**非对角**且 ridge MSE 仍高 → 支持 channel semantic mismatch。

### 综合与后续建议

- 若 **l2/abs_mean 比值**远离 1 且逐通道 std 比两极化 → **scale** 问题突出，优先 **LayerNorm / 可学习 1×1 / channel-wise scale**。

- 若直方图形状差异大、但 LayerNorm 后统计接近 → **distribution** 多为尺度/偏移，可先 **normalization** 再 fusion。

- 若 ridge（尤其归一化后）仍高 MSE、热图杂乱 → **channel semantic** 突出，优先 **concat + 投影、FiLM、或与 graspnet 分支并行再融合**。
