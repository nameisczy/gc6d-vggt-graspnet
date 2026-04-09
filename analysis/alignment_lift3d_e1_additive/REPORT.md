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
  "cond_expand_to_graspnet_seed_ridge": {
    "mse": 1.4442635776076501e+181,
    "cosine_global": 0.5001194674298646,
    "fro_W": 2.8142777651186194e+92
  },
  "cond_expand_to_graspnet_seed_ridge_normalized": {
    "mse": 1.9213145047520468e+194,
    "cosine_global": -0.004700899418786603,
    "fro_W": 2.641620491318799e+98
  }
}

## 四、结论（根据本次运行自动摘要，需结合图核对）

### Scale mismatch

- **l2 比值** `adapter_ckpt__cond_vs_graspnet_seed_features`: 0.0202（encoder 相对 graspnet）
- **l2 比值** `adapter_ckpt__cond_expand_vs_graspnet_seed_features`: 0.6466（encoder 相对 graspnet）
- **l2 比值** `adapter_ckpt__c_raw_vs_graspnet_seed_features`: 0.0202（encoder 相对 graspnet）
- **l2 比值** `adapter_ckpt__seed_features_vs_graspnet_seed_features`: 0.9686（encoder 相对 graspnet）
- **l2 比值** `adapter_ckpt__seed_xyz_vs_graspnet_seed_features`: 0.0966（encoder 相对 graspnet）
- **l2 比值** `adapter_ckpt__seed_after_fusion_vs_graspnet_seed_features`: 1.2882（encoder 相对 graspnet）
- **逐通道 std 中位比** `adapter_ckpt__cond_expand`: median=0.3381, 通道数>2x: 45, <0.5x: 148
- **逐通道 std 中位比** `adapter_ckpt__seed_features`: median=2.3188, 通道数>2x: 142, <0.5x: 2
- **逐通道 std 中位比** `adapter_ckpt__seed_after_fusion`: median=2.4340, 通道数>2x: 145, <0.5x: 1

### Distribution mismatch

- LayerNorm 后 L2 与 graspnet 之比 `adapter_ckpt__cond_expand`: 1.0000
- LayerNorm 后 L2 与 graspnet 之比 `adapter_ckpt__seed_features`: 1.0000
- LayerNorm 后 L2 与 graspnet 之比 `adapter_ckpt__seed_after_fusion`: 1.0000

### Channel semantic mismatch（线性可对齐性 + 热图）

- `cond_expand_to_graspnet_seed_ridge`: MSE=14442635776076501268316956044889033064651367892338379081599589562115262627159144122282895961817744184519922416246112670464848294696539682386478676014811307885407858250205825899429888.000000, cosine=0.5001, ||W||_F=281427776511861938650411536051041847651530142127333446008669703014884341711771966753791279104.00
- `cond_expand_to_graspnet_seed_ridge_normalized`: MSE=192131450475204678217633579977331353428541435417100429865258275160797488896363144250470031015280422173987197143077889205662629139003633773170119962282655595922010971037889516119833487684596137984.000000, cosine=-0.0047, ||W||_F=264162049131879893851428948646416213428201501102691951399548415370687125736133840820680719616442368.00
- 通道余弦热图：若**非对角**且 ridge MSE 仍高 → 支持 channel semantic mismatch。

### 综合与后续建议

- 若 **l2/abs_mean 比值**远离 1 且逐通道 std 比两极化 → **scale** 问题突出，优先 **LayerNorm / 可学习 1×1 / channel-wise scale**。

- 若直方图形状差异大、但 LayerNorm 后统计接近 → **distribution** 多为尺度/偏移，可先 **normalization** 再 fusion。

- 若 ridge（尤其归一化后）仍高 MSE、热图杂乱 → **channel semantic** 突出，优先 **concat + 投影、FiLM、或与 graspnet 分支并行再融合**。
