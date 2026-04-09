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
    "mse": 3.9391938170893604e+128,
    "cosine_global": 0.0791823419622634,
    "fro_W": 6.4323765889527496e+66
  },
  "cond_expand_to_graspnet_seed_ridge_normalized": {
    "mse": 7.447386115493813e+185,
    "cosine_global": -0.0007963373598268914,
    "fro_W": 1.692142376330677e+94
  }
}

## 四、结论（根据本次运行自动摘要，需结合图核对）

### Scale mismatch

- **l2 比值** `vggt_adapter_ckpt__cond_vs_graspnet_seed_features`: 0.0021（encoder 相对 graspnet）
- **l2 比值** `vggt_adapter_ckpt__cond_expand_vs_graspnet_seed_features`: 0.1359（encoder 相对 graspnet）
- **l2 比值** `vggt_adapter_ckpt__c_raw_vs_graspnet_seed_features`: 0.0042（encoder 相对 graspnet）
- **l2 比值** `vggt_adapter_ckpt__seed_features_vs_graspnet_seed_features`: 0.9965（encoder 相对 graspnet）
- **l2 比值** `vggt_adapter_ckpt__seed_xyz_vs_graspnet_seed_features`: 0.0966（encoder 相对 graspnet）
- **l2 比值** `vggt_adapter_ckpt__seed_after_fusion_vs_graspnet_seed_features`: 1.0407（encoder 相对 graspnet）
- **逐通道 std 中位比** `vggt_adapter_ckpt__cond_expand`: median=0.0000, 通道数>2x: 0, <0.5x: 256
- **逐通道 std 中位比** `vggt_adapter_ckpt__seed_features`: median=2.4913, 通道数>2x: 144, <0.5x: 3
- **逐通道 std 中位比** `vggt_adapter_ckpt__seed_after_fusion`: median=2.4913, 通道数>2x: 144, <0.5x: 3

### Distribution mismatch

- LayerNorm 后 L2 与 graspnet 之比 `vggt_adapter_ckpt__cond_expand`: 0.9982
- LayerNorm 后 L2 与 graspnet 之比 `vggt_adapter_ckpt__seed_features`: 1.0000
- LayerNorm 后 L2 与 graspnet 之比 `vggt_adapter_ckpt__seed_after_fusion`: 1.0000

### Channel semantic mismatch（线性可对齐性 + 热图）

- `cond_expand_to_graspnet_seed_ridge`: MSE=393919381708936040057284552811202866454017586212121247763037813990358254844965712760567169074576768790532010454369844392387149824.000000, cosine=0.0792, ||W||_F=6432376588952749561019997644890419760902877186607702540841681682432.00
- `cond_expand_to_graspnet_seed_ridge_normalized`: MSE=744738611549381342174374496128796781904481392339431829466271069249605284930274513024711521747110130212141758528145496207636077461641084598330137873344139152491801136373607323241027731456.000000, cosine=-0.0008, ||W||_F=16921423763306769603881449462123368656808930364937021023980841654331477338990752520533929099264.00
- 通道余弦热图：若**非对角**且 ridge MSE 仍高 → 支持 channel semantic mismatch。

### 综合与后续建议

- 若 **l2/abs_mean 比值**远离 1 且逐通道 std 比两极化 → **scale** 问题突出，优先 **LayerNorm / 可学习 1×1 / channel-wise scale**。

- 若直方图形状差异大、但 LayerNorm 后统计接近 → **distribution** 多为尺度/偏移，可先 **normalization** 再 fusion。

- 若 ridge（尤其归一化后）仍高 MSE、热图杂乱 → **channel semantic** 突出，优先 **concat + 投影、FiLM、或与 graspnet 分支并行再融合**。
