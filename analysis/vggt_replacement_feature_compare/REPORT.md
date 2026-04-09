# VGGT replacement vs GraspNet reference（vpmodule 输入）

## 摘要（自动根据数值指标给出）

### 哪一种在 **scale（全局 std 比值接近 1）** 上最接近 backbone？

**ln_affine_adapter (|std_ratio-1| 最小, ratio=0.6286)**

### 哪一种在 **整体分布形状（与 reference 的一维 Wasserstein 距离更小）** 上最接近？

**ln_affine_adapter (W1=0.227723)**

### 哪一种在 **PCA 空间（样本到 reference 质心的平均距离更小）** 上更接近？

**ln_adapter (dist=3.913939)**

### 通道余弦热图（对角线均值更高通常表示通道对齐更好）上哪一种更好？

**ln_affine (diag_mean=0.0214)**

### LayerNorm / affine / adapter 的对比（脚本内启发式）

- **layernorm_affine vs layernorm_adapter（看 affine 是否关键）**：
  在 |std_ratio-1|、W1、PCA 距离（越小越好）与通道余弦对角线（越大越好）四项上，layernorm_affine 胜 1 项，layernorm_adapter 胜 3 项。若 affine 明显占优，更支持「LayerNorm+affine 对对齐关键」；若接近则需结合图看。
- **layernorm_affine vs layernorm_affine_adapter（看 adapter 是否有帮助）**：
  同上四项：layernorm_affine 胜 1 项，layernorm_affine_adapter 胜 3 项。若 adapter 版本在多项上更好，支持「adapter 有额外帮助」。
- **与 GraspNet backbone 是否仍有明显 gap**：
  当前三种 replacement 相对 reference：max|std_ratio-1|=1.2935，max W1=0.676263，max PCA 均距=14.545235。若这些量仍显著大于 0，则通常认为 **仍存在明显 gap**（具体阈值请结合业务敏感度）。

## 指标表（摘录）

```json
{
  "std_ratio_vs_ref": {
    "vggt_replacement_layernorm_affine": 2.293487325892956,
    "vggt_replacement_ln_affine_adapter": 0.6286137921679413,
    "vggt_replacement_ln_adapter": 0.6286137921679413
  },
  "wasserstein_1d_vs_ref": {
    "vggt_replacement_layernorm_affine": 0.6762631542286924,
    "vggt_replacement_ln_affine_adapter": 0.2277233889406659,
    "vggt_replacement_ln_adapter": 0.2277233889406659
  },
  "pca_mean_distance_to_ref_centroid": {
    "vggt_replacement_layernorm_affine": 14.545234680175781,
    "vggt_replacement_ln_affine_adapter": 3.914655923843384,
    "vggt_replacement_ln_adapter": 3.913938522338867
  },
  "channel_cosine_diag_mean": {
    "vggt_replacement_layernorm_affine": 0.021423079073429108,
    "vggt_replacement_ln_affine_adapter": 0.012564420700073242,
    "vggt_replacement_ln_adapter": 0.012564420700073242
  }
}
```
