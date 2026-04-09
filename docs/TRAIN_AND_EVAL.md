# 训练与评估指令

训练和评估为**两条独立指令**：训练只用你划分的 train/val（17D loss 验证），最后测试用 `eval_benchmark.py` 算 AP。

---

## 一、训练指令

数据需有 `index_train_{camera}.jsonl`、`index_val_{camera}.jsonl`。训练过程在**验证集**上算 17D matching loss，不打 AP。

### 1. LIFT3D + Adapter + GraspNet（Stage1→2→3→4）

```bash
cd /path/to/gc6d_grasp_pipeline

# 环境（与 run 脚本一致）
export GRASPNET_BASELINE="${GRASPNET_BASELINE:-$HOME/graspnet-baseline}"
export LIFT3D_ROOT="${LIFT3D_ROOT:-$HOME/LIFT3D}"
export PYTHONPATH="$GRASPNET_BASELINE:$GRASPNET_BASELINE/pointnet2:$GRASPNET_BASELINE/utils:$GRASPNET_BASELINE/knn:$PYTHONPATH"
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null):$LD_LIBRARY_PATH"

DATA="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified"   # 含 index_train_*.jsonl, index_val_*.jsonl
MODE=1sample   # 或 small | full

# 单数据点过拟合
MODE=1sample ./run_train_adapter_lift3d.sh

# 小批量（例如 100 条）
MODE=small ./run_train_adapter_lift3d.sh

# 全量
MODE=full ./run_train_adapter_lift3d.sh
```

**单 stage 手动示例（LIFT3D）：**

```bash
# Stage 1：只训 adapter
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder lift3d --stage 1 \
  --graspnet_ckpt "$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar" \
  --graspnet_root "$GRASPNET_BASELINE" --lift3d_root "$LIFT3D_ROOT" \
  --max_samples 1 --batch_size 4 --lr 1e-3 \
  --save_name "gc6d_lift3d_adapter_graspnet_s1_1sample"

# Stage 2：接 Stage1 的 ckpt，训 adapter + grasp_net
python train_adapter_graspnet.py \
  --data_dir "$DATA" --encoder lift3d --stage 2 \
  --graspnet_ckpt "$GRASPNET_BASELINE/logs/log_rs/checkpoint-rs.tar" \
  --graspnet_root "$GRASPNET_BASELINE" --lift3d_root "$LIFT3D_ROOT" \
  --max_samples 1 --batch_size 4 --lr 1e-3 \
  --load_ckpt checkpoints/gc6d_lift3d_adapter_graspnet_s1_1sample.pt \
  --save_name "gc6d_lift3d_adapter_graspnet_s2_1sample"

# Stage 3 / 4 同理，--load_ckpt 用上一 stage 的 .pt，--save_name 对应 s3/s4
```

**VGGT Base（仅 Stage1 + Stage2，Stage2 步数 4k+2k）：**

```bash
MODE=1sample ./run_train_adapter_vggt_base.sh
# 或 MODE=small / full
```

**VGGT Ft（Stage1→2→3→4）：**

```bash
MODE=1sample ./run_train_adapter_vggt_ft.sh
# 需 --vggt_ckpt 指向微调后的 VGGT
```

训练期验证日志示例：`[Stage2] step 200 train_same=0.xxx val=0.xxx`，结束时：`[Stage2] final train_same=... val=...`。

**每 stage 默认步数**（可按 `--steps` 覆盖）：
- **全量**（`max_samples=0`）：Stage1=1000, Stage2=4000, Stage3/4=2000（几千 step）
- **小批量**（如 `max_samples=100`）：每 stage 约 200~400 step（几百）
- **单样本**（`max_samples=1`）：每 stage 约 50~80 step（几十），避免过拟合跑太久

**单样本过拟合**：理论上 1 条数据上 train_same 应能压到接近 0（<0.01）。若卡在 ~0.02 以上：
- **Stage 1** 只训 adapter，有非零下界属正常；看 Stage 2 及以后。
- **Stage 2+** 若 train_same 仍卡在 ~0.02：可适当**加步数**（如 `--steps 150`）、或略调大 **lr**（如 `--lr 3e-3`），或加 `--log_grad_norm` 看梯度是否还充足。

---

## 二、评估指令（最后测试，算 AP）

训练完成后，用 **eval_benchmark.py** 在**测试集**上 dump 预测并调用 GraspClutter6D API 得到 AP/AP0.4/AP0.8。需提供完整 GC6D 数据集根目录（含 `scenes/`、`models_m/`、`split_info/`）。

```bash
cd /path/to/gc6d_grasp_pipeline

DATA="/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified"       # index 与 npz 所在目录
DATASET_ROOT="/mnt/ssd/ziyaochen/GraspClutter6D"              # 官方数据集根目录，供 API 读 scenes/models_m/split_info
CKPT="checkpoints/gc6d_lift3d_adapter_graspnet_s2_1sample.pt"

python eval_benchmark.py \
  --data_dir "$DATA" \
  --checkpoint "$CKPT" \
  --split test \
  --dataset_root "$DATASET_ROOT" \
  --camera realsense-d415 \
  --out_dir eval_out
```

- **LIFT3D** 需加：`--lift3d_root $LIFT3D_ROOT`
- 输出：终端打印 `AP=xx.xx AP0.4=xx.xx AP0.8=xx.xx`，并写入 `eval_out/<encoder_type>/summary_test.json`

**VGGT 评估示例：**

```bash
python eval_benchmark.py \
  --data_dir "$DATA" \
  --checkpoint checkpoints/gc6d_vggt_base_adapter_graspnet_s2_1sample.pt \
  --split test \
  --dataset_root "$DATASET_ROOT" \
  --camera realsense-d415
```

**仅在 val 上跑 AP（不推荐，正式结果用 test）：**

```bash
python eval_benchmark.py --data_dir "$DATA" --checkpoint "$CKPT" --split val --dataset_root "$DATASET_ROOT"
```

---

## 三、与 GraspNet-1B 方法在 GC6D 上的 AP 对照

Contact-GraspNet、GraspNet-Baseline、ScaleBalancedGrasp、EconomicGrasp 等在 **GraspNet-1B 上训练、在 GraspClutter6D 上测试** 时，AP 约为 **十几**。若本仓库在 GC6D 上训练得到的 AP 明显低于该水平，可逐项核对以下内容，保证**评估协议一致**后再比较：

1. **TOP_K**  
   - 本仓库默认 `--top_k 50`，与 GraspClutter6D API 的 TOP_K 一致。  
   - 若论文/官方 benchmark 使用不同 TOP_K（如 10 或 1），请用相同值：  
     `python eval_benchmark.py ... --top_k 10`

2. **测试集与点云来源**  
   - 本仓库使用 `offline_unified` 的 `index_test_<camera>.jsonl` 及对应 npz 中的 `point_cloud`（GC6D 深度转点云）。  
   - 确认对比方法在 GC6D 上报告 AP 时，是否使用**同一 test split、同一相机、同一点云生成方式**（坐标系、下采样、范围等）。若对方使用官方提供的另一种点云或图像生成点云，结果会不可直接比。

3. **纯 GraspNet-Baseline 在本仓库数据上的 AP**  
   - 用 **仅在 GraspNet-1B 上预训练、不加载本仓库 adapter、不微调** 的 GraspNet，在**同一** `offline_unified` test 数据上跑 `eval_benchmark.py`（需单独写一个只加载 grasp_net 的入口或脚本），得到 AP。  
   - 若该 AP 也是 3～4，说明当前**点云/数据格式**与论文中“十几”的设置可能不一致，需对齐数据或评估脚本。  
   - 若该 AP 为十几，而本仓库 adapter 微调后仍为 3～4，则问题更可能在**训练设置**（数据量、步数、学习率等）或**adapter 引入后的分布偏移**。

4. **全量训练**  
   - 当前 small100（约 100 条）仅作快速试验，AP 约 3～4 属预期。  
   - 要逼近或超过“十几”的参考值，需用 **全量 GC6D 训练**（`MODE=full`，`max_samples=0`，每 stage 数千步），再在 test 上评估。

---

## 四、GC6D 的 GSR/DR 与训练可能问题

### 4.1 GSR / DR 是如何得到的

- **GraspClutter6D** 官方 benchmark 通常有两类指标：
  - **GSR (Grasping Success Rate)**：单次抓取尝试的成功率（成功次数 / 尝试次数），需在**仿真或真机**中执行预测的抓取并判定是否成功（如力闭合、物体被拿起等）。
  - **DR (Declutter Rate)**：场景级“清障率”，即成功抓走并移除的物体数 / 场景内总物体数，衡量多轮抓取后场景被清空的比例。

- 本仓库当前评估使用的是 **GraspClutter6D API 的 AP**（`eval_benchmark.py` → `GraspClutter6DEval.eval_all` / `eval_scene`），属于**离线、基于预测与 GT 匹配**的指标（如 Precision@k、力闭合判定等），**不**直接跑仿真或机器人，因此：
  - **AP**：由官方 API 根据我们 dump 的 17D 预测与 GT 的匹配规则（含 TOP_K、摩擦系数等）计算得到；具体 GSR/DR 若在论文/官方代码中有定义，需查看 **GraspClutter6D 数据集论文或 graspclutter6dAPI 的文档/源码**（如 `eval_all` 内部是否聚合为 GSR/DR，或是否另有脚本跑仿真得到 GSR/DR）。
  - 若你需要的是**论文里报告的 GSR/DR**，需确认其是用同一套 API 的 AP 换算而来，还是用**仿真/真机执行**得到的；若是后者，本仓库目前只提供 AP，GSR/DR 需在官方提供的仿真或评估流程中复现。

### 4.2 除评估外，训练可能存在的问题

在评估流程已用 `--eval_gt_dump` 验证为正常（GT 的 AP 很高）的前提下，若 AP 仍偏低，可从训练侧排查：

| 可能问题 | 说明与建议 |
|----------|------------|
| **数据量过少** | small100 / 1sample 仅作调试用；要接近“十几”需**全量** GC6D（`max_samples=0`）且每 stage 足够步数（数千步）。 |
| **17D loss 与 AP 目标不一致** | 训练优化的是 **w/R/t 上的 MSE matching**（`_cost_17d_per_pair`），而 AP 由**力闭合、Precision@k** 等规则决定；两者并非同一度量。可尝试：适当加大 `best_gt_weight` 让主 GT 更主导；或确认官方是否有“用 AP 或 GSR 相关目标做训练”的设定。 |
| **pred2gt_agg="min"** | 当前默认对“预测→GT”分支取 **min**（一个预测匹配即可），有利于单样本过拟合，但可能让模型只学会“蒙对一个”；若全量训练后 AP 仍不高，可尝试 **pred2gt_agg="mean"** 或配合 `pred2gt_top_frac`，使多预测整体质量被约束。 |
| **学习率 / 步数** | 全量时若 loss 已平稳但 AP 不高，可尝试略调 **lr**（如 3e-4～1e-3）或增加 Stage2/3/4 的 **steps**；小批量时避免 lr 过大导致不稳定。 |
| **点云与 GT 的坐标系/归一化** | 确认 `offline_unified` 的 `point_cloud` 与 GT 17D 的 **t/R** 是否在同一坐标系（与 GC6D 官方一致）；若数据预处理与官方 benchmark 不一致，会同时影响训练和评估。 |
| **Stage 顺序与冻结** | Stage1 只训 adapter、Stage2 训 adapter+grasp_net、Stage3/4 再解冻 encoder LoRA；若某 stage 未收敛就进入下一 stage，或冻结/解冻与论文不一致，可能限制最终性能。 |
| **GraspNet 预训练域与 GC6D 域差** | GraspNet 在 GraspNet-1B 上预训练，与 GC6D 的 clutter/相机/物体分布有差异；adapter 若容量或数据不足，可能难以完全弥补，导致 AP 天花板低于“纯 1B 预训练在 GC6D 上零样本”的十几。 |

建议优先做：**全量训练 + 与纯 GraspNet-Baseline（无 adapter）在同一 test 数据上的 AP 对比**；若 baseline 在我们数据上也是 3～4，则主要矛盾在数据/评估协议；若 baseline 为十几而我们有 adapter 仍低，则重点查上述训练项。

---

## 五、对照小结

| 阶段       | 指令/脚本                  | 数据 split      | 指标        |
|------------|----------------------------|-----------------|-------------|
| 训练       | train_adapter_graspnet.py  | train（训练）   | 17D loss    |
| 训练中验证 | 同上（每 val_every 步）    | val（验证集）   | 17D loss    |
| 最后测试   | eval_benchmark.py --split test | test（测试集） | AP/AP0.4/AP0.8 |
