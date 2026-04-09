# Benchmark 调试：关键代码片段

用于直接定位「哪个字段顺序错、坐标系要不要乘 T、R 的轴怎么换、width/clamp 怎么做」。

---

## 可直接运行的命令（不要复制文档里的 `...`）

**单样本 + 打印第一条 grasp 字段**（脚本默认已带 `--debug_first_grasp`）：
```bash
./run_benchmark_one_sample.sh
```

**导出 PLY（点云 + grasp 坐标系 + 物体）**：
```bash
EXPORT_PLY=eval_out/lift3d_clip/first_grasp.ply ./run_benchmark_one_sample.sh
```

**用 GT 作为 pred 跑 benchmark**：
```bash
USE_GT=1 ./run_benchmark_one_sample.sh
```

**不用脚本、完整参数示例**：
```bash
python eval_benchmark.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --checkpoint checkpoints/gc6d_lift3d_overfit_test.pt \
  --split train --max_samples 1 \
  --dataset_root /mnt/ssd/ziyaochen/GraspClutter6D \
  --lift3d_root "$LIFT3D_ROOT" \
  --debug_first_grasp \
  --export_ply eval_out/lift3d_clip/first_grasp.ply
```

**GT 作为 pred**（把上面命令加上 `--use_gt_as_pred` 即可）：
```bash
python eval_benchmark.py \
  --data_dir /mnt/ssd/ziyaochen/GraspClutter6D/offline_unified \
  --checkpoint checkpoints/gc6d_lift3d_overfit_test.pt \
  --split train --max_samples 1 \
  --dataset_root /mnt/ssd/ziyaochen/GraspClutter6D \
  --lift3d_root "$LIFT3D_ROOT" \
  --use_gt_as_pred
```

---

## 可直接运行的命令（勿复制文档里的 ...）

- 单样本+打印第一条 grasp: `./run_benchmark_one_sample.sh`
- 导出 PLY: `EXPORT_PLY=eval_out/lift3d_clip/first_grasp.ply ./run_benchmark_one_sample.sh`
- 用 GT 作为 pred: `USE_GT=1 ./run_benchmark_one_sample.sh`

完整命令见脚本内 echo 或本文件末尾。

## 1) 我们把 pred (K,10) / 单条 (10,) 转成 GraspGroup 的代码

**文件**: `utils/action2grasp.py`

```python
# 10D: [t(3), R_col1(3), R_col2(3), width(1)]
t0 = a[0:3].astype(np.float32)
c1 = a[3:6].astype(np.float32)   # R 第 1 列
c2 = a[6:9].astype(np.float32)   # R 第 2 列
w0 = float(a[9])

if clip_t:
    t0 = np.clip(t0, [-0.5, -0.5, 0.0], [0.5, 0.5, 1.5])

# Gram–Schmidt 正交化，第三列 = c1 × c2
c1 = c1 / (np.linalg.norm(c1) + eps)
c2 = c2 - c1 * np.dot(c1, c2)
c2 = c2 / (np.linalg.norm(c2) + eps)
c3 = np.cross(c1, c2)
c3 = c3 / (np.linalg.norm(c3) + eps)
R = np.stack([c1, c2, c3], axis=1)   # 3x3, columns = 三轴

# GraspGroup 每行 17 维: score(1), w(1), height(1), depth(1), R_9(row-major), t(3), object_id(1)
arr[:, 0] = score
arr[:, 1] = w
arr[:, 2] = height
arr[:, 3] = depth
arr[:, 4:13] = R.reshape(1, 9).repeat(num_grasps, axis=0)   # R 按行展平
arr[:, 13:16] = t
arr[:, 16] = object_id
return GraspGroup(arr)
```

eval 时我们只传**一条** 10D action，`action10_to_graspgroup` 会在 t/w 上加噪声展开成 `top_k` 条 grasp，再交给 benchmark。

---

## 2) Benchmark 读取我们 pred 的调用

**文件**: `eval_benchmark.py`

```python
# 用 pred 或 GT 得到一条 10D
action_np = (actions_gt[i].cpu().numpy() if args.use_gt_as_pred else actions_pred[i].cpu().numpy())
pc_np = pcs[i].cpu().numpy()

# 转成 GraspGroup（内部会展开成 top_k 条、并填 17 维）
grasp_group = action10_to_graspgroup(
    action_np,
    pc_np,
    num_grasps=args.top_k,
    score_mode="centroid",
)

# 调用官方 eval
res = eval_grasp(
    grasp_group=grasp_group,
    models=models_np,
    dexnet_models=dexmodel_list,
    poses=pose_list,
    config=config,
    table=None,
    voxel_size=0.008,
    TOP_K=args.top_k,
)
grasp_list, score_list, collision_mask_list = res
```

`eval_grasp` 内部会按 `grasp_group` 的 17 维格式（含 R 的存储顺序、t 的坐标系）做碰撞检测；若我们写的 R/t/顺序或坐标系与 benchmark 预期不一致，就会 0% 成功。

---

检查要点（对照上面两段）：

- **字段顺序**: 我们 10D 是 `t(3), col1(3), col2(3), w(1)`；GraspGroup 里 R 是 row-major 9 维。若 benchmark 期望 column-major 或先 t 后 R，需改。
- **坐标系**: `t` 是否要乘场景的 `T_world_cam` 或 `T_cam_world`。
- **R 的轴**: 我们是 `[c1, c2, c3]` 列；若 benchmark 期望 z 轴为 approach，可能需 swap 列或取反某一列。
- **width**: 我们已 clip 到 `[width_min, width_max]`；若 benchmark 期望米或毫米，需缩放。
