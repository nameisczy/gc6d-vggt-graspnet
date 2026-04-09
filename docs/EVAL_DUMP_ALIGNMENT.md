# GC6D 评估：dump 流程与 `gc6d_graspnet_repro` 对齐说明

目标：在**同一** `GraspClutter6DEval`（`eval_scene` / `eval_all`）下，使 **AP = f(dump)** 中的 dump 可比。不修改 evaluator 内部 AP 定义。

## 1. 两侧入口

| 项目 | 入口 | dump 路径布局 |
|------|------|----------------|
| **gc6d_grasp_pipeline** | `eval_benchmark.py` | `dump_<split>/<scene_6d>/<camera>/<img_id>.npy` |
| **gc6d_graspnet_repro** | `src/data/gc6d_to_baseline_adapter.py` + `src/models/baseline_infer.py` | 同上（由 subprocess 调 baseline 写 npy） |

API 调用：`GraspClutter6DEval.eval_scene(..., TOP_K=..., background_filter=...)`；test 集可用 `eval_all`。

## 2. dump 内容（17D）

与 `graspclutter6dAPI.grasp.GraspGroup` 一致：`[score, width, height, depth, R(9), t(3), obj_id]`。

## 3. 已对齐的关键差异（pipeline 侧修改摘要）

1. **`pred_decode_17d` 截断前必须按 score 降序**  
   Baseline `pred_decode` 输出顺序为 seed 顺序，不是全局高分优先。若直接取前 `K` 行再 pad，会丢掉后排高分 grasp。现与可微分支一致：**先按第 0 列降序，再取前 `max_dump_grasps`**。

2. **可选与 repro 一致的 dump 前碰撞**  
   repro 的 `baseline_infer` 在写 npy 前使用 `ModelFreeCollisionDetector`（`collision_thresh`、`voxel_size` 与 yaml 一致）。  
   `eval_benchmark.py` 默认 **`--pre_dump_collision_filter`**（可用 `--no_pre_dump_collision_filter` 关闭）。

3. **去掉 tensor pad 产生的全零行**  
   保存前丢弃 `score <= 1e-8` 的行，避免把 padding 当 grasp 写入。

4. **显式传递 `background_filter`**  
   与 repro 的 `eval_gc6d` 一致，默认 **True**（前景： grasp 中心到物体采样点最近距离 `< 0.05m`，在 **API 内** 执行）。可用 `--no_background_filter` 关闭。

5. **参数**  
   - `--max_dump_grasps`（默认 4096）：排序后最多写入条数，与 repro「碰撞后整表保存」同一量级时可设大。  
   - `--top_k`：仅影响 **evaluator** 的 `TOP_K`，不是 dump 行数上限。

6. **相机 `ann_id → img_id`**  
   必须与数据与 repro 配置一致（如 repro 常用 `realsense-d435`，pipeline 若仍用 `realsense-d415` 则**同一 ann_id 对应不同文件**，对比时务必统一 `--camera`）。

## 4. 调试与可视化

- **各阶段数量**（raw → 去 pad → 预碰撞 → 写入行数 → 与 `eval_scene` 一致的 foreground 粗算）  
  ```bash
  python scripts/debug_dump_pipeline_trace.py \
    --checkpoint ... --data_dir ... --dataset_root $GC6D_ROOT \
    --scene_id 42 --ann_id 0 --camera realsense-d435
  ```
  仅对比两份 npy 统计量：  
  `--repro_npy ... --pipeline_npy ...`

- **同帧 top-K 可视化（repro vs pipeline）**  
  ```bash
  python scripts/visualize_compare_two_dumps.py \
    --scene_id 42 --ann_id 0 --camera realsense-d435 \
    --repro_npy ... --pipeline_npy ... --top 20
  ```

- **实现细节**  
  `utils/dump_pipeline_trace.py`：`trace_encoder_graspnet_dump`、`count_foreground_like_eval_scene`（复现 API 内 `background_filter` 距离阈值 **5cm**）。

## 5. `summary_*.json`

`eval_benchmark` 会写入 `dump_alignment` 字段（`max_dump_grasps`、`top_k_eval`、预碰撞与 `api_background_filter` 等），便于与 repro 的 yaml 对照。

## 6. 依赖提示

dump 前碰撞需要 **graspnet-baseline** 的 `collision_detector`（`GRASPNET_BASELINE` / `--graspnet_root`）。

封装抓取姿态时：**优先** `graspnetAPI.GraspGroup`（与官方 baseline 一致）；若环境未安装 `graspnetAPI`，代码会**自动回退**到已安装的 `graspclutter6dAPI.grasp.GraspGroup`（同为 17D，`ModelFreeCollisionDetector` 所需属性一致）。

可选：在 graspnet 环境中 `pip install graspnetAPI`（或按 baseline 文档安装），与 repro 子进程环境完全一致。
