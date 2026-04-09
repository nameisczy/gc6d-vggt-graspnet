# 本地 SCP dump 并用 GraspClutter6D API 可视化

在服务器上跑完 `eval_benchmark.py` 得到 dump 后，把 **dump 目录** 和 **GraspClutter6D 数据集** 拉到本地，用 graspclutter6dAPI 的示例或本仓库提供的脚本做可视化。

---

## 一、服务器上要 SCP 的内容

### 1. Dump 目录（必选）

评估脚本输出的预测 npy 所在目录，例如：

- 路径：`eval_out/{encoder_type}/dump_{split}`  
  例如 `eval_out/lift3d/dump_test` 或 `eval_out/lift3d_clip/dump_val`
- 结构：`dump_xxx/000001/realsense-d415/000001.npy`（scene 6 位 / camera / img_num 6 位 .npy）

```bash
# 在本地执行，把服务器上的整个 dump 目录拉下来
scp -r 用户名@服务器:~/gc6d_grasp_pipeline/eval_out/lift3d/dump_test ./dump_test
```

### 2. GraspClutter6D 数据集根目录（可视化必选）

`exam_vis.py` 和下面的 `vis_dump_locally.py` 都需要 **GC6D 数据集根目录**（用来读场景点云、相机参数等）。根目录下应有：

- `scenes/`（各场景点云、相机等）
- `split_info/`（如 grasp_train_scene_ids.json）
- `models_m/`（物体 mesh，若要做 6D 位姿可视化）

若服务器上数据集很大，可以只拉要可视化的少量场景，例如只拉 `scenes/scene_000001`、`scenes/scene_000002` 和 `split_info/`、`models_m/`（或按需裁剪）。

```bash
# 示例：拉整个数据集根目录（体积大）
scp -r 用户名@服务器:/mnt/ssd/ziyaochen/GraspClutter6D ./GraspClutter6D

# 或只拉部分场景（替换为你需要的 scene id）
mkdir -p ./GraspClutter6D
scp -r 用户名@服务器:/mnt/ssd/ziyaochen/GraspClutter6D/split_info ./GraspClutter6D/
scp -r 用户名@服务器:/mnt/ssd/ziyaochen/GraspClutter6D/scenes/scene_000001 ./GraspClutter6D/scenes/
scp -r 用户名@服务器:/mnt/ssd/ziyaochen/GraspClutter6D/models_m ./GraspClutter6D/  # 如需物体模型
```

---

## 二、本地环境：安装 graspclutter6dAPI

```bash
cd /path/to/your/local/workdir
git clone https://github.com/xxx/graspclutter6dAPI.git   # 或你使用的仓库地址
cd graspclutter6dAPI
pip install -e .
```

依赖：Python 3、numpy、open3d、opencv 等（见 graspclutter6dAPI 的 requirements）。

---

## 三、设置环境变量并跑官方示例（看 GT 场景抓取）

```bash
export GC6D_ROOT=/path/to/GraspClutter6D   # 你本地放数据集的路径
cd graspclutter6dAPI
python examples/exam_vis.py
```

- `exam_vis.py` 会：
  - 用 `GraspClutter6D(GC6D_ROOT, camera, split)` 读 **数据集自带的 GT**；
  - `showObjGrasp`：物体级抓取；
  - `show6DPose`：场景 6D 位姿；
  - `showSceneGrasp(sceneId, camera, annId, ...)`：**该帧的 GT 场景抓取**（不是你的 dump）。

所以 **只看官方示例** 时，只需：本地有 `GC6D_ROOT`、装好 graspclutter6dAPI，然后运行上面命令即可；**不需要** dump。

---

## 四、可视化「你的 dump」预测（用本仓库脚本）

要可视化 **eval_benchmark 生成的 dump**（预测抓取），需要：

1. 本地有 **GC6D_ROOT**（同上）；
2. 本地有 **dump 目录**（第一节 SCP 下来的 `dump_test` 等）；
3. 使用本仓库提供的脚本（见下），或自己按同样逻辑写一小段脚本。

把下面脚本保存到本地，例如 `vis_dump_locally.py`（与 graspclutter6dAPI 同目录或在其 `examples/` 下均可），然后运行。

```bash
export GC6D_ROOT=/path/to/GraspClutter6D
export DUMP_FOLDER=/path/to/dump_test   # 你 SCP 下来的 dump 目录

python vis_dump_locally.py --scene_id 0 --camera realsense-d415 --ann_id 0 --num_grasp 30
```

脚本内容见下一节；参数含义：

- `--dump_folder`：dump 根目录（或用环境变量 `DUMP_FOLDER`）；
- `--scene_id`：场景 id（整数，如 0、1）；
- `--camera`：与 dump 一致，如 `realsense-d415`；
- `--ann_id`：该场景下的帧/标注 id（0～12）；
- `--num_grasp`：最多画多少条抓取。

---

## 五、`vis_dump_locally.py` 脚本内容

在本地创建 `vis_dump_locally.py`，内容如下（依赖 graspclutter6dAPI 和 GC6D_ROOT）：

```python
#!/usr/bin/env python3
"""本地可视化 eval_benchmark 的 dump：需设 GC6D_ROOT，并安装 graspclutter6dAPI。"""
import os
import argparse

def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    img_id = ann_id * 4
    if camera == "realsense-d415": img_id += 1
    elif camera == "realsense-d435": img_id += 2
    elif camera == "azure-kinect": img_id += 3
    elif camera == "zivid": img_id += 4
    return img_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_folder", default=os.environ.get("DUMP_FOLDER", "./dump_test"), help="dump 根目录")
    ap.add_argument("--scene_id", type=int, default=0)
    ap.add_argument("--camera", default="realsense-d415")
    ap.add_argument("--ann_id", type=int, default=0)
    ap.add_argument("--num_grasp", type=int, default=50)
    ap.add_argument("--max_width", type=float, default=0.14)
    args = ap.parse_args()

    if "GC6D_ROOT" not in os.environ:
        print("请设置 GC6D_ROOT，例如: export GC6D_ROOT=/path/to/GraspClutter6D")
        return
    gc6d_root = os.environ["GC6D_ROOT"]
    dump_folder = os.path.abspath(args.dump_folder)
    scene_name = "%06d" % args.scene_id
    img_num = ann_id_to_img_id(args.ann_id, args.camera)
    npy_path = os.path.join(dump_folder, scene_name, args.camera, "%06d.npy" % img_num)
    if not os.path.isfile(npy_path):
        print("Dump 文件不存在:", npy_path)
        return

    from graspclutter6dAPI import GraspClutter6D
    from graspclutter6dAPI.grasp import GraspGroup
    import open3d as o3d

    g = GraspClutter6D(gc6d_root, camera=args.camera, split="test")
    scene_pcd = g.loadScenePointCloud(sceneId=args.scene_id, camera=args.camera, annId=args.ann_id, align=False)
    gg = GraspGroup().from_npy(npy_path)
    gg = gg.nms(translation_thresh=0.03, rotation_thresh=15.0 / 180.0 * 3.14159265)
    import numpy as np
    w = gg.grasp_group_array[:, 1]
    gg.grasp_group_array = gg.grasp_group_array[w <= args.max_width]
    gg = gg[: args.num_grasp]
    geoms = [scene_pcd] + gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
```

---

## 六、小结

| 目的 | 需要 SCP | 本地命令 |
|------|----------|----------|
| 只跑官方 exam_vis（看 GT） | 仅 GraspClutter6D 数据集根目录 | `export GC6D_ROOT=...` 后 `python examples/exam_vis.py` |
| 可视化你的 dump | dump 目录 + GraspClutter6D 数据集根目录 | `export GC6D_ROOT=... DUMP_FOLDER=...` 后 `python vis_dump_locally.py --scene_id 0 --camera realsense-d415 --ann_id 0` |

- **Dump 目录**：服务器上 `eval_out/{encoder}/dump_{split}` 整目录拉下来即可。
- **数据集**：本地 `GC6D_ROOT` 指向的目录需包含 `scenes/`、`split_info/`（以及按需的 `models_m/`）。
- **exam_vis.py** 看的是数据集自带的 **GT**；要看 **预测结果** 用上面的 **vis_dump_locally.py**（或等价逻辑）。
