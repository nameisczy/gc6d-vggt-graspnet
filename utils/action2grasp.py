# -*- coding: utf-8 -*-
"""
10D action -> GraspGroup，供 GC6D benchmark 使用。
与 LIFT3D 中 action_to_graspgroup_sampled 逻辑一致，此处独立实现不依赖 LIFT3D。
"""

import numpy as np


def grasp_group_row_to_action10(row: np.ndarray) -> np.ndarray:
    """从 gt_grasp_group 的一行 (17,) 提取 10D action。与 data.dataset 中逻辑一致，用于 round-trip 测试。"""
    row = np.asarray(row, dtype=np.float32).ravel()
    if row.size < 17:
        return np.zeros(10, dtype=np.float32)
    t = row[13:16]
    R = row[4:13].reshape(3, 3)
    c1, c2 = R[:, 0], R[:, 1]
    w = row[1:2]
    return np.concatenate([t, c1, c2, w], axis=0).astype(np.float32)


def action10_to_t_R_w(action10: np.ndarray, clip_t: bool = True):
    """
    从 10D action 解析 t(3), R(3x3), width(1)。与 action10_to_graspgroup 内逻辑一致，用于调试/导出。
    action10: (10,) [t(3), R_col1(3), R_col2(3), width(1)]
    returns: t (3,), R (3,3), w (float)
    """
    a = np.asarray(action10, dtype=np.float32).reshape(-1)
    if a.shape[0] != 10:
        raise ValueError(f"action must be 10D, got {a.shape}")
    t0 = a[0:3].astype(np.float32)
    c1 = a[3:6].astype(np.float32)
    c2 = a[6:9].astype(np.float32)
    w0 = float(a[9])
    if clip_t:
        t0 = np.clip(
            t0,
            np.array([-0.5, -0.5, 0.0], dtype=np.float32),
            np.array([0.5, 0.5, 1.5], dtype=np.float32),
        )
    eps = 1e-8
    c1 = c1 / (np.linalg.norm(c1) + eps)
    c2 = c2 - c1 * float(np.dot(c1, c2))
    c2 = c2 / (np.linalg.norm(c2) + eps)
    c3 = np.cross(c1, c2)
    c3 = c3 / (np.linalg.norm(c3) + eps)
    R = np.stack([c1, c2, c3], axis=1).astype(np.float32)
    return t0, R, w0


def action10_to_graspgroup(
    action10: np.ndarray,
    pc_xyz: np.ndarray,
    num_grasps: int = 50,
    t_sigma: float = 0.01,
    width_sigma: float = 0.01,
    width_min: float = 0.01,
    width_max: float = 0.12,
    height: float = 0.02,
    depth: float = 0.04,
    object_id: int = -1,
    seed: int = 0,
    clip_t: bool = True,
    score_mode: str = "centroid",
    w_penalty: float = 0.2,
    R_flatten: str = "row",
    return_arr: bool = False,
    R_permute: str = "012",
):
    """
    action10: (10,) [t(3), R_col1(3), R_col2(3), width(1)]
    pc_xyz: (N, 3) 点云，用于 centroid 打分
    R_flatten: "row" = R.reshape(9) 行优先, "col" = R.T.reshape(9) 列优先
    return_arr: 若 True 返回 (GraspGroup, arr) 便于打印 w/h/d/object_id
    R_permute: 列排列，3 个字符 "0","1","2" 的排列。R 原列 [c1,c2,c3]=0,1,2。
      "012"=[c1,c2,c3], "021"=[c1,c3,c2], "102"=[c2,c1,c3], "120"=[c2,c3,c1], "201"=[c3,c1,c2], "210"=[c3,c2,c1]
    返回: GraspGroup(arr) 或 (GraspGroup(arr), arr)
    """
    try:
        from graspclutter6dAPI.grasp import GraspGroup
    except ImportError:
        from graspclutter6dAPI.grasp import GraspGroup

    rng = np.random.RandomState(int(seed))
    a = np.asarray(action10, dtype=np.float32).reshape(-1)
    if a.shape[0] != 10:
        raise ValueError(f"action must be 10D, got {a.shape}")

    t0 = a[0:3].astype(np.float32)
    c1 = a[3:6].astype(np.float32)
    c2 = a[6:9].astype(np.float32)
    w0 = float(a[9])

    if clip_t:
        t0 = np.clip(
            t0,
            np.array([-0.5, -0.5, 0.0], dtype=np.float32),
            np.array([0.5, 0.5, 1.5], dtype=np.float32),
        )

    eps = 1e-8
    c1 = c1 / (np.linalg.norm(c1) + eps)
    c2 = c2 - c1 * float(np.dot(c1, c2))
    c2 = c2 / (np.linalg.norm(c2) + eps)
    c3 = np.cross(c1, c2)
    c3 = c3 / (np.linalg.norm(c3) + eps)
    R = np.stack([c1, c2, c3], axis=1).astype(np.float32)
    if R_permute != "012":
        perm = [int(R_permute[0]), int(R_permute[1]), int(R_permute[2])]
        R = R[:, perm].astype(np.float32)

    dt = rng.randn(int(num_grasps), 3).astype(np.float32) * float(t_sigma)
    t = t0.reshape(1, 3) + dt
    dw = rng.randn(int(num_grasps)).astype(np.float32) * float(width_sigma)
    w = np.clip(w0 + dw, float(width_min), float(width_max)).astype(np.float32)

    if score_mode == "centroid":
        pc = np.asarray(pc_xyz, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"pc_xyz must be (N,3), got {pc.shape}")
        centroid = pc.mean(axis=0).astype(np.float32)
        dist = np.linalg.norm(t - centroid.reshape(1, 3), axis=1).astype(np.float32)
        wdev = np.abs(w - float(w0)).astype(np.float32)
        score = -(dist + float(w_penalty) * wdev).astype(np.float32)
    elif score_mode == "t0_only":
        score = -np.linalg.norm(t, axis=1).astype(np.float32)
    else:
        score = -np.linalg.norm(dt, axis=1).astype(np.float32)

    arr = np.zeros((int(num_grasps), 17), dtype=np.float32)
    arr[:, 0] = score
    arr[:, 1] = w
    arr[:, 2] = float(height)
    arr[:, 3] = float(depth)
    if R_flatten == "col":
        arr[:, 4:13] = np.ascontiguousarray(R.T).reshape(1, 9).repeat(int(num_grasps), axis=0)
    else:
        arr[:, 4:13] = R.reshape(1, 9).repeat(int(num_grasps), axis=0)
    arr[:, 13:16] = t
    arr[:, 16] = float(object_id)

    if return_arr:
        return GraspGroup(arr), arr
    return GraspGroup(arr)


def action10_batch_to_graspgroup(
    actions_K: np.ndarray,
    width_min: float = 0.01,
    width_max: float = 0.12,
    height: float = 0.02,
    depth: float = 0.04,
    object_id: int = -1,
    clip_t: bool = True,
    R_flatten: str = "row",
    R_permute: str = "012",
):
    """
    将 K 个 10D action 转为 GraspGroup（K 行），无噪声；供评估时提交 K 个 proposal 作为多候选。
    actions_K: (K, 10) [t(3), R_col1(3), R_col2(3), width(1)] 每行一个 proposal
    返回: GraspGroup(arr)，arr 为 (K, 17)。score 设为 0，由 benchmark 自行排序。
    """
    try:
        from graspclutter6dAPI.grasp import GraspGroup
    except ImportError:
        from graspclutter6dAPI.grasp import GraspGroup

    actions_K = np.asarray(actions_K, dtype=np.float32)
    if actions_K.ndim != 2 or actions_K.shape[1] != 10:
        raise ValueError(f"actions_K must be (K, 10), got {actions_K.shape}")
    K = actions_K.shape[0]
    arr = np.zeros((K, 17), dtype=np.float32)
    for k in range(K):
        t0, R, w0 = action10_to_t_R_w(actions_K[k], clip_t=clip_t)
        if R_permute != "012":
            perm = [int(R_permute[0]), int(R_permute[1]), int(R_permute[2])]
            R = R[:, perm].astype(np.float32)
        w0 = np.clip(float(w0), float(width_min), float(width_max))
        arr[k, 0] = 0.0
        arr[k, 1] = w0
        arr[k, 2] = float(height)
        arr[k, 3] = float(depth)
        if R_flatten == "col":
            arr[k, 4:13] = np.ascontiguousarray(R.T).reshape(9)
        else:
            arr[k, 4:13] = R.reshape(9)
        arr[k, 13:16] = t0
        arr[k, 16] = float(object_id)
    return GraspGroup(arr)


def r6_to_R(r6: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """R6d（前两列 6 个数）→ 正交阵 R (3,3)。与 head 内建 R 方式一致，用于 11D→17D。"""
    r6 = np.asarray(r6, dtype=np.float32).ravel()
    if r6.size < 6:
        return np.eye(3, dtype=np.float32)
    c1 = r6[0:3].astype(np.float32)
    c2 = r6[3:6].astype(np.float32)
    c1 = c1 / (np.linalg.norm(c1) + eps)
    c2 = c2 - c1 * float(np.dot(c1, c2))
    c2 = c2 / (np.linalg.norm(c2) + eps)
    c3 = np.cross(c1, c2)
    c3 = c3 / (np.linalg.norm(c3) + eps)
    return np.stack([c1, c2, c3], axis=1).astype(np.float32)


def proposals_11d_to_graspgroup(
    proposals_K: np.ndarray,
    width_min: float = 0.01,
    width_max: float = 0.12,
    height: float = 0.02,
    depth: float = 0.04,
    object_id: int = -1,
    clip_t: bool = True,
    R_row_major: bool = True,
):
    """
    GraspNet head 原始 11D 直接转 GC6D 17D，不经 10D，与 benchmark 格式一致。
    proposals_K: (K, 11) 每行 [t(3), R6d(6), width(1), score(1)]
    返回: GraspGroup(arr)，arr (K, 17)。R 按 GC6D 行优先存，无 R_permute 歧义。
    """
    try:
        from graspclutter6dAPI.grasp import GraspGroup
    except ImportError:
        from graspclutter6dAPI.grasp import GraspGroup

    proposals_K = np.asarray(proposals_K, dtype=np.float32)
    if proposals_K.ndim != 2 or proposals_K.shape[1] != 11:
        raise ValueError(f"proposals_K must be (K, 11), got {proposals_K.shape}")
    K = proposals_K.shape[0]
    arr = np.zeros((K, 17), dtype=np.float32)
    for k in range(K):
        t0 = proposals_K[k, 0:3].astype(np.float32)
        if clip_t:
            t0 = np.clip(
                t0,
                np.array([-0.5, -0.5, 0.0], dtype=np.float32),
                np.array([0.5, 0.5, 1.5], dtype=np.float32),
            )
        r6 = proposals_K[k, 3:9]
        R = r6_to_R(r6)
        w0 = np.clip(float(proposals_K[k, 9]), float(width_min), float(width_max))
        score = float(proposals_K[k, 10])
        arr[k, 0] = score
        arr[k, 1] = w0
        arr[k, 2] = float(height)
        arr[k, 3] = float(depth)
        if R_row_major:
            arr[k, 4:13] = R.reshape(9)
        else:
            arr[k, 4:13] = np.ascontiguousarray(R.T).reshape(9)
        arr[k, 13:16] = t0
        arr[k, 16] = float(object_id)
    return GraspGroup(arr)
