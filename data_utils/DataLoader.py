import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from hilbertcurve.hilbertcurve import HilbertCurve
import sys
import glob
import re
import time
import h5py
import random
# sys.path.append("./superpoint_graph/partition/cut-pursuit/build/src")
# sys.path.append("./superpoint_graph/partition/ply_c")
# sys.path.append("./superpoint_graph/partition")
from data_utils.superpoint_multi_generation import superpoint_generation_multi
def normalize_pointcloud(pc):
    centroid = pc[:,:3].mean(axis=0)
    pc[:,:3] = pc[:,:3] - centroid
    scale = np.max(np.linalg.norm(pc[:,:3], axis=1))
    pc[:,:3] = pc[:,:3] / scale
    return pc

import numpy as np
from numpy.lib.stride_tricks import as_strided

import numpy as np

import numpy as np



def split_superpoints_with_overlap_fast_idx_adaptive(
    raw_array,
    init_seq_length=128,
    overlap=0.5,
    max_superpoints=2000,
    min_seq_length=32,
    sp_col_idx=-2
):
    """
    自适应滑窗切片：确保每个分段内的 superpoint 数量不超过 max_superpoints

    Args:
        raw_array (np.ndarray): 输入数据，形状为 (N, D)，其中包含 sp_label 列
        init_seq_length (int): 初始分段长度
        overlap (float): 相邻分段的重叠率 [0, 1)
        max_superpoints (int): 每段中允许的最大 superpoint 数量
        min_seq_length (int): 每段允许的最小长度
        sp_col_idx (int): sp_label 所在列的索引（默认倒数第二列）

    Returns:
        idx_list (List[Tuple[int, int]]): 每个片段的 (start, end) 索引对
    """
    N = raw_array.shape[0]
    stride = int(init_seq_length * (1 - overlap))
    stride = max(stride, 1)

    idx_list = []

    # 主体滑窗阶段
    i = 0
    while i + init_seq_length <= N:
        seq_length = init_seq_length
        segment = raw_array[i:i + seq_length]
        num_sp = len(np.unique(segment[:, sp_col_idx]))

        while num_sp > max_superpoints and seq_length > min_seq_length:
            seq_length = int(seq_length * 0.9)
            seq_length = max(seq_length, min_seq_length)
            segment = raw_array[i:i + seq_length]
            num_sp = len(np.unique(segment[:, sp_col_idx]))

        idx_list.append((i, i + seq_length))
        i += stride

    # 补尾部阶段（从末尾继续按 stride 补齐）
    start = idx_list[-1][1] if idx_list else 0
    while start < N:
        end = N
        seq_length = end - start
        segment = raw_array[start:end]
        num_sp = len(np.unique(segment[:, sp_col_idx]))

        while num_sp > max_superpoints and seq_length > min_seq_length:
            seq_length = int(seq_length * 0.9)
            seq_length = max(seq_length, min_seq_length)
            start = end - seq_length
            segment = raw_array[start:end]
            num_sp = len(np.unique(segment[:, sp_col_idx]))
        assert end-start != init_seq_length, f"Invalid segment length: {end-start} != {init_seq_length}"
        idx_list.append((start, end))
        start += stride

    return idx_list



def split_superpoints_with_overlap_fast_idx(S, seq_length=128, overlap=0.5):
    """
    返回 superpoint 切片的起始索引，而不实际返回数据

    Args:
        S (int): 数据总长度（sp_array 的第 0 维大小）
        seq_length (int): 每个片段的长度
        overlap (float): 重叠比例

    Returns:
        idx_list (List[Tuple[int, int]]): 每段的 (start_idx, end_idx)
    """
    assert 0 <= overlap < 1, "overlap must be in [0, 1)"
    stride = int(seq_length * (1 - overlap))
    stride = max(stride, 1)

    idx_list = []

    for start in range(0, S - seq_length + 1, stride):
        end = start + seq_length
        idx_list.append((start, end))

    # 最后一段补全（覆盖所有点）
    if not idx_list or idx_list[-1][1] < S:
        idx_list.append((max(0, S - seq_length), S))

    return idx_list # [(start_idx, end_idx)]

def split_hsp_with_overlap_fast(sp_array, seq_length=128, overlap=0.5, sp_col_idx=None, ca_K=3):
    """
    使用 as_strided 加速 superpoint 分段采样（支持重叠）
    """
    assert 0 <= overlap < 1, "overlap must be in [0, 1)"
    stride = int(seq_length * (1 - overlap))
    stride = max(stride, 1)

    sp_array = np.ascontiguousarray(sp_array)  # 保证内存连续
    S, D = sp_array.shape

    # 计算可以提取多少段
    num_segments = (S - seq_length) // stride + 1
    if num_segments <= 0:
        # 太短，返回最后一段补齐
        pad = np.zeros((seq_length - S, D), dtype=sp_array.dtype)
        return np.expand_dims(np.concatenate([pad, sp_array], axis=0), axis=0)

    # 构建新视图，性能远超 for 循环
    new_shape = (num_segments, seq_length, D)
    new_strides = (sp_array.strides[0] * stride, sp_array.strides[0], sp_array.strides[1])
    segments = as_strided(sp_array, shape=new_shape, strides=new_strides)

    # 检查是否还需要补最后一段
    last_end = stride * (num_segments - 1) + seq_length
    if last_end < S:
        last_segment = sp_array[-seq_length:]
        segments = np.concatenate([segments, last_segment[None]], axis=0)
    
    neighbor_sp_idx_lists = []
    for segment in tqdm(segments):
        # print(f"Processing segment with shape: {segment.shape}")
        segment_torch = torch.tensor(segment, device='cuda', dtype=torch.float32)
        raw_xyz = segment_torch[:, :3]
        neigh_list = []

        for i in range(len(sp_col_idx)):
            sp_label = segment_torch[:, sp_col_idx[i]].to(torch.int64)
            neighbor_sp_idx = get_sp_neighbors_torch(raw_xyz, sp_label, K=ca_K)
            neigh_list.append(neighbor_sp_idx.cpu().numpy())

        neighbor_sp_idx_lists.append(neigh_list)

    return segments, np.array(neighbor_sp_idx_lists)


import numpy as np
from sklearn.neighbors import KDTree

def farthest_point_sampling(xyz: np.ndarray, num_samples: int) -> np.ndarray:
    """
    xyz: (N, 3) numpy array of xyz coordinates
    num_samples: number of points to sample
    Return:
        sampled_idx: (num_samples,) array of sampled point indices
    """
    N, _ = xyz.shape
    sampled_idx = np.zeros((num_samples,), dtype=np.int64)
    distances = np.full(N, np.inf)
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        sampled_idx[i] = farthest
        dist = np.sum((xyz - xyz[farthest])**2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return sampled_idx

def build_knn_samples_with_random(points: np.ndarray, S: int, K: int) -> np.ndarray:
    """
    构建随机采样 + KNN 样本

    Args:
        points: (N, D) numpy array，前3列是 xyz
        S: 采样点数
        K: 每个样本的 KNN 点数

    Returns:
        sample_idx: (S, K) int64 numpy array，表示每个采样点的邻域索引
    """
    assert points.ndim == 2 and points.shape[1] >= 3, "points 必须是 (N,D)，前3列是 xyz"
    xyz = points[:, :3]
    N = xyz.shape[0]

    # ✅ Step 1: 随机采样 S 个中心点索引
    center_idx = np.random.choice(N, size=S, replace=False)  # (S,)

    # ✅ Step 2: KDTree 查询 KNN
    kdtree = KDTree(xyz)
    _, neighbor_idx = kdtree.query(xyz[center_idx], k=K)  # (S, K)

    return neighbor_idx  # shape: (S, K)

def build_knn_samples_with_fps(points: np.ndarray, S: int, K: int) -> np.ndarray:
    """
    构建 KNN 样本

    Args:
        points: (N, D) numpy array, 前3列是xyz
        S: 采样点数（生成的样本数）
        K: 每个样本的 KNN 点数（包括自身）

    Returns:
        sample_idx: (S, K) numpy int64 array，表示每个样本对应的 K 个点的索引
    """
    assert points.ndim == 2 and points.shape[1] >= 3, "points 必须是 (N,D)，前3列是 xyz"
    xyz = points[:, :3]
    N = xyz.shape[0]

    # Step 1: FPS 采样得到 S 个中心点的索引
    center_idx = farthest_point_sampling(xyz, S)  # (S,)

    # Step 2: 构建 KDTree 并查询每个中心的KNN邻域
    kdtree = KDTree(xyz)
    _, neighbor_idx = kdtree.query(xyz[center_idx], k=K)  # (S, K)

    return neighbor_idx  # shape (S, K)


def split_superpoints_with_overlap_fast(sp_array, seq_length=128, overlap=0.5):
    """
    使用 as_strided 加速 superpoint 分段采样（支持重叠）
    """
    assert 0 <= overlap < 1, "overlap must be in [0, 1)"
    stride = int(seq_length * (1 - overlap))
    stride = max(stride, 1)

    sp_array = np.ascontiguousarray(sp_array)  # 保证内存连续
    S, D = sp_array.shape

    # 计算可以提取多少段
    num_segments = (S - seq_length) // stride + 1
    if num_segments <= 0:
        # 太短，返回最后一段补齐
        pad = np.zeros((seq_length - S, D), dtype=sp_array.dtype)
        return np.expand_dims(np.concatenate([pad, sp_array], axis=0), axis=0)

    # 构建新视图，性能远超 for 循环
    new_shape = (num_segments, seq_length, D)
    new_strides = (sp_array.strides[0] * stride, sp_array.strides[0], sp_array.strides[1])
    segments = as_strided(sp_array, shape=new_shape, strides=new_strides)

    # 检查是否还需要补最后一段
    last_end = stride * (num_segments - 1) + seq_length
    if last_end < S:
        last_segment = sp_array[-seq_length:]
        segments = np.concatenate([segments, last_segment[None]], axis=0)

    return segments # size: (num_segments, seq_length, D)


import numpy as np

def split_superpoints_with_overlap(sp_array, seq_length=128, overlap=0.5, as_array=False):
    """
    将 superpoint array 按照 seq_length 和 overlap 分段采样，返回一个 list，
    每段 shape=(seq_length, D)。确保最后一段总是对齐到结尾（向前回填）。
    """
    assert 0 <= overlap < 1, "overlap should be [0, 1)"
    S, D = sp_array.shape
    assert S >= seq_length, f"S({S}) must be >= seq_length({seq_length})"

    stride = max(int(seq_length * (1 - overlap)), 1)

    segments = []
    last_start = None
    # 常规滑窗
    for start in range(0, S - seq_length + 1, stride):
        segments.append(sp_array[start:start + seq_length, :])
        last_start = start

    # 若还有尾巴没覆盖，则追加“对齐到末尾”的最后一窗
    if (last_start is None) or (last_start + seq_length < S):
        segments.append(sp_array[-seq_length:, :])  # 例如会得到 [3,870,000 : 4,070,000)

    # 是否堆成一个三维数组
    if as_array:
        return np.stack(segments, axis=0)
    segments = np.array(segments)
    return segments





def grid_sampling(points, voxel_size=0.1, method="first"):
    """
    快速点云 grid 降采样 (voxel downsampling)
    
    参数:
        points: np.ndarray, 形状 (N, d)，至少包含 (x,y,z)
        voxel_size: float, 体素大小
        method: str, "centroid" 或 "first"
    
    返回:
        np.ndarray, 降采样后的点云
    """
    if points.shape[1] < 3:
        raise ValueError("点云必须至少包含 (x,y,z)")

    # 将坐标映射到 voxel 网格
    coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)

    # 给每个 voxel 分配唯一 id（哈希）
    # 为了避免冲突，这里用坐标的线性组合
    h = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791

    # 找到每个 voxel 的第一个点/所有点索引
    _, inv, counts = np.unique(h, return_inverse=True, return_counts=True)

    if method == "first":
        # 直接取每个 voxel 的第一个点
        idx = np.unique(inv, return_index=True)[1]
        return points[idx]

    elif method == "centroid":
        # 向量化质心计算：用 np.bincount
        sampled = np.zeros((len(counts), points.shape[1]), dtype=np.float32)
        for dim in range(points.shape[1]):
            sampled[:, dim] = np.bincount(inv, weights=points[:, dim], minlength=len(counts)) / counts
        return sampled

    else:
        raise ValueError("method 必须是 'centroid' 或 'first'")


from collections import Counter

def generate_superpoints(data):
    """
    Args:
        data: numpy array of shape (N, D), 最后一列是 gt_label，倒数第二列是 sp_label。
    
    Returns:
        superpoints: numpy array of shape (num_superpoints, D-1)
                     每行是一个 superpoint 的平均特征 + 主 gt_label。
    """
    # 分离特征、superpoint标签和gt标签
    features = data[:, :-2]          # (N, D-2)
    sp_labels = data[:, -2].astype(int)
    gt_labels = data[:, -1].astype(int)

    unique_sps = np.unique(sp_labels)
    superpoint_features = []
    
    for sp in unique_sps:
        mask = sp_labels == sp
        sp_feats = features[mask]
        sp_gt_labels = gt_labels[mask]

        # 平均 pooling 特征
        pooled_feature = sp_feats.mean(axis=0)

        # 找出出现最多的 gt_label
        most_common_label = Counter(sp_gt_labels).most_common(1)[0][0]

        # 拼接平均特征和gt_label
        superpoint_features.append(np.concatenate([pooled_feature, [sp], [most_common_label]]))

    return np.array(superpoint_features)

import numpy as np
import pandas as pd

def generate_superpoints_fast(data):
    """
    Args:
        data: numpy array of shape (N, D), 最后一列是 gt_label，倒数第二列是 sp_label。
    
    Returns:
        superpoints: numpy array of shape (num_superpoints, D-1)
    """
    df = pd.DataFrame(data)
    feature_cols = df.columns[:-2]
    sp_col = df.columns[-2]
    gt_col = df.columns[-1]

    # 平均池化特征
    pooled = df.groupby(sp_col)[feature_cols].mean()

    # 统计每个 sp 的主 gt_label
    mode_labels = (
        df.groupby(sp_col)[gt_col]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("major_gt")
    )

    # 拼接
    result = pd.concat([pooled, mode_labels], axis=1).reset_index()
  

    return result.to_numpy()



from sklearn.neighbors import NearestNeighbors
import torch
from torch.nn.functional import one_hot
# GPU 版本的 get_sp_neighbors_hilbert 函数

def align_first_column_with_B_fast(A: torch.Tensor, B: torch.Tensor):
    """
    高效对齐 A 的每一行的第一个元素与 B：若不一致，交换位置，无 for 循环。
    
    参数:
        A: (N, K) Tensor
        B: (N,) Tensor
    
    返回:
        A_aligned: (N, K) Tensor，已就位的 A
    """
    N, K = A.shape
    assert B.shape[0] == N, "B 和 A 的行数不一致"

    # 步骤 1: 判断 A[:, 0] 是否已经等于 B
    match_mask = (A[:, 0] == B)  # (N,)
    not_matched = ~match_mask  # 需要处理的行数

    if not not_matched.any():
        return A  # 所有都匹配，无需操作

    # 步骤 2: 对于不匹配的行，找到每行 B[i] 在 A[i, :] 中的位置
    A_mismatch = A[not_matched]             # shape: (M, K)
    B_mismatch = B[not_matched].unsqueeze(1)  # shape: (M, 1)

    # 掩码：每行中 B[i] 在 A[i, :] 中的位置为 True
    match_pos_mask = (A_mismatch == B_mismatch)  # (M, K)

    # 确保每行至少有一个 True
    assert match_pos_mask.any(dim=1).all(), "某些 B[i] 不在对应 A[i] 中"

    # 找到第一个匹配的位置索引（每行）
    match_pos = match_pos_mask.float().argmax(dim=1)  # (M,)

    # 现在构造一个 scatter 操作，把 B[i] 对应值换到 A[i, 0]
    A_updated = A.clone()

    # 获取要处理的行号（原 A 中的行号）
    idx_rows = not_matched.nonzero(as_tuple=False).squeeze(1)  # (M,)

    # 获取当前要交换的列索引
    idx_cols = match_pos  # (M,)

    # 进行交换 A[i, 0] <-> A[i, idx_cols]
    temp = A_updated[idx_rows, 0].clone()
    A_updated[idx_rows, 0] = A_updated[idx_rows, idx_cols]
    A_updated[idx_rows, idx_cols] = temp
    

    return A_updated

from torch_scatter import scatter_add

def get_sp_neighbors_torch_fast(raw_xyz: torch.Tensor,
                            sp_label: torch.Tensor,
                            K: int = 3) -> torch.Tensor:
    """
    raw_xyz : (N, 3)  float32, CUDA
    sp_label: (N,)    int64,   CUDA (任意不连续 ID)
    返回值   : (N, K)  int64,   CUDA  每行表示当前点所属 super-point 与其 K-1 个最近邻 super-point
    """
    unique_ids, inverse = torch.unique(sp_label, return_inverse=True)
    S = unique_ids.size(0)

    # ------- 1. 计算 super-point 质心 (S, 3) -------
    # 方法 A: scatter_add（推荐）
    sums = scatter_add(raw_xyz, inverse, dim=0, dim_size=S)       # (S,3)
    counts = torch.bincount(inverse, minlength=S).unsqueeze(1)    # (S,1)
    sp_xyz = sums / counts.clamp_min(1)

    # 方法 B: 纯 torch.index_add（若不想依赖 torch-scatter）
    #   sp_xyz = torch.zeros(S, 3, device=raw_xyz.device)
    #   ones   = torch.ones_like(inverse, dtype=raw_xyz.dtype).unsqueeze(-1)
    #   counts = torch.zeros(S, 1, device=raw_xyz.device).index_add_(0, inverse, ones)
    #   sp_xyz = sp_xyz.index_add_(0, inverse, raw_xyz).div_(counts.clamp_min(1))

    # ------- 2. 找 super-point 互相最近的 K 个 -------
    # 大多数场景 S <= 8k；若更大可分块 (appendix) ↓
    dist = torch.cdist(sp_xyz, sp_xyz)          # (S,S)
    _, nn_idx = dist.topk(K, largest=False)     # (S,K)  —> 每个 super-point 的 K 近邻（含自身）

    # ------- 3. 映射回原始点 -------
    neighbors = nn_idx[inverse]                 # (N,K)
    # 把第一列强制对齐为自身所属 super-point（O(1) 原子操作）
    neighbors[:, 0] = inverse                  # inverse 已是 [0, S-1] 连续 ID
    return neighbors 

from torch_scatter import scatter_add
def get_sp_neighbors_torch_fast_2(raw_xyz: torch.Tensor,
                                 sp_label: torch.Tensor,
                                 K: int = 3) -> torch.Tensor:
    """
    raw_xyz : (N, 3)  float32, CUDA
    sp_label: (N,)    int64,   CUDA（不要求连续）
    返回值   : (N, K)  int64,   CUDA，每行是当前点的 super-point 及其 K-1 个最近 super-point
    """
    

    device = raw_xyz.device
    unique_ids, inverse = torch.unique(sp_label, return_inverse=True)
    S = unique_ids.size(0)

    # ------- 1. 计算 super-point 质心 (S, 3) -------
    sums = scatter_add(raw_xyz, inverse, dim=0, dim_size=S)
    counts = torch.bincount(inverse, minlength=S).unsqueeze(1)
    sp_xyz = sums / counts.clamp_min(1)

    # ------- 2. 用批次化 knn 计算 K 个最近 super-point -------
    # 避免 S×S 占用显存，使用块状 knn 查询（适配大 S）

    B = 4096  # 可调批次大小，防止爆显存
    all_knn = []

    for i in range(0, S, B):
        q = sp_xyz[i:i+B]                      # query (B, 3)
        dists = torch.norm(q[:, None, :] - sp_xyz[None, :, :], dim=-1)  # (B, S)
        _, knn_idx = dists.topk(K, largest=False, dim=1)                # (B, K)
        all_knn.append(knn_idx)

    nn_idx = torch.cat(all_knn, dim=0)         # (S, K)

    # ------- 3. 映射回原始点 -------
    neighbors = nn_idx[inverse]               # (N, K)
    neighbors[:, 0] = inverse                 # 第一个是自身 super-point
    return neighbors


def get_sp_neighbors_torch(raw_xyz, sp_label, K=3):
    """
    raw_xyz: [N, 3] - 原始点坐标 (torch.Tensor, GPU)
    sp_label: [N] - 每个 raw 点所属 superpoint 的 ID（可非连续, torch.Tensor, GPU）
    K: 每个 raw 点关联的 superpoints 数量（含自身）
    """
    # 获取唯一的 superpoint ID 和对应的索引
    unique_sp_ids, inverse = torch.unique(sp_label, return_inverse=True)  # inverse shape [N]
    # print('unique_sp_ids.shape: ', unique_sp_ids.shape)

    # 计算每个 superpoint 的中心点坐标
    sp_xyz = torch.zeros((unique_sp_ids.size(0), 3), device=raw_xyz.device)
    for d in range(3):
        sp_xyz[:, d] = torch.bincount(inverse, weights=raw_xyz[:, d]) / torch.bincount(inverse)

    # 使用 PyTorch 的最近邻查找
    distances = torch.cdist(sp_xyz, sp_xyz)  # [S, S]
    _, indices = torch.topk(-distances, K, dim=1)  # 取最近的 K 个点（负号取最小值）

    # 将 superpoint 的邻居映射回原始点
    neighbors_label = indices[inverse]  # [N, K]

    neighbors_label = align_first_column_with_B_fast(neighbors_label, sp_label)

    return neighbors_label

def get_sp_neighbors(raw_xyz, sp_label, K=3):
    """
    raw_xyz: [N, 3] - 原始点坐标
    sp_label: [N] - 每个 raw 点所属 superpoint 的 ID（可非连续）
    K: 每个 raw 点关联的 superpoints 数量（含自身）
    grid_size: 用于 Hilbert 编码的 voxel 缩放倍数（越大越稠密）
    default_sp: 不足 K 时的默认补充 superpoint ID
    """

    
    
    unique_sp_ids, inverse = np.unique(sp_label, return_inverse=True)  # inverse shape [N]

    sp_xyz = np.stack([
        np.bincount(inverse, weights=raw_xyz[:, d]) / np.bincount(inverse)
        for d in range(3)
    ], axis=1)  # shape: [S, 3]

    # print(f"unique IDs: {len(unique_sp_ids)}")



    knn = NearestNeighbors(n_neighbors=K, algorithm='auto')
    knn.fit(sp_xyz) 
    distances, indices = knn.kneighbors(sp_xyz)  # [S, K] both
    # print('indices shape:', indices.shape)  # [S, K]

    neighbors_label = indices[sp_label]

    
    return neighbors_label  # shape [N, K]



def build_knn_samples_with_fps(points: np.ndarray, fps_idx: np.ndarray, K: int) -> np.ndarray:
    """
    构建随机采样 + KNN 样本

    Args:
        points: (N, D) numpy array，前3列是 xyz
        S: 采样点数
        K: 每个样本的 KNN 点数

    Returns:
        sample_idx: (S, K) int64 numpy array，表示每个采样点的邻域索引
    """
    assert points.ndim == 2 and points.shape[1] >= 3, "points 必须是 (N,D)，前3列是 xyz"
    xyz = points[:, :3]
    N = xyz.shape[0]

    # ✅ Step 1: 随机采样 S 个中心点索引
    center_idx = fps_idx  # (S,)

    # ✅ Step 2: KDTree 查询 KNN
    kdtree = KDTree(xyz)
    _, neighbor_idx = kdtree.query(xyz[center_idx], k=K)  # (S, K)

    return neighbor_idx  # shape: (S, K)
def farthest_point_sampling(points_np, n_samples):
    """
    GPU-based Farthest Point Sampling (FPS) with manual GPU selection.
    
    Args:
        points_np: (N, 3) numpy array or torch.Tensor (CPU), float32
        n_samples: int, number of samples to select
        gpu_id: int, index of the GPU to use (e.g., 0, 1, ...)
    
    Returns:
        fps_idx: (n_samples,) torch.LongTensor on CPU
    """
    assert isinstance(points_np, (torch.Tensor,)), "Input must be a torch.Tensor"
    assert points_np.shape[1] == 3, "Input must be of shape (N, 3)"

    
    
    device = points_np.device
    points = points_np

    N = points.shape[0]
    fps_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)

    for i in range(n_samples):
        fps_idx[i] = farthest
        centroid = points[farthest].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distances = torch.min(distances, dist)
        farthest = torch.argmax(distances)

    return fps_idx.cpu()  # 转回 CPU，方便后续索引操作

class Toronto3D_HSP_Dataset_fast_fps_knn(Dataset):
    def __init__(self, args, split='train', label_number=8):
        super().__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   
        self.seq_len = args.npoint
        self.overalp = args.sample_overlap
        data_root = args.data_root
        self.split = split

        self.sp_n_max = int(self.seq_len/1) #60
  
        self.label_number = label_number
        self.p_dim = args.p_dim  # x y z r g b intensity geof1 geof2 geof3 geof4 sp_label gt_label
        self.ca_K = args.ca_K

        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if room.endswith('.npy')]

        if split == 'train':
            self.rooms_split = ['L001', 'L003', 'L004']
        else:
            self.rooms_split = [ 'L002' ]



        grid_size = 500
        p = 10  # 每个轴的比特数
        n = 3   # 维度 (3D)
        hilbert_curve = HilbertCurve(p, n)
        
        labelweights = np.zeros(label_number)


        self.sorted_data = []
        self.data_sample_idx = []

        area_account = 0
        for room_name_root in tqdm(self.rooms_split, desc=f"Loading {split} data"):
            # print(f"Processing room: {room_name_root}")
            pattern = osp.join(data_root, f"{room_name_root}_reggo_*.npy")
            npy_files = glob.glob(pattern)

            # 提取 reggo_ 后的数字并排序
            def extract_reggo_num(filename):
                match = re.search(r"reggo_([0-9.]+)_regf", filename)
                return float(match.group(1)) if match else -1
            
            start_time = time.time()

            npy_files_sorted = sorted(npy_files, key=extract_reggo_num, reverse=False)
            # print('npy files name:', npy_files_sorted)
            self.room_data_list = []
            for npy_path in npy_files_sorted:
                data = np.load(npy_path)
                # print('sp labels:', np.unique(data[:,-2]).shape)
     
                assert data.shape[-1] == self.p_dim, f"Invalid data shape: {data.shape}"
                self.room_data_list.append(data)
            
            # 初始化合并后的数组
            merged_data = self.room_data_list[0][:, :-2]  # 取前 11 列
            merged_sp_idx = []  # 用于存储拼接后的 sp_idx
            merged_gt_label = self.room_data_list[0][:, -1:]  # 取最后一列

            # 遍历 self.room_data_list 中的每个数组
            for data in self.room_data_list:
                sp_idx = data[:, -2:-1]  # 提取 sp_idx 列
                # print('unique sp idx:', np.unique(sp_idx).shape)
                merged_sp_idx.append(sp_idx)  # 拼接 sp_idx

            # 将所有 sp_idx 列拼接为一列
            merged_sp_idx = np.concatenate(merged_sp_idx, axis=-1)

            # 将前 11 列、拼接后的 sp_idx 和最后一列合并为最终数组
            data = np.concatenate([merged_data, merged_sp_idx, merged_gt_label], axis=1)
            # print(f"Processed data shape: {data.shape}")


            data = normalize_pointcloud(data)#[:1000000,:]


            gt_labels = data[:, -1].astype(int) - 1
           
            tmp, _ = np.histogram(gt_labels, range(label_number+1))
            labelweights += tmp


  
            overlap = self.overalp
            if split == 'test':
                overlap = 0.9
            
            # print('split data')
            self.sorted_data.append(data) #sorted_data

            data_torch = torch.tensor(data, dtype=torch.float32).cuda()
            # print(f"Data shape: {data_torch.shape}, dtype: {data_torch.dtype}")

            sample_idx = farthest_point_sampling(data_torch[:,:3], n_samples=int(data.shape[0]/self.seq_len * 1/(1-overlap)))
            # print(f"Sampled indices: {sample_idx.shape}")

            time_after_fps = time.time()
            data_seg_idx = build_knn_samples_with_fps(data, sample_idx, K=self.seq_len)
            time_after_knn = time.time()
            # print(f"Time taken for FPS: {time_after_fps - start_time:.2f} seconds")
            print(f"Time taken for KNN: {time_after_knn - time_after_fps:.2f} seconds")
            # print(f"Sampled points KNN shape: {data_seg_idx.shape}")
            # data_seg_idx = split_superpoints_with_overlap_fast_idx(sorted_data.shape[0], seq_length=self.seq_len, overlap=overlap)
            # data_seg_idx = build_knn_samples_with_random(data, S = int(data.shape[0]/self.seq_len * 1/(1-overlap)), K=self.seq_len)  # 输出 shape (S, K)
            # data_seg_idx = split_superpoints_with_overlap_fast_idx(S = data.shape[0], seq_length=self.seq_len, overlap=overlap)
            data_seg_idx += area_account
            area_account += data.shape[0]  # 更新 area_account
            # print('split data done')
            assert max(np.unique(data[:,-1])) <= self.label_number, f"Invalid gt_labels: {np.unique(data[:,-1])}"
            end_time = time.time()
            print(f"Time taken for processing room {room_name_root}: {end_time - start_time:.2f} seconds")
            self.data_sample_idx.append(data_seg_idx)
        
        self.sorted_data = np.concatenate(self.sorted_data, axis=0)  # 合并所有房间的数据
        self.data_sample_idx = np.concatenate(self.data_sample_idx, axis=0)
        self.data_sample_idx = np.array(self.data_sample_idx, dtype=np.int64)  # 确保索引是整数类型


        self.labelweights = np.ones(label_number)
        # if split == 'train':
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        



    def __len__(self):
  
        return self.data_sample_idx.shape[0]

   

    def __getitem__(self, idx):
        start_time = time.time()
        indices = self.data_sample_idx[idx]  # shape: (K,)
        segment = self.sorted_data[indices]  # shape: (K, D)

        feature_channel = self.p_dim - 2
        points = segment[:, :feature_channel]  # 纯 numpy
        gt_labels = segment[:, -1].astype(int) - 1

        ab_sp_idx = []
        neighbor_sp_idx_list = []

        for i in range(len(self.room_data_list)):
            sp_label = segment[:, -1 - len(self.room_data_list) + i].astype(int)
            ab_sp_idx.append(sp_label)
            _, sp_label_contiguous = np.unique(sp_label, return_inverse=True)

            if self.ca_K > 0:
                neighbor_sp_idx = np.tile(sp_label_contiguous[:, None], (1, 2))
            else:
                neighbor_sp_idx = np.tile(sp_label_contiguous[:, None], (1, 2))

            neighbor_sp_idx_list.append(neighbor_sp_idx)

        ab_sp1_idx = np.array(ab_sp_idx)
        end_time = time.time()
        # print(f"getitem time: {end_time - start_time:.4f} seconds")
        return points, np.array(neighbor_sp_idx_list), gt_labels, ab_sp1_idx

