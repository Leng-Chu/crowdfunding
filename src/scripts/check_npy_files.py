#!/usr/bin/env python3
"""
检查npy文件的大小和内容的工具脚本
此脚本用于加载和分析npy文件，显示形状、数据类型、统计信息等
"""

import os
import sys
import numpy as np
from pathlib import Path
import random


def check_embedding_matrix(data, name="array"):
    """
    对 embedding 矩阵做专项检查
    """
    print("\n[Embedding 检查]")
    print(f"维度数: {data.ndim}")

    # 常见错误 shape 提示
    if data.ndim >= 3:
        print("⚠️ 警告: embedding 通常应为 2D (N, D)，当前 shape 可能未 flatten")

    if data.ndim == 2:
        N, D = data.shape
        print(f"样本数 N = {N}, 向量维度 D = {D}")

        # 每行 L2 norm
        norms = np.linalg.norm(data, axis=1)
        print(f"L2 norm (per row): min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")

        if norms.max() < 1e-3:
            print("❌ 所有向量范数极小，疑似未归一化或全零")
        elif norms.min() < 1e-6:
            print("❌ 存在范数接近 0 的向量（严重异常）")
        elif norms.min() > 0.9 and norms.max() < 1.1:
            print("✅ 看起来是 L2-normalized embedding")
        else:
            print("ℹ️ 向量范数不统一（未归一化或刻意保留尺度）")

        # unique 值数量（快速发现 0/1 / mask）
        unique_count = np.unique(data).size
        print(f"unique 值数量: {unique_count}")
        if unique_count <= 10:
            print("⚠️ unique 值很少，疑似被二值化 / mask / sign")

        # 每行 min/max 的极端值
        row_min = data.min(axis=1)
        row_max = data.max(axis=1)

        idx_min = np.argmin(row_min)
        idx_max = np.argmax(row_max)

        print(f"最小值出现在样本 {idx_min}: {row_min[idx_min]:.6f}")
        print(f"最大值出现在样本 {idx_max}: {row_max[idx_max]:.6f}")

    print("[Embedding 检查结束]\n")


def check_npy_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件 {file_path} 不存在")
        return

    if file_path.suffix.lower() != '.npy':
        print(f"警告: {file_path} 不是.npy文件")
        return

    try:
        data = np.load(file_path)

        print("=" * 70)
        print(f"文件路径: {file_path}")
        size = os.path.getsize(file_path)
        print(f"文件大小: {size} 字节 ({size/1024:.2f} KB, {size/(1024*1024):.2f} MB)")
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"元素总数: {data.size}")
        print(f"数组内存占用: {data.nbytes} 字节 ({data.nbytes/1024:.2f} KB)")

        if np.issubdtype(data.dtype, np.number):
            print("\n[全局统计]")
            print(f"最小值: {data.min()}")
            print(f"最大值: {data.max()}")
            print(f"平均值: {data.mean()}")
            print(f"标准差: {data.std()}")

            # embedding 专项检查
            check_embedding_matrix(data)

            # 内容预览
            print("[数组内容预览]")
            flat = data.ravel()
            print(flat[:20])

        else:
            print("非数值类型数组，跳过统计")

        print("=" * 70)

    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")


def check_npy_directory(directory_path, recursive=False):
    directory_path = Path(directory_path)

    if not directory_path.exists():
        print(f"错误: 目录 {directory_path} 不存在")
        return

    if recursive:
        npy_files = list(directory_path.rglob("*.npy"))
    else:
        npy_files = list(directory_path.glob("*.npy"))

    if not npy_files:
        print(f"在目录 {directory_path} 中没有找到.npy文件")
        return

    print(f"在目录 {directory_path} 中找到 {len(npy_files)} 个.npy文件\n")

    for f in sorted(npy_files):
        check_npy_file(f)


def check_random_subdirs(base_directory, num_dirs=5):
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"错误: 目录 {base_path} 不存在")
        return
    
    # 获取所有子目录
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"目录 {base_path} 下没有子目录")
        return
    
    # 随机选择最多5个子目录
    selected_dirs = random.sample(subdirs, min(num_dirs, len(subdirs)))
    
    print(f"从 {len(subdirs)} 个子目录中随机选择了 {len(selected_dirs)} 个进行检查:")
    for d in selected_dirs:
        print(f"  - {d.name}")
    print()
    
    # 检查每个选中的子目录中的npy文件
    for subdir in selected_dirs:
        print(f">>> 正在检查子目录: {subdir}")
        check_npy_directory(subdir, recursive=True)


def main():
    # 默认检查 /home/zlc/crowdfunding/data/projects/now 目录下的随机5个子文件夹
    target_path = "/home/zlc/crowdfunding/data/projects/now"
    check_random_subdirs(target_path, 5)


if __name__ == "__main__":
    main()