#!/usr/bin/env python3
"""
检查npy文件的大小和内容的工具脚本
此脚本用于加载和分析npy文件，显示形状、数据类型、统计信息等
"""

import os
import sys
import numpy as np
from pathlib import Path


def check_npy_file(file_path):
    """
    检查单个npy文件的大小和内容
    
    Args:
        file_path (str): npy文件路径
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    if file_path.suffix.lower() != '.npy':
        print(f"警告: {file_path} 不是.npy文件")
        return
    
    try:
        # 加载npy文件
        data = np.load(file_path)
        
        print("="*60)
        print(f"文件路径: {file_path}")
        print(f"文件大小: {os.path.getsize(file_path)} 字节 ({os.path.getsize(file_path)/1024:.2f} KB, {os.path.getsize(file_path)/(1024*1024):.2f} MB)")
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"元素总数: {data.size}")
        
        # 计算内存占用
        memory_size = data.nbytes
        print(f"数组内存占用: {memory_size} 字节 ({memory_size/1024:.2f} KB, {memory_size/(1024*1024):.2f} MB)")
        
        # 如果是数值类型，显示统计信息
        if np.issubdtype(data.dtype, np.number):
            print(f"最小值: {data.min()}")
            print(f"最大值: {data.max()}")
            print(f"平均值: {data.mean()}")
            print(f"标准差: {data.std()}")
            
            # 如果数组不太大，显示实际内容
            if data.size <= 100:
                print(f"数组内容:\n{data}")
            else:
                print(f"数组内容 (前20个元素):\n{data.flat[:min(20, data.size)]}")
        else:
            # 非数值类型，简单显示内容
            if data.size <= 20:
                print(f"数组内容:\n{data}")
            else:
                print(f"数组内容 (前20个元素):\n{data.flat[:min(20, data.size)]}")
        
        print("="*60)
        
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")


def check_npy_directory(directory_path, recursive=False):
    """
    检查目录下所有npy文件
    
    Args:
        directory_path (str): 目录路径
        recursive (bool): 是否递归搜索子目录
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"错误: 目录 {directory_path} 不存在")
        return
    
    if not directory_path.is_dir():
        print(f"错误: {directory_path} 不是目录")
        return
    
    # 查找所有npy文件
    if recursive:
        npy_files = list(directory_path.rglob("*.npy"))
    else:
        npy_files = list(directory_path.glob("*.npy"))
    
    if not npy_files:
        print(f"在目录 {directory_path} 中没有找到.npy文件")
        return
    
    print(f"在目录 {directory_path} 中找到 {len(npy_files)} 个.npy文件\n")
    
    for npy_file in sorted(npy_files):
        check_npy_file(npy_file)


def main():
    """
    主函数，处理命令行参数并执行相应的检查
    """
    if len(sys.argv) < 2:
        print("用法:")
        print("  检查单个文件: python check_npy_files.py <file_path>")
        print("  检查目录:     python check_npy_files.py <directory_path> [-r]")
        print("  递归检查目录: python check_npy_files.py <directory_path> -r")
        sys.exit(1)
    
    target_path = sys.argv[1]
    
    # 检查是否是目录
    if Path(target_path).is_dir():
        recursive = '-r' in sys.argv or '--recursive' in sys.argv
        check_npy_directory(target_path, recursive)
    else:
        # 检查单个文件
        check_npy_file(target_path)


if __name__ == "__main__":
    main()