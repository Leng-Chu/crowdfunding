import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path


def _get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


REPO_ROOT = _get_repo_root()

# 读取原始CSV文件
csv_path = REPO_ROOT / "data" / "metadata" / "now.csv"
df = pd.read_csv(csv_path)

# 确保 launched_at 是 datetime 类型并提取年份
df["launched_at"] = pd.to_datetime(df["launched_at"], utc=True)
df["year"] = df["launched_at"].dt.year

# 分析每年的成功和失败项目数量
print("原始数据中每年成功/失败项目数量:")
for year in sorted(df["year"].unique()):
    yearly_data = df[df["year"] == year]
    successful_count = len(yearly_data[yearly_data["state"] == "successful"])
    failed_count = len(yearly_data[yearly_data["state"] == "failed"])
    print(f"{year}年 - 成功: {successful_count}, 失败: {failed_count}")

# 创建一个副本用于处理
processed_df = df.copy()

# 遍历每年，平衡成功和失败项目的数量
moved_projects = []  # 记录被移动的项目

for year in df["year"].unique():
    yearly_data = processed_df[processed_df["year"] == year]
    
    successful_projects = yearly_data[yearly_data["state"] == "successful"]
    failed_projects = yearly_data[yearly_data["state"] == "failed"]
    
    successful_count = len(successful_projects)
    failed_count = len(failed_projects)
    
    # 计算需要移动多少成功项目以达到平衡
    if successful_count > failed_count:
        # 需要移动多余的成功项目
        excess_successful = successful_count - failed_count
        if excess_successful > 0:
            # 随机选择要移动的成功项目
            projects_to_move = successful_projects.sample(n=min(excess_successful, successful_count), random_state=42)
            
            # 添加到被移动项目列表
            moved_projects.extend(projects_to_move.index.tolist())
            
            # 从数据框中删除这些项目
            processed_df = processed_df.drop(projects_to_move.index)

    elif failed_count > successful_count:
        # 需要移动多余的失败项目
        excess_failed = failed_count - successful_count
        if excess_failed > 0:
            # 随机选择要移动的失败项目
            projects_to_move = failed_projects.sample(n=min(excess_failed, failed_count), random_state=42)
            
            # 添加到被移动项目列表
            moved_projects.extend(projects_to_move.index.tolist())
            
            # 从数据框中删除这些项目
            processed_df = processed_df.drop(projects_to_move.index)

# 输出处理后的统计信息
print("\n处理后每年成功/失败项目数量:")
for year in sorted(processed_df["year"].unique()):
    yearly_data = processed_df[processed_df["year"] == year]
    successful_count = len(yearly_data[yearly_data["state"] == "successful"])
    failed_count = len(yearly_data[yearly_data["state"] == "failed"])
    print(f"{year}年 - 成功: {successful_count}, 失败: {failed_count}")

# 获取被移动的项目详细信息
moved_df = df.loc[moved_projects].copy()

# 移动项目文件夹
if not moved_df.empty:
    source_folder = REPO_ROOT / "data" / "projects" / "now"
    target_folder = REPO_ROOT / "data" / "projects" / "move2"
    
    # 确保目标目录存在
    target_folder.mkdir(parents=True, exist_ok=True)
    
    # 遍历被移动的项目
    for index, row in moved_df.iterrows():
        project_id = row['project_id']
        source_path = source_folder / str(project_id)
        target_path = target_folder / str(project_id)
        
        # 如果源目录存在，则移动
        if source_path.exists():
            print(f"正在移动项目 {project_id} 从 {source_path} 到 {target_path}")
            shutil.move(str(source_path), str(target_path))
        else:
            print(f"警告：项目 {project_id} 的目录不存在: {source_path}")

# 删除临时的 year 列，因为我们不再需要它
processed_df = processed_df.drop(columns=["year"])

# 保存处理后的数据到新的CSV文件
balanced_csv_path = REPO_ROOT / "data" / "metadata" / "now.csv"
processed_df.to_csv(balanced_csv_path, index=False)

# 保存被移动的项目到另一个CSV文件
moved_csv_path = REPO_ROOT / "data" / "metadata" / "move2.csv"
moved_df.to_csv(moved_csv_path, index=False)

print(f"\n已创建平衡后的CSV文件: {balanced_csv_path}")
print(f"已创建被移动项目的CSV文件: {moved_csv_path}")
print(f"原始数据行数: {len(df)}")
print(f"平衡后数据行数: {len(processed_df)}")
print(f"被移动的项目数: {len(moved_df)}")
