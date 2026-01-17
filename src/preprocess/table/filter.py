import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

# 读取原始CSV文件
csv_path = "/home/zlc/crowdfunding/data/metadata/all.csv"
df = pd.read_csv(csv_path)

# 获取原始数据的备份，用于后续处理
original_df = df.copy()

# 检查是否有缺失的category的行
missing_category = df[df['category_parent'].isna() | (df['category_parent'] == '')]

# 计算duration_days
df_temp = df.copy()
df_temp["launched_at"] = pd.to_datetime(df_temp["launched_at"], utc=True)
df_temp["deadline"] = pd.to_datetime(df_temp["deadline"], utc=True)
df_temp["duration_days"] = (df_temp["deadline"] - df_temp["launched_at"]).dt.total_seconds() / 86400.0

# 找出duration_days小于5的行
short_duration_rows = df_temp[df_temp['duration_days'] < 5]

# 计算currency和category的出现频率
currency_counts = df['currency'].value_counts()
country_counts = df['country'].value_counts()
category_counts = df['category'].value_counts()

# 找出出现频率小于100的currency和category
low_freq_currency = set(currency_counts[currency_counts < 100].index)
low_freq_category = set(category_counts[category_counts < 50].index)
low_freq_country = set(country_counts[country_counts < 100].index)

# 找出属于低频currency或低频category的行
low_frequency_rows = df[
    df['currency'].isin(low_freq_currency) |
    df['category'].isin(low_freq_category) |
    df['country'].isin(low_freq_country)
]

# 合并需要删除的行（缺失category 或 属于低频currency/category 或 duration_days<5）
rows_to_process = pd.concat([missing_category, low_frequency_rows, short_duration_rows]).drop_duplicates()

if not rows_to_process.empty:
    # 移动这些项目文件夹到新位置
    source_folder = "/home/zlc/crowdfunding/data/projects/now"
    target_folder = "/home/zlc/crowdfunding/data/projects/move1"
    
    # 确保目标目录存在
    os.makedirs(target_folder, exist_ok=True)
    
    # 遍历需要处理的行
    for index, row in rows_to_process.iterrows():
        project_id = row['project_id']
        source_path = os.path.join(source_folder, str(project_id))
        target_path = os.path.join(target_folder, str(project_id))
        
        # 如果源目录存在，则移动
        if os.path.exists(source_path):
            print(f"正在移动项目 {project_id} 从 {source_path} 到 {target_path}")
            shutil.move(source_path, target_path)
        else:
            print(f"警告：项目 {project_id} 的目录不存在: {source_path}")
    
    # 从原始数据中删除这些行，创建新的CSV
    filtered_df = df.drop(rows_to_process.index)
    
    # 保存新的CSV文件（包含原始列）- 过滤后的数据
    new_csv_path = "/home/zlc/crowdfunding/data/metadata/now.csv"
    filtered_df.to_csv(new_csv_path, index=False)
    
    # 保存被移动的行到另一个CSV文件
    moved_rows_csv_path = "/home/zlc/crowdfunding/data/metadata/move1.csv"
    rows_to_process.to_csv(moved_rows_csv_path, index=False)
    
    print(f"已创建新的CSV文件（过滤后的数据）: {new_csv_path}")
    print(f"已创建CSV文件（被移动的行）: {moved_rows_csv_path}")
    print(f"原始数据行数: {len(original_df)}")
    print(f"过滤后的数据行数: {len(filtered_df)}")
    print(f"被移动的行数: {len(rows_to_process)}")
    
    # 显示一些统计信息
    print(f"低频currency数量: {len(low_freq_currency)}")
    print(f"低频category数量: {len(low_freq_category)}")
    print(f"低频country数量: {len(low_freq_country)}")
    print(f"因缺失category_parent被删除的行数: {len(missing_category)}")
    print(f"因低频数据被删除的行数: {len(low_frequency_rows)}")
    print(f"因duration_days<5被删除的行数: {len(short_duration_rows)}")
else:
    print("没有找到需要处理的数据行")

