import pandas as pd
import numpy as np

csv_path = "/home/zlc/crowdfunding/data/metadata/now.csv"
df = pd.read_csv(csv_path)

# ===== 1. 提取需要保留的列 =====
df_original = pd.read_csv(csv_path)  # 重新加载以获取完整的原始数据
df_original['launched_at'] = pd.to_datetime(df_original['launched_at'], utc=True)
project_info = df_original[['project_id', 'launched_at']].copy()
project_info['year'] = df_original['launched_at'].dt.year

# ===== 2. 删除不需要的列 =====
drop_cols = [
    "content_status", "download_status", "backers_count", "percent_funded", "usd_pledged",
    "project_id", "creator_id", "project_url", "cover_url", "creator_profile_url",
    "title", "blurb", "staff_pick", "static_usd_rate", "category_parent"
]
df.drop(columns=drop_cols, inplace=True)

# ===== 3. 创建标签 =====
y = (df["state"] == "successful").astype(int)
X = df.drop(columns=["state"])

# ===== 4. 从'launched_at'和'deadline'创建'duration_days' =====
X["launched_at"] = pd.to_datetime(X["launched_at"], utc=True)
X["deadline"] = pd.to_datetime(X["deadline"], utc=True)
X["duration_days"] = (X["deadline"] - X["launched_at"]).dt.total_seconds() / 86400.0

# ===== 新增: 检查 launched_at 分别在 2023, 2024, 2025 的成功/失败项目数量 =====
# 筛选出 2023, 2024, 2025 年的数据
original_for_stats = pd.read_csv(csv_path)  # 再次加载用于统计的原始数据
original_for_stats['launched_at'] = pd.to_datetime(original_for_stats['launched_at'], utc=True)
original_for_stats['year'] = original_for_stats['launched_at'].dt.year

years_of_interest = [2023, 2024, 2025]
filtered_df = original_for_stats[original_for_stats['year'].isin(years_of_interest)]

# 按年份和项目状态分组统计
yearly_success_fail_counts = filtered_df.groupby(['year', 'state']).size().unstack(fill_value=0)

# 输出结果
print("每年成功/失败项目数量:")
print(yearly_success_fail_counts)

X.drop(columns=["launched_at", "deadline"], inplace=True)

# ===== 5. 对'usd_goal'应用对数变换 =====
X["log_usd_goal"] = np.log1p(X["usd_goal"])
X.drop(columns=["usd_goal"], inplace=True)

# ===== 6. 合并保留的列和编码后的特征 =====
# 只保留project_id和year，不包含launched_at
project_info_reduced = project_info[['project_id', 'year']].copy()
final_data = pd.concat([project_info_reduced.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

# ===== 7. 保存到新的CSV文件 =====
output_csv_path = "/home/zlc/crowdfunding/data/metadata/now_processed.csv"
final_data['state'] = y.values  # 添加标签列
final_data.to_csv(output_csv_path, index=False)
print(f"已保存处理后的数据到: {output_csv_path}")

