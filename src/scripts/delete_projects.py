import pandas as pd
import os
import shutil
from pathlib import Path

def delete_unsuccessful_projects(start_row=1, end_row=5000):
    """
    删除content_status不为success的项目的文件夹，并从CSV中移除对应行
    """
    # 可修改的起点和终点变量（从1开始计数的闭区间）
    START_ROW = start_row  # 起点行号（包含），从1开始
    END_ROW = end_row      # 终点行号（包含），从1开始
    
    csv_path = "data/metadata/years/2023_10844.csv"
    projects_base_path = "data/projects/2023"
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"CSV文件 {csv_path} 不存在")
        return
    
    if not os.path.exists(projects_base_path):
        print(f"项目文件夹 {projects_base_path} 不存在")
        return
    
    # 循环处理，直到指定范围内的content_status均为success
    while True:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 将从1开始的行号转换为从0开始的索引（pandas DataFrame使用从0开始的索引）
        start_idx = START_ROW - 1  # 从1开始的行号转为从0开始的索引
        end_idx = min(END_ROW, len(df))  # 确保不超过DataFrame长度
        end_idx -= 1  # 转换为从0开始的索引（包含原来END_ROW位置的数据）
        
        # 获取指定范围内的数据
        if start_idx < len(df):
            if end_idx < len(df):
                range_df = df.iloc[start_idx:end_idx+1]  # +1 因为iloc右边界不包含
            else:
                range_df = df.iloc[start_idx:]
        else:
            print(f"起始行号 {START_ROW} 超出数据范围，处理完成")
            break
        
        if range_df.empty:
            print(f"指定范围({START_ROW}到{END_ROW})内没有数据，处理完成")
            break
        
        # 查找content_status不为success的行
        non_success_rows = range_df[range_df.get('content_status', pd.Series()) != 'success']
        
        if len(non_success_rows) == 0:
            print(f"第{START_ROW}到{END_ROW}行的content_status均为success，处理完成")
            break
        
        print(f"发现 {len(non_success_rows)} 个content_status不为success的项目")
        
        # 遍历所有content_status不为success的项目
        for index, row in non_success_rows.iterrows():
            project_id = row.get('project_id')
            
            if pd.isna(project_id):
                continue
                
            project_folder_path = os.path.join(projects_base_path, str(project_id))
            
            # 检查项目文件夹是否存在
            if os.path.exists(project_folder_path):
                try:
                    # 删除项目文件夹
                    shutil.rmtree(project_folder_path)
                    print(f"已删除项目文件夹: {project_folder_path}")
                except Exception as e:
                    print(f"删除项目文件夹失败 {project_folder_path}: {str(e)}")
            else:
                print(f"项目文件夹不存在: {project_folder_path}")
        
        # 从DataFrame中删除content_status不为success的行
        df = df.drop(non_success_rows.index)
        
        # 保存更新后的CSV文件
        df.to_csv(csv_path, index=False)
        print(f"已更新CSV文件，删除了 {len(non_success_rows)} 行")
    
    # 根据实际项目数重命名CSV文件
    final_row_count = len(df)
    original_dir = os.path.dirname(csv_path)
    original_filename = os.path.basename(csv_path)
    year = original_filename.split('_')[0]  # 提取年份，如"2024"
    
    # 创建新文件名
    new_filename = f"{year}_{final_row_count}.csv"
    new_csv_path = os.path.join(original_dir, new_filename)
    
    # 重命名CSV文件
    os.rename(csv_path, new_csv_path)
    print(f"CSV文件已重命名为: {new_filename} (项目数: {final_row_count})")
    
    print("所有操作已完成")


if __name__ == "__main__":
    # 可以通过修改这两个参数来定义处理的起点和终点（从1开始计数的闭区间）
    START_ROW = 1    # 定义起点（从1开始计数）
    END_ROW = 11000   # 定义终点（从1开始计数）
    
    delete_unsuccessful_projects(START_ROW, END_ROW)