import os
import pandas as pd

def merge_result_csv():
    """
    遍历experiments/late目录下除_cache外的所有子文件夹的子文件夹，
    找到其中的result.csv文件，合并为一个CSV文件并按test_f1排序
    """
    # 定义根目录路径
    base_dir = "experiments/late"
    
    # 存储所有CSV数据的列表
    all_csv_data = []
    
    # 遍历base_dir下的所有子文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # 跳过_cache文件夹
        if folder_name == "_cache":
            continue
        
        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            # 遍历第二层子文件夹
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                
                if os.path.isdir(subfolder_path):
                    # 在每个子文件夹中寻找result.csv
                    result_csv_path = os.path.join(subfolder_path, "result.csv")
                    
                    if os.path.exists(result_csv_path):
                        print(f"正在读取: {result_csv_path}")
                        df = pd.read_csv(result_csv_path)
                        
                        # 添加两列标识来源文件夹，方便后续追踪
                        df['first_folder'] = folder_name
                        df['second_folder'] = subfolder_name
                        
                        all_csv_data.append(df)
                    else:
                        print(f"未找到: {result_csv_path}")
    
    if not all_csv_data:
        print("没有找到任何result.csv文件")
        return
    
    # 合并所有CSV数据
    merged_df = pd.concat(all_csv_data, ignore_index=True)
    
    # 按照test_f1列排序（降序，高分在前）
    if 'test_f1' in merged_df.columns:
        merged_df = merged_df.sort_values(by='test_f1', ascending=False)
    else:
        print("警告: 没有找到test_f1列，将按原始顺序输出")
    
    # 保存合并后的CSV文件
    output_path = "experiments/late/merged_results.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"已将所有result.csv文件合并到: {output_path}")
    print(f"总共包含 {len(merged_df)} 行数据")
    
    # 显示前几行数据作为预览
    print("\n前5行数据预览:")
    print(merged_df.head())


if __name__ == "__main__":
    merge_result_csv()