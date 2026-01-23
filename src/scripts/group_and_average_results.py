import os
import pandas as pd


def group_and_average_results(input_path=None, output_path=None):
    """
    根据mode, baseline_mode, image_embedding_type, text_embedding_type进行分组，
    计算test_accuracy, test_precision, test_recall, test_f1, test_auc的平均值，
    并删除first_folder, second_folder, threshold列
    """
    # 如果没有指定输入路径，则使用默认路径
    if input_path is None:
        input_path = "experiments/ch1/ch1.csv"
    
    # 如果没有指定输出路径，则使用默认路径
    if output_path is None:
        output_path = "experiments/ch1/averaged_results.csv"
    
    # 读取CSV文件
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    print(f"正在读取文件: {input_path}")
    df = pd.read_csv(input_path)
    
    # 检查必要的列是否存在
    required_groupby_cols = ['mode', 'baseline_mode', 'image_embedding_type', 'text_embedding_type']
    required_avg_cols = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    cols_to_remove = ['first_folder', 'second_folder', 'threshold']
    
    missing_groupby_cols = [col for col in required_groupby_cols if col not in df.columns]
    missing_avg_cols = [col for col in required_avg_cols if col not in df.columns]
    missing_cols_to_remove = [col for col in cols_to_remove if col not in df.columns]
    
    if missing_groupby_cols:
        print(f"警告: 以下分组所需的列不存在: {missing_groupby_cols}")
        required_groupby_cols = [col for col in required_groupby_cols if col in df.columns]
    
    if missing_avg_cols:
        print(f"警告: 以下需要平均的列不存在: {missing_avg_cols}")
        required_avg_cols = [col for col in required_avg_cols if col in df.columns]
    
    cols_to_remove = [col for col in cols_to_remove if col in df.columns]
    
    if not required_groupby_cols:
        print("错误: 没有找到任何有效的分组列")
        return
    
    if not required_avg_cols:
        print("错误: 没有找到任何需要平均的列")
        return
    
    print(f"按以下列进行分组: {required_groupby_cols}")
    print(f"对以下列计算平均值: {required_avg_cols}")
    print(f"将删除以下列: {cols_to_remove}")
    
    # 删除不需要的列
    df_filtered = df.drop(columns=cols_to_remove)
    
    # 定义聚合字典，对指定列计算平均值
    agg_dict = {}
    for col in required_avg_cols:
        agg_dict[col] = 'mean'
    
    # 对其他非分组列也应用适当的聚合方式（例如取第一个值）
    other_cols = [col for col in df_filtered.columns if col not in required_groupby_cols + required_avg_cols]
    for col in other_cols:
        # 对于非数值列，使用第一个值；对于数值列，也可以使用均值
        if pd.api.types.is_numeric_dtype(df_filtered[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'
    
    # 分组并计算平均值
    grouped_df = df_filtered.groupby(required_groupby_cols).agg(agg_dict).reset_index()
    
    # 按test_f1列排序（降序，高分在前）
    if 'test_f1' in grouped_df.columns:
        grouped_df = grouped_df.sort_values(by='test_f1', ascending=False)
        print("结果已按test_f1列降序排序")
    else:
        print("警告: 没有找到test_f1列，无法按此列排序")
    
    # 保存结果
    grouped_df.to_csv(output_path, index=False)
    print(f"已将分组平均后的结果保存到: {output_path}")
    print(f"总共包含 {len(grouped_df)} 行数据")
    
    # 显示前几行数据作为预览
    print("\n前5行数据预览:")
    print(grouped_df.head())
    
    # 显示统计摘要
    print(f"\n分组列的唯一值计数:")
    for col in required_groupby_cols:
        print(f"{col}: {grouped_df[col].nunique()} 个唯一值")


if __name__ == "__main__":
    group_and_average_results()