import pandas as pd
from pathlib import Path
import shutil


def clean_projects(csv_path, projects_base_path, check_download_status=True, remove_csv_rows=True):
    """
    根据CSV中的数据清理项目文件夹
    
    :param csv_path: CSV文件路径
    :param projects_base_path: 项目基础路径
    :param check_download_status: 是否检查download_status
    :param remove_csv_rows: 是否从CSV中删除对应的行
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建备份副本以便后续操作
    original_len = len(df)
    
    # 遍历每一行
    rows_to_remove = []
    for idx, row in df.iterrows():
        project_id = str(row.get('project_id', ''))  # 确保project_id为字符串
        
        # 检查是否存在project_id对应的子文件夹
        project_folder = Path(projects_base_path) / project_id
        if not project_folder.exists():
            # 如果项目文件夹不存在，标记此行需要删除
            if remove_csv_rows:
                print(f"项目文件夹不存在，将从CSV中删除该行 [CSV第{idx+1}行] (项目ID: {project_id})")
            rows_to_remove.append(idx)
        else:
            # 检查content_status和download_status
            content_status = row.get('content_status', '')
            download_status = row.get('download_status', '')
            
            # 判断是否需要删除文件夹
            should_delete_folder = (
                content_status != 'success' or 
                (check_download_status and download_status != 'success')
            )
            
            if should_delete_folder:
                print(f"删除项目文件夹 {project_folder} [CSV第{idx+1}行] (项目ID: {project_id}, "
                      f"content_status: {content_status}, download_status: {download_status})")
                
                # 删除文件夹
                shutil.rmtree(project_folder)
                rows_to_remove.append(idx)


    # 如果需要删除CSV中的行
    if remove_csv_rows and rows_to_remove:
        print(f"从CSV中删除 {len(rows_to_remove)} 行")
        df = df.drop(rows_to_remove).reset_index(drop=True)
        
        # 保存更新后的CSV
        csv_path_obj = Path(csv_path)
        stem = csv_path_obj.stem
        suffix = csv_path_obj.suffix
        
        # 检查原始文件名是否包含下划线后的数字，并替换为当前行数
        parts = stem.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            # 替换最后一个部分为当前行数
            new_stem = '_'.join(parts[:-1]) + f"_{len(df)}"
        else:
            # 如果没有下划线数字，添加行数
            new_stem = f"{stem}_{len(df)}"
        
        new_csv_path = csv_path_obj.parent / f"{new_stem}{suffix}"
        df.to_csv(new_csv_path, index=False)
        print(f"已保存更新后的CSV到 {new_csv_path}")
    
    print(f"处理完成，原始行数: {original_len}, 现在行数: {len(df)}")


if __name__ == "__main__":
    CSV_PATH = "data/metadata/years/2024_10642.csv"
    PROJECTS_BASE_PATH = "data/projects/2024"
    CHECK_DOWNLOAD_STATUS = False   # 控制是否要判断download_status不为success
    REMOVE_CSV_ROWS = True         # 控制是否要从csv中删掉这一行
    
    clean_projects(
        csv_path=CSV_PATH,
        projects_base_path=PROJECTS_BASE_PATH,
        check_download_status=CHECK_DOWNLOAD_STATUS,
        remove_csv_rows=REMOVE_CSV_ROWS
    )