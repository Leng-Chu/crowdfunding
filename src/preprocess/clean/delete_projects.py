import pandas as pd
from pathlib import Path
import shutil


def clean_projects(csv_path, projects_base_path, trash_folder_path=None, check_download_status=True, separate_deleted_rows=True):
    """
    根据CSV中的数据清理项目文件夹，将要删除的文件夹移到垃圾文件夹中
    
    :param csv_path: CSV文件路径
    :param projects_base_path: 项目基础路径
    :param trash_folder_path: 垃圾文件夹路径，用于存放要删除的项目文件夹
    :param check_download_status: 是否检查download_status
    :param separate_deleted_rows: 是否将要删除的行单独保存到新CSV中
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建垃圾文件夹路径（如果未提供则默认创建）
    if trash_folder_path is None:
        trash_folder_path = Path(projects_base_path) / "trash"
    else:
        trash_folder_path = Path(trash_folder_path)
    
    trash_folder_path.mkdir(exist_ok=True)
    
    # 创建备份副本以便后续操作
    original_len = len(df)
    
    # 用于存储要删除的行
    rows_to_remove = []
    removed_dataframes = []
    
    for idx, row in df.iterrows():
        project_id = str(row.get('project_id', ''))  # 确保project_id为字符串
        
        # 检查是否存在project_id对应的子文件夹
        project_folder = Path(projects_base_path) / project_id
        if not project_folder.exists():
            # 如果项目文件夹不存在，标记此行需要删除
            print(f"项目文件夹不存在，将把对应行加入删除列表 (项目ID: {project_id})")
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
                # 移动文件夹到垃圾文件夹
                trash_project_folder = trash_folder_path / project_id
                print(f"移动项目文件夹 {project_folder} 到 {trash_project_folder} [CSV第{idx+1}行] (项目ID: {project_id}, "
                      f"content_status: {content_status}, download_status: {download_status})")
                
                # 移动文件夹
                shutil.move(str(project_folder), str(trash_project_folder))
                rows_to_remove.append(idx)
    
    # 分离需要删除的行
    if rows_to_remove:
        removed_df = df.iloc[rows_to_remove].copy()
        removed_dataframes.append(removed_df)
    
    # 获取剩余的行
    remaining_df = df.drop(rows_to_remove).reset_index(drop=True)
    
    # 保存需要删除的行到新的CSV文件
    if separate_deleted_rows and rows_to_remove:
        csv_path_obj = Path(csv_path)
        stem = csv_path_obj.stem
        suffix = csv_path_obj.suffix
        
        # 创建删除行的新CSV文件名
        deleted_csv_path = csv_path_obj.parent / f"{stem}_deleted{suffix}"
        removed_df_combined = pd.concat(removed_dataframes, ignore_index=True)
        removed_df_combined.to_csv(deleted_csv_path, index=False)
        print(f"已保存需要删除的行到 {deleted_csv_path} (共 {len(removed_df_combined)} 行)")
    
    # 保存更新后的剩余CSV
    if separate_deleted_rows:
        csv_path_obj = Path(csv_path)
        stem = csv_path_obj.stem
        suffix = csv_path_obj.suffix
        
        # 检查原始文件名是否包含下划线后的数字，并替换为当前行数
        parts = stem.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            # 替换最后一个部分为当前行数
            new_stem = '_'.join(parts[:-1]) + f"_{len(remaining_df)}"
        else:
            # 如果没有下划线数字，添加行数
            new_stem = f"{stem}_{len(remaining_df)}"
        
        new_csv_path = csv_path_obj.parent / f"{new_stem}{suffix}"
        remaining_df.to_csv(new_csv_path, index=False)
        print(f"已保存更新后的剩余CSV到 {new_csv_path} (共 {len(remaining_df)} 行)")
    else:
        # 如果不需要分离删除行，则覆盖原CSV
        df.to_csv(csv_path, index=False)
        print(f"已更新原CSV文件 {csv_path}")
    
    print(f"处理完成，原始行数: {original_len}, 保留行数: {len(remaining_df)}, 删除行数: {len(pd.concat(removed_dataframes, ignore_index=True)) if rows_to_remove else 0}")


if __name__ == "__main__":
    CSV_PATH = "/home/zlc/crowdfunding/data/metadata/add_11337.csv"
    PROJECTS_BASE_PATH = "/home/zlc/crowdfunding/data/projects/add"
    TRASH_FOLDER_PATH = "/home/zlc/crowdfunding/data/projects/tmp"  # 新增参数，指定垃圾文件夹路径
    CHECK_DOWNLOAD_STATUS = True   # 控制是否要判断download_status不为success
    SEPARATE_DELETED_ROWS = True    # 控制是否要将删除的行单独保存到新CSV中
    
    clean_projects(
        csv_path=CSV_PATH,
        projects_base_path=PROJECTS_BASE_PATH,
        trash_folder_path=TRASH_FOLDER_PATH,
        check_download_status=CHECK_DOWNLOAD_STATUS,
        separate_deleted_rows=SEPARATE_DELETED_ROWS
    )