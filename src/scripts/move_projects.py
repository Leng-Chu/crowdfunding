import os
import csv
import shutil

def move_projects_from_later_rows():
    # 定义路径
    csv_file_path = "data/metadata/years/2024_20752.csv"
    source_folder = "data/projects/2024"
    target_folder = "data/projects/2024_5001"
    begin = 5001
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    
    # 读取CSV文件，获取第begin+1行及之后的所有project_id
    project_ids = set()  # 使用集合避免重复
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row_num = 0
        for row in reader:
            row_num += 1
            # 如果行号大于begin，则记录project_id
            if row_num >= begin:
                project_id = row['project_id']
                project_ids.add(project_id)
    
    print(f"Found {len(project_ids)} unique project IDs from rows >= begin:")
    for pid in list(project_ids)[:10]:  # 只显示前10个作为示例
        print(f"  - {pid}")
    if len(project_ids) > 10:
        print(f"  ... and {len(project_ids) - 10} more")
    
    # 移动对应的文件夹
    moved_count = 0
    for project_id in project_ids:
        source_path = os.path.join(source_folder, str(project_id))
        target_path = os.path.join(target_folder, str(project_id))
        
        if os.path.exists(source_path):
            if os.path.exists(target_path):
                print(f"Warning: Target directory {target_path} already exists. Skipping {project_id}.")
            else:
                shutil.move(source_path, target_path)
                print(f"Moved {source_path} to {target_path}")
                moved_count += 1
    
    print(f"\nSuccessfully moved {moved_count} directories.")
    print(f"Total unique project IDs from rows >= begin: {len(project_ids)}")

if __name__ == "__main__":
    move_projects_from_later_rows()