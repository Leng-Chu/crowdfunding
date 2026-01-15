import pandas as pd
from pathlib import Path
import json

def check_content_status(project_folder):
    content_json_path = project_folder / "content.json"
    if not content_json_path.exists():
        return False
    try:
        with open(content_json_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
            
        # 检查content_sequence字段是否存在且长度大于0
        if 'content_sequence' in content_data and isinstance(content_data['content_sequence'], list):
            if len(content_data['content_sequence']) > 0:
                return True
            else:
                return False
    except Exception as e:
        print(f"读取content.json出错: {e}")
        return False
    return False
    
def check_download_status(project_folder, download_video=False):
    content_json_path = project_folder / "content.json"
    if not content_json_path.exists():
        return False, []
    with open(content_json_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    required_paths = []
    cover_image_url = content.get("cover_image")
    video_url = content.get("video")
    if cover_image_url:
        required_paths.append(project_folder / "cover" / "cover_image.jpg")
    if video_url and download_video:
        required_paths.append(project_folder / "cover" / "project_video.mp4")
    for item in content.get("content_sequence", []):
        if item.get("type") != "image":
            continue
        filename = item.get("filename")
        if not filename:
            continue
        required_paths.append(project_folder / filename)
    missing = [str(path) for path in required_paths if not Path(path).exists()]
    return not missing, missing

def update_status(csv_path, projects_base_path, update_content_status=True, update_download_status=True, download_video=False):
    """
    根据实际检查结果更新CSV中的状态字段
    
    :param csv_path: CSV文件路径
    :param projects_base_path: 项目基础路径
    :param update_content_status: 是否更新content_status字段
    :param update_download_status: 是否更新download_status字段
    :param download_video: 检查下载状态时是否包括视频
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"开始更新CSV文件: {csv_path}，共{len(df)}行数据")
    
    # 初始化新的状态列（如果不存在）
    if update_content_status and 'content_status' not in df.columns:
        df['content_status'] = ''
    if update_download_status and 'download_status' not in df.columns:
        df['download_status'] = ''
    
    # 统计更新情况
    content_updated = 0
    download_updated = 0
    
    # 遍历每一行
    for idx, row in df.iterrows():
        project_id = str(row.get('project_id', ''))  # 确保project_id为字符串
        
        # 检查是否存在project_id对应的子文件夹
        project_folder = Path(projects_base_path) / project_id
        if not project_folder.exists():
            print(f"项目文件夹不存在 (项目ID: {project_id})")
            continue
            
        # 使用实际检查函数来检查项目状态
        actual_content_status = check_content_status(project_folder)
        actual_download_status, _ = check_download_status(project_folder, download_video=download_video)
        
        # 更新CSV中的状态字段
        if update_content_status:
            old_content_status = str(row.get('content_status', ''))
            new_content_status = 'success' if actual_content_status else 'failed'
            
            if old_content_status != new_content_status:
                df.at[idx, 'content_status'] = new_content_status
                content_updated += 1
                print(f"[CSV第{idx+1}行] 项目ID: {project_id} content_status: '{old_content_status}' -> '{new_content_status}'")
        
        if update_download_status:
            old_download_status = str(row.get('download_status', ''))
            new_download_status = 'success' if actual_download_status else 'failed'
            
            if old_download_status != new_download_status:
                df.at[idx, 'download_status'] = new_download_status
                download_updated += 1
                print(f"[CSV第{idx+1}行] 项目ID: {project_id} download_status: '{old_download_status}' -> '{new_download_status}'")

    # 保存更新后的CSV
    csv_path_obj = Path(csv_path)
    original_path = csv_path_obj
    
    # 生成新的文件名，保留原文件名的基本部分
    stem = csv_path_obj.stem
    suffix = csv_path_obj.suffix
    
    # 添加"_updated"标识
    new_stem = f"{stem}_updated"
    
    new_csv_path = csv_path_obj.parent / f"{new_stem}{suffix}"
    df.to_csv(new_csv_path, index=False)
    print(f"已保存更新后的CSV到 {new_csv_path}")
    
    print(f"更新完成:")
    print(f"- content_status 更新了 {content_updated} 项")
    print(f"- download_status 更新了 {download_updated} 项")
    print(f"- 原始行数: {len(df)}, 文件保存至: {new_csv_path}")


if __name__ == "__main__":
    CSV_PATH = "/home/zlc/crowdfunding/data/metadata/add_8266.csv"
    PROJECTS_BASE_PATH = "/home/zlc/crowdfunding/data/projects/add"
    UPDATE_CONTENT_STATUS = True      # 控制是否更新content_status
    UPDATE_DOWNLOAD_STATUS = True     # 控制是否更新download_status
    DOWNLOAD_VIDEO = False            # 控制检查下载状态时是否包括视频
    
    update_status(
        csv_path=CSV_PATH,
        projects_base_path=PROJECTS_BASE_PATH,
        update_content_status=UPDATE_CONTENT_STATUS,
        update_download_status=UPDATE_DOWNLOAD_STATUS,
        download_video=DOWNLOAD_VIDEO
    )