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