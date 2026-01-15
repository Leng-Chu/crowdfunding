import os
import json
from pathlib import Path
from PIL import Image 
import shutil
import warnings
warnings.filterwarnings("ignore")


def get_image_format_by_pil(image_path):
    """
    使用PIL获取图像的真实格式
    """
    try:
        with Image.open(image_path) as img:
            # 返回PIL识别出的格式，如 'JPEG', 'PNG', 'WEBP' 等
            return img.format
    except Exception as e:
        print(f"Error detecting image format for {image_path}: {e}")
        return None


def rename_cover_image_if_needed(cover_path):
    """
    检查cover目录下的cover_image文件后缀是否正确，如果不正确则修改
    """
    # 查找cover目录下以cover_image开头的文件
    cover_dir = Path(cover_path)
    if not cover_dir.exists() or not cover_dir.is_dir():
        return True
        
    for cover_file in cover_dir.iterdir():
        if cover_file.is_file() and cover_file.name.startswith("cover_image"):
            detected_format = get_image_format_by_pil(cover_file)
            
            if detected_format is not None:
                # 获取正确的扩展名
                correct_extension = "." + detected_format.lower()
                current_extension = cover_file.suffix
                
                # 如果扩展名不匹配，则重命名
                if current_extension != correct_extension:
                    base_name = cover_file.stem  # 文件名不带扩展名
                    new_name = base_name + correct_extension
                    new_path = cover_file.parent / new_name
                    
                    # 检查目标文件是否已经存在
                    if new_path.exists() and new_path != cover_file:
                        print(f"Warning: Target file {new_path} already exists, skipping...")
                        continue
                        
                    # 重命名文件
                    cover_file.rename(new_path)
                    # print(f"Renamed cover image {cover_file} to {new_path}")
                return False
            else:
                print(f"Could not detect image format for {cover_file}")
    
    return True


def rename_image_in_json(projects_base_path):
    """
    遍历项目基础路径下的所有子文件夹，查找content.json文件，
    将其中所有以错误后缀结尾的图片路径改为真实格式的后缀，
    并将对应的文件重命名为正确的扩展名
    同时检查项目目录下的cover/cover_image文件后缀是否正确
    """
    projects_base_path = Path(projects_base_path)
    cnt = 0
    
    # 遍历所有子文件夹
    for folder in projects_base_path.iterdir():
        if not folder.is_dir():
            continue
            
        content_json_path = folder / "content.json"
        
        # 检查是否存在content.json文件
        if not content_json_path.exists():
            continue
        
        # 检查并修正cover目录下的封面图片
        cover_path = folder / "cover"
        
        # 读取content.json文件
        try:
            with open(content_json_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
        except Exception as e:
            print(f"Error reading {content_json_path}: {e}")
            continue
        
        # 标记是否对JSON进行了更改
        json_changed = False
        error_project = rename_cover_image_if_needed(cover_path)
        
        # 遍历content_sequence数组
        if 'content_sequence' in content_data:
            for item in content_data['content_sequence']:
                if item.get('type') == 'image' and 'filename' in item:
                    old_filename = item['filename']
                    old_extension = os.path.splitext(old_filename)[1]

                    # 获取原始文件名（去掉错误的扩展名）
                    base_name = old_filename.rsplit('.', 1)[0]
                    
                    # 获取真实文件路径
                    old_file_path = folder / old_filename
                    
                    if old_file_path.exists():
                        # 使用PIL检测真实格式
                        detected_format = get_image_format_by_pil(old_file_path)
                        
                        # 只有在成功检测到格式且格式与扩展名不匹配时才重命名
                        if detected_format is not None:
                            new_extension = "." + detected_format.lower()
                            
                            if new_extension != old_extension:
                                try:
                                    new_filename = base_name + new_extension
                                    new_file_path = folder / new_filename
                                    
                                    # 检查目标文件是否已经存在
                                    if new_file_path.exists() and new_file_path != old_file_path:
                                        print(f"Warning: Target file {new_file_path} already exists, skipping...")
                                        continue
                                        
                                    old_file_path.rename(new_file_path)
                                    item['filename'] = new_filename
                                    json_changed = True
                                    #print(f"Renamed {old_file_path} to {new_file_path}")
                                except Exception as e:
                                    print(f"Error renaming file {old_file_path}: {e}")
                                    error_project = True
                            # else: 文件扩展名已经是正确的，无需重命名
                        else:
                            error_project = True
                            print(f"Could not detect image format for {old_file_path}, skipping...")
                    else:
                        error_project = True
                        #print(f"Warning: File {old_file_path} does not exist")

        
        # 如果JSON被修改，则保存回文件
        if json_changed:
            try:
                with open(content_json_path, 'w', encoding='utf-8') as f:
                    json.dump(content_data, f, ensure_ascii=False, indent=2)
                #print(f"Updated {content_json_path}")
            except Exception as e:
                print(f"Error writing {content_json_path}: {e}")
        if error_project:
            print(f"Error project: {folder}")
            cnt += 1
            # 删除文件夹
            shutil.rmtree(folder)
    print(f"Error project count: {cnt}")


def main():
    projects_base_path = "/home/zlc/crowdfunding/data/projects/2025"
    rename_image_in_json(projects_base_path)


if __name__ == "__main__":
    main()