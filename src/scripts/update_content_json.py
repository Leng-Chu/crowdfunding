import os
import json
import pandas as pd
from pathlib import Path


def find_cover_image_file(project_dir):
    """
    查找项目目录下的封面图片文件，并返回相对路径格式的filename
    """
    cover_dir = os.path.join(project_dir, 'cover')
    if not os.path.exists(cover_dir):
        return None
    
    # 查找cover目录下的所有图片文件
    for file in os.listdir(cover_dir):
        if file.startswith('cover_image'):
            file_path = os.path.join(cover_dir, file)
            # 获取图片尺寸（如果可能）
            try:
                from PIL import Image
                img = Image.open(file_path)
                width, height = img.size
            except Exception:
                # 如果无法获取尺寸，则设为默认值
                width, height = 0, 0
            
            # 返回相对路径格式的filename
            relative_filename = os.path.relpath(file_path, start=os.path.dirname(cover_dir))
            return {
                'url': '',  # 从CSV中获取或者保持原样
                'filename': relative_filename,  # 使用相对路径格式
                'width': width,
                'height': height
            }
    
    return None


def update_content_json(test_csv_path, all_csv_path, projects_base_dir):
    """
    根据CSV信息更新每个项目的content.json文件
    """
    # 读取test.csv和all.csv文件
    test_df = pd.read_csv(test_csv_path)
    all_df = pd.read_csv(all_csv_path)
    
    # 创建一个字典，以便快速查找all_df中的数据
    all_dict = {}
    for _, row in all_df.iterrows():
        # 使用整数类型的project_id作为键，而不是字符串
        all_dict[int(row['project_id'])] = row
    
    # 遍历test.csv中的每一行
    for _, row in test_df.iterrows():
        # 尝试将project_id转换为整数以匹配字典中的键
        try:
            project_id = int(row['project_id'])
        except ValueError:
            print(f"无法转换project_id为整数: {row['project_id']}")
            continue
        
        project_dir = os.path.join(projects_base_dir, str(row['project_id']))
        
        # 检查项目目录是否存在
        if not os.path.exists(project_dir):
            print(f"项目目录不存在: {project_dir}")
            continue
        
        content_json_path = os.path.join(project_dir, 'content.json')
        
        # 检查content.json是否存在
        if not os.path.exists(content_json_path):
            print(f"content.json不存在: {content_json_path}")
            continue
        
        # 从all_df中获取项目详细信息
        if project_id not in all_dict:
            print(f"在all.csv中找不到项目 {project_id} 的详细信息")
            continue
        
        project_details = all_dict[project_id]
        
        # 读取现有的content.json
        with open(content_json_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        
        # 保存原始的cover_image（如果它是字符串形式的URL）
        original_cover_image = None
        if 'cover_image' in content_data and isinstance(content_data['cover_image'], str):
            original_cover_image = content_data['cover_image']
        
        # 创建新的content.json数据结构，按照指定顺序
        new_content_data = {}
        
        # 首先添加基本字段（除了video和content_sequence）
        if 'project_url' in content_data:
            new_content_data['project_url'] = content_data['project_url']
        
        # 然后添加title（如果CSV中有相关列）
        if pd.notna(project_details.get('title')):
            new_content_data['title'] = {
                'content': project_details['title'],
                'content_length': len(str(project_details['title']))
            }
        
        # 然后添加blurb（如果CSV中有相关列）
        if pd.notna(project_details.get('blurb')):
            new_content_data['blurb'] = {
                'content': project_details['blurb'],
                'content_length': len(str(project_details['blurb']))
            }
        
        # 然后更新cover_image信息
        cover_image_info = find_cover_image_file(project_dir)
        if cover_image_info:
            # 使用CSV中的cover_url，如果没有则保留原来的
            if pd.notna(project_details.get('cover_url')) and project_details['cover_url']:
                cover_image_info['url'] = project_details['cover_url']
            elif original_cover_image:
                cover_image_info['url'] = original_cover_image
            
            new_content_data['cover_image'] = cover_image_info
        else:
            # 如果没有找到本地图片文件，则使用CSV中的URL
            url = ''
            if pd.notna(project_details.get('cover_url')) and project_details['cover_url']:
                url = project_details['cover_url']
            elif original_cover_image:
                url = original_cover_image
                
            new_content_data['cover_image'] = {
                'url': url,
                'filename': '',  # 使用相对路径格式
                'width': 0,
                'height': 0
            }
        
        # 然后添加video
        if 'video' in content_data:
            new_content_data['video'] = content_data['video']
        
        # 最后添加content_sequence
        if 'content_sequence' in content_data:
            new_content_data['content_sequence'] = content_data['content_sequence']
        
        # 写回更新后的content.json
        with open(content_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_content_data, f, ensure_ascii=False, indent=2)
        
        #print(f"已更新项目 {project_id} 的content.json")


if __name__ == "__main__":
    test_csv_path = "data/metadata/move2.csv"
    all_csv_path = "data/metadata/all.csv"
    projects_base_dir = "data/projects/move2"
    
    update_content_json(test_csv_path, all_csv_path, projects_base_dir)