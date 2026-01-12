import os
import json
import numpy as np
from pathlib import Path
import dashscope
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import embedding_clip, embedding_qwen, embedding_bge

def _get_vectors_filename(model: str, vector_type: str = "text") -> str:
    """根据后端和向量类型生成文件名"""
    return f"{vector_type}_{model}.npy"

def validate_content_item(item: Dict[str, Any], project_folder: Path) -> bool:
    """验证内容项的有效性"""
    item_type = item.get("type")
    
    if item_type == "text":
        content = item.get("content")
        if not content or not content.strip():
            print(f"文本内容为空，停止处理整个项目。")
            return False
    elif item_type == "image":
        filename = item.get("filename")
        if not filename or not filename.strip():
            print(f"图片文件名为空，停止处理整个项目。")
            return False
        
        img_path = project_folder / Path(filename)
        if not img_path.exists():
            print(f"图片文件不存在: {img_path}，停止处理整个项目。")
            return False
    
    return True

def get_text_backend(text_model: str) -> str:
    """根据模型名称返回对应的后端"""
    if "qwen" in text_model:
        return embedding_qwen.vectorize_sequence
    elif "clip" in text_model:
        return embedding_clip.vectorize_sequence
    elif "bge" in text_model:
        return embedding_bge.vectorize_sequence
    else:
        print(f"未知的文本向量化后端: {text_model}")
        return None
    
def get_image_backend(image_model: str) -> str:
    """根据模型名称返回对应的后端"""
    if "qwen" in image_model:
        return embedding_qwen.vectorize_sequence
    elif "clip" in image_model:
        return embedding_clip.vectorize_sequence
    else:
        print(f"未知的图像向量化后端: {image_model}")
        return None

def process_single_project(project_folder: Path, 
                           text_model: str, image_model: str, 
                           enable_text_vector: bool = True, 
                           enable_image_vector: bool = True) -> bool:
    """处理单个项目"""
    content_json_path = project_folder / "content.json"
    print(f"正在处理项目: {project_folder.name}")

    try:
        # 读取content.json文件
        with open(content_json_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        content_sequence = content_data.get("content_sequence", [])
        if not content_sequence:
            print(f"跳过 {project_folder.name}, 因为 content_sequence 为空")
            return False
        
        # 验证所有内容项
        for item in content_sequence:
            if not validate_content_item(item, project_folder):
                return False
    
        # 根据开关分别处理文本和图像向量
        text_success = True
        image_success = True
        
        if enable_text_vector:
            text_contents = [item["content"] for item in content_sequence if item.get("type") == "text"]
            if text_contents:
                text_backend = get_text_backend(text_model)
                text_vector_list = text_backend(content_sequence = text_contents, vector_type = "text")
                if text_vector_list:
                    text_vectors_path = project_folder / _get_vectors_filename(text_model, "text")
                    text_vectors_matrix = np.stack(text_vector_list)
                    np.save(text_vectors_path, text_vectors_matrix)
                    # print(f"文本向量已保存到 {text_vectors_path}")
                else:
                    text_success = False
        
        if enable_image_vector:
            # 提取图像内容并转换为完整路径
            image_paths = [(project_folder / Path(item["filename"])).resolve() 
                                for item in content_sequence if item.get("type") == "image"]
            image_backend = get_image_backend(image_model)
            if image_paths:
                image_vector_list = image_backend(content_sequence = image_paths, vector_type = "image")
                if image_vector_list:
                    image_vectors_path = project_folder / _get_vectors_filename(image_model, "image")
                    image_vectors_matrix = np.stack(image_vector_list)
                    np.save(image_vectors_path, image_vectors_matrix)
                    # print(f"图像向量已保存到 {image_vectors_path}")
                else:
                    image_success = False
        
        # 开启的开关都必须成功才算整体成功
        success = (not enable_text_vector or text_success) and (not enable_image_vector or image_success)
        if success:
            # print(f"完成项目 {project_folder.name} 的向量化")
            return True
        else:
            print(f"项目 {project_folder.name} 向量化失败")
            return False
        
    except Exception as e:
        print(f"处理项目 {project_folder.name} 时发生错误: {str(e)}")
        return False


def main():
    """主函数，遍历projects_root中的所有子文件夹，使用多线程处理"""
    dashscope.api_key = "xxx"
    max_workers = 1  
    
    # 添加开关控制
    enable_text_vector = True  # 控制是否生成文本向量
    enable_image_vector = True  # 控制是否生成图像向量
    text_model = "bge"
    image_model = "clip"

    projects_root = Path("data/projects/test")
    
    if not projects_root.exists():
        print(f"错误: 目录 {projects_root} 不存在")
        return
    
    # 获取所有项目文件夹
    project_folders = [folder for folder in projects_root.iterdir() if folder.is_dir()]
    
    if not project_folders:
        print(f"没有找到需要处理的项目")
        return
    
    print(f"开始使用 {max_workers} 个线程处理 {len(project_folders)} 个项目")
    print(f"文本向量生成: {'开启' if enable_text_vector else '关闭'}, "
          f"图像向量生成: {'开启' if enable_image_vector else '关闭'}")
    
    processed_count = 0
    error_count = 0

    if max_workers > 1:
        # 使用线程池处理项目
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_project = {
                executor.submit(process_single_project, project_folder, text_model, image_model,
                               enable_text_vector, enable_image_vector): project_folder 
                for project_folder in project_folders
            }
            
            # 等待所有任务完成
            for future in as_completed(future_to_project):
                project_folder = future_to_project[future]
                try:
                    success = future.result()
                    if success:
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"处理项目 {project_folder.name} 时发生异常: {str(e)}")
                    error_count += 1
    else:
        for project_folder in project_folders:
            try:
                success = process_single_project(project_folder, text_model, image_model,
                                           enable_text_vector, enable_image_vector)
                if success:
                    processed_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"处理项目 {project_folder.name} 时发生异常: {str(e)}")
                error_count += 1
    
    print(f"\n处理完成! 成功处理 {processed_count} 个项目，{error_count} 个项目出现错误")


if __name__ == "__main__":
    main()