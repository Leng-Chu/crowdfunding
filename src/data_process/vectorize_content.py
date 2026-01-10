import os
import json
import numpy as np
from pathlib import Path
import dashscope
import base64
from http import HTTPStatus
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def encode_image_to_base64(image_path: str) -> str:
    """将本地图片文件编码为base64格式"""
    image_format = 'jpg'
    
    with open(image_path, "rb") as image_file:
        # 读取文件并转换为Base64
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 返回data URI格式的数据
    image_data = f"data:image/{image_format};base64,{base64_image}"
    return image_data


def vectorize_content_sequence(project_folder, content_sequence: List[Dict[str, Any]]) -> List[np.ndarray]:
    """使用qwen2.5-vl-embedding模型对content_sequence进行向量化"""
    vector_list = []
    
    for idx, item in enumerate(content_sequence):
        #print(f"正在处理序列中的第 {idx+1}/{len(content_sequence)} 项...")
        
        # 准备输入数据
        input_data = []
        
        # 根据item类型处理内容
        item_type = item.get("type", "")
        
        if item_type == "text":
            # 处理文本内容
            text_content = item.get("content", "")
            if text_content:
                input_data.append({"text": text_content})
            else:
                print(f"项目中有文本内容为空，停止处理整个项目")
                return []  # 整个项目不处理
                
        elif item_type == "image":
            # 处理图片内容 - 使用本地文件路径
            filename = item.get("filename", "")
            if filename:
                # 将相对路径转换为绝对路径
                image_path = project_folder / Path(filename)
                if image_path.exists():
                    # 将本地图片编码为base64格式
                    image_data = encode_image_to_base64(str(image_path))
                    input_data.append({"image": image_data})
                else:
                    print(f"项目中图片文件 {image_path} 不存在，停止处理整个项目")
                    return []  # 整个项目不处理
            else:
                print(f"项目中有图片filename为空，停止处理整个项目")
                return []  # 整个项目不处理
        else:
            print(f"未知的项目类型 '{item_type}'，停止处理整个项目")
            return []  # 整个项目不处理
        
        # 调用多模态向量模型
        try:
            response = dashscope.MultiModalEmbedding.call(
                model="qwen2.5-vl-embedding",
                input=input_data,
                parameters={
                    'dimension': 1024  # 使用1024维向量
                }
            )
            
            if response.status_code == HTTPStatus.OK:
                # 提取向量数据
                embeddings = response.output['embeddings']
                
                # 提取向量并转换为numpy数组
                for emb in embeddings:
                    vector_array = np.array(emb['embedding'], dtype=np.float32)
                    vector_list.append(vector_array)
            else:
                print(f"向量化失败，状态码: {response.status_code}, 错误信息: {getattr(response, 'message', '')}")
                return []  # 整个项目不处理
                
        except Exception as e:
            print(f"向量化过程中发生错误: {str(e)}")
            return []  # 整个项目不处理
    
    return vector_list


def process_single_project(project_folder: Path) -> bool:
    """处理单个项目"""
    content_json_path = project_folder / "content.json"
    vectors_path = project_folder / "vectors.npy"
    
    # 检查是否已经存在vectors.npy文件
    if vectors_path.exists():
        print(f"跳过 {project_folder.name}, 因为 {vectors_path} 已存在")
        return True
    
    if not content_json_path.exists():
        print(f"跳过 {project_folder.name}, 因为 content.json 不存在")
        return False
    
    print(f"正在处理项目: {project_folder.name}")
    
    try:
        # 读取content.json文件
        with open(content_json_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        
        # 获取content_sequence
        content_sequence = content_data.get("content_sequence", [])
        
        if not content_sequence:
            print(f"跳过 {project_folder.name}, 因为 content_sequence 为空")
            return False
        
        # 对content_sequence进行向量化
        vector_list = vectorize_content_sequence(project_folder, content_sequence)
        
        # 只有当所有元素都成功向量化时才保存文件
        if vector_list:  # 如果向量化成功且有结果
            # 将所有向量堆叠成一个矩阵
            vectors_matrix = np.stack(vector_list)
            np.save(vectors_path, vectors_matrix)
            print(f"完成项目 {project_folder.name} 的向量化，向量已保存到 {vectors_path}")
            return True
        else:
            print(f"项目 {project_folder.name} 向量化失败，未生成向量文件")
            return False
        
    except Exception as e:
        print(f"处理项目 {project_folder.name} 时发生错误: {str(e)}")
        return False


def main():
    """主函数，遍历data/projects/2025中的所有项目文件夹，使用多线程处理"""
    # 设置dashscope API key
    dashscope.api_key = "sk-9744aae8ea1e4ca88d2307c6c7e84600"
    # 配置并发参数
    max_workers = 2  
    # 遍历projects_root中的所有子文件夹
    projects_root = Path("data/projects/test")
    
    if not projects_root.exists():
        print(f"错误: 目录 {projects_root} 不存在")
        return
    
    # 获取所有项目文件夹
    project_folders = []
    for project_folder in projects_root.iterdir():
        if project_folder.is_dir():
            content_json_path = project_folder / "content.json"
            vectors_path = project_folder / "vectors.npy"
            
            # 检查content.json是否存在，以及vectors.npy是否不存在
            if content_json_path.exists() and not vectors_path.exists():
                project_folders.append(project_folder)
    
    if not project_folders:
        print(f"没有找到需要处理的项目（没有vectors.npy文件的项目）")
        return
    
    print(f"开始使用 {max_workers} 个线程处理 {len(project_folders)} 个项目")
    
    processed_count = 0
    error_count = 0
    
    # 使用线程池处理项目
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_project = {
            executor.submit(process_single_project, project_folder): project_folder 
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
    
    print(f"\n处理完成! 成功处理 {processed_count} 个项目，{error_count} 个项目出现错误")


if __name__ == "__main__":
    main()