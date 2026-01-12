import os
import json
import numpy as np
from pathlib import Path
import dashscope
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import embedding_clip as embedding_clip
import embedding_qwen as embedding_qwen

def _get_vectors_filename(backend: str) -> str:
    if "qwen" in backend:
        return "vectors_qwen.npy"
    if "clip" in backend:
        return "vectors_clip.npy"
    return "vectors.npy"

def process_single_project(project_folder: Path, backend: str) -> bool:
    """处理单个项目"""
    vectors_filename = _get_vectors_filename(backend)
    content_json_path = project_folder / "content.json"
    vectors_path = project_folder / vectors_filename
    
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
        if "qwen" in backend:
            vector_list = embedding_qwen.vectorize_sequence(project_folder, content_sequence)
        elif "clip" in backend:
            vector_list = embedding_clip.vectorize_sequence(project_folder, content_sequence)
        else:
            print(f"未知的向量化后端: {backend}")
            return False
        
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
    """主函数，遍历projects_root中的所有子文件夹，使用多线程处理"""
    dashscope.api_key = "xxx"
    max_workers = 1  
    backend = "clip"
    vectors_filename = _get_vectors_filename(backend)
    projects_root = Path("data/projects/test")
    
    if not projects_root.exists():
        print(f"错误: 目录 {projects_root} 不存在")
        return
    
    # 获取所有项目文件夹
    project_folders = []
    for project_folder in projects_root.iterdir():
        if project_folder.is_dir():
            content_json_path = project_folder / "content.json"
            vectors_path = project_folder / vectors_filename
            
            # 检查content.json是否存在，以及向量文件是否不存在
            if content_json_path.exists() and not vectors_path.exists():
                project_folders.append(project_folder)
    
    if not project_folders:
        print(f"没有找到需要处理的项目")
        return
    
    print(f"开始使用 {max_workers} 个线程处理 {len(project_folders)} 个项目")
    
    processed_count = 0
    error_count = 0

    if max_workers > 1:
        # 使用线程池处理项目
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_project = {
                executor.submit(process_single_project, project_folder, backend): project_folder 
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
                success = process_single_project(project_folder, backend)
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
