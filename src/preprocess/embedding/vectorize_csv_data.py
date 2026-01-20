import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from backends import get_text_backend, get_image_backend 


def _get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


REPO_ROOT = _get_repo_root()

def _get_vectors_filename(model: str, vector_type: str = "text") -> str:
    """根据模型和向量类型生成文件名"""
    return f"{vector_type}_{model}.npy"

def process_single_row(
    project_id: str,
    title: str,
    blurb: str,
    projects_root: Path,
    text_model: str,
    image_model: str
) -> bool:
    """处理单行数据"""
    project_folder = projects_root / project_id
    cover_folder = project_folder / "cover"
    
    # 检查项目文件夹是否存在
    if not project_folder.exists():
        print(f"项目文件夹不存在: {project_folder}")
        return False

    # 检查封面图片是否存在
    cover_image_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        potential_path = cover_folder / f"cover_image{ext}"
        if potential_path.exists():
            cover_image_path = potential_path
            break
    
    if not cover_image_path or not cover_image_path.exists():
        print(f"封面图片不存在: {cover_folder}/cover_image.[jpg|jpeg|png|webp]")
        return False

    try:
        text_success = True
        image_success = True
        
        # 处理标题和摘要向量化
        text_contents = [title, blurb] if title and blurb else ([title] if title else ([blurb] if blurb else []))
        if text_contents:
            # 移除空或只包含空白字符的文本
            text_contents = [content for content in text_contents if content and content.strip()]
            
            if text_contents:
                text_vectors_path = project_folder / _get_vectors_filename(text_model, "title_blurb")
                
                # 检查文本向量文件是否已经存在
                if text_vectors_path.exists():
                    print(f"标题和摘要向量文件已存在，跳过: {text_vectors_path}")
                else:
                    text_backend = get_text_backend(text_model)
                    text_vector_list = text_backend(content_sequence=text_contents, vector_type="text")
                    if text_vector_list:
                        text_vectors_matrix = np.stack(text_vector_list)
                        np.save(text_vectors_path, text_vectors_matrix)
                        # print(f"标题和摘要向量已保存到 {text_vectors_path}")
                    else:
                        text_success = False
                        print(f"标题和摘要向量化失败: {project_id}")
        
        # 处理封面图片向量化
        if cover_image_path:
            image_vectors_path = project_folder / _get_vectors_filename(image_model, "cover_image")
            
            # 检查图像向量文件是否已经存在
            if image_vectors_path.exists():
                print(f"封面图片向量文件已存在，跳过: {image_vectors_path}")
            else:
                image_backend = get_image_backend(image_model)
                image_vector_list = image_backend(content_sequence=[cover_image_path], vector_type="image")
                if image_vector_list:
                    image_vectors_matrix = np.stack(image_vector_list)
                    np.save(image_vectors_path, image_vectors_matrix)
                    # print(f"封面图片向量已保存到 {image_vectors_path}")
                else:
                    image_success = False
                    print(f"封面图片向量化失败: {project_id}")
        
        # 开启的开关都必须成功才算整体成功
        success = text_success and image_success
        if success:
            # print(f"完成项目 {project_id} 的向量化")
            return True
        else:
            print(f"项目 {project_id} 向量化失败")
            return False

    except Exception as e:
        print(f"处理项目 {project_id} 时发生错误: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='根据CSV数据生成向量表示')
    parser.add_argument('--dataset', type=str, default='test', help='数据集名称，默认为 test')
    parser.add_argument('--text-model', type=str, default='siglip', help='文本模型名称，默认为 siglip')
    parser.add_argument('--image-model', type=str, default='siglip', help='图像模型名称，默认为 siglip')
    # CUDA_VISIBLE_DEVICES=3 TRANSFORMERS_OFFLINE=1 python src/preprocess/embedding/vectorize_csv_data.py --dataset 2023 --text-model bge --image-model clip
    args = parser.parse_args()
    
    # 读取CSV文件
    projects_root = REPO_ROOT / "data" / "projects" / args.dataset
    csv_path = REPO_ROOT / "data" / "metadata" / f"{args.dataset}.csv"
    if not csv_path.exists():
        print(f"错误: CSV文件 {csv_path} 不存在")
        return
    
    df = pd.read_csv(csv_path)
    
    # 检查必要的列是否存在
    required_columns = ['project_id', 'title', 'blurb']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件缺少必要列 {col}")
            return
    
    if not projects_root.exists():
        print(f"错误: 数据根目录 {projects_root} 不存在")
        return

    # 获取要处理的项目
    projects_to_process = []
    for _, row in df.iterrows():
        project_id = str(row['project_id'])
        title = str(row['title']) if pd.notna(row['title']) else ""
        blurb = str(row['blurb']) if pd.notna(row['blurb']) else ""
        projects_to_process.append((project_id, title, blurb))

    print(f"共找到 {len(projects_to_process)} 个项目需要处理")

    processed_count = 0
    error_count = 0

    # 顺序处理每个项目
    for idx, (project_id, title, blurb) in enumerate(projects_to_process):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在处理第 {idx + 1}/{len(projects_to_process)} 个项目: {project_id}")        
        try:
            success = process_single_row(
                project_id=project_id,
                title=title,
                blurb=blurb,
                projects_root=projects_root,
                text_model=args.text_model,
                image_model=args.image_model
            )
            if success:
                processed_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"处理项目 {project_id} 时发生异常: {str(e)}")
            error_count += 1

    print(f"\n处理完成! 成功处理 {processed_count} 个项目，{error_count} 个项目出现错误")


if __name__ == "__main__":
    main()
