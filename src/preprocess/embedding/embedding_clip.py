from pathlib import Path
from typing import Any, Dict, List
import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image

@torch.no_grad()
def clip_text_embeddings(processor, model, texts, device):
    """
    texts: list[str]
    return: torch.FloatTensor [N, D]
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = model.get_text_features(**inputs)  # [N, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()

@torch.no_grad()
def clip_image_embeddings(processor, model, image_paths, device):
    """
    image_paths: list[str]
    return: torch.FloatTensor [N, D]
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)  # [N, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


def vectorize_sequence(
    project_folder: Path,
    content_sequence: List[Dict[str, Any]],
    model_name: str = "openai/clip-vit-base-patch32"
) -> List[np.ndarray]:
    
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)   

    # 准备用于批量处理的文本和图像列表
    text_items = []  # 存储文本内容及原始索引
    image_items = []  # 存储图像路径及原始索引
    
    # 记录原始顺序的占位符列表
    vector_list: List[np.ndarray] = [None] * len(content_sequence)
    
    # 遍历内容序列，验证内容并记录索引位置
    for idx, item in enumerate(content_sequence):
        item_type = item.get("type", "")
        if item_type == "text":
            text_content = item.get("content", "")
            if not text_content:
                print("文本内容为空，停止处理整个项目。")
                return []
            text_items.append((idx, text_content))

        elif item_type == "image":
            filename = item.get("filename", "")
            if not filename:
                print("图片文件名为空，停止处理整个项目。")
                return []
            image_path = project_folder / Path(filename)
            if not image_path.exists():
                print(f"图片文件不存在: {image_path}，停止处理整个项目。")
                return []
            image_items.append((idx, str(image_path)))

        else:
            print(f"未知的项目类型 '{item_type}'，停止处理整个项目。")
            return []

    # 批量处理文本
    if text_items:
        texts = [item[1] for item in text_items]
        text_embeddings = clip_text_embeddings(processor, model, texts, device)
        
        # 将文本向量按原始顺序放置
        for i, (orig_idx, _) in enumerate(text_items):
            vector_list[orig_idx] = text_embeddings[i].numpy()

    # 批量处理图像
    if image_items:
        image_paths = [item[1] for item in image_items]
        image_embeddings = clip_image_embeddings(processor, model, image_paths, device)
        
        # 将图像向量按原始顺序放置
        for i, (orig_idx, _) in enumerate(image_items):
            vector_list[orig_idx] = image_embeddings[i].numpy()

    return vector_list