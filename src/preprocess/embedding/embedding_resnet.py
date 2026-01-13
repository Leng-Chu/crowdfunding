from pathlib import Path
from typing import List
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

@torch.no_grad()
def resnet_image_embeddings(processor, model, image_paths, device):
    images = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # HuggingFace ResNetModel 一般会给 pooler_output: [N, 2048]
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feats = outputs.pooler_output
    else:
        # 兜底：last_hidden_state 通常是 [N, C, H, W]，做全局平均池化
        x = outputs.last_hidden_state
        feats = x.mean(dim=(-2, -1))

    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu()

def vectorize_sequence(
    content_sequence: List[str],
    model_name: str = "microsoft/resnet-50",
    vector_type: str = "image"
) -> List[np.ndarray]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    if vector_type == "image":
        image_embeddings = resnet_image_embeddings(processor, model, content_sequence, device)
        return [emb.numpy() for emb in image_embeddings]
    else:
        print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
        return []
