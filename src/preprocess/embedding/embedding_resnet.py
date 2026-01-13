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
        feats = outputs.pooler_output.flatten(1)   # [N,2048]
    else:
        x = outputs.last_hidden_state              # [N,2048,H,W]
        feats = x.mean(dim=(-2, -1))               # [N,2048]

    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu()


class EmbeddingModel:
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    @torch.no_grad()
    def embed_image(self, image_paths: List[str]) -> List[np.ndarray]:
        image_embeddings = resnet_image_embeddings(self.processor, self.model, image_paths, self.device)
        return [emb.numpy() for emb in image_embeddings]

    def __call__(self, content_sequence: List[str], vector_type: str = "image") -> List[np.ndarray]:
        if vector_type == "image":
            return self.embed_image(content_sequence)
        else:
            print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
            return []
