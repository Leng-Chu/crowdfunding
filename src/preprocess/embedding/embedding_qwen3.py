from typing import List
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image

@torch.no_grad()
def qwen3_vl_text_embeddings(processor, model, texts, device):
    """
    texts: list[str]
    return: torch.FloatTensor [N, D]
    """
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model(**inputs)

    # Qwen3-VL-Embedding-2B 会直接给 embedding
    if hasattr(outputs, "embeddings"):
        feats = outputs.embeddings          # [N, D]
    else:
        raise RuntimeError("Model output does not contain `embeddings`")

    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu()


@torch.no_grad()
def qwen3_vl_image_embeddings(processor, model, image_paths, device):
    images = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))

    inputs = processor(
        images=images,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)

    if hasattr(outputs, "embeddings"):
        feats = outputs.embeddings          # [N, D]
    else:
        raise RuntimeError("Model output does not contain `embeddings`")

    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu()


class EmbeddingModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    @torch.no_grad()
    def embed_text(self, content_sequence: List[str]) -> List[np.ndarray]:
        embs = qwen3_vl_text_embeddings(self.processor, self.model, content_sequence, self.device)
        return [e.numpy() for e in embs]

    @torch.no_grad()
    def embed_image(self, image_paths: List[str]) -> List[np.ndarray]:
        embs = qwen3_vl_image_embeddings(self.processor, self.model, image_paths, self.device)
        return [e.numpy() for e in embs]

    def __call__(self, content_sequence: List[str], vector_type: str = "text") -> List[np.ndarray]:
        if vector_type == "text":
            return self.embed_text(content_sequence)
        elif vector_type == "image":
            return self.embed_image(content_sequence)
        else:
            print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
            return []

