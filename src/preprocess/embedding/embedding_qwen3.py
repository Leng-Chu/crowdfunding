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


def vectorize_sequence(
    content_sequence: List[str],
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    vector_type: str = "text"
) -> List[np.ndarray]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if vector_type == "text":
        embs = qwen3_vl_text_embeddings(processor, model, content_sequence, device)
        return [e.numpy() for e in embs]

    elif vector_type == "image":
        embs = qwen3_vl_image_embeddings(processor, model, content_sequence, device)
        return [e.numpy() for e in embs]

    else:
        print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
        return []
