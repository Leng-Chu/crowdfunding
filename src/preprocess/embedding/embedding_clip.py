from pathlib import Path
from typing import List
import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image

@torch.no_grad()
def clip_text_embeddings(processor, model, texts, device):
    """
    texts: list[str]
    return: torch.FloatTensor [N, D]
    自动处理超过 CLIP 最大长度的文本：滑窗切块 -> 块向量平均 -> 归一化
    """
    tok = processor.tokenizer
    max_len = tok.model_max_length  # CLIP 通常是 77
    stride = 32  # 固定滑窗重叠步长（不新增参数，写死在函数内部）

    # 先判定哪些是长文本
    is_long = []
    input_ids_list = []
    for t in texts:
        ids = tok(t, truncation=False, add_special_tokens=True)["input_ids"]
        input_ids_list.append(ids)
        is_long.append(len(ids) > max_len)

    # 1) 短文本：保持你原来的批处理路径（快）
    short_idx = [i for i, flag in enumerate(is_long) if not flag]
    short_feats = {}
    if short_idx:
        short_texts = [texts[i] for i in short_idx]
        inputs = processor(text=short_texts, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        feats = model.get_text_features(**inputs)  # [Ns, D]
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feats = feats.cpu()
        for k, i in enumerate(short_idx):
            short_feats[i] = feats[k]

    # 2) 长文本：逐条滑窗切块 -> 块向量平均（稳）
    long_feats = {}
    for i, flag in enumerate(is_long):
        if not flag:
            continue

        ids = input_ids_list[i]
        # 生成滑窗 chunks（token id）
        chunks = []
        start = 0
        while start < len(ids):
            end = min(start + max_len, len(ids))
            chunk_ids = torch.tensor(ids[start:end], dtype=torch.long).unsqueeze(0)  # [1, L]
            attn = torch.ones_like(chunk_ids)
            chunks.append((chunk_ids, attn))
            if end == len(ids):
                break
            start += max_len - stride

        # 逐块求 embedding
        feats_list = []
        for chunk_ids, attn in chunks:
            chunk_ids = chunk_ids.to(device)
            attn = attn.to(device)
            f = model.get_text_features(input_ids=chunk_ids, attention_mask=attn)  # [1, D]
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            feats_list.append(f[0])

        feats_stack = torch.stack(feats_list, dim=0)         # [K, D]
        f = feats_stack.mean(dim=0, keepdim=True)            # [1, D]
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        long_feats[i] = f[0].cpu()

    # 3) 按原输入顺序组装输出 [N, D]
    out = []
    for i in range(len(texts)):
        if i in short_feats:
            out.append(short_feats[i])
        else:
            out.append(long_feats[i])
    return torch.stack(out, dim=0)

@torch.no_grad()
def clip_image_embeddings(processor, model, image_paths, device):
    images = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)  # [N, D]
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu()

def vectorize_sequence(
    content_sequence: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
    vector_type: str = "text"
) -> List[np.ndarray]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    if vector_type == "text":
        text_embeddings = clip_text_embeddings(processor, model, content_sequence, device)
        return [emb.numpy() for emb in text_embeddings]
    elif vector_type == "image":
        image_embeddings = clip_image_embeddings(processor, model, content_sequence, device)
        return [emb.numpy() for emb in image_embeddings]
    else:
        print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
        return []
