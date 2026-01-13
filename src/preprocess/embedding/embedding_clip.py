from pathlib import Path
from typing import List
import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image

@torch.no_grad()
def clip_text_embeddings(processor, model, texts, device):
    tok = processor.tokenizer

    # 用模型的硬上限更可靠（CLIP=77）
    max_len = int(getattr(model.config, "max_position_embeddings", tok.model_max_length))
    # 给 BOS/EOS 留位置
    content_max = max_len - 2
    stride = 32
    stride = min(stride, content_max - 1)  # 防止 stride 太大导致死循环

    # 拿 special token id（不同实现命名可能不同）
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    if bos_id is None or eos_id is None:
        # CLIP tokenizer 通常有
        raise ValueError("Tokenizer missing BOS/EOS token id; cannot safely chunk for CLIP.")

    out_feats = [None] * len(texts)

    # 短文本批处理（用 truncation=True 保证不越界）
    # 这里用 tokenizer 先判断长度：不加 special 更直观
    lengths = []
    content_ids_list = []
    for t in texts:
        content_ids = tok(t, truncation=False, add_special_tokens=False, verbose=False)["input_ids"]
        content_ids_list.append(content_ids)
        lengths.append(len(content_ids))

    short_idx = [i for i, L in enumerate(lengths) if L <= content_max]
    if short_idx:
        short_texts = [texts[i] for i in short_idx]
        inputs = processor(text=short_texts, return_tensors="pt",
                           padding=True, truncation=True, max_length=max_len).to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feats = feats.cpu()
        for k, i in enumerate(short_idx):
            out_feats[i] = feats[k]

    # 长文本逐条切块（对 content_ids 切）
    long_idx = [i for i in range(len(texts)) if i not in short_idx]
    for i in long_idx:
        content_ids = content_ids_list[i]

        feats_list = []
        start = 0
        while start < len(content_ids):
            end = min(start + content_max, len(content_ids))
            chunk_content = content_ids[start:end]
            # 手动加 BOS/EOS，保证总长<=max_len
            chunk_ids = [bos_id] + chunk_content + [eos_id]
            # 保险：再截一次
            chunk_ids = chunk_ids[:max_len]

            chunk_ids = torch.tensor(chunk_ids, dtype=torch.long).unsqueeze(0).to(device)
            attn = torch.ones_like(chunk_ids)
            f = model.get_text_features(input_ids=chunk_ids, attention_mask=attn)
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            feats_list.append(f[0])

            if end == len(content_ids):
                break
            start += (content_max - stride)

        feats_stack = torch.stack(feats_list, dim=0)
        f = feats_stack.mean(dim=0, keepdim=True)
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        out_feats[i] = f[0].cpu()

    return torch.stack(out_feats, dim=0)


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


class EmbeddingModel:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    @torch.no_grad()
    def embed_text(self, content_sequence: List[str]) -> List[np.ndarray]:
        text_embeddings = clip_text_embeddings(self.processor, self.model, content_sequence, self.device)
        return [emb.numpy() for emb in text_embeddings]

    @torch.no_grad()
    def embed_image(self, image_paths: List[str]) -> List[np.ndarray]:
        image_embeddings = clip_image_embeddings(self.processor, self.model, image_paths, self.device)
        return [emb.numpy() for emb in image_embeddings]

    def __call__(self, content_sequence: List[str], vector_type: str = "text") -> List[np.ndarray]:
        if vector_type == "text":
            return self.embed_text(content_sequence)
        elif vector_type == "image":
            return self.embed_image(content_sequence)
        else:
            print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
            return []

