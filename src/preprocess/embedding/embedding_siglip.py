from typing import List
import torch
import numpy as np
from PIL import Image
from transformers import SiglipModel, AutoProcessor


@torch.no_grad()
def siglip_text_embeddings(processor, model, texts, device):
    """
    texts: list[str]
    return: torch.FloatTensor [N, D]
    自动处理超过 SigLIP 最大长度的文本：滑窗切块 -> 块向量平均 -> 归一化
    """
    tok = processor.tokenizer
    max_len = tok.model_max_length  # SigLIP 通常比 CLIP 更短（常见 64）
    stride = 32  # 固定滑窗重叠步长（不新增参数，写死在函数内部）

    # 先判定哪些是长文本
    is_long = []
    input_ids_list = []
    for t in texts:
        ids = tok(t, truncation=False, add_special_tokens=False, verbose=False)["input_ids"]
        input_ids_list.append(ids)
        is_long.append(len(ids) > max_len)

    # 1) 短文本：批处理（快）
    short_feats = {}
    if any(not f for f in is_long):
        short_idx = [i for i, flag in enumerate(is_long) if not flag]
        short_texts = [texts[i] for i in short_idx]

        inputs = processor(
            text=short_texts,
            return_tensors="pt",
            padding="max_length",     # SigLIP 推荐
            truncation=True
        ).to(device)

        # 尽量走 get_text_features；如果版本不支持就从 outputs 里拿 text_embeds
        if hasattr(model, "get_text_features"):
            feats = model.get_text_features(**inputs)  # [Ns, D]
        else:
            outputs = model(**inputs)
            feats = outputs.text_embeds  # [Ns, D]

        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feats = feats.cpu()
        for k, i in enumerate(short_idx):
            short_feats[i] = feats[k]

    # 2) 长文本：逐条滑窗切块 -> 块向量平均
    long_feats = {}
    for i, flag in enumerate(is_long):
        if not flag:
            continue

        ids = input_ids_list[i]

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

        feats_list = []
        for chunk_ids, attn in chunks:
            chunk_ids = chunk_ids.to(device)
            attn = attn.to(device)

            # SigLIP 推荐 padding="max_length"：这里手动 pad 到 max_len
            if chunk_ids.size(1) < max_len:
                pad_len = max_len - chunk_ids.size(1)
                pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
                pad = torch.full((1, pad_len), pad_id, device=device, dtype=chunk_ids.dtype)
                chunk_ids = torch.cat([chunk_ids, pad], dim=1)
                attn = torch.cat([attn, torch.zeros_like(pad)], dim=1)

            if hasattr(model, "get_text_features"):
                f = model.get_text_features(input_ids=chunk_ids, attention_mask=attn)  # [1, D]
            else:
                outputs = model(input_ids=chunk_ids, attention_mask=attn)
                f = outputs.text_embeds  # [1, D]

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
def siglip_image_embeddings(processor, model, image_paths, device):
    images = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt").to(device)

    if hasattr(model, "get_image_features"):
        feats = model.get_image_features(**inputs)  # [N, D]
    else:
        outputs = model(**inputs)
        feats = outputs.image_embeds  # [N, D]

    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats.cpu()


def vectorize_sequence(
    content_sequence: List[str],
    model_name: str = "google/siglip-base-patch16-224",
    vector_type: str = "text"
) -> List[np.ndarray]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    if vector_type == "text":
        text_embeddings = siglip_text_embeddings(processor, model, content_sequence, device)
        return [emb.numpy() for emb in text_embeddings]
    elif vector_type == "image":
        image_embeddings = siglip_image_embeddings(processor, model, content_sequence, device)
        return [emb.numpy() for emb in image_embeddings]
    else:
        print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
        return []
