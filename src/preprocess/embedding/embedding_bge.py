from pathlib import Path
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np


@torch.no_grad()
def bge_m3_text_embeddings(tokenizer, model, texts, device):
    """
    texts: list[str]
    return: torch.FloatTensor [N, D]
    """
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    outputs = model(**inputs)

    # mean pooling with attention mask
    last_hidden = outputs.last_hidden_state  # [B, L, D]
    mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
    feats = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-12)

    # normalize
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    return feats.cpu()


def vectorize_sequence(
    content_sequence: List[str],
    model_name: str = "BAAI/bge-m3",
    vector_type: str = "text"
) -> List[np.ndarray]:

    if vector_type != "text":
        print("vector_type must be 'text'")
        return []
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    text_embeddings = bge_m3_text_embeddings(
        tokenizer, model, content_sequence, device
    )
    
    return [emb.numpy() for emb in text_embeddings]
