import os
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


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        local_snapshot = os.environ.get("BGE_M3_SNAPSHOT", "").strip()

        # 添加use_local参数以支持离线模式
        try:
            source = local_snapshot or model_name
            self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            self.model = AutoModel.from_pretrained(source, local_files_only=True).to(self.device).eval()
        except OSError:
            # 如果本地没有模型，则尝试在线下载
            print(f"本地未找到模型 {model_name}，正在尝试在线获取...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embed_text(self, content_sequence: List[str]) -> List[np.ndarray]:
        text_embeddings = bge_m3_text_embeddings(
            self.tokenizer, self.model, content_sequence, self.device
        )
        return [emb.numpy() for emb in text_embeddings]

    def __call__(self, content_sequence: List[str], vector_type: str = "text") -> List[np.ndarray]:
        if vector_type != "text":
            print("参数 vector_type 必须为 'text'")
            return []
        return self.embed_text(content_sequence)

