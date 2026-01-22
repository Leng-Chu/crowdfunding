# -*- coding: utf-8 -*-
"""
模型定义（seq）：

Chapter 1：图文内容块序列建模（顺序是否重要）

约束：
- 所有 baseline 的 token 输入 X 必须完全一致（不含任何位置信息）
- 顺序信息仅允许通过 position encoding 注入（仅对 trm_pos / trm_pos_shuffled）
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from config import SeqConfig


class MetaMLPEncoder(nn.Module):
    """metadata 特征 -> FC -> Dropout，输出一个定长向量（与 mlp baseline 对齐）。"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim 需要 > 0，但得到 {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 需要 > 0，但得到 {hidden_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.fc = nn.Linear(int(input_dim), int(hidden_dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.output_dim = int(hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc(x))
        z = self.drop(z)
        return z


class TokenEncoder(nn.Module):
    """
    内容块统一表示（不含位置信息）：
    x_i = proj(e_i) + type_embedding(t_i) + attr_proj(a_i)
    """

    def __init__(
        self,
        image_embedding_dim: int,
        text_embedding_dim: int,
        d_model: int,
        token_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_embedding_dim <= 0:
            raise ValueError(f"image_embedding_dim 需要 > 0，但得到 {image_embedding_dim}")
        if text_embedding_dim <= 0:
            raise ValueError(f"text_embedding_dim 需要 > 0，但得到 {text_embedding_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        if token_dropout < 0.0 or token_dropout >= 1.0:
            raise ValueError("token_dropout 需要在 [0, 1) 之间")

        self.img_proj = nn.Linear(int(image_embedding_dim), int(d_model))
        self.txt_proj = nn.Linear(int(text_embedding_dim), int(d_model))
        self.type_emb = nn.Embedding(2, int(d_model))  # 0=text，1=image
        self.attr_proj = nn.Linear(1, int(d_model))
        self.drop = nn.Dropout(p=float(token_dropout))
        self.d_model = int(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.img_proj.weight)
        nn.init.xavier_uniform_(self.txt_proj.weight)
        nn.init.xavier_uniform_(self.attr_proj.weight)
        if self.img_proj.bias is not None:
            nn.init.zeros_(self.img_proj.bias)
        if self.txt_proj.bias is not None:
            nn.init.zeros_(self.txt_proj.bias)
        if self.attr_proj.bias is not None:
            nn.init.zeros_(self.attr_proj.bias)

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        seq_type: torch.Tensor,
        seq_attr: torch.Tensor,
    ) -> torch.Tensor:
        if x_img.ndim != 3 or x_txt.ndim != 3:
            raise ValueError(f"x_img/x_txt 需要为 3 维 (B, L, D)，但得到 {tuple(x_img.shape)} / {tuple(x_txt.shape)}")
        if seq_type.ndim != 2 or seq_attr.ndim != 2:
            raise ValueError(
                f"seq_type/seq_attr 需要为 2 维 (B, L)，但得到 {tuple(seq_type.shape)} / {tuple(seq_attr.shape)}"
            )

        img_mask = (seq_type == 1).unsqueeze(-1).to(dtype=x_img.dtype)
        txt_mask = (seq_type == 0).unsqueeze(-1).to(dtype=x_txt.dtype)

        img_feat = torch.relu(self.img_proj(x_img))
        txt_feat = torch.relu(self.txt_proj(x_txt))
        content = img_feat * img_mask + txt_feat * txt_mask

        type_feat = self.type_emb(seq_type.to(torch.long))
        attr_feat = self.attr_proj(seq_attr.to(dtype=content.dtype).unsqueeze(-1))
        x = content + type_feat + attr_feat
        x = self.drop(x)
        return x


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """masked mean pooling：mask=True 为有效位置。"""
    if x.ndim != 3 or mask.ndim != 2:
        raise ValueError(f"x/mask 形状不合法：x={tuple(x.shape)} mask={tuple(mask.shape)}")
    m = mask.unsqueeze(-1).to(dtype=x.dtype)
    s = torch.sum(x * m, dim=1)
    d = torch.sum(m, dim=1).clamp(min=1.0)
    return s / d


class SetAttentionPooling(nn.Module):
    """单个全局 query 的单头 attention pooling（不含位置信息）。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        self.query = nn.Parameter(torch.zeros(int(d_model)))
        self.scale = float(1.0 / math.sqrt(float(d_model)))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError(f"x/mask 形状不合法：x={tuple(x.shape)} mask={tuple(mask.shape)}")
        scores = torch.einsum("bld,d->bl", x, self.query) * float(self.scale)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class SinusoidalPositionalEncoding(nn.Module):
    """标准 sinusoidal 位置编码。"""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        if max_len <= 0:
            raise ValueError(f"max_len 需要 > 0，但得到 {max_len}")
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")

        pe = torch.zeros(int(max_len), int(d_model), dtype=torch.float32)
        position = torch.arange(int(max_len), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, int(d_model), 2, dtype=torch.float32) * (-math.log(10000.0) / float(d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError(f"x/mask 形状不合法：x={tuple(x.shape)} mask={tuple(mask.shape)}")
        L = int(x.shape[1])
        pos = self.pe[:L].unsqueeze(0).to(device=x.device, dtype=x.dtype)
        return x + pos * mask.unsqueeze(-1).to(dtype=x.dtype)


class SeqBinaryClassifier(nn.Module):
    """序列模型 + 可选 meta 融合，输出 logits。"""

    def __init__(
        self,
        baseline_mode: str,
        use_meta: bool,
        meta_input_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        d_model: int,
        token_dropout: float,
        max_seq_len: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float,
        meta_hidden_dim: int,
        meta_dropout: float,
        fusion_hidden_dim: int,
        fusion_dropout: float,
    ) -> None:
        super().__init__()
        self.baseline_mode = str(baseline_mode or "").strip().lower()
        self.use_meta = bool(use_meta)

        self.token = TokenEncoder(
            image_embedding_dim=int(image_embedding_dim),
            text_embedding_dim=int(text_embedding_dim),
            d_model=int(d_model),
            token_dropout=float(token_dropout),
        )

        self.set_attn_pool = SetAttentionPooling(int(d_model))

        self.transformer: Optional[nn.Module] = None
        self.pos: Optional[nn.Module] = None

        if self.baseline_mode in {"trm_no_pos", "trm_pos", "trm_pos_shuffled"}:
            if int(d_model) % int(transformer_num_heads) != 0:
                raise ValueError(f"d_model={d_model} 必须能被 num_heads={transformer_num_heads} 整除。")
            layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(transformer_num_heads),
                dim_feedforward=int(transformer_dim_feedforward),
                dropout=float(transformer_dropout),
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=int(transformer_num_layers))

        if self.baseline_mode in {"trm_pos", "trm_pos_shuffled"}:
            # 统一使用 sinusoidal 位置编码，不提供可选项。
            self.pos = SinusoidalPositionalEncoding(int(max_seq_len), int(d_model))

        self.meta: Optional[MetaMLPEncoder] = None
        if self.use_meta:
            self.meta = MetaMLPEncoder(
                input_dim=int(meta_input_dim),
                hidden_dim=int(meta_hidden_dim),
                dropout=float(meta_dropout),
            )

        fusion_in_dim = int(d_model) + (int(meta_hidden_dim) if self.use_meta else 0)
        if int(fusion_hidden_dim) <= 0:
            fusion_hidden_dim = int(2 * fusion_in_dim)

        self.fusion_fc = nn.Linear(int(fusion_in_dim), int(fusion_hidden_dim))
        self.fusion_drop = nn.Dropout(p=float(fusion_dropout))
        self.head = nn.Linear(int(fusion_hidden_dim), 1)
        self.fusion_in_dim = int(fusion_in_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fusion_fc.weight)
        if self.fusion_fc.bias is not None:
            nn.init.zeros_(self.fusion_fc.bias)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        seq_type: torch.Tensor,
        seq_attr: torch.Tensor,
        seq_mask: torch.Tensor,
        x_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token(x_img=x_img, x_txt=x_txt, seq_type=seq_type, seq_attr=seq_attr)

        if self.baseline_mode == "set_mean":
            pooled = masked_mean_pool(x, seq_mask)
        elif self.baseline_mode == "set_attn":
            pooled = self.set_attn_pool(x, seq_mask)
        elif self.baseline_mode == "trm_no_pos":
            if self.transformer is None:
                raise RuntimeError("transformer 未初始化。")
            z = self.transformer(x, src_key_padding_mask=~seq_mask)
            pooled = masked_mean_pool(z, seq_mask)
        elif self.baseline_mode in {"trm_pos", "trm_pos_shuffled"}:
            if self.transformer is None or self.pos is None:
                raise RuntimeError("transformer/pos 未初始化。")
            x2 = self.pos(x, seq_mask)
            z = self.transformer(x2, src_key_padding_mask=~seq_mask)
            pooled = masked_mean_pool(z, seq_mask)
        else:
            raise ValueError(
                "不支持的 baseline_mode="
                f"{self.baseline_mode!r}，可选：set_mean/set_attn/trm_no_pos/trm_pos/trm_pos_shuffled"
            )

        feats = [pooled]
        if self.use_meta:
            if self.meta is None or x_meta is None:
                raise ValueError("use_meta=True 但 x_meta 为空。")
            feats.append(self.meta(x_meta))

        fused = torch.cat(feats, dim=1)
        fused = torch.relu(self.fusion_fc(fused))
        fused = self.fusion_drop(fused)
        logits = self.head(fused).squeeze(-1)
        return logits


def build_seq_model(
    cfg: SeqConfig,
    meta_input_dim: int,
    image_embedding_dim: int,
    text_embedding_dim: int,
) -> SeqBinaryClassifier:
    return SeqBinaryClassifier(
        baseline_mode=str(getattr(cfg, "baseline_mode", "set_mean")),
        use_meta=bool(getattr(cfg, "use_meta", False)),
        meta_input_dim=int(meta_input_dim) if bool(getattr(cfg, "use_meta", False)) else 0,
        image_embedding_dim=int(image_embedding_dim),
        text_embedding_dim=int(text_embedding_dim),
        d_model=int(getattr(cfg, "d_model", 256)),
        token_dropout=float(getattr(cfg, "token_dropout", 0.0)),
        max_seq_len=int(getattr(cfg, "max_seq_len", 128)),
        transformer_num_layers=int(getattr(cfg, "transformer_num_layers", 2)),
        transformer_num_heads=int(getattr(cfg, "transformer_num_heads", 4)),
        transformer_dim_feedforward=int(getattr(cfg, "transformer_dim_feedforward", 512)),
        transformer_dropout=float(getattr(cfg, "transformer_dropout", 0.1)),
        meta_hidden_dim=int(getattr(cfg, "meta_hidden_dim", 256)),
        meta_dropout=float(getattr(cfg, "meta_dropout", 0.3)),
        fusion_hidden_dim=int(getattr(cfg, "fusion_hidden_dim", 0)),
        fusion_dropout=float(getattr(cfg, "fusion_dropout", 0.9)),
    )
