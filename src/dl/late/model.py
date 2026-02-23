# -*- coding: utf-8 -*-
"""
模型定义（late）：
- 图像集合与文本集合分别编码
- baseline_mode:
  - mean_pool：masked mean pooling（无位置编码）
  - attn_pool：集合 attention pooling（无位置编码）
  - trm_no_pos：Transformer（无位置编码）
  - trm_pos：Transformer（图/文各自使用模态内序号的 sinusoidal 位置编码）
- 晚期融合（concat）后接二分类头（Linear→ReLU→Dropout→Linear）
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from config import LateConfig


def _normalize_baseline_mode(baseline_mode: str) -> str:
    mode = str(baseline_mode or "").strip().lower()
    if mode not in {"mean_pool", "attn_pool", "trm_no_pos", "trm_pos"}:
        raise ValueError(f"不支持的 baseline_mode={mode!r}，可选：mean_pool/attn_pool/trm_no_pos/trm_pos")
    return mode


class MetaMLPEncoder(nn.Module):
    """meta 特征 -> FC -> Dropout，输出一个定长向量。"""

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


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """lengths -> mask（True=有效，False=padding）。"""
    max_len = int(max_len)
    if max_len <= 0:
        return lengths.new_zeros((int(lengths.shape[0]), 0), dtype=torch.bool)
    lengths = lengths.to(torch.int64)
    lengths = torch.clamp(lengths, min=0, max=max_len)
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    数值稳定的 masked softmax：
    - scores: [B, S]
    - mask:   [B, S]（True=有效）
    返回：weights=[B, S]，对无效位置输出 0；当某行全为无效时，该行全 0。
    """
    if scores.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"masked_softmax 期望 scores/mask 为 2 维，但得到 {scores.ndim} / {mask.ndim}")
    if scores.shape != mask.shape:
        raise ValueError(f"scores/mask 形状不一致：{tuple(scores.shape)} vs {tuple(mask.shape)}")

    mask = mask.to(torch.bool)
    if scores.numel() == 0:
        return scores

    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    max_scores = scores.max(dim=1, keepdim=True).values
    exp = torch.exp(scores - max_scores) * mask.to(scores.dtype)
    denom = exp.sum(dim=1, keepdim=True)
    return exp / (denom + float(eps))


class AttentionPoolingSetEncoder(nn.Module):
    """
    集合编码器：attention pooling（单头，手写实现，不使用任何位置信息）。
    每个模态一个可学习全局 query：q ∈ R^{d_model}。
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 必须在 [0, 1) 范围内")

        self.d_model = int(d_model)
        self.q = nn.Parameter(torch.empty(self.d_model))
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)
        self.drop = nn.Dropout(p=float(dropout))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.q, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        if self.k.bias is not None:
            nn.init.zeros_(self.k.bias)
        if self.v.bias is not None:
            nn.init.zeros_(self.v.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入 x 需要为 3 维张量 (B, S, D)，但得到 {tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"输入 mask 需要为 2 维张量 (B, S)，但得到 {tuple(mask.shape)}")

        B, S, D = x.shape
        if int(D) != int(self.d_model):
            raise ValueError(f"d_model 不一致：期望 {self.d_model}，但输入为 {D}")

        if int(S) <= 0:
            return x.new_zeros((int(B), int(D)))

        x = self.drop(x)
        K = self.k(x)
        V = self.v(x)
        q = self.q.view(1, 1, int(D)).expand(int(B), 1, int(D))
        scores = (K * q).sum(dim=-1) / math.sqrt(float(D))  # [B, S]
        weights = _masked_softmax(scores, mask=mask)  # [B, S]
        pooled = (weights.unsqueeze(-1) * V).sum(dim=1)  # [B, D]
        pooled = self.drop(pooled)
        return pooled


class MeanPoolingSetEncoder(nn.Module):
    """masked mean pooling（不含位置信息）。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入 x 需要为 3 维张量 (B, S, D)，但得到 {tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"输入 mask 需要为 2 维张量 (B, S)，但得到 {tuple(mask.shape)}")

        B, S, D = x.shape
        if int(D) != int(self.d_model):
            raise ValueError(f"d_model 不一致：期望 {self.d_model}，但输入为 {D}")

        if int(S) <= 0:
            return x.new_zeros((int(B), int(D)))

        mask = mask.to(torch.bool)
        all_pad = ~mask.any(dim=1)  # [B]

        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        summed = (x * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom

        if bool(torch.any(all_pad)):
            pooled = torch.where(all_pad.unsqueeze(-1), pooled.new_zeros((int(B), int(D))), pooled)
        return pooled


class SinusoidalPositionalEncoding(nn.Module):
    """标准 sinusoidal 位置编码（按模态内序号，padding 位置不加）。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or mask.ndim != 2:
            raise ValueError(f"x/mask 形状不合法：x={tuple(x.shape)} mask={tuple(mask.shape)}")
        B, S, D = x.shape
        if int(D) != int(self.d_model):
            raise ValueError(f"d_model 不一致：期望 {self.d_model}，但输入为 {D}")
        if int(S) <= 0:
            return x

        mask = mask.to(torch.bool)
        pos = torch.arange(int(S), device=x.device, dtype=torch.float32).unsqueeze(1)  # [S, 1]
        div_term = torch.exp(
            torch.arange(0, int(D), 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / float(D))
        )  # [ceil(D/2)]

        pe = torch.zeros((int(S), int(D)), device=x.device, dtype=torch.float32)  # [S, D]
        pe[:, 0::2] = torch.sin(pos * div_term)
        if int(D) > 1:
            pe[:, 1::2] = torch.cos(pos * div_term[: int(D) // 2])

        pe = pe.unsqueeze(0).to(dtype=x.dtype)  # [1, S, D]
        return x + pe * mask.unsqueeze(-1).to(dtype=x.dtype)


class TransformerNoPosSetEncoder(nn.Module):
    """集合编码器：Transformer Encoder（不使用任何 position encoding），最后做 masked mean pooling。"""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        if n_layers <= 0:
            raise ValueError(f"n_layers 需要 > 0，但得到 {n_layers}")
        if n_heads <= 0:
            raise ValueError(f"n_heads 需要 > 0，但得到 {n_heads}")
        if int(d_model) % int(n_heads) != 0:
            raise ValueError(f"d_model={d_model} 需要能被 n_heads={n_heads} 整除")
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim 需要 > 0，但得到 {ffn_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        # Transformer 使用 pre-LN（norm_first=True），更稳定。
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入 x 需要为 3 维张量 (B, S, D)，但得到 {tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"输入 mask 需要为 2 维张量 (B, S)，但得到 {tuple(mask.shape)}")

        B, S, D = x.shape
        if int(D) != int(self.d_model):
            raise ValueError(f"d_model 不一致：期望 {self.d_model}，但输入为 {D}")

        if int(S) <= 0:
            return x.new_zeros((int(B), int(D)))

        mask = mask.to(torch.bool)
        all_pad = ~mask.any(dim=1)  # [B]

        src_key_padding_mask = ~mask  # True=padding
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, S, D]

        mask_f = mask.to(out.dtype).unsqueeze(-1)
        summed = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom

        if bool(torch.any(all_pad)):
            pooled = torch.where(all_pad.unsqueeze(-1), pooled.new_zeros((int(B), int(D))), pooled)
        return pooled


class TransformerPosSetEncoder(nn.Module):
    """集合编码器：Transformer Encoder（使用 sinusoidal 位置编码），最后做 masked mean pooling。"""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        if n_layers <= 0:
            raise ValueError(f"n_layers 需要 > 0，但得到 {n_layers}")
        if n_heads <= 0:
            raise ValueError(f"n_heads 需要 > 0，但得到 {n_heads}")
        if int(d_model) % int(n_heads) != 0:
            raise ValueError(f"d_model={d_model} 需要能被 n_heads={n_heads} 整除")
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim 需要 > 0，但得到 {ffn_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.pos = SinusoidalPositionalEncoding(int(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入 x 需要为 3 维张量 (B, S, D)，但得到 {tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"输入 mask 需要为 2 维张量 (B, S)，但得到 {tuple(mask.shape)}")

        B, S, D = x.shape
        if int(D) != int(self.d_model):
            raise ValueError(f"d_model 不一致：期望 {self.d_model}，但输入为 {D}")

        if int(S) <= 0:
            return x.new_zeros((int(B), int(D)))

        mask = mask.to(torch.bool)
        all_pad = ~mask.any(dim=1)  # [B]

        x2 = self.pos(x, mask)
        src_key_padding_mask = ~mask  # True=padding
        out = self.encoder(x2, src_key_padding_mask=src_key_padding_mask)  # [B, S, D]

        mask_f = mask.to(out.dtype).unsqueeze(-1)
        summed = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom

        if bool(torch.any(all_pad)):
            pooled = torch.where(all_pad.unsqueeze(-1), pooled.new_zeros((int(B), int(D))), pooled)
        return pooled


from dataclasses import dataclass

@dataclass(frozen=True)
class EncoderSpec:
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    dropout: float

def _build_set_encoder(mode: str, spec: EncoderSpec) -> nn.Module:
    mode = _normalize_baseline_mode(mode)
    if mode == "mean_pool":
        return MeanPoolingSetEncoder(d_model=spec.d_model)
    if mode == "attn_pool":
        return AttentionPoolingSetEncoder(d_model=spec.d_model, dropout=spec.dropout)
    if mode == "trm_no_pos":
        return TransformerNoPosSetEncoder(
            d_model=spec.d_model,
            n_layers=spec.n_layers,
            n_heads=spec.n_heads,
            ffn_dim=spec.ffn_dim,
            dropout=spec.dropout,
        )
    if mode == "trm_pos":
        return TransformerPosSetEncoder(
            d_model=spec.d_model,
            n_layers=spec.n_layers,
            n_heads=spec.n_heads,
            ffn_dim=spec.ffn_dim,
            dropout=spec.dropout,
        )
    raise ValueError(f"不支持的 baseline_mode={mode!r}")

def _assign_encoders(mode: str, spec: EncoderSpec, share: bool) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None]:
    """
    返回 (shared_encoder, img_encoder, txt_encoder)
    - share=True: 仅 shared_encoder 非空
    - share=False: img/txt 分别一套
    """
    if share:
        return _build_set_encoder(mode, spec), None, None
    return None, _build_set_encoder(mode, spec), _build_set_encoder(mode, spec)


class LateFusionBinaryClassifier(nn.Module):
    """图像/文本分别编码 + 晚期融合二分类（输出 logits）。"""

    def __init__(
        self,
        use_meta: bool,
        use_attr: bool,
        meta_input_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        baseline_mode: str = "attn_pool",
        d_model: int = 256,
        token_dropout: float = 0.0,
        share_encoder: bool = True,
        transformer_n_layers: int = 2,
        transformer_n_heads: int = 8,
        transformer_ffn_dim: int = 512,
        transformer_dropout: float = 0.1,
        meta_hidden_dim: int = 256,
        meta_dropout: float = 0.3,
        fusion_hidden_dim: Optional[int] = None,
        fusion_dropout: float = 0.9,
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
        if fusion_dropout < 0.0 or fusion_dropout >= 1.0:
            raise ValueError("fusion_dropout 需要在 [0, 1) 之间")

        self.use_meta = bool(use_meta)
        self.use_attr = bool(use_attr)
        self.baseline_mode = _normalize_baseline_mode(baseline_mode)
        self.d_model = int(d_model)
        self.share_encoder = bool(share_encoder)

        # 3.1 Modality Projection
        self.img_proj = nn.Linear(int(image_embedding_dim), int(self.d_model))
        self.txt_proj = nn.Linear(int(text_embedding_dim), int(self.d_model))
        # 每个 token 的数值属性（文本字数 / 图片面积）可选投影到 d_model
        self.img_attr_proj: Optional[nn.Linear] = (
            nn.Linear(1, int(self.d_model)) if self.use_attr else None
        )
        self.txt_attr_proj: Optional[nn.Linear] = (
            nn.Linear(1, int(self.d_model)) if self.use_attr else None
        )
        self.img_ln = nn.LayerNorm(int(self.d_model))
        self.txt_ln = nn.LayerNorm(int(self.d_model))
        # 在 token 级别对输入表示做 dropout（有效 token 由 mask/Pooling 约束）。
        self.token_drop = nn.Dropout(p=float(token_dropout))

        # 3.2 集合编码（模态内）
        spec = EncoderSpec(
            d_model=int(self.d_model),
            n_layers=int(transformer_n_layers),
            n_heads=int(transformer_n_heads),
            ffn_dim=int(transformer_ffn_dim),
            dropout=float(transformer_dropout),
        )
        self.shared_encoder, self.img_encoder, self.txt_encoder = _assign_encoders(
            mode=self.baseline_mode,
            spec=spec,
            share=bool(share_encoder),
        )

        # meta 分支（可选）
        self.meta: Optional[MetaMLPEncoder] = None
        meta_out_dim = 0
        if self.use_meta:
            self.meta = MetaMLPEncoder(
                input_dim=int(meta_input_dim),
                hidden_dim=int(meta_hidden_dim),
                dropout=float(meta_dropout),
            )
            meta_out_dim = int(self.meta.output_dim)

        # 3.3 晚期融合与分类
        fusion_in_dim = int(self.d_model) * 2 + int(meta_out_dim)
        if fusion_in_dim <= 0:
            raise ValueError("fusion_in_dim 需要 > 0")
        if fusion_hidden_dim is None:
            fusion_hidden_dim = int(fusion_in_dim * 2)
        if int(fusion_hidden_dim) <= 0:
            raise ValueError(f"fusion_hidden_dim 需要 > 0，但得到 {fusion_hidden_dim}")

        self.fusion_in_dim = int(fusion_in_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.fusion_fc = nn.Linear(int(self.fusion_in_dim), int(self.fusion_hidden_dim))
        self.fusion_in_ln = nn.LayerNorm(int(self.fusion_in_dim))
        self.fusion_drop = nn.Dropout(p=float(fusion_dropout))
        self.head = nn.Linear(int(self.fusion_hidden_dim), 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.img_proj.weight)
        nn.init.xavier_uniform_(self.txt_proj.weight)
        if self.img_attr_proj is not None:
            nn.init.xavier_uniform_(self.img_attr_proj.weight)
        if self.txt_attr_proj is not None:
            nn.init.xavier_uniform_(self.txt_attr_proj.weight)
        if self.img_proj.bias is not None:
            nn.init.zeros_(self.img_proj.bias)
        if self.txt_proj.bias is not None:
            nn.init.zeros_(self.txt_proj.bias)
        if self.img_attr_proj is not None and self.img_attr_proj.bias is not None:
            nn.init.zeros_(self.img_attr_proj.bias)
        if self.txt_attr_proj is not None and self.txt_attr_proj.bias is not None:
            nn.init.zeros_(self.txt_attr_proj.bias)

        nn.init.xavier_uniform_(self.fusion_fc.weight)
        if self.fusion_fc.bias is not None:
            nn.init.zeros_(self.fusion_fc.bias)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_meta: torch.Tensor | None = None,
        x_image: torch.Tensor | None = None,
        len_image: torch.Tensor | None = None,
        attr_image: torch.Tensor | None = None,
        x_text: torch.Tensor | None = None,
        len_text: torch.Tensor | None = None,
        attr_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_image is None or x_text is None:
            raise ValueError("x_image/x_text 不能为空。")
        if len_image is None or len_text is None:
            raise ValueError("len_image/len_text 不能为空。")
        if self.use_attr and (attr_image is None or attr_text is None):
            raise ValueError("use_attr=True 时，attr_image/attr_text 不能为空。")

        if x_image.ndim != 3:
            raise ValueError(f"x_image 需要为 3 维 (B, S, D)，但得到 {tuple(x_image.shape)}")
        if x_text.ndim != 3:
            raise ValueError(f"x_text 需要为 3 维 (B, S, D)，但得到 {tuple(x_text.shape)}")
        if self.use_attr:
            if attr_image is None or attr_text is None:
                raise ValueError("use_attr=True 时，attr_image/attr_text 不能为空。")
            if attr_image.ndim != 2:
                raise ValueError(f"attr_image 需要为 2 维 (B, S)，但得到 {tuple(attr_image.shape)}")
            if attr_text.ndim != 2:
                raise ValueError(f"attr_text 需要为 2 维 (B, S)，但得到 {tuple(attr_text.shape)}")

        img_mask = _lengths_to_mask(len_image, max_len=int(x_image.shape[1]))
        txt_mask = _lengths_to_mask(len_text, max_len=int(x_text.shape[1]))

        img_base = torch.relu(self.img_proj(x_image))
        if self.use_attr:
            if self.img_attr_proj is None or attr_image is None:
                raise RuntimeError("use_attr=True 但 img_attr_proj 或 attr_image 不可用。")
            img_attr = self.img_attr_proj(attr_image.to(dtype=img_base.dtype).unsqueeze(-1))
            Img = self.img_ln(img_base + img_attr)
        else:
            Img = self.img_ln(img_base)
        Img = self.token_drop(Img)

        txt_base = torch.relu(self.txt_proj(x_text))
        if self.use_attr:
            if self.txt_attr_proj is None or attr_text is None:
                raise RuntimeError("use_attr=True 但 txt_attr_proj 或 attr_text 不可用。")
            txt_attr = self.txt_attr_proj(attr_text.to(dtype=txt_base.dtype).unsqueeze(-1))
            Txt = self.txt_ln(txt_base + txt_attr)
        else:
            Txt = self.txt_ln(txt_base)
        Txt = self.token_drop(Txt)

        if self.shared_encoder is not None:
            h_img = self.shared_encoder(Img, img_mask)
            h_txt = self.shared_encoder(Txt, txt_mask)
        else:
            if self.img_encoder is None or self.txt_encoder is None:
                raise RuntimeError("img_encoder/txt_encoder 未初始化。")
            h_img = self.img_encoder(Img, img_mask)
            h_txt = self.txt_encoder(Txt, txt_mask)

        fused = torch.cat([h_img, h_txt], dim=1)
        if self.use_meta:
            if self.meta is None or x_meta is None:
                raise ValueError("meta 分支已开启，但 x_meta 为空。")
            fused = torch.cat([fused, self.meta(x_meta)], dim=1)

        fused = self.fusion_in_ln(fused)
        fused = torch.relu(self.fusion_fc(fused))
        fused = self.fusion_drop(fused)
        logits = self.head(fused).squeeze(-1)
        return logits


def build_late_model(
    cfg: LateConfig,
    use_meta: bool,
    meta_input_dim: int,
    image_embedding_dim: int,
    text_embedding_dim: int,
) -> LateFusionBinaryClassifier:
    return LateFusionBinaryClassifier(
        use_meta=bool(use_meta),
        use_attr=bool(getattr(cfg, "use_attr", True)),
        meta_input_dim=int(meta_input_dim) if use_meta else 0,
        image_embedding_dim=int(image_embedding_dim),
        text_embedding_dim=int(text_embedding_dim),
        baseline_mode=str(getattr(cfg, "baseline_mode", "attn_pool")),
        d_model=int(getattr(cfg, "d_model", 256)),
        token_dropout=float(getattr(cfg, "token_dropout", 0.0)),
        share_encoder=bool(getattr(cfg, "share_encoder", True)),
        transformer_n_layers=int(getattr(cfg, "transformer_n_layers", 2)),
        transformer_n_heads=int(getattr(cfg, "transformer_n_heads", 8)),
        transformer_ffn_dim=int(getattr(cfg, "transformer_ffn_dim", 512)),
        transformer_dropout=float(getattr(cfg, "transformer_dropout", 0.1)),
        meta_hidden_dim=int(getattr(cfg, "meta_hidden_dim", 256)),
        meta_dropout=float(getattr(cfg, "meta_dropout", 0.3)),
        fusion_hidden_dim=getattr(cfg, "fusion_hidden_dim", None),
        fusion_dropout=float(getattr(cfg, "fusion_dropout", 0.9)),
    )

