# -*- coding: utf-8 -*-
"""
模型定义（dcan）：

- 固定使用 image + text 两路输入（不提供 use_image/use_text 开关）
- 仅保留 use_meta：meta 分支与 mlp baseline 完全一致，仅在融合阶段 concat，不参与 DCAN 注意力交互

DCAN 结构（baseline）：
1) Modality Projection：Linear -> ReLU，并融合 token 属性（图像面积/文本长度）
2) Query Generator：K/V 生成 + masked mean pooling 得 raw_q + 单头 scaled dot attention 得 refined_q
3) Cross-Attention Blocks：堆叠 L 层，对称更新 img_q / txt_q
4) 融合分类：concat(img_q, txt_q, [meta_h]) -> MLP head 输出 logits
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from config import DcanConfig


def _masked_mean_pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B, L, D]
    mask: [B, L]，True 表示有效位置
    返回: [B, D]
    """
    if x.ndim != 3:
        raise ValueError(f"x 需要是 3 维张量 (B,L,D)，但得到 {tuple(x.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask 需要是 2 维张量 (B,L)，但得到 {tuple(mask.shape)}")
    if x.shape[0] != mask.shape[0] or x.shape[1] != mask.shape[1]:
        raise ValueError(f"x/mask 维度不匹配：x={tuple(x.shape)} mask={tuple(mask.shape)}")

    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B, L, 1]
    summed = (x * mask_f).sum(dim=1)  # [B, D]
    denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return summed / denom


def _single_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    单头 scaled dot-product attention（显式手写，不使用 MultiheadAttention）。

    query: [B, D]
    key/value: [B, L, D]
    mask: [B, L]，True 表示有效位置；padding 在 softmax 前置为 -inf
    返回: ctx [B, D]
    """
    if query.ndim != 2:
        raise ValueError(f"query 需要是 2 维张量 (B,D)，但得到 {tuple(query.shape)}")
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError(f"key/value 需要是 3 维张量 (B,L,D)，但得到 key={tuple(key.shape)} value={tuple(value.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask 需要是 2 维张量 (B,L)，但得到 {tuple(mask.shape)}")
    if key.shape != value.shape:
        raise ValueError(f"key/value 形状不一致：{tuple(key.shape)} vs {tuple(value.shape)}")
    if query.shape[0] != key.shape[0] or key.shape[0] != mask.shape[0] or key.shape[1] != mask.shape[1]:
        raise ValueError(
            f"batch/length 不匹配：query={tuple(query.shape)} key={tuple(key.shape)} mask={tuple(mask.shape)}"
        )
    if query.shape[1] != key.shape[2]:
        raise ValueError(f"维度不匹配：query_dim={query.shape[1]} key_dim={key.shape[2]}")

    d = float(query.shape[1])
    scores = (key * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(max(d, 1.0))  # [B, L]

    # padding 位置 softmax 前置为 -inf
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)  # [B, L]
    attn = torch.nan_to_num(attn, nan=0.0)

    ctx = (attn.unsqueeze(-1) * value).sum(dim=1)  # [B, D]
    return ctx


class MetaMLPEncoder(nn.Module):
    """metadata 特征 -> FC -> Dropout，输出一个定长向量"""

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


class QueryGenerator(nn.Module):
    """从序列中生成 K/V，并得到可用于跨模态检索的全局 query。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")

        self.k_fc = nn.Linear(int(d_model), int(d_model))
        self.v_fc = nn.Linear(int(d_model), int(d_model))
        self.pool_fc = nn.Linear(int(d_model), int(d_model))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, L, d_model]
        mask: [B, L]，True 表示有效位置
        返回：K [B, L, d_model], V [B, L, d_model], refined_q [B, d_model]
        """
        if x.ndim != 3:
            raise ValueError(f"x 需要为 3 维张量 (B,L,D)，但得到 {tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"mask 需要为 2 维张量 (B,L)，但得到 {tuple(mask.shape)}")
        if x.shape[0] != mask.shape[0] or x.shape[1] != mask.shape[1]:
            raise ValueError(f"x/mask 形状不匹配：x={tuple(x.shape)} mask={tuple(mask.shape)}")

        K = torch.relu(self.k_fc(x))
        V = torch.relu(self.v_fc(x))

        pooled_in = torch.relu(self.pool_fc(x))
        raw_q = _masked_mean_pooling(pooled_in, mask=mask)

        refined_q = _single_head_attention(query=raw_q, key=K, value=V, mask=mask)
        return K, V, refined_q


class CrossAttentionBlock(nn.Module):
    """单层 Cross-Attention Block：对称更新 img_q 与 txt_q。"""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.ffn_img = nn.Sequential(nn.Linear(int(d_model), int(d_model)), nn.Dropout(p=float(dropout)))
        self.ffn_txt = nn.Sequential(nn.Linear(int(d_model), int(d_model)), nn.Dropout(p=float(dropout)))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        img_q: torch.Tensor,
        txt_q: torch.Tensor,
        K_img: torch.Tensor,
        V_img: torch.Tensor,
        img_mask: torch.Tensor,
        K_txt: torch.Tensor,
        V_txt: torch.Tensor,
        txt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 文本 -> 图像：用 txt_q 作为 query，对图像 K/V 做 attention
        ctx_img = _single_head_attention(query=txt_q, key=K_img, value=V_img, mask=img_mask)
        img_q = img_q + self.ffn_img(ctx_img)

        # 图像 -> 文本：用更新后的 img_q 作为 query，对文本 K/V 做 attention
        ctx_txt = _single_head_attention(query=img_q, key=K_txt, value=V_txt, mask=txt_mask)
        txt_q = txt_q + self.ffn_txt(ctx_txt)

        return img_q, txt_q


class DCANBinaryClassifier(nn.Module):
    """DCAN 二分类器（输出 logits）。"""

    def __init__(
        self,
        use_meta: bool,
        use_attr: bool,
        meta_input_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        d_model: int = 256,
        num_cross_layers: int = 2,
        cross_ffn_dropout: float = 0.1,
        meta_hidden_dim: int = 256,
        meta_dropout: float = 0.3,
        fusion_dropout: float = 0.9,
    ) -> None:
        super().__init__()

        if image_embedding_dim <= 0:
            raise ValueError(f"image_embedding_dim 需要 > 0，但得到 {image_embedding_dim}")
        if text_embedding_dim <= 0:
            raise ValueError(f"text_embedding_dim 需要 > 0，但得到 {text_embedding_dim}")
        if int(d_model) <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        if int(num_cross_layers) <= 0:
            raise ValueError(f"num_cross_layers 需要 > 0，但得到 {num_cross_layers}")
        if cross_ffn_dropout < 0.0 or cross_ffn_dropout >= 1.0:
            raise ValueError("cross_ffn_dropout 需要在 [0, 1) 之间")
        if fusion_dropout < 0.0 or fusion_dropout >= 1.0:
            raise ValueError("fusion_dropout 需要在 [0, 1) 之间")

        self.use_meta = bool(use_meta)
        self.use_attr = bool(use_attr)
        self.d_model = int(d_model)
        self.num_cross_layers = int(num_cross_layers)

        # 模块 1：Modality Projection
        self.img_proj = nn.Sequential(nn.Linear(int(image_embedding_dim), int(d_model)), nn.ReLU())
        self.txt_proj = nn.Sequential(nn.Linear(int(text_embedding_dim), int(d_model)), nn.ReLU())
        self.img_attr_proj = nn.Linear(1, int(d_model))
        self.txt_attr_proj = nn.Linear(1, int(d_model))
        self.img_ln = nn.LayerNorm(int(d_model))
        self.txt_ln = nn.LayerNorm(int(d_model))

        # 模块 2：Query Generator（每个模态一个）
        self.img_query_gen = QueryGenerator(d_model=int(d_model))
        self.txt_query_gen = QueryGenerator(d_model=int(d_model))

        # 模块 3：Cross-Attention Blocks
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model=int(d_model), dropout=float(cross_ffn_dropout)) for _ in range(int(num_cross_layers))]
        )

        # 模块 4：融合与分类（沿用 mlp baseline 风格）
        fusion_in_dim = int(2 * int(d_model))
        self.meta: Optional[MetaMLPEncoder] = None
        if self.use_meta:
            if int(meta_input_dim) <= 0:
                raise ValueError("use_meta=True 时，meta_input_dim 需要 > 0。")
            self.meta = MetaMLPEncoder(
                input_dim=int(meta_input_dim),
                hidden_dim=int(meta_hidden_dim),
                dropout=float(meta_dropout),
            )
            fusion_in_dim += int(self.meta.output_dim)

        fusion_hidden_dim = int(fusion_in_dim * 2)
        self.fusion_fc = nn.Linear(int(fusion_in_dim), int(fusion_hidden_dim))
        self.fusion_drop = nn.Dropout(p=float(fusion_dropout))
        self.head = nn.Linear(int(fusion_hidden_dim), 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        img_emb: torch.Tensor,
        img_attr: torch.Tensor | None,
        txt_emb: torch.Tensor,
        txt_attr: torch.Tensor | None,
        img_mask: torch.Tensor,
        txt_mask: torch.Tensor,
        meta_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        img_emb: [B, N, D_img]
        img_attr: [B, N]（log(max(1, area))）
        txt_emb: [B, M, D_txt]
        txt_attr: [B, M]（log(max(1, length))）
        img_mask/txt_mask: [B, N]/[B, M]，True 表示有效位置
        meta_features: [B, F]（可选）
        返回 logits: [B]
        """
        if img_emb.ndim != 3 or txt_emb.ndim != 3:
            raise ValueError(f"img_emb/txt_emb 需要是 3 维张量，但得到 {tuple(img_emb.shape)} / {tuple(txt_emb.shape)}")
        if img_mask.ndim != 2 or txt_mask.ndim != 2:
            raise ValueError(f"img_mask/txt_mask 需要是 2 维张量，但得到 {tuple(img_mask.shape)} / {tuple(txt_mask.shape)}")

        if img_emb.shape[0] != txt_emb.shape[0]:
            raise ValueError(f"batch_size 不一致：img={img_emb.shape[0]} txt={txt_emb.shape[0]}")
        if img_emb.shape[0] != img_mask.shape[0] or img_emb.shape[1] != img_mask.shape[1]:
            raise ValueError(f"img_emb/img_mask 形状不匹配：img={tuple(img_emb.shape)} mask={tuple(img_mask.shape)}")
        if txt_emb.shape[0] != txt_mask.shape[0] or txt_emb.shape[1] != txt_mask.shape[1]:
            raise ValueError(f"txt_emb/txt_mask 形状不匹配：txt={tuple(txt_emb.shape)} mask={tuple(txt_mask.shape)}")
        if self.use_attr:
            if img_attr is None or txt_attr is None:
                raise ValueError("use_attr=True 时，img_attr/txt_attr 不能为空。")
            if img_attr.ndim != 2 or txt_attr.ndim != 2:
                raise ValueError(f"img_attr/txt_attr 需要是 2 维张量，但得到 {tuple(img_attr.shape)} / {tuple(txt_attr.shape)}")
            if img_emb.shape[0] != img_attr.shape[0] or img_emb.shape[1] != img_attr.shape[1]:
                raise ValueError(f"img_emb/img_attr 形状不匹配：img={tuple(img_emb.shape)} attr={tuple(img_attr.shape)}")
            if txt_emb.shape[0] != txt_attr.shape[0] or txt_emb.shape[1] != txt_attr.shape[1]:
                raise ValueError(f"txt_emb/txt_attr 形状不匹配：txt={tuple(txt_emb.shape)} attr={tuple(txt_attr.shape)}")

        img_mask = img_mask.to(dtype=torch.bool)
        txt_mask = txt_mask.to(dtype=torch.bool)

        # 1) Projection
        img_feat = self.img_proj(img_emb)
        txt_feat = self.txt_proj(txt_emb)
        if self.use_attr:
            img_feat = img_feat + self.img_attr_proj(img_attr.unsqueeze(-1))  # [B, N, d_model]
            txt_feat = txt_feat + self.txt_attr_proj(txt_attr.unsqueeze(-1))  # [B, M, d_model]
        img_feat = self.img_ln(img_feat)
        txt_feat = self.txt_ln(txt_feat)

        # 2) Query Generator
        K_img, V_img, img_q = self.img_query_gen(img_feat, img_mask)
        K_txt, V_txt, txt_q = self.txt_query_gen(txt_feat, txt_mask)

        # 3) Cross-Attention Blocks
        for block in self.cross_blocks:
            img_q, txt_q = block(
                img_q=img_q,
                txt_q=txt_q,
                K_img=K_img,
                V_img=V_img,
                img_mask=img_mask,
                K_txt=K_txt,
                V_txt=V_txt,
                txt_mask=txt_mask,
            )

        fuse = torch.cat([img_q, txt_q], dim=1)
        if self.use_meta:
            if self.meta is None or meta_features is None:
                raise ValueError("use_meta=True 但 meta_features 为空。")
            meta_h = self.meta(meta_features)
            fuse = torch.cat([fuse, meta_h], dim=1)

        z = torch.relu(self.fusion_fc(fuse))
        z = self.fusion_drop(z)
        logits = self.head(z).squeeze(-1)
        return logits


def build_dcan_model(
    cfg: DcanConfig,
    use_meta: bool,
    meta_input_dim: int,
    image_embedding_dim: int,
    text_embedding_dim: int,
) -> DCANBinaryClassifier:
    return DCANBinaryClassifier(
        use_meta=bool(use_meta),
        use_attr=bool(getattr(cfg, "use_attr", True)),
        meta_input_dim=int(meta_input_dim) if bool(use_meta) else 0,
        image_embedding_dim=int(image_embedding_dim),
        text_embedding_dim=int(text_embedding_dim),
        d_model=int(getattr(cfg, "d_model", 256)),
        num_cross_layers=int(getattr(cfg, "num_cross_layers", 2)),
        cross_ffn_dropout=float(getattr(cfg, "cross_ffn_dropout", 0.1)),
        meta_hidden_dim=int(getattr(cfg, "meta_hidden_dim", 256)),
        meta_dropout=float(getattr(cfg, "meta_dropout", 0.3)),
        fusion_dropout=float(getattr(cfg, "fusion_dropout", 0.5)),
    )


