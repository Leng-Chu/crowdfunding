# -*- coding: utf-8 -*-
"""
模型定义（res）：

三路分支（与需求一致）：
- v_meta：metadata 表格特征（one-hot + 数值标准化，来自 now_processed.csv）
- h_seq：正文 content_sequence 的序列表示（仅正文 token；不包含 title_blurb/cover_image token）
- v_key：第一印象表示（title_blurb + cover_image embedding 的融合；缺失用零向量由 data.py 处理并计数）

两种 baseline_mode：
1) baseline_mode="mlp"
   logit = Head( concat( LN(h_seq), LN(meta_proj(v_meta)), LN(key_proj(v_key)) ) )
2) baseline_mode="res"
   z_base = MLP_base( concat( LN(h_seq), LN(meta_proj(v_meta)) ) )
   z_res  = MLP_prior( concat( LN(key_proj(v_key)), LN(meta_proj(v_meta)), LN(key_proj(v_key)⊙meta_proj(v_meta)) ) )
   logit  = z_base + delta_scale * z_res

其中：
- meta_proj/key_proj：把向量投影到与 h_seq 相同维度 d（若已为 d 则 identity）
- delta_scale：可学习标量，初始化为 0（或 1e-3）
- 关键：MLP_base 的结构必须与 src/dl/seq/model.py 的融合头一致（LN -> Linear -> ReLU -> Dropout -> Linear(1)）。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from config import ResConfig


def _init_linear(m: nn.Linear) -> None:
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


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
        self.ln = nn.LayerNorm(int(d_model))
        self.drop = nn.Dropout(p=float(token_dropout))
        self.d_model = int(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_linear(self.img_proj)
        _init_linear(self.txt_proj)
        _init_linear(self.attr_proj)

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
        x = self.ln(x)
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


class SinusoidalPositionalEncoding(nn.Module):
    """标准 sinusoidal 位置编码（固定）。"""

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


class FirstImpressionEncoder(nn.Module):
    """
    FirstImpressionEncoder（沿用 gate 的做法）：

    输入：
    - title_blurb: [B, 2, D_txt]（顺序固定 [title, blurb]）
    - cover: [B, D_img]

    输出：
    - v_key: [B, d]
    """

    def __init__(
        self,
        image_embedding_dim: int,
        text_embedding_dim: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_embedding_dim <= 0:
            raise ValueError(f"image_embedding_dim 需要 > 0，但得到 {image_embedding_dim}")
        if text_embedding_dim <= 0:
            raise ValueError(f"text_embedding_dim 需要 > 0，但得到 {text_embedding_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.text_proj = nn.Linear(int(text_embedding_dim), int(d_model))
        self.cover_proj = nn.Linear(int(image_embedding_dim), int(d_model))
        self.text_ln = nn.LayerNorm(int(d_model))
        self.cover_ln = nn.LayerNorm(int(d_model))
        self.c1_ln = nn.LayerNorm(int(d_model))
        self.drop = nn.Dropout(p=float(dropout))

        self.alpha_fc = nn.Linear(int(3 * d_model), 1)
        self.beta_fc = nn.Linear(int(3 * d_model), 1)

        self.text_agg_fc = nn.Linear(int(2 * d_model), int(d_model))
        self.text_agg_ln = nn.LayerNorm(int(d_model))

        self.key_fc = nn.Linear(int(3 * d_model), int(d_model))
        self.key_ln = nn.LayerNorm(int(d_model))

        self.d_model = int(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.text_proj, self.cover_proj, self.alpha_fc, self.beta_fc, self.text_agg_fc, self.key_fc]:
            _init_linear(m)

    def forward(self, title_blurb: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        if title_blurb.ndim != 3:
            raise ValueError(f"title_blurb 需要为 3 维 (B,2,D_txt)，但得到 {tuple(title_blurb.shape)}")
        if int(title_blurb.shape[1]) != 2:
            raise ValueError(f"title_blurb 的第 2 维必须为 2（固定 [title, blurb]），但得到 {tuple(title_blurb.shape)}")
        if cover.ndim != 2:
            raise ValueError(f"cover 需要为 2 维 (B,D_img)，但得到 {tuple(cover.shape)}")
        if int(title_blurb.shape[0]) != int(cover.shape[0]):
            raise ValueError(f"title_blurb/cover batch size 不一致：{tuple(title_blurb.shape)} vs {tuple(cover.shape)}")

        T = self.text_ln(torch.relu(self.text_proj(title_blurb)))  # [B, 2, d]
        c = self.cover_ln(torch.relu(self.cover_proj(cover)))  # [B, d]
        T = self.drop(T)
        c = self.drop(c)
        c2 = c.unsqueeze(1).expand(-1, 2, -1)  # [B, 2, d]

        alpha_in = torch.cat([T, c2, T * c2], dim=-1)  # [B, 2, 3d]
        alpha = torch.sigmoid(0.5 * self.alpha_fc(alpha_in))  # [B, 2, 1]
        t_tilde = torch.sum(alpha * T, dim=1)  # [B, d]
        c1 = self.c1_ln(c + t_tilde)  # [B, d]
        c1 = self.drop(c1)

        beta_in = torch.cat([c2, T, c2 * T], dim=-1)  # [B, 2, 3d]
        beta = torch.sigmoid(0.5 * self.beta_fc(beta_in))  # [B, 2, 1]
        T1 = T + beta * c2  # [B, 2, d]
        T1 = self.drop(T1)

        t_cat = torch.cat([T1[:, 0, :], T1[:, 1, :]], dim=-1)  # [B, 2d]
        t1 = self.text_agg_ln(self.text_agg_fc(t_cat))  # [B, d]
        t1 = self.drop(t1)

        key_in = torch.cat([t1, c1, 0.5 * (t1 * c1)], dim=-1)  # [B, 3d]
        v_key = self.key_ln(self.key_fc(key_in))  # [B, d]
        v_key = self.drop(v_key)
        return v_key


class ContentSeqEncoder(nn.Module):
    """正文 content_sequence 的 trm_pos：TokenEncoder + sinusoidal PE + Transformer + masked mean pooling。"""

    def __init__(
        self,
        image_embedding_dim: int,
        text_embedding_dim: int,
        d_model: int,
        token_dropout: float,
        max_seq_len: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float,
    ) -> None:
        super().__init__()
        if int(d_model) % int(transformer_num_heads) != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={transformer_num_heads} 整除。")

        self.token = TokenEncoder(
            image_embedding_dim=int(image_embedding_dim),
            text_embedding_dim=int(text_embedding_dim),
            d_model=int(d_model),
            token_dropout=float(token_dropout),
        )
        self.pos = SinusoidalPositionalEncoding(int(max_seq_len), int(d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(transformer_num_heads),
            dim_feedforward=int(transformer_dim_feedforward),
            dropout=float(transformer_dropout),
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(transformer_num_layers), enable_nested_tensor=False)
        self.d_model = int(d_model)

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        seq_type: torch.Tensor,
        seq_attr: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token(x_img=x_img, x_txt=x_txt, seq_type=seq_type, seq_attr=seq_attr)
        x2 = self.pos(x, seq_mask)
        z = self.transformer(x2, src_key_padding_mask=~seq_mask)
        pooled = masked_mean_pool(z, seq_mask)
        return pooled


def _activation_module(name: str) -> nn.Module:
    name = str(name or "").strip().lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"不支持的 activation={name!r}，可选：relu/gelu")


class TwoLayerMLPHead(nn.Module):
    """2-layer MLP：Linear -> (ReLU/GELU) -> Dropout -> Linear(1)"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, activation: str = "relu") -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError(f"input_dim 需要 > 0，但得到 {input_dim}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim 需要 > 0，但得到 {hidden_dim}")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.act = _activation_module(str(activation))
        self.drop = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), 1)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


class SeqFusionHead(nn.Module):
    """
    与 src/dl/seq/model.py 融合头保持结构一致：
    LN(input) -> Linear -> ReLU -> Dropout -> Linear(1)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError(f"input_dim 需要 > 0，但得到 {input_dim}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim 需要 > 0，但得到 {hidden_dim}")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.fusion_fc = nn.Linear(int(input_dim), int(hidden_dim))
        self.fusion_in_ln = nn.LayerNorm(int(input_dim))
        self.fusion_drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(hidden_dim), 1)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_linear(self.fusion_fc)
        _init_linear(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fusion_in_ln(x)
        x = torch.relu(self.fusion_fc(x))
        x = self.fusion_drop(x)
        return self.head(x).squeeze(-1)


class ResBinaryClassifier(nn.Module):
    """三分支 + 两种 baseline（mlp/res），输出二分类 logits。"""

    def __init__(
        self,
        baseline_mode: str,
        meta_input_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        d_model: int,
        token_dropout: float,
        key_dropout: float,
        max_seq_len: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float,
        head_hidden_dim: int,
        head_dropout: float,
        head_activation: str,
        base_hidden_dim: int,
        base_dropout: float,
        prior_hidden_dim: int,
        prior_dropout: float,
        prior_activation: str,
        delta_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        self.baseline_mode = str(baseline_mode or "").strip().lower()
        if self.baseline_mode not in {"mlp", "res"}:
            raise ValueError(f"不支持的 baseline_mode={self.baseline_mode!r}，可选：mlp/res")

        if int(meta_input_dim) <= 0:
            raise ValueError("meta_input_dim 需要 > 0（res 模块默认始终使用 meta 分支）。")

        self.d_model = int(d_model)

        # 三个分支编码器（与 gate 对齐）
        self.key = FirstImpressionEncoder(
            image_embedding_dim=int(image_embedding_dim),
            text_embedding_dim=int(text_embedding_dim),
            d_model=int(d_model),
            dropout=float(key_dropout),
        )
        self.seq = ContentSeqEncoder(
            image_embedding_dim=int(image_embedding_dim),
            text_embedding_dim=int(text_embedding_dim),
            d_model=int(d_model),
            token_dropout=float(token_dropout),
            max_seq_len=int(max_seq_len),
            transformer_num_layers=int(transformer_num_layers),
            transformer_num_heads=int(transformer_num_heads),
            transformer_dim_feedforward=int(transformer_dim_feedforward),
            transformer_dropout=float(transformer_dropout),
        )

        # 投影到同一维度 d
        self.meta_proj: nn.Module
        if int(meta_input_dim) == int(d_model):
            self.meta_proj = nn.Identity()
        else:
            self.meta_proj = nn.Linear(int(meta_input_dim), int(d_model))
            _init_linear(self.meta_proj)  # type: ignore[arg-type]

        # v_key 已经输出 d；保留 key_proj 以满足“若已为 d 则 identity”的约束
        self.key_proj: nn.Module = nn.Identity()

        # 分支 LN（按需求：在 concat 前对各向量做归一化）
        self.ln_seq = nn.LayerNorm(int(d_model))
        self.ln_meta = nn.LayerNorm(int(d_model))
        self.ln_key = nn.LayerNorm(int(d_model))
        self.ln_key_meta_mul = nn.LayerNorm(int(d_model))

        # heads
        self.head = TwoLayerMLPHead(
            input_dim=int(3 * d_model),
            hidden_dim=int(head_hidden_dim),
            dropout=float(head_dropout),
            activation=str(head_activation),
        )
        self.head_hidden_dim = int(head_hidden_dim)

        self.mlp_base = SeqFusionHead(
            input_dim=int(2 * d_model),
            hidden_dim=int(base_hidden_dim),
            dropout=float(base_dropout),
        )
        self.base_hidden_dim = int(base_hidden_dim)

        self.mlp_prior = TwoLayerMLPHead(
            input_dim=int(3 * d_model),
            hidden_dim=int(prior_hidden_dim),
            dropout=float(prior_dropout),
            activation=str(prior_activation),
        )
        self.prior_hidden_dim = int(prior_hidden_dim)

        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init), dtype=torch.float32))

        # 冻结未使用分支，避免 AdamW weight_decay 在“无梯度参数”上产生漂移。
        if self.baseline_mode == "mlp":
            self.mlp_base.requires_grad_(False)
            self.mlp_prior.requires_grad_(False)
            self.delta_scale.requires_grad_(False)
        else:
            self.head.requires_grad_(False)

    def forward(
        self,
        title_blurb: torch.Tensor,
        cover: torch.Tensor,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        seq_type: torch.Tensor,
        seq_attr: torch.Tensor,
        seq_mask: torch.Tensor,
        x_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_meta is None:
            raise ValueError("x_meta 不能为空（res 模块默认始终使用 meta 分支）。")

        if self.baseline_mode == "mlp":
            v_key = self.key(title_blurb=title_blurb, cover=cover)  # [B, d]
            h_seq = self.seq(
                x_img=x_img,
                x_txt=x_txt,
                seq_type=seq_type,
                seq_attr=seq_attr,
                seq_mask=seq_mask,
            )  # [B, d]
            v_meta_d = self.meta_proj(x_meta)  # [B, d]
            v_key_d = self.key_proj(v_key)  # [B, d]
            fused = torch.cat(
                [
                    self.ln_seq(h_seq),
                    self.ln_meta(v_meta_d),
                    self.ln_key(v_key_d),
                ],
                dim=1,
            )
            return self.head(fused)

        # baseline_mode == "res"
        z, _z_base, _delta = self.forward_res_parts(
            title_blurb=title_blurb,
            cover=cover,
            x_img=x_img,
            x_txt=x_txt,
            seq_type=seq_type,
            seq_attr=seq_attr,
            seq_mask=seq_mask,
            x_meta=x_meta,
        )
        return z

    def forward_res_parts(
        self,
        title_blurb: torch.Tensor,
        cover: torch.Tensor,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        seq_type: torch.Tensor,
        seq_attr: torch.Tensor,
        seq_mask: torch.Tensor,
        x_meta: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        仅用于 baseline_mode="res" 的调试/分析：

        返回：
        - z：最终 logits
        - z_base：基线 logits
        - delta：残差项（delta = delta_scale * z_res）
        """
        if self.baseline_mode != "res":
            raise RuntimeError("forward_res_parts 仅支持 baseline_mode='res'。")
        if x_meta is None:
            raise ValueError("x_meta 不能为空（res 模块默认始终使用 meta 分支）。")

        v_key = self.key(title_blurb=title_blurb, cover=cover)  # [B, d]
        h_seq = self.seq(
            x_img=x_img,
            x_txt=x_txt,
            seq_type=seq_type,
            seq_attr=seq_attr,
            seq_mask=seq_mask,
        )  # [B, d]
        v_meta_d = self.meta_proj(x_meta)  # [B, d]
        v_key_d = self.key_proj(v_key)  # [B, d]

        base_in = torch.cat([self.ln_seq(h_seq), self.ln_meta(v_meta_d)], dim=1)
        z_base = self.mlp_base(base_in)

        km = v_key_d * v_meta_d
        prior_in = torch.cat([self.ln_key(v_key_d), self.ln_meta(v_meta_d), self.ln_key_meta_mul(km)], dim=1)
        z_res = self.mlp_prior(prior_in)
        delta = self.delta_scale * z_res
        z = z_base + delta
        return z, z_base, delta


def build_res_model(
    cfg: ResConfig,
    meta_input_dim: int,
    image_embedding_dim: int,
    text_embedding_dim: int,
) -> ResBinaryClassifier:
    return ResBinaryClassifier(
        baseline_mode=str(getattr(cfg, "baseline_mode", "res")),
        meta_input_dim=int(meta_input_dim),
        image_embedding_dim=int(image_embedding_dim),
        text_embedding_dim=int(text_embedding_dim),
        d_model=int(getattr(cfg, "d_model", 256)),
        token_dropout=float(getattr(cfg, "token_dropout", 0.0)),
        key_dropout=float(getattr(cfg, "key_dropout", 0.0)),
        max_seq_len=int(getattr(cfg, "max_seq_len", 40)),
        transformer_num_layers=int(getattr(cfg, "transformer_num_layers", 2)),
        transformer_num_heads=int(getattr(cfg, "transformer_num_heads", 4)),
        transformer_dim_feedforward=int(getattr(cfg, "transformer_dim_feedforward", 512)),
        transformer_dropout=float(getattr(cfg, "transformer_dropout", 0.1)),
        head_hidden_dim=int(getattr(cfg, "head_hidden_dim", 512)),
        head_dropout=float(getattr(cfg, "head_dropout", 0.5)),
        head_activation=str(getattr(cfg, "head_activation", "relu")),
        base_hidden_dim=int(getattr(cfg, "base_hidden_dim", 512)),
        base_dropout=float(getattr(cfg, "base_dropout", 0.5)),
        prior_hidden_dim=int(getattr(cfg, "prior_hidden_dim", 512)),
        prior_dropout=float(getattr(cfg, "prior_dropout", 0.5)),
        prior_activation=str(getattr(cfg, "prior_activation", "relu")),
        delta_scale_init=float(getattr(cfg, "delta_scale_init", 0.0)),
    )
