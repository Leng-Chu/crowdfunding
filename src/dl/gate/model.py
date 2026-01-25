# -*- coding: utf-8 -*-
"""
模型定义（gate / Chapter 2）：三分支 + Two-stage gated fusion。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from config import GateConfig


class MetaEncoder(nn.Module):
    """
    MetaEncoder（必须与约束一致）：
    Linear(d_meta_in→d) → ReLU → Dropout(p_meta) → Linear(d→d) → ReLU
    输出 v_meta ∈ R^{B×d}
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim 需要 > 0，但得到 {input_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.fc1 = nn.Linear(int(input_dim), int(d_model))
        self.drop = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(int(d_model), int(d_model))
        self.output_dim = int(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc1(x))
        z = self.drop(z)
        z = torch.relu(self.fc2(z))
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
    FirstImpressionEncoder（必须按约束实现）：

    输入：
    - title_blurb: E_tb [B, 2, D_txt]（顺序固定 [title, blurb]）
    - cover: e_c [B, D_img]

    输出：
    - v_key ∈ R^{B×d}
    """

    def __init__(self, image_embedding_dim: int, text_embedding_dim: int, d_model: int) -> None:
        super().__init__()
        if image_embedding_dim <= 0:
            raise ValueError(f"image_embedding_dim 需要 > 0，但得到 {image_embedding_dim}")
        if text_embedding_dim <= 0:
            raise ValueError(f"text_embedding_dim 需要 > 0，但得到 {text_embedding_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model 需要 > 0，但得到 {d_model}")

        # TextProj / CoverProj
        self.text_proj = nn.Linear(int(text_embedding_dim), int(d_model))
        self.cover_proj = nn.Linear(int(image_embedding_dim), int(d_model))

        # Cover <- Text gated pooling
        self.alpha_fc = nn.Linear(int(3 * d_model), 1)

        # Text <- Cover gated update
        self.beta_fc = nn.Linear(int(3 * d_model), 1)

        # TextAgg: concat+proj
        self.text_agg_fc = nn.Linear(int(2 * d_model), int(d_model))
        self.text_agg_ln = nn.LayerNorm(int(d_model))

        # v_key
        self.key_fc = nn.Linear(int(3 * d_model), int(d_model))
        self.key_ln = nn.LayerNorm(int(d_model))

        self.d_model = int(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.text_proj, self.cover_proj, self.alpha_fc, self.beta_fc, self.text_agg_fc, self.key_fc]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, title_blurb: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        if title_blurb.ndim != 3:
            raise ValueError(f"title_blurb 需要为 3 维 (B,2,D_txt)，但得到 {tuple(title_blurb.shape)}")
        if int(title_blurb.shape[1]) != 2:
            raise ValueError(f"title_blurb 的第 2 维必须为 2（固定 [title, blurb]），但得到 {tuple(title_blurb.shape)}")
        if cover.ndim != 2:
            raise ValueError(f"cover 需要为 2 维 (B,D_img)，但得到 {tuple(cover.shape)}")
        if int(title_blurb.shape[0]) != int(cover.shape[0]):
            raise ValueError(f"title_blurb/cover batch size 不一致：{tuple(title_blurb.shape)} vs {tuple(cover.shape)}")

        # TextProj / CoverProj
        T = torch.relu(self.text_proj(title_blurb))  # [B, 2, d]
        c = torch.relu(self.cover_proj(cover))  # [B, d]
        c2 = c.unsqueeze(1).expand(-1, 2, -1)  # [B, 2, d]

        # Cover <- Text gated pooling
        alpha_in = torch.cat([T, c2, T * c2], dim=-1)  # [B, 2, 3d]
        alpha = torch.sigmoid(self.alpha_fc(alpha_in))  # [B, 2, 1]
        t_tilde = torch.sum(alpha * T, dim=1)  # [B, d]
        c1 = c + t_tilde  # [B, d]

        # Text <- Cover gated update
        beta_in = torch.cat([c2, T, c2 * T], dim=-1)  # [B, 2, 3d]
        beta = torch.sigmoid(self.beta_fc(beta_in))  # [B, 2, 1]
        T1 = T + beta * c2  # [B, 2, d]

        # TextAgg: concat+proj + LN
        t_cat = torch.cat([T1[:, 0, :], T1[:, 1, :]], dim=-1)  # [B, 2d]
        t1 = self.text_agg_ln(self.text_agg_fc(t_cat))  # [B, d]

        # v_key: LN(Linear(concat(t1, c1, t1*c1)))
        key_in = torch.cat([t1, c1, t1 * c1], dim=-1)  # [B, 3d]
        v_key = self.key_ln(self.key_fc(key_in))  # [B, d]
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
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(transformer_num_layers))
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


class GateBinaryClassifier(nn.Module):
    """三分支 +（可切换的）融合策略，输出二分类 logits。"""

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
        meta_dropout: float,
        head_hidden_dim: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.baseline_mode = str(baseline_mode or "").strip().lower()
        if self.baseline_mode not in {"late_concat", "stage1_only", "stage2_only", "two_stage"}:
            raise ValueError(
                f"不支持的 baseline_mode={self.baseline_mode!r}，可选：late_concat/stage1_only/stage2_only/two_stage"
            )
        self.use_meta = bool(use_meta)
        self.d_model = int(d_model)

        # 三个分支
        self.key = FirstImpressionEncoder(
            image_embedding_dim=int(image_embedding_dim),
            text_embedding_dim=int(text_embedding_dim),
            d_model=int(d_model),
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

        self.meta: Optional[MetaEncoder] = None
        if self.use_meta:
            if int(meta_input_dim) <= 0:
                raise ValueError("use_meta=True 时 meta_input_dim 需要 > 0。")
            self.meta = MetaEncoder(int(meta_input_dim), int(d_model), dropout=float(meta_dropout))

        # -----------------------------
        # 融合（Two-stage gated fusion）与 baseline
        # -----------------------------
        # Stage1
        self.stage1_gate_fc = nn.Linear(int(2 * d_model), int(d_model))  # g1
        self.stage1_p_fc = nn.Linear(int(3 * d_model), int(d_model))  # concat(v_key, v_meta2, v_key*v_meta2)
        self.stage1_p_ln = nn.LayerNorm(int(d_model))

        # Stage2
        self.stage2_gate_fc = nn.Linear(int(2 * d_model), int(d_model))  # g2
        self.stage2_ln = nn.LayerNorm(int(d_model))

        # baselines
        self.late_fc = nn.Linear(int(3 * d_model), int(d_model))

        self.stage1only_fuse_fc = nn.Linear(int(2 * d_model), int(d_model))
        self.stage1only_fuse_ln = nn.LayerNorm(int(d_model))

        self.stage2only_p_fc = nn.Linear(int(2 * d_model), int(d_model))
        self.stage2only_p_ln = nn.LayerNorm(int(d_model))

        # -----------------------------
        # Head（与 baseline 对齐）
        # -----------------------------
        if int(head_hidden_dim) <= 0:
            head_hidden_dim = int(2 * d_model)
        self.head_fc = nn.Linear(int(d_model), int(head_hidden_dim))
        self.head_drop = nn.Dropout(p=float(head_dropout))
        self.head_out = nn.Linear(int(head_hidden_dim), 1)
        self.head_hidden_dim = int(head_hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [
            self.stage1_gate_fc,
            self.stage1_p_fc,
            self.stage2_gate_fc,
            self.late_fc,
            self.stage1only_fuse_fc,
            self.stage2only_p_fc,
            self.head_fc,
            self.head_out,
        ]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_v_meta(self, x_meta: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if self.use_meta:
            if self.meta is None or x_meta is None:
                raise ValueError("use_meta=True 但 x_meta 为空。")
            return self.meta(x_meta)
        # 不使用 metadata 时，按约定用 0 向量占位，保持融合公式不变。
        return torch.zeros((int(ref.shape[0]), int(self.d_model)), device=ref.device, dtype=ref.dtype)

    def _stage1_prior(self, v_key: torch.Tensor, v_meta: torch.Tensor) -> torch.Tensor:
        g1 = torch.sigmoid(self.stage1_gate_fc(torch.cat([v_key, v_meta], dim=1)))
        v_meta2 = g1 * v_meta
        p_in = torch.cat([v_key, v_meta2, v_key * v_meta2], dim=1)
        p = self.stage1_p_ln(self.stage1_p_fc(p_in))
        return p

    def _stage2_fuse(self, h_seq: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        g2 = torch.sigmoid(self.stage2_gate_fc(torch.cat([p, h_seq], dim=1)))
        h_final = self.stage2_ln(h_seq + g2 * p)
        return h_final

    def forward(
        self,
        title_blurb: torch.Tensor,
        cover: torch.Tensor,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        seq_type: torch.Tensor,
        seq_attr: torch.Tensor,
        seq_mask: torch.Tensor,
        x_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v_key = self.key(title_blurb=title_blurb, cover=cover)
        v_meta = self._get_v_meta(x_meta=x_meta, ref=v_key)
        h_seq = self.seq(x_img=x_img, x_txt=x_txt, seq_type=seq_type, seq_attr=seq_attr, seq_mask=seq_mask)

        if self.baseline_mode == "late_concat":
            h = self.late_fc(torch.cat([h_seq, v_key, v_meta], dim=1))
        elif self.baseline_mode == "stage1_only":
            p = self._stage1_prior(v_key=v_key, v_meta=v_meta)
            h = self.stage1only_fuse_ln(self.stage1only_fuse_fc(torch.cat([h_seq, p], dim=1)))
        elif self.baseline_mode == "stage2_only":
            p0 = self.stage2only_p_ln(self.stage2only_p_fc(torch.cat([v_key, v_meta], dim=1)))
            h = self._stage2_fuse(h_seq=h_seq, p=p0)
        elif self.baseline_mode == "two_stage":
            p = self._stage1_prior(v_key=v_key, v_meta=v_meta)
            h = self._stage2_fuse(h_seq=h_seq, p=p)
        else:
            raise RuntimeError(f"未覆盖的 baseline_mode：{self.baseline_mode!r}")

        z = torch.relu(self.head_fc(h))
        z = self.head_drop(z)
        logits = self.head_out(z).squeeze(-1)
        return logits


def build_gate_model(
    cfg: GateConfig,
    meta_input_dim: int,
    image_embedding_dim: int,
    text_embedding_dim: int,
) -> GateBinaryClassifier:
    return GateBinaryClassifier(
        baseline_mode=str(getattr(cfg, "baseline_mode", "two_stage")),
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
        meta_dropout=float(getattr(cfg, "meta_dropout", 0.3)),
        head_hidden_dim=int(getattr(cfg, "head_hidden_dim", 0)),
        head_dropout=float(getattr(cfg, "head_dropout", 0.9)),
    )
