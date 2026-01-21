# -*- coding: utf-8 -*-
"""
模型定义（gate）：

三分支：
- meta：表格元数据编码（对齐 mlp 的 MetaMLPEncoder）
- 第一印象：title_blurb + cover_image -> MLP 融合得到向量
- 图文序列：text 序列 + image 序列 -> 编码后用 MLP 融合得到向量

最终融合：
- 两阶段门控网络：
  1) fuse(meta, impression) -> 初步认知向量
  2) fuse(初步认知, sequence) -> 最终融合向量
- 线性分类层输出 logits（训练使用 BCEWithLogitsLoss）
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config import GateConfig


class MetaMLPEncoder(nn.Module):
    """metadata 特征 -> FC -> Dropout，输出一个定长向量。"""

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


class ImageCNNEncoder(nn.Module):
    """图片嵌入序列 -> 1D CNN -> 全局 max pooling，输出定长向量。"""

    def __init__(
        self,
        embedding_dim: int,
        conv_channels: int = 256,
        conv_kernel_size: int = 3,
        input_dropout: float = 0.0,
        dropout: float = 0.5,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim 需要 > 0，但得到 {embedding_dim}")
        if conv_channels <= 0:
            raise ValueError(f"conv_channels 需要 > 0，但得到 {conv_channels}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size 需要为正奇数（便于 same padding）")
        if input_dropout < 0.0 or input_dropout >= 1.0:
            raise ValueError("input_dropout 需要在 [0, 1) 之间")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        pad = int(conv_kernel_size // 2)

        self.input_drop = nn.Dropout(p=float(input_dropout))

        self.conv1 = nn.Conv1d(
            in_channels=int(embedding_dim),
            out_channels=int(conv_channels),
            kernel_size=int(conv_kernel_size),
            padding=pad,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(int(conv_channels)) if use_batch_norm else None
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.conv2 = nn.Conv1d(
            in_channels=int(conv_channels),
            out_channels=int(conv_channels),
            kernel_size=int(conv_kernel_size),
            padding=pad,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(int(conv_channels)) if use_batch_norm else None
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.drop2 = nn.Dropout(p=float(dropout))

        self.output_dim = int(conv_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")

    @staticmethod
    def _pool_len(lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        return (lengths + 1) // 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入 x 需要为 3 维张量 (B, L, D)，但得到 {tuple(x.shape)}")

        z = x.transpose(1, 2).contiguous()
        z = self.input_drop(z)

        z = self.conv1(z)
        if self.bn1 is not None:
            z = self.bn1(z)
        z = torch.relu(z)
        z = self.pool1(z)
        z = self.drop1(z)

        z = self.conv2(z)
        if self.bn2 is not None:
            z = self.bn2(z)
        z = torch.relu(z)
        z = self.pool2(z)
        z = self.drop2(z)

        if lengths is not None:
            l2 = self._pool_len(self._pool_len(lengths.to(z.device)))
            max_len = int(z.shape[-1])
            idx = torch.arange(max_len, device=z.device).unsqueeze(0)
            mask = idx < l2.unsqueeze(1)
            z = z.masked_fill(~mask.unsqueeze(1), -1e9)

        feat = torch.amax(z, dim=-1)
        if lengths is not None:
            valid = (lengths.to(feat.device) > 0).to(feat.dtype).unsqueeze(1)
            feat = feat * valid
        return feat


class TextCNNEncoder(nn.Module):
    """文本嵌入序列 -> 1D CNN -> 全局 max pooling，输出定长向量。"""

    def __init__(
        self,
        embedding_dim: int,
        conv_kernel_size: int = 3,
        input_dropout: float = 0.0,
        dropout: float = 0.3,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim 需要 > 0，但得到 {embedding_dim}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size 需要为正奇数（便于 same padding）")
        if input_dropout < 0.0 or input_dropout >= 1.0:
            raise ValueError("input_dropout 需要在 [0, 1) 之间")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        pad = int(conv_kernel_size // 2)
        channels = (96, 128, 256)

        self.input_drop = nn.Dropout(p=float(input_dropout))

        self.conv1 = nn.Conv1d(
            in_channels=int(embedding_dim),
            out_channels=int(channels[0]),
            kernel_size=int(conv_kernel_size),
            padding=pad,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(int(channels[0])) if use_batch_norm else None
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.conv2 = nn.Conv1d(
            in_channels=int(channels[0]),
            out_channels=int(channels[1]),
            kernel_size=int(conv_kernel_size),
            padding=pad,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(int(channels[1])) if use_batch_norm else None
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.drop2 = nn.Dropout(p=float(dropout))

        self.conv3 = nn.Conv1d(
            in_channels=int(channels[1]),
            out_channels=int(channels[2]),
            kernel_size=int(conv_kernel_size),
            padding=pad,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(int(channels[2])) if use_batch_norm else None
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.drop3 = nn.Dropout(p=float(dropout))

        self.output_dim = int(channels[-1])
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")

    @staticmethod
    def _pool_len(lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        return (lengths + 1) // 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入 x 需要为 3 维张量 (B, L, D)，但得到 {tuple(x.shape)}")

        z = x.transpose(1, 2).contiguous()
        z = self.input_drop(z)

        z = self.conv1(z)
        if self.bn1 is not None:
            z = self.bn1(z)
        z = torch.relu(z)
        z = self.pool1(z)
        z = self.drop1(z)

        z = self.conv2(z)
        if self.bn2 is not None:
            z = self.bn2(z)
        z = torch.relu(z)
        z = self.pool2(z)
        z = self.drop2(z)

        z = self.conv3(z)
        if self.bn3 is not None:
            z = self.bn3(z)
        z = torch.relu(z)
        z = self.pool3(z)
        z = self.drop3(z)

        if lengths is not None:
            l3 = self._pool_len(self._pool_len(self._pool_len(lengths.to(z.device))))
            max_len = int(z.shape[-1])
            idx = torch.arange(max_len, device=z.device).unsqueeze(0)
            mask = idx < l3.unsqueeze(1)
            z = z.masked_fill(~mask.unsqueeze(1), -1e9)

        feat = torch.amax(z, dim=-1)
        if lengths is not None:
            valid = (lengths.to(feat.device) > 0).to(feat.dtype).unsqueeze(1)
            feat = feat * valid
        return feat


class MLPProjector(nn.Module):
    """两层 MLP：Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim 需要 > 0，但得到 {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 需要 > 0，但得到 {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim 需要 > 0，但得到 {output_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(output_dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.output_dim = int(output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc1(x))
        z = self.drop(z)
        z = torch.relu(self.fc2(z))
        z = self.drop(z)
        return z


class GatedFusion(nn.Module):
    """
    两路向量门控融合：
    - gate = sigmoid(MLP([a;b]))，gate 维度与向量维度一致
    - fused = gate * a + (1 - gate) * b
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, use_layer_norm: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim 需要 > 0，但得到 {dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 需要 > 0，但得到 {hidden_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        self.fc1 = nn.Linear(int(dim * 2), int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.norm = nn.LayerNorm(int(dim)) if bool(use_layer_norm) else None
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"输入需要为 2D 张量 (B, D)，但得到 a={tuple(a.shape)} b={tuple(b.shape)}")
        if int(a.shape[0]) != int(b.shape[0]) or int(a.shape[1]) != int(b.shape[1]):
            raise ValueError(f"a/b 形状不一致：a={tuple(a.shape)} b={tuple(b.shape)}")

        x = torch.cat([a, b], dim=1)
        gate = torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))
        fused = gate * a + (1.0 - gate) * b
        if self.norm is not None:
            fused = self.norm(fused)
        fused = self.drop(fused)
        return fused


class GateBinaryClassifier(nn.Module):
    """三分支 + 两阶段门控融合的二分类模型（输出 logits）。"""

    def __init__(
        self,
        meta_input_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        meta_hidden_dim: int,
        meta_dropout: float,
        image_conv_channels: int,
        image_conv_kernel_size: int,
        image_input_dropout: float,
        image_dropout: float,
        image_use_batch_norm: bool,
        text_conv_kernel_size: int,
        text_input_dropout: float,
        text_dropout: float,
        text_use_batch_norm: bool,
        gate_dim: int,
        impression_mlp_hidden_dim: int,
        impression_dropout: float,
        seq_fuse_hidden_dim: int,
        seq_fuse_dropout: float,
        gate_hidden_dim: int,
        gate_dropout: float,
        gate_use_layer_norm: bool,
        classifier_dropout: float,
    ) -> None:
        super().__init__()

        self.meta = MetaMLPEncoder(input_dim=int(meta_input_dim), hidden_dim=int(meta_hidden_dim), dropout=float(meta_dropout))
        self.meta_proj: Optional[nn.Module]
        if int(self.meta.output_dim) != int(gate_dim):
            self.meta_proj = nn.Sequential(
                nn.Linear(int(self.meta.output_dim), int(gate_dim)),
                nn.ReLU(),
                nn.Dropout(p=float(meta_dropout)),
            )
            nn.init.xavier_uniform_(self.meta_proj[0].weight)
            if self.meta_proj[0].bias is not None:
                nn.init.zeros_(self.meta_proj[0].bias)
        else:
            self.meta_proj = None

        # 第一印象：title_blurb 与 cover_image 也先过 encoder，再做 MLP 融合
        self.impression_image_encoder = ImageCNNEncoder(
            embedding_dim=int(image_embedding_dim),
            conv_channels=int(image_conv_channels),
            conv_kernel_size=int(image_conv_kernel_size),
            input_dropout=float(image_input_dropout),
            dropout=float(image_dropout),
            use_batch_norm=bool(image_use_batch_norm),
        )
        self.impression_text_encoder = TextCNNEncoder(
            embedding_dim=int(text_embedding_dim),
            conv_kernel_size=int(text_conv_kernel_size),
            input_dropout=float(text_input_dropout),
            dropout=float(text_dropout),
            use_batch_norm=bool(text_use_batch_norm),
        )
        self.impression = MLPProjector(
            input_dim=int(self.impression_image_encoder.output_dim) + int(self.impression_text_encoder.output_dim),
            hidden_dim=int(impression_mlp_hidden_dim),
            output_dim=int(gate_dim),
            dropout=float(impression_dropout),
        )

        self.image_encoder = ImageCNNEncoder(
            embedding_dim=int(image_embedding_dim),
            conv_channels=int(image_conv_channels),
            conv_kernel_size=int(image_conv_kernel_size),
            input_dropout=float(image_input_dropout),
            dropout=float(image_dropout),
            use_batch_norm=bool(image_use_batch_norm),
        )
        self.text_encoder = TextCNNEncoder(
            embedding_dim=int(text_embedding_dim),
            conv_kernel_size=int(text_conv_kernel_size),
            input_dropout=float(text_input_dropout),
            dropout=float(text_dropout),
            use_batch_norm=bool(text_use_batch_norm),
        )

        self.seq_fuse = MLPProjector(
            input_dim=int(self.image_encoder.output_dim) + int(self.text_encoder.output_dim),
            hidden_dim=int(seq_fuse_hidden_dim),
            output_dim=int(gate_dim),
            dropout=float(seq_fuse_dropout),
        )

        self.gate1 = GatedFusion(
            dim=int(gate_dim),
            hidden_dim=int(gate_hidden_dim),
            dropout=float(gate_dropout),
            use_layer_norm=bool(gate_use_layer_norm),
        )
        self.gate2 = GatedFusion(
            dim=int(gate_dim),
            hidden_dim=int(gate_hidden_dim),
            dropout=float(gate_dropout),
            use_layer_norm=bool(gate_use_layer_norm),
        )

        self.cls_drop = nn.Dropout(p=float(classifier_dropout))
        self.head = nn.Linear(int(gate_dim), 1)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_meta: torch.Tensor,
        x_cover: torch.Tensor,
        len_cover: torch.Tensor,
        x_title_blurb: torch.Tensor,
        len_title_blurb: torch.Tensor,
        x_image: torch.Tensor,
        len_image: torch.Tensor,
        x_text: torch.Tensor,
        len_text: torch.Tensor,
    ) -> torch.Tensor:
        h_meta = self.meta(x_meta)
        if self.meta_proj is not None:
            h_meta = self.meta_proj(h_meta)

        h_cover = self.impression_image_encoder(x_cover, lengths=len_cover)
        h_title = self.impression_text_encoder(x_title_blurb, lengths=len_title_blurb)
        h_impression = self.impression(torch.cat([h_title, h_cover], dim=1))

        h_img = self.image_encoder(x_image, lengths=len_image)
        h_txt = self.text_encoder(x_text, lengths=len_text)
        h_seq = self.seq_fuse(torch.cat([h_img, h_txt], dim=1))

        h_cognition = self.gate1(h_meta, h_impression)
        h_fused = self.gate2(h_cognition, h_seq)

        logits = self.head(self.cls_drop(h_fused)).squeeze(-1)
        return logits


def build_gate_model(
    cfg: GateConfig,
    meta_input_dim: int,
    image_embedding_dim: int,
    text_embedding_dim: int,
) -> GateBinaryClassifier:
    """根据配置构建 gate 模型。"""
    return GateBinaryClassifier(
        meta_input_dim=int(meta_input_dim),
        image_embedding_dim=int(image_embedding_dim),
        text_embedding_dim=int(text_embedding_dim),
        meta_hidden_dim=int(getattr(cfg, "meta_hidden_dim", 256)),
        meta_dropout=float(getattr(cfg, "meta_dropout", 0.3)),
        image_conv_channels=int(getattr(cfg, "image_conv_channels", 256)),
        image_conv_kernel_size=int(getattr(cfg, "image_conv_kernel_size", 3)),
        image_input_dropout=float(getattr(cfg, "image_input_dropout", 0.1)),
        image_dropout=float(getattr(cfg, "image_dropout", 0.5)),
        image_use_batch_norm=bool(getattr(cfg, "image_use_batch_norm", True)),
        text_conv_kernel_size=int(getattr(cfg, "text_conv_kernel_size", 3)),
        text_input_dropout=float(getattr(cfg, "text_input_dropout", 0.1)),
        text_dropout=float(getattr(cfg, "text_dropout", 0.3)),
        text_use_batch_norm=bool(getattr(cfg, "text_use_batch_norm", True)),
        gate_dim=int(getattr(cfg, "gate_dim", 256)),
        impression_mlp_hidden_dim=int(getattr(cfg, "impression_mlp_hidden_dim", 512)),
        impression_dropout=float(getattr(cfg, "impression_dropout", 0.3)),
        seq_fuse_hidden_dim=int(getattr(cfg, "seq_fuse_hidden_dim", 512)),
        seq_fuse_dropout=float(getattr(cfg, "seq_fuse_dropout", 0.3)),
        gate_hidden_dim=int(getattr(cfg, "gate_hidden_dim", 256)),
        gate_dropout=float(getattr(cfg, "gate_dropout", 0.1)),
        gate_use_layer_norm=bool(getattr(cfg, "gate_use_layer_norm", True)),
        classifier_dropout=float(getattr(cfg, "classifier_dropout", 0.2)),
    )
