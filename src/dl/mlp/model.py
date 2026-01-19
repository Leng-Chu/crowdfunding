# -*- coding: utf-8 -*-
"""
模型定义（多模态三路分支 + 融合）：
- metadata：FC(256) -> Dropout(0.3)
- image：1D CNN（两层 Conv + Pool + Dropout）
- text：1D CNN（三层 Conv + Pool + Dropout）
- 融合：concat -> FC(1536) -> Dropout(0.9) -> logits

说明：
- image/text 的输入均为堆叠后的嵌入序列，形状 (B, L, D)
- 使用 lengths 做 mask，降低 padding 对全局池化的影响
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import MlpDLConfig


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
        """对应 MaxPool1d(kernel=2,stride=2,ceil_mode=True) 的长度变化：ceil(L/2)。"""
        lengths = lengths.to(torch.int64)
        return (lengths + 1) // 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, L, D)
        lengths: (B,) 表示每个样本的真实 L（用于 mask）
        """
        if x.ndim != 3:
            raise ValueError(f"输入 x 需要为 3 维张量 (B, L, D)，但得到 {tuple(x.shape)}")

        # (B, L, D) -> (B, D, L)
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
            idx = torch.arange(max_len, device=z.device).unsqueeze(0)  # (1, T)
            mask = idx < l2.unsqueeze(1)  # (B, T)
            z = z.masked_fill(~mask.unsqueeze(1), -1e9)

        feat = torch.amax(z, dim=-1)  # (B, C)
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
            idx = torch.arange(max_len, device=z.device).unsqueeze(0)  # (1, T)
            mask = idx < l3.unsqueeze(1)  # (B, T)
            z = z.masked_fill(~mask.unsqueeze(1), -1e9)

        feat = torch.amax(z, dim=-1)  # (B, C)
        return feat


class MultiModalBinaryClassifier(nn.Module):
    """三路编码器 + 融合 MLP，输出二分类 logits。"""

    def __init__(
        self,
        meta_input_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        meta_hidden_dim: int = 256,
        meta_dropout: float = 0.3,
        image_conv_channels: int = 256,
        image_conv_kernel_size: int = 3,
        image_input_dropout: float = 0.0,
        image_dropout: float = 0.5,
        image_use_batch_norm: bool = False,
        text_conv_kernel_size: int = 3,
        text_input_dropout: float = 0.0,
        text_dropout: float = 0.3,
        text_use_batch_norm: bool = False,
        fusion_hidden_dim: int = 1536,
        fusion_dropout: float = 0.9,
    ) -> None:
        super().__init__()

        self.meta = MetaMLPEncoder(
            input_dim=int(meta_input_dim),
            hidden_dim=int(meta_hidden_dim),
            dropout=float(meta_dropout),
        )
        self.image = ImageCNNEncoder(
            embedding_dim=int(image_embedding_dim),
            conv_channels=int(image_conv_channels),
            conv_kernel_size=int(image_conv_kernel_size),
            input_dropout=float(image_input_dropout),
            dropout=float(image_dropout),
            use_batch_norm=bool(image_use_batch_norm),
        )
        self.text = TextCNNEncoder(
            embedding_dim=int(text_embedding_dim),
            conv_kernel_size=int(text_conv_kernel_size),
            input_dropout=float(text_input_dropout),
            dropout=float(text_dropout),
            use_batch_norm=bool(text_use_batch_norm),
        )

        fusion_in_dim = int(meta_hidden_dim) + int(self.image.output_dim) + int(self.text.output_dim)
        if fusion_hidden_dim <= 0:
            raise ValueError(f"fusion_hidden_dim 需要 > 0，但得到 {fusion_hidden_dim}")
        if fusion_dropout < 0.0 or fusion_dropout >= 1.0:
            raise ValueError("fusion_dropout 需要在 [0, 1) 之间")

        self.fusion_fc = nn.Linear(int(fusion_in_dim), int(fusion_hidden_dim))
        self.fusion_drop = nn.Dropout(p=float(fusion_dropout))
        self.head = nn.Linear(int(fusion_hidden_dim), 1)
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
        x_meta: torch.Tensor,
        x_image: torch.Tensor,
        len_image: torch.Tensor | None = None,
        x_text: torch.Tensor | None = None,
        len_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_text is None:
            raise ValueError("x_text 不能为空")

        meta_feat = self.meta(x_meta)
        image_feat = self.image(x_image, lengths=len_image)
        text_feat = self.text(x_text, lengths=len_text)

        fused = torch.cat([meta_feat, image_feat, text_feat], dim=1)
        fused = torch.relu(self.fusion_fc(fused))
        fused = self.fusion_drop(fused)
        logits = self.head(fused).squeeze(-1)
        return logits


def build_model(cfg: MlpDLConfig, meta_input_dim: int, image_embedding_dim: int, text_embedding_dim: int) -> MultiModalBinaryClassifier:
    """根据配置创建多模态三路融合模型。"""
    return MultiModalBinaryClassifier(
        meta_input_dim=int(meta_input_dim),
        image_embedding_dim=int(image_embedding_dim),
        text_embedding_dim=int(text_embedding_dim),
        meta_hidden_dim=int(getattr(cfg, "meta_hidden_dim", 256)),
        meta_dropout=float(getattr(cfg, "meta_dropout", 0.3)),
        image_conv_channels=int(getattr(cfg, "image_conv_channels", 256)),
        image_conv_kernel_size=int(getattr(cfg, "image_conv_kernel_size", 3)),
        image_input_dropout=float(getattr(cfg, "image_input_dropout", 0.0)),
        image_dropout=float(getattr(cfg, "image_dropout", 0.5)),
        image_use_batch_norm=bool(getattr(cfg, "image_use_batch_norm", False)),
        text_conv_kernel_size=int(getattr(cfg, "text_conv_kernel_size", 3)),
        text_input_dropout=float(getattr(cfg, "text_input_dropout", 0.0)),
        text_dropout=float(getattr(cfg, "text_dropout", 0.3)),
        text_use_batch_norm=bool(getattr(cfg, "text_use_batch_norm", False)),
        fusion_hidden_dim=int(getattr(cfg, "fusion_hidden_dim", 1536)),
        fusion_dropout=float(getattr(cfg, "fusion_dropout", 0.9)),
    )

