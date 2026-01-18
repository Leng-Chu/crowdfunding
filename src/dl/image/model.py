# -*- coding: utf-8 -*-
"""
模型定义模块：使用 PyTorch 1D CNN 做二分类（与示意图一致）。

输入：
- 堆叠后的图片嵌入序列，形状为 (B, L, D)
  - L：向量条数（cover=1；若存在 image_{type}.npy，则 L=1+N）
  - D：嵌入维度（clip=512 / siglip=768 / resnet=2048）

结构（简化版）：
stacking -> Conv(256) -> MaxPool -> Dropout(0.5) -> Conv(256) -> MaxPool -> Dropout(0.5)
-> GlobalMaxPool -> 全连接 -> Dropout -> 分类 -> logits
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ImageDLConfig


class ImageCNNBinaryClassifier(nn.Module):
    """图片嵌入序列 -> 1D CNN -> 二分类 logits。"""

    def __init__(
        self,
        embedding_dim: int,
        conv_channels: int = 256,
        conv_kernel_size: int = 3,
        fc_hidden_dim: int = 0,
        input_dropout: float = 0.0,
        dropout: float = 0.5,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim 需要 > 0，当前为 {embedding_dim}")
        if conv_channels <= 0:
            raise ValueError(f"conv_channels 需要 > 0，当前为 {conv_channels}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size 需要为正奇数（便于 same padding）")
        if input_dropout < 0.0 or input_dropout >= 1.0:
            raise ValueError("input_dropout 需要在 [0, 1) 之间")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        pad = int(conv_kernel_size // 2)
        fc_dim = int(fc_hidden_dim) if int(fc_hidden_dim) > 0 else int(conv_channels)

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

        self.fc = nn.Linear(int(conv_channels), int(fc_dim))
        self.fc_drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(fc_dim), 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化（conv 使用 kaiming，linear 用 xavier）。"""
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _pool_len(lengths: torch.Tensor) -> torch.Tensor:
        """对应 MaxPool1d(kernel=2,stride=2,ceil_mode=True) 的长度变化：ceil(L/2)。"""
        lengths = lengths.to(torch.int64)
        return (lengths + 1) // 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, L, D)
        lengths: (B,) 表示每个样本的真实 L（用于 mask，避免 padding 影响全局池化）
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

        # mask 掉 padding 的位置，再做全局 max pooling
        if lengths is not None:
            l2 = self._pool_len(self._pool_len(lengths.to(z.device)))
            max_len = int(z.shape[-1])
            idx = torch.arange(max_len, device=z.device).unsqueeze(0)  # (1, T)
            mask = idx < l2.unsqueeze(1)  # (B, T)
            z = z.masked_fill(~mask.unsqueeze(1), -1e9)

        feat = torch.amax(z, dim=-1)  # (B, C)
        feat = torch.relu(self.fc(feat))
        feat = self.fc_drop(feat)
        logits = self.head(feat).squeeze(-1)
        return logits


def build_cnn(cfg: ImageDLConfig, embedding_dim: int) -> ImageCNNBinaryClassifier:
    """根据配置创建 PyTorch 1D CNN 二分类模型。"""
    return ImageCNNBinaryClassifier(
        embedding_dim=int(embedding_dim),
        conv_channels=int(getattr(cfg, "conv_channels", 256)),
        conv_kernel_size=int(getattr(cfg, "conv_kernel_size", 3)),
        fc_hidden_dim=int(getattr(cfg, "fc_hidden_dim", 0)),
        input_dropout=float(getattr(cfg, "input_dropout", 0.0)),
        dropout=float(getattr(cfg, "dropout", 0.5)),
        use_batch_norm=bool(getattr(cfg, "use_batch_norm", False)),
    )
