# -*- coding: utf-8 -*-
"""
模型定义（mlp）：
- 单分支：
  - meta：MLPBinaryClassifier
  - image：ImageCNNBinaryClassifier
  - text：TextCNNBinaryClassifier
- 多分支（两路/三路）：
  - 使用 encoder + concat + fusion head 的方式（对齐 src/dl/mlp 的融合思路）
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from config import MlpConfig


# -----------------------------
# meta（单分支）
# -----------------------------


def _build_activation(name: str) -> nn.Module:
    key = (name or "").strip().lower()
    if key in {"relu"}:
        return nn.ReLU()
    if key in {"tanh"}:
        return nn.Tanh()
    if key in {"sigmoid", "logistic"}:
        return nn.Sigmoid()
    raise ValueError(f"不支持的 activation={name!r}，可选：relu/tanh/sigmoid")


class MLPBinaryClassifier(nn.Module):
    """可配置的 MLP 二分类器（输出 logits）。"""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim 需要 > 0，当前为 {input_dim}")

        hidden_list: List[int] = [int(h) for h in hidden_sizes]
        if not hidden_list:
            raise ValueError("hidden_layer_sizes 不能为空，例如：(256, 128, 64)")
        if any(h <= 0 for h in hidden_list):
            raise ValueError(f"hidden_layer_sizes 必须全为正整数：{hidden_list}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间")

        act = _build_activation(activation)

        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for h in hidden_list:
            layers.append(nn.Linear(in_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act.__class__())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(-1)


def build_meta_mlp(cfg: MlpConfig, input_dim: int) -> MLPBinaryClassifier:
    hidden_sizes = getattr(cfg, "hidden_layer_sizes", None)
    if hidden_sizes is None:
        hidden_sizes = (int(getattr(cfg, "meta_hidden_dim", 256)),)

    activation = str(getattr(cfg, "activation", "relu"))
    dropout = float(getattr(cfg, "dropout", getattr(cfg, "meta_dropout", 0.0)))
    use_batch_norm = bool(getattr(cfg, "use_batch_norm", False))

    return MLPBinaryClassifier(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )


# -----------------------------
# image（单分支）
# -----------------------------


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
        feat = torch.relu(self.fc(feat))
        feat = self.fc_drop(feat)
        logits = self.head(feat).squeeze(-1)
        return logits


def build_image_cnn(cfg: MlpConfig, embedding_dim: int) -> ImageCNNBinaryClassifier:
    return ImageCNNBinaryClassifier(
        embedding_dim=int(embedding_dim),
        conv_channels=int(getattr(cfg, "image_conv_channels", 256)),
        conv_kernel_size=int(getattr(cfg, "image_conv_kernel_size", 3)),
        fc_hidden_dim=int(getattr(cfg, "fc_hidden_dim", 0)),
        input_dropout=float(getattr(cfg, "image_input_dropout", 0.0)),
        dropout=float(getattr(cfg, "image_dropout", 0.5)),
        use_batch_norm=bool(getattr(cfg, "image_use_batch_norm", False)),
    )


# -----------------------------
# text（单分支）
# -----------------------------


class TextCNNBinaryClassifier(nn.Module):
    """文本嵌入序列 -> 1D CNN -> 二分类 logits。"""

    def __init__(
        self,
        embedding_dim: int,
        conv_kernel_size: int = 3,
        fc_hidden_dim: int = 0,
        input_dropout: float = 0.0,
        dropout: float = 0.3,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim 需要 > 0，当前为 {embedding_dim}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size 需要为正奇数（便于 same padding）。")
        if input_dropout < 0.0 or input_dropout >= 1.0:
            raise ValueError("input_dropout 需要在 [0, 1) 之间。")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout 需要在 [0, 1) 之间。")

        pad = int(conv_kernel_size // 2)
        channels = (96, 128, 256)
        fc_dim = int(fc_hidden_dim) if int(fc_hidden_dim) > 0 else int(channels[-1])

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

        self.fc = nn.Linear(int(channels[2]), int(fc_dim))
        self.fc_drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(fc_dim), 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

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
        feat = torch.relu(self.fc(feat))
        feat = self.fc_drop(feat)
        logits = self.head(feat).squeeze(-1)
        return logits


def build_text_cnn(cfg: MlpConfig, embedding_dim: int) -> TextCNNBinaryClassifier:
    return TextCNNBinaryClassifier(
        embedding_dim=int(embedding_dim),
        conv_kernel_size=int(getattr(cfg, "text_conv_kernel_size", 3)),
        fc_hidden_dim=int(getattr(cfg, "fc_hidden_dim", 0)),
        input_dropout=float(getattr(cfg, "text_input_dropout", 0.0)),
        dropout=float(getattr(cfg, "text_dropout", 0.3)),
        use_batch_norm=bool(getattr(cfg, "text_use_batch_norm", False)),
    )


# -----------------------------
# multimodal（两路/三路）
# -----------------------------


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
        return feat


class MultiModalBinaryClassifier(nn.Module):
    """可选分支的多模态融合网络（输出 logits）。"""

    def __init__(
        self,
        use_meta: bool,
        use_image: bool,
        use_text: bool,
        meta_input_dim: int = 0,
        image_embedding_dim: int = 0,
        text_embedding_dim: int = 0,
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
        fusion_hidden_dim: int | None = None,
        fusion_dropout: float = 0.9,
    ) -> None:
        super().__init__()

        self.use_meta = bool(use_meta)
        self.use_image = bool(use_image)
        self.use_text = bool(use_text)
        if not (self.use_meta or self.use_image or self.use_text):
            raise ValueError("至少需要开启一个分支。")

        self.meta: Optional[MetaMLPEncoder] = None
        self.image: Optional[ImageCNNEncoder] = None
        self.text: Optional[TextCNNEncoder] = None

        fusion_in_dim = 0
        if self.use_meta:
            self.meta = MetaMLPEncoder(
                input_dim=int(meta_input_dim),
                hidden_dim=int(meta_hidden_dim),
                dropout=float(meta_dropout),
            )
            fusion_in_dim += int(self.meta.output_dim)
        if self.use_image:
            self.image = ImageCNNEncoder(
                embedding_dim=int(image_embedding_dim),
                conv_channels=int(image_conv_channels),
                conv_kernel_size=int(image_conv_kernel_size),
                input_dropout=float(image_input_dropout),
                dropout=float(image_dropout),
                use_batch_norm=bool(image_use_batch_norm),
            )
            fusion_in_dim += int(self.image.output_dim)
        if self.use_text:
            self.text = TextCNNEncoder(
                embedding_dim=int(text_embedding_dim),
                conv_kernel_size=int(text_conv_kernel_size),
                input_dropout=float(text_input_dropout),
                dropout=float(text_dropout),
                use_batch_norm=bool(text_use_batch_norm),
            )
            fusion_in_dim += int(self.text.output_dim)

        if fusion_in_dim <= 0:
            raise ValueError("fusion_in_dim 需要 > 0")
        if fusion_hidden_dim is None:
            fusion_hidden_dim = int(fusion_in_dim * 2)
        if fusion_hidden_dim <= 0:
            raise ValueError(f"fusion_hidden_dim 需要 > 0，但得到 {fusion_hidden_dim}")
        if fusion_dropout < 0.0 or fusion_dropout >= 1.0:
            raise ValueError("fusion_dropout 需要在 [0, 1) 之间")

        self.fusion_in_dim = int(fusion_in_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)

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
        x_meta: torch.Tensor | None = None,
        x_image: torch.Tensor | None = None,
        len_image: torch.Tensor | None = None,
        x_text: torch.Tensor | None = None,
        len_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []

        if self.use_meta:
            if self.meta is None or x_meta is None:
                raise ValueError("meta 分支已开启，但 x_meta 为空。")
            feats.append(self.meta(x_meta))

        if self.use_image:
            if self.image is None or x_image is None:
                raise ValueError("image 分支已开启，但 x_image 为空。")
            feats.append(self.image(x_image, lengths=len_image))

        if self.use_text:
            if self.text is None or x_text is None:
                raise ValueError("text 分支已开启，但 x_text 为空。")
            feats.append(self.text(x_text, lengths=len_text))

        fused = torch.cat(feats, dim=1)
        fused = torch.relu(self.fusion_fc(fused))
        fused = self.fusion_drop(fused)
        logits = self.head(fused).squeeze(-1)
        return logits


def build_multimodal_model(
    cfg: MlpConfig,
    use_meta: bool,
    use_image: bool,
    use_text: bool,
    meta_input_dim: int = 0,
    image_embedding_dim: int = 0,
    text_embedding_dim: int = 0,
) -> MultiModalBinaryClassifier:
    return MultiModalBinaryClassifier(
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        meta_input_dim=int(meta_input_dim) if use_meta else 0,
        image_embedding_dim=int(image_embedding_dim) if use_image else 0,
        text_embedding_dim=int(text_embedding_dim) if use_text else 0,
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
        fusion_dropout=float(getattr(cfg, "fusion_dropout", 0.9)),
    )
