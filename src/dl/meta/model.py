# -*- coding: utf-8 -*-
"""
模型定义模块：
这里使用 PyTorch 手写 MLP（多层感知机）来做二分类。

说明：
- hidden_layer_sizes 控制网络层数与每层神经元数量
- 激活函数 activation：relu / tanh / sigmoid 等
- 输出为 logits，训练时使用 BCEWithLogitsLoss（等价于二分类交叉熵）
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

from config import MetaDLConfig


def _build_activation(name: str) -> nn.Module:
    """把字符串激活函数名映射为 PyTorch 模块。"""
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

        # 二分类：输出 1 个 logit
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """对线性层做一个相对稳定的初始化，便于训练收敛。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(-1)


def build_mlp(cfg: MetaDLConfig, input_dim: int) -> MLPBinaryClassifier:
    """根据配置创建 PyTorch MLP 二分类模型。"""
    return MLPBinaryClassifier(
        input_dim=input_dim,
        hidden_sizes=cfg.hidden_layer_sizes,
        activation=cfg.activation,
        dropout=getattr(cfg, "dropout", 0.0),
        use_batch_norm=getattr(cfg, "use_batch_norm", False),
    )
