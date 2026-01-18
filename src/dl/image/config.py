# -*- coding: utf-8 -*-
"""
配置文件：
- 仅使用图片嵌入做二分类（预测项目是否成功，state=1）
- 不使用文本与项目元数据特征；CSV 只用于读取 project_id 与标签 state
- 支持三种图片嵌入类型：clip / siglip / resnet

输入构建规则：
- 必须存在 cover_image_{type}.npy（形状通常为 (1, D)）
- image_{type}.npy 可以缺失；若存在且形状为 (N, D)，则在 cover 后拼接，形成 (1+N, D) 的“堆叠向量集合”

模型结构（如你给的示意图）：
stacking -> Conv(256) -> MaxPool -> Dropout(0.5) -> Conv(256) -> MaxPool -> Dropout(0.5) -> 预测

默认使用：
- data/metadata/test.csv
- data/projects/test/<project_id>/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class ImageDLConfig:
    # -----------------------------
    # 运行相关
    # -----------------------------
    run_name: Optional[str] = None

    # -----------------------------
    # 数据与路径
    # -----------------------------
    data_csv: str = "data/metadata/test.csv"
    projects_root: str = "data/projects/test"
    experiment_root: str = "experiments/image_dl"

    # -----------------------------
    # 列配置
    # -----------------------------
    id_col: str = "project_id"
    target_col: str = "state"

    # -----------------------------
    # 图片嵌入配置
    # -----------------------------
    # 可选：clip / siglip / resnet
    embedding_type: str = "clip"
    # 当必须文件缺失时的处理策略（例如：项目目录缺失 / cover_image 缺失）：
    # - skip：跳过该样本
    # - error：直接报错
    missing_strategy: str = "error"

    # -----------------------------
    # 划分策略
    # -----------------------------
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle_before_split: bool = False

    # -----------------------------
    # 模型（PyTorch 1D CNN）超参
    # -----------------------------
    conv_channels: int = 256
    conv_kernel_size: int = 3
    dropout: float = 0.5
    use_batch_norm: bool = False

    alpha: float = 1e-4  # Adam 的 weight_decay（L2）
    learning_rate_init: float = 5e-4
    batch_size: int = 256

    max_epochs: int = 50
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    metric_for_best: str = "val_accuracy"  # val_accuracy / val_auc / val_loss

    # 学习率自适应
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    reset_early_stop_on_lr_change: bool = True

    # 训练稳定性
    max_grad_norm: float = 0.0

    # 其他
    threshold: float = 0.5
    random_seed: int = 42
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
