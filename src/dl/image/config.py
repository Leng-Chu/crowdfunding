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
    # image_{type}.npy 最多使用多少个向量（不含 cover），<=0 表示全部使用
    # 适当限制可以降低过拟合，并显著减少 padding 与训练显存/内存占用
    max_image_vectors: int = 12
    # 当 image 向量数量 > max_image_vectors 时的截取策略：first / random
    # random 会基于 random_seed + project_id 做“可复现”的抽样
    image_select_strategy: str = "first"
    # 当必须文件缺失时的处理策略（例如：项目目录缺失 / cover_image 缺失）：
    # - skip：跳过该样本
    # - error：直接报错
    missing_strategy: str = "error"

    # -----------------------------
    # 数据缓存（加速：避免每次都逐个 np.load）
    # -----------------------------
    use_cache: bool = True
    cache_dir: str = "experiments/image_dl/_cache"
    # True：忽略缓存并强制重建
    refresh_cache: bool = False
    # True：np.savez_compressed（更省空间但更慢）；False：np.savez（更快）
    cache_compress: bool = False

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
    # 减小通道数、加大 dropout/weight_decay 通常能缓解过拟合
    conv_channels: int = 128
    conv_kernel_size: int = 3
    # GlobalMaxPool 之后的全连接隐藏层维度（<=0 表示等于 conv_channels）
    fc_hidden_dim: int = 128
    # 输入层 dropout（对 embedding 做 dropout，进一步正则化）
    input_dropout: float = 0.1
    dropout: float = 0.6
    use_batch_norm: bool = True

    alpha: float = 5e-4  # weight_decay（L2）
    learning_rate_init: float = 3e-4
    batch_size: int = 256

    max_epochs: int = 50
    early_stop_patience: int = 5
    early_stop_min_epochs: int = 3
    # 过拟合时 val_accuracy 可能不敏感，优先用 val_loss
    metric_for_best: str = "val_loss"  # val_accuracy / val_auc / val_loss

    # 学习率自适应
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    reset_early_stop_on_lr_change: bool = False

    # 训练稳定性
    max_grad_norm: float = 0.0

    # 其他
    threshold: float = 0.5
    random_seed: int = 42
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
