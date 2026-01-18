# -*- coding: utf-8 -*-
"""
配置文件：
- 仅使用文本嵌入做二分类（预测项目是否成功，state=1）
- CSV 只用于读取 project_id 与标签 state
- 支持三种文本嵌入类型：bge / clip / siglip

输入构建规则：
- 必须存在 title_blurb_{type}.npy（通常为 (1, D) 或 (2, D)）
- text_{type}.npy 可以缺失；若存在且形状为 (N, D)，则拼接在 title_blurb 后，形成 (L, D) 的“堆叠向量集合”
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class TextDLConfig:
    # -----------------------------
    # 运行相关
    # -----------------------------
    run_name: Optional[str] = None

    # -----------------------------
    # 数据与路径
    # -----------------------------
    data_csv: str = "data/metadata/now_processed.csv"
    projects_root: str = "data/projects/now"
    experiment_root: str = "experiments/text_bge"

    # -----------------------------
    # 列配置
    # -----------------------------
    id_col: str = "project_id"
    target_col: str = "state"

    # -----------------------------
    # 文本嵌入配置
    # -----------------------------
    # 可选：bge / clip / siglip
    embedding_type: str = "bge"
    # text_{type}.npy 最多使用多少个向量（不含 title_blurb），<=0 表示全部使用
    max_text_vectors: int = 200
    # 当 text 向量数量 > max_text_vectors 时的截取策略：first / random
    # random 会基于 random_seed + project_id 做“可复现”的抽样
    text_select_strategy: str = "first"
    # 当必需文件缺失时的处理策略（例如：项目目录缺失 / title_blurb 缺失）：
    # - skip：跳过该样本
    # - error：直接报错
    missing_strategy: str = "error"

    # -----------------------------
    # 数据缓存（加速：避免每次都逐个 np.load）
    # -----------------------------
    use_cache: bool = True
    cache_dir: str = "experiments/text_bge/_cache"
    refresh_cache: bool = False
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
    # 结构（最后一个全连接层之前）：
    # Conv(96) -> MaxPool -> Dropout(0.3)
    # -> Conv(128) -> MaxPool -> Dropout(0.3)
    # -> Conv(256) -> MaxPool -> Dropout(0.3)
    # -----------------------------
    conv_kernel_size: int = 3
    # GlobalMaxPool 之后的全连接隐藏层维度（<=0 表示等于 256）
    fc_hidden_dim: int = 128
    input_dropout: float = 0.1
    dropout: float = 0.3
    use_batch_norm: bool = True

    alpha: float = 5e-4  # weight_decay（L2）
    learning_rate_init: float = 3e-4
    batch_size: int = 256

    max_epochs: int = 100
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    metric_for_best: str = "val_accuracy"  # val_accuracy / val_auc / val_loss

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

