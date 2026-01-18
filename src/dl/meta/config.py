# -*- coding: utf-8 -*-
"""
配置文件：
- 统一管理路径与超参数，避免在主程序里硬编码
- 默认按你的需求：2023/2024 训练，2025 验证+测试
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MetaDLConfig:
    # -----------------------------
    # 运行相关
    # -----------------------------
    # 不想在命令行里写任何参数的话，就在这里改：
    # - None：run_id 只用时间戳
    # - 字符串：会拼到 run_id 后面，便于区分实验
    run_name: Optional[str] = None

    # -----------------------------
    # 数据与路径
    # -----------------------------
    data_csv: str = "data/metadata/now_processed.csv"
    experiment_root: str = "experiments/meta_dl"

    train_years: Tuple[int, ...] = (2023, 2024)
    eval_year: int = 2025

    # -----------------------------
    # 列配置
    # -----------------------------
    drop_cols: Tuple[str, ...] = ("project_id",)
    categorical_cols: Tuple[str, ...] = ("category", "country", "currency")
    numeric_cols: Tuple[str, ...] = ("duration_days", "log_usd_goal")
    target_col: str = "state"

    # -----------------------------
    # 划分策略
    # -----------------------------
    # 你要求“2025 作为测试集和验证集”，默认直接复用同一份 2025 数据
    use_same_eval_for_val_and_test: bool = False
    # 如果不复用，则从 2025 中再切一刀：val_ratio_in_eval 给验证集比例，其余为测试集
    val_ratio_in_eval: float = 0.5

    # -----------------------------
    # 模型（PyTorch MLP）超参
    # -----------------------------
    # 网络结构：每个数字代表一层隐藏层的宽度，例如 (256,128,64)
    hidden_layer_sizes: Tuple[int, ...] = (256, 128, 64)
    activation: str = "relu"
    # 对应 Adam 的 weight_decay（L2 正则），保留字段名 alpha 以兼容旧配置
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    batch_size: int = 256
    # 结构可调项：dropout / batchnorm
    dropout: float = 0.0
    use_batch_norm: bool = False

    max_epochs: int = 50
    early_stop_patience: int = 10
    # 可选：val_auc / val_loss
    metric_for_best: str = "val_auc"

    # -----------------------------
    # 其他
    # -----------------------------
    threshold: float = 0.5
    random_seed: int = 42
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
