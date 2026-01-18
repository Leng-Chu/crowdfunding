# -*- coding: utf-8 -*-
"""
配置文件：
- 统一管理路径与超参数，避免在主程序里硬编码
- 默认按你的需求：按 CSV 的时间顺序做比例划分（不再依赖 year 列）
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

    # -----------------------------
    # 列配置
    # -----------------------------
    # now_processed.csv 已无 year，新增 time（已按时间排序）；time 不参与训练，直接丢弃
    drop_cols: Tuple[str, ...] = ("project_id", "time")
    categorical_cols: Tuple[str, ...] = ("category", "country", "currency")
    numeric_cols: Tuple[str, ...] = ("duration_days", "log_usd_goal")
    target_col: str = "state"

    # -----------------------------
    # 划分策略
    # -----------------------------
    # 训练/验证/测试比例（需要相加为 1.0）
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 划分开关：
    # - False：按 CSV 原始顺序切分（适用于按时间排序的数据）
    # - True：随机打乱后再切分（可复现实验由 random_seed 控制）
    shuffle_before_split: bool = False

    # -----------------------------
    # 模型（PyTorch MLP）超参
    # -----------------------------
    # 网络结构：每个数字代表一层隐藏层的宽度，例如 (256,128,64)
    hidden_layer_sizes: Tuple[int, ...] = (256, 128, 64)
    activation: str = "relu"
    # 对应 Adam 的 weight_decay（L2 正则），保留字段名 alpha 以兼容旧配置
    alpha: float = 1e-4
    # 经验上 5e-4 在该数据上更稳定，不容易在前几轮就过拟合
    learning_rate_init: float = 5e-4
    batch_size: int = 256
    # 结构可调项：dropout / batchnorm
    # 轻量正则：避免 val_loss 早早反弹
    dropout: float = 0.1
    use_batch_norm: bool = False

    max_epochs: int = 50
    # 早停：连续多少个 epoch 没有提升就停止
    early_stop_patience: int = 10
    # 早停：最少训练多少个 epoch 才允许触发（避免过早停止）
    early_stop_min_epochs: int = 5
    # 可选：val_accuracy / val_auc / val_loss
    # 你已保证类别绝对平衡，因此默认用 accuracy 做早停
    metric_for_best: str = "val_accuracy"

    # -----------------------------
    # 学习率自适应（自动调参的一种：只调学习率）
    # -----------------------------
    # 当验证集 logloss 长时间不下降时，自动降低学习率，避免“看起来不降就提前停”
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    # 当学习率发生下降时，是否把早停的 bad 计数清零，给模型“再试一次”的机会
    reset_early_stop_on_lr_change: bool = True

    # -----------------------------
    # 训练稳定性
    # -----------------------------
    # 梯度裁剪：>0 时启用（建议从 1.0 / 5.0 试起）
    max_grad_norm: float = 0.0

    # -----------------------------
    # 其他
    # -----------------------------
    threshold: float = 0.5
    random_seed: int = 42
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
