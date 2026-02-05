# -*- coding: utf-8 -*-
"""
mlp 配置（单文件 / 单类）：
- 三路输入：metadata / image / text
- 通过 use_meta / use_image / use_text 三个开关自由组合分支
- `fusion_hidden_dim` 在代码里根据实际启用的分支自动计算

运行方式（在项目根目录）：
- 使用默认配置：
  `conda run -n crowdfunding python src/dl/mlp/main.py`
- 指定 run_name / 嵌入类型 / 显卡：
  `conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --image-embedding-type clip --text-embedding-type clip --device cuda:0`
- 强制使用 CPU：
  `conda run -n crowdfunding python src/dl/mlp/main.py --device cpu`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MlpConfig:
    # -----------------------------
    # 运行相关
    # -----------------------------
    run_name: Optional[str] = "clip"
    device: str = "auto"  # auto / cpu / cuda / cuda:0 / cuda:1 ...

    # -----------------------------
    # 数据与路径
    # -----------------------------
    data_csv: str = "data/metadata/now_processed.csv"
    projects_root: str = "data/projects/now"
    experiment_root: str = "experiments/mlp"

    # -----------------------------
    # 分支开关
    # -----------------------------
    use_meta: bool = True
    use_image: bool = True
    use_text: bool = True

    # -----------------------------
    # 列配置（CSV）
    # -----------------------------
    id_col: str = "project_id"
    target_col: str = "state"
    drop_cols: Tuple[str, ...] = ("project_id", "time")
    categorical_cols: Tuple[str, ...] = ("category", "country", "currency")
    numeric_cols: Tuple[str, ...] = ("duration_days", "log_usd_goal")

    # -----------------------------
    # 划分策略
    # -----------------------------
    # 按比例切分 train/val/test（可选是否在切分前打乱）
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    shuffle_before_split: bool = False

    # -----------------------------
    # 嵌入配置（图片）
    # -----------------------------
    image_embedding_type: str = "clip"  # clip / siglip / resnet
    max_image_vectors: int = 20
    image_select_strategy: str = "first"  # first / random

    # -----------------------------
    # 嵌入配置（文本）
    # -----------------------------
    text_embedding_type: str = "clip"  # bge / clip / siglip
    max_text_vectors: int = 20
    text_select_strategy: str = "first"  # first / random

    # -----------------------------
    # 缺失处理
    # -----------------------------
    # 当项目目录或必须嵌入文件缺失时：
    # - skip：跳过该样本
    # - error：直接报错
    missing_strategy: str = "error"

    # -----------------------------
    # 模型结构超参
    # -----------------------------
    # meta 分支
    meta_hidden_dim: int = 256
    meta_dropout: float = 0.3

    # image 分支
    image_conv_channels: int = 256
    image_conv_kernel_size: int = 3
    image_input_dropout: float = 0.1
    image_dropout: float = 0.5
    image_use_batch_norm: bool = True

    # text 分支
    text_conv_kernel_size: int = 3
    text_input_dropout: float = 0.1
    text_dropout: float = 0.3
    text_use_batch_norm: bool = True

    # 融合 head（fusion_hidden_dim 自动计算）
    fusion_dropout: float = 0.9

    # -----------------------------
    # 训练超参
    # -----------------------------
    alpha: float = 5e-4  # weight_decay（L2）
    learning_rate_init: float = 5e-4
    batch_size: int = 1024

    max_epochs: int = 100
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    metric_for_best: str = "val_accuracy"  # val_accuracy / val_auc / val_loss

    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    reset_early_stop_on_lr_change: bool = False

    max_grad_norm: float = 0.0

    threshold: float = 0.5
    random_seed: int = 42
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
