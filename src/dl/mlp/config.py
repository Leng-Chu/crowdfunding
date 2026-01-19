# -*- coding: utf-8 -*-
"""
配置文件：
- 三路输入：metadata / image / text
- metadata 来自 data/metadata/*.csv（表格特征）
- image/text 来自 data/projects/.../<project_id>/ 下的 .npy 嵌入

嵌入类型选项（与现有分支保持一致）：
- 图片：clip / siglip / resnet
- 文本：bge / clip / siglip

网络结构（参考示意图，自上而下三路依次为 metadata、image、text）：
1) metadata：FC(256) -> Dropout(0.3)
2) image：stacking -> Conv(256) -> MaxPool -> Dropout(0.5) -> Conv(256) -> MaxPool -> Dropout(0.5)
3) text：stacking -> Conv(96) -> MaxPool -> Dropout(0.3)
            -> Conv(128) -> MaxPool -> Dropout(0.3)
            -> Conv(256) -> MaxPool -> Dropout(0.3)
融合：concat -> FC(1536) -> Dropout(0.9) -> 输出 logits
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MlpDLConfig:
    # -----------------------------
    # 运行相关
    # -----------------------------
    run_name: Optional[str] = None

    # -----------------------------
    # 数据与路径
    # -----------------------------
    data_csv: str = "data/metadata/now_processed.csv"
    projects_root: str = "data/projects/now"
    experiment_root: str = "experiments/mlp_dl"

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
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle_before_split: bool = False

    # -----------------------------
    # 嵌入配置（图片）
    # -----------------------------
    image_embedding_type: str = "siglip"  # clip / siglip / resnet
    max_image_vectors: int = 20
    image_select_strategy: str = "first"  # first / random

    # -----------------------------
    # 嵌入配置（文本）
    # -----------------------------
    text_embedding_type: str = "bge"  # bge / clip / siglip
    max_text_vectors: int = 200
    text_select_strategy: str = "first"  # first / random

    # -----------------------------
    # 缺失处理
    # -----------------------------
    # 当项目目录或必需嵌入文件缺失时：
    # - skip：跳过该样本
    # - error：直接报错
    missing_strategy: str = "error"

    # -----------------------------
    # 数据缓存（加速：避免每次都逐个 np.load）
    # -----------------------------
    use_cache: bool = True
    cache_dir: str = "experiments/mlp_dl/_cache"
    refresh_cache: bool = False
    cache_compress: bool = False

    # -----------------------------
    # 模型结构超参（按示意图默认）
    # -----------------------------
    # metadata 分支
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

    # 融合与分类头
    fusion_hidden_dim: int = 1536
    fusion_dropout: float = 0.9

    # -----------------------------
    # 训练超参
    # -----------------------------
    alpha: float = 5e-4  # weight_decay（L2）
    learning_rate_init: float = 3e-4
    batch_size: int = 256

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

