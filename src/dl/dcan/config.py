# -*- coding: utf-8 -*-
"""
DCAN 配置（图文 Cross-Attention baseline）：

目标：
- 固定使用 image + text 双模态输入（不提供 use_image/use_text 开关）
- 仅保留 use_meta 开关：meta 分支仅在融合阶段 concat，不参与 DCAN 注意力交互

运行方式（在项目根目录）：
- 使用默认配置：
  `conda run -n crowdfunding python src/dl/dcan/main.py`
- 指定 run_name / 嵌入类型 / 显卡：
  `conda run -n crowdfunding python src/dl/dcan/main.py --run-name clip --image-embedding-type clip --text-embedding-type clip --device cuda:0`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DcanConfig:
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
    experiment_root: str = "experiments/meta"

    # -----------------------------
    # 分支开关
    # -----------------------------
    use_meta: bool = True
    use_attr: bool = True  # 是否使用 token 属性（文本长度/图片面积）

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
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    shuffle_before_split: bool = True

    # -----------------------------
    # 嵌入类型
    # -----------------------------
    image_embedding_type: str = "clip"  # clip / siglip / resnet
    text_embedding_type: str = "clip"  # bge / clip / siglip

    # -----------------------------
    # 统一序列截断（prefix 先并入统一序列，再整体截断）
    # -----------------------------
    max_seq_len: int = 40
    truncation_strategy: str = "first"  # first / random（random 需可复现）

    # -----------------------------
    # 缺失处理
    # -----------------------------
    # 当项目目录或必需文件缺失时：
    # - skip：跳过该样本
    # - error：直接报错（默认，更利于保证实验一致性）
    missing_strategy: str = "error"

    # -----------------------------
    # 模型结构超参
    # -----------------------------
    # meta 分支
    meta_hidden_dim: int = 256
    meta_dropout: float = 0.3

    # DCAN（图文交互）
    d_model: int = 256
    num_cross_layers: int = 2
    cross_ffn_dropout: float = 0.1

    # 融合 head（可手动配置；<=0 时按 2 * fusion_in_dim 自动计算）
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.5

    # -----------------------------
    # 训练超参
    # -----------------------------
    alpha: float = 4e-4  # weight_decay（AdamW）
    learning_rate_init: float = 5e-4
    batch_size: int = 256

    max_epochs: int = 80
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    lr_scheduler_min_lr: float = 1e-5

    max_grad_norm: float = 1.0
    random_seed: int = 42
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
