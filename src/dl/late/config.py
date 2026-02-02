# -*- coding: utf-8 -*-
"""
late 配置（单文件 / 单类）：
- 图像集合 + 文本集合分别建模
- 晚期融合（concat）后接与 mlp baseline 一致的分类头

运行方式（在项目根目录）：
- 使用默认配置：
  `conda run -n crowdfunding python src/dl/late/main.py`
- 指定嵌入类型 / baseline 模式 / 显卡：
  `conda run -n crowdfunding python src/dl/late/main.py --image-embedding-type clip --text-embedding-type bge --baseline-mode attn_pool --device cuda:0`
- 仅指定 GPU 序号（等价于 --device cuda:N）：
  `conda run -n crowdfunding python src/dl/late/main.py --gpu 1`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class LateConfig:
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
    experiment_root: str = "experiments/late"

    # -----------------------------
    # 分支开关
    # -----------------------------
    use_meta: bool = True

    # -----------------------------
    # 列配置（CSV）
    # -----------------------------
    id_col: str = "project_id"
    target_col: str = "state"
    drop_cols: Tuple[str, ...] = ("project_id", "time")
    categorical_cols: Tuple[str, ...] = ("category", "country", "currency")
    numeric_cols: Tuple[str, ...] = ("duration_days", "log_usd_goal")

    # -----------------------------
    # 划分策略（与 mlp baseline 对齐）
    # -----------------------------
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    shuffle_before_split: bool = True

    # -----------------------------
    # 嵌入类型（与 src/preprocess/embedding 对齐）
    # -----------------------------
    image_embedding_type: str = "clip"  # clip / siglip / resnet
    text_embedding_type: str = "clip"  # bge / clip / siglip

    # -----------------------------
    # 统一序列截断（按 content_sequence）
    # -----------------------------
    max_seq_len: int = 40
    truncation_strategy: str = "first"  # first / random（random 需可复现）

    # -----------------------------
    # 缺失处理
    # -----------------------------
    # 当项目目录或必须文件缺失时：
    # - skip：跳过该样本
    # - error：直接报错（默认，更利于保证实验一致性）
    missing_strategy: str = "error"

    # -----------------------------
    # 模型结构超参
    # -----------------------------
    baseline_mode: str = "attn_pool"  # mean_pool / attn_pool / trm_no_pos / trm_pos

    d_model: int = 256
    token_dropout: float = 0.33
    share_encoder: bool = True  # 与 seq 容量对齐：图/文共享同一套 set encoder 权重
    
    transformer_n_layers: int = 2
    transformer_n_heads: int = 4
    transformer_ffn_dim: int = 512
    transformer_dropout: float = 0.1

    # meta 分支
    meta_hidden_dim: int = 128
    meta_dropout: float = 0.2

    # 融合 head（Linear→ReLU→Dropout→Linear）
    fusion_hidden_dim: Optional[int] = 512  # 降低 head 容量，缓解轻度过拟合
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

