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
    experiment_root: str = "experiments/ch1"

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
    shuffle_before_split: bool = False

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
    baseline_mode: str = "attn_pool"  # attn_pool / trm_no_pos

    d_model: int = 256

    # trm_no_pos（容量对照：不使用任何 position encoding）
    transformer_n_layers: int = 2
    transformer_n_heads: int = 8
    transformer_ffn_dim: int = 512
    transformer_dropout: float = 0.1

    # meta 分支（与 mlp baseline 一致）
    meta_hidden_dim: int = 256
    meta_dropout: float = 0.3

    # 融合 head（与 mlp baseline 一致：Linear→ReLU→Dropout→Linear）
    fusion_hidden_dim: Optional[int] = None  # None 表示自动取 2 * fusion_in_dim
    fusion_dropout: float = 0.9

    # -----------------------------
    # 训练超参（与 mlp baseline 对齐）
    # -----------------------------
    alpha: float = 5e-4  # weight_decay（L2）
    learning_rate_init: float = 5e-4
    batch_size: int = 1024

    max_epochs: int = 50
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    metric_for_best: str = "val_accuracy"  # val_accuracy / val_auc / val_loss

    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-5
    reset_early_stop_on_lr_change: bool = False

    max_grad_norm: float = 0.0

    threshold: float = 0.5
    random_seed: int = 22
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

