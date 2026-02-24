# -*- coding: utf-8 -*-
"""
seq 配置（Chapter 1：图文内容块序列建模）：

- 任务：Kickstarter 项目二分类（成功/失败），loss = BCEWithLogitsLoss
- 输入：预先计算好的 image/text embedding（不涉及原始模态特征）
- 可选：是否融合 meta 表格特征
- 命令行仅覆盖常用参数：--run-name / --baseline-mode / --use-meta / --use-prefix / --use-attr / --image-embedding-type / --text-embedding-type / --device / --gpu
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SeqConfig:
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
    experiment_root: str = "experiments/nometa"

    # -----------------------------
    # 模式
    # -----------------------------
    baseline_mode: str = "set_mean"  # set_mean / set_attn / trm_no_pos / trm_pos / trm_pos_shuffled

    # -----------------------------
    # meta 分支开关
    # -----------------------------
    use_meta: bool = True

    # -----------------------------
    # prefix token 开关
    # -----------------------------
    use_prefix: bool = True  # 是否使用 title_blurb 和 cover_image 作为 prefix token

    # -----------------------------
    # 序列属性开关
    # -----------------------------
    use_attr: bool = True  # 是否注入文本长度/图片面积属性（seq_attr）

    # -----------------------------
    # 列配置
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
    # 嵌入配置
    # -----------------------------
    image_embedding_type: str = "clip"  # clip / siglip / resnet
    text_embedding_type: str = "clip"  # bge / clip / siglip

    # 统一交替序列截断
    max_seq_len: int = 40
    truncation_strategy: str = "first"  # first / random

    # -----------------------------
    # 缺失处理
    # -----------------------------
    missing_strategy: str = "error"  # skip / error

    # -----------------------------
    # 模型结构超参
    # -----------------------------
    d_model: int = 256
    token_dropout: float = 0.33

    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    # meta encoder
    meta_hidden_dim: int = 128
    meta_dropout: float = 0.2

    # 分类头（Linear -> ReLU -> Dropout -> Linear(->1)）
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.5

    # -----------------------------
    # 训练超参
    # -----------------------------
    alpha: float = 4e-4  # weight_decay
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
        """将配置转换为字典格式"""
        return asdict(self)

