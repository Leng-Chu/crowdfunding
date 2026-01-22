# -*- coding: utf-8 -*-
"""
seq 配置（Chapter 1：图文内容块序列建模）：

- 任务：Kickstarter 项目二分类（成功/失败），loss = BCEWithLogitsLoss
- 输入：预先计算好的 image/text embedding（不涉及原始模态特征）
- 可选：是否融合 meta 表格特征（结构与处理方式与 mlp baseline 对齐）

说明：
- 本目录代码可独立运行，不得 import 或复用 `src/dl/mlp` 的代码
- 命令行仅覆盖常用参数：--run-name / --baseline-mode / --use-meta / --image-embedding-type / --text-embedding-type / --device / --gpu
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
    experiment_root: str = "experiments/seq"

    # -----------------------------
    # 模式（实验组）
    # -----------------------------
    # 仅修改 baseline_mode 即可复现实验组（其余保持一致）
    baseline_mode: str = "set_mean"  # set_mean / set_attn / trm_no_pos / trm_pos / trm_pos_shuffled

    # -----------------------------
    # meta 分支开关
    # -----------------------------
    use_meta: bool = True

    # -----------------------------
    # 列配置（CSV，与 mlp baseline 对齐）
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
    # 嵌入配置（必须与其它 baseline 使用同一套 embedding）
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
    # 数据缓存（加速：避免每次逐个 np.load / 解析图片）
    # -----------------------------
    use_cache: bool = True
    cache_dir: str = "experiments/seq/_cache"

    # -----------------------------
    # 模型结构超参
    # -----------------------------
    d_model: int = 256
    token_dropout: float = 0.0

    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    # meta encoder（与 mlp baseline 对齐）
    meta_hidden_dim: int = 256
    meta_dropout: float = 0.3

    # 分类头（与 mlp baseline 对齐：Linear -> ReLU -> Dropout -> Linear(->1)）
    fusion_hidden_dim: int = 0  # <=0 表示自动取 2 * fusion_in_dim
    fusion_dropout: float = 0.9

    # -----------------------------
    # 训练超参（与 mlp baseline 训练流程对齐）
    # -----------------------------
    alpha: float = 5e-4  # weight_decay（L2）
    learning_rate_init: float = 5e-4
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
