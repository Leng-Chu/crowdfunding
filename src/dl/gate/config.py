# -*- coding: utf-8 -*-
"""
gate 配置（Chapter 2：三分支 + Two-stage gated fusion）：

- 任务：Kickstarter 项目二分类（成功/失败），loss = BCEWithLogitsLoss
- 输入：预先计算好的 image/text embedding（不涉及原始模态特征）
- 三分支：
  1) Meta 分支（可选 use_meta）：表格特征 one-hot + 数值标准化 + 2 层 MLP
  2) First Impression 分支：title+blurb 与 cover 的跨模态交互
  3) Content Seq 分支：仅正文 content_sequence 的 trm_pos（sinusoidal PE + Transformer + masked mean）

说明：
- 本目录代码可独立运行，不得 import 或复用 `src/dl/seq` 的代码
- 命令行仅覆盖常用参数：--run-name / --baseline-mode / --use-meta / --image-embedding-type / --text-embedding-type / --device / --gpu
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GateConfig:
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
    experiment_root: str = "experiments/gate"

    # -----------------------------
    # 模式（实验组）
    # -----------------------------
    # 仅修改 baseline_mode 即可复现实验组（其余保持一致）
    baseline_mode: str = "two_stage"  # late_concat / stage1_only / stage2_only / two_stage / seq_only / key_only / meta_only

    # -----------------------------
    # meta 分支开关
    # -----------------------------
    use_meta: bool = True

    # -----------------------------
    # 列配置（CSV，与 mlp/seq 对齐）
    # -----------------------------
    id_col: str = "project_id"
    target_col: str = "state"
    drop_cols: Tuple[str, ...] = ("project_id", "time")
    categorical_cols: Tuple[str, ...] = ("category", "country", "currency")
    numeric_cols: Tuple[str, ...] = ("duration_days", "log_usd_goal")

    # -----------------------------
    # 划分策略（与 seq 对齐）
    # -----------------------------
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    shuffle_before_split: bool = False

    # -----------------------------
    # 嵌入配置
    # -----------------------------
    image_embedding_type: str = "clip"  # clip / siglip / resnet
    text_embedding_type: str = "clip"  # bge / clip / siglip

    # 正文 content_sequence 截断（仅作用于正文 tokens）
    max_seq_len: int = 40
    truncation_strategy: str = "first"  # first / random

    # -----------------------------
    # 缺失处理
    # -----------------------------
    missing_strategy: str = "error"  # skip / error

    # -----------------------------
    # 模型结构超参
    # -----------------------------
    d_model: int = 192
    token_dropout: float = 0.15

    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_dim_feedforward: int = 384
    transformer_dropout: float = 0.2

    # First Impression（title/blurb/cover）分支的 dropout
    key_dropout: float = 0.2

    # 融合表征的 dropout
    fusion_dropout: float = 0.2

    # MetaEncoder：Linear(d_meta_in->d)->ReLU->Dropout->Linear(d->d)->ReLU
    meta_dropout: float = 0.4

    # Head：Linear(d->d_head)->ReLU->Dropout->Linear(d_head->1)
    head_hidden_dim: int = 0  # <=0 表示自动取 2 * d_model
    head_dropout: float = 0.6

    # -----------------------------
    # 训练超参（与 seq 训练流程对齐）
    # -----------------------------
    alpha: float = 5e-3  # weight_decay（L2）
    learning_rate_init: float = 2e-4
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

    max_grad_norm: float = 1.0
    debug_gate_stats: bool = True

    threshold: float = 0.5
    random_seed: int = 42
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
