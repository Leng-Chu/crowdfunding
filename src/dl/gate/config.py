# -*- coding: utf-8 -*-
"""
gate 配置（两阶段门控融合）：

目标：在 `src/dl/mlp` 的工程化框架基础上，提供一种固定三分支结构的多模态二分类模型：

1) meta 分支：与 mlp 中一致（表格元数据 -> MLP 编码）
2) 第一印象分支：title_blurb + cover_image 的嵌入，经 MLP 融合得到向量
3) 图文序列分支：text + image（不包含 title_blurb 与 cover_image）的嵌入序列，经编码后用 MLP 融合得到向量
4) 最终融合：两阶段门控网络融合三路向量，并用线性分类层输出成功概率（logits）

运行方式（在项目根目录）：
- 使用默认配置：
  `conda run -n crowdfunding python src/dl/gate/main.py`
- 指定 run_name / 嵌入类型 / 显卡：
  `conda run -n crowdfunding python src/dl/gate/main.py --run-name gate --image-embedding-type clip --text-embedding-type clip --device cuda:0`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GateConfig:
    # -----------------------------
    # 运行相关
    # -----------------------------
    run_name: Optional[str] = "gate"
    device: str = "auto"  # auto / cpu / cuda / cuda:0 / cuda:1 ...

    # -----------------------------
    # 数据与路径
    # -----------------------------
    data_csv: str = "data/metadata/now_processed.csv"
    projects_root: str = "data/projects/now"
    experiment_root: str = "experiments/gate"

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
    # split_mode:
    # - ratio:  按 (train/val/test) 比例切分（可选是否打乱）
    # - kfold:  打乱后仅动态切分 train/test 做 K 折交叉验证（每折的 val 会复用 train）
    split_mode: str = "ratio"  # ratio / kfold
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle_before_split: bool = False

    # K 折交叉验证（仅 split_mode=kfold 时生效）
    k_folds: int = 5
    k_fold_index: int = -1  # -1 表示跑全部折；>=0 表示只跑指定折（便于调试）
    kfold_shuffle: bool = True
    kfold_stratify: bool = True

    # -----------------------------
    # 嵌入配置
    # -----------------------------
    image_embedding_type: str = "clip"  # clip / siglip / resnet
    text_embedding_type: str = "clip"  # bge / clip / siglip

    # 图文序列分支：对 image/text 序列做截断
    max_image_vectors: int = 20
    image_select_strategy: str = "first"  # first / random
    max_text_vectors: int = 20
    text_select_strategy: str = "first"  # first / random

    # -----------------------------
    # 缺失处理
    # -----------------------------
    # 当项目目录或必须嵌入文件（cover_image/title_blurb）缺失时：
    # - skip：跳过该样本
    # - error：直接报错
    #
    # 注意：图文序列（image/text）允许缺失；缺失时将该序列视为空序列（长度=0）。
    missing_strategy: str = "error"

    # -----------------------------
    # 数据缓存（加速：避免每次逐个 np.load）
    # -----------------------------
    use_cache: bool = True
    cache_dir: str = "experiments/gate/_cache"

    # -----------------------------
    # 模型结构超参
    # -----------------------------
    # meta 分支
    meta_hidden_dim: int = 256
    meta_dropout: float = 0.3

    # 图文序列：编码器（对齐 mlp 中的 CNN Encoder）
    image_conv_channels: int = 256
    image_conv_kernel_size: int = 3
    image_input_dropout: float = 0.1
    image_dropout: float = 0.5
    image_use_batch_norm: bool = True

    text_conv_kernel_size: int = 3
    text_input_dropout: float = 0.1
    text_dropout: float = 0.3
    text_use_batch_norm: bool = True

    # 三路向量统一维度（门控融合工作在该维度）
    gate_dim: int = 256

    # 第一印象分支：MLP 融合
    impression_mlp_hidden_dim: int = 512
    impression_dropout: float = 0.3

    # 图文序列分支：MLP 融合
    seq_fuse_hidden_dim: int = 512
    seq_fuse_dropout: float = 0.3

    # 两阶段门控
    gate_hidden_dim: int = 256
    gate_dropout: float = 0.1
    gate_use_layer_norm: bool = True

    # 分类层
    classifier_dropout: float = 0.2

    # -----------------------------
    # 训练超参
    # -----------------------------
    alpha: float = 5e-4  # weight_decay（L2）
    learning_rate_init: float = 3e-4
    batch_size: int = 512

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

