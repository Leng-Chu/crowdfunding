# -*- coding: utf-8 -*-
"""
res 配置（Residual Baselines）：

- 任务：Kickstarter 项目二分类（成功/失败），loss = BCEWithLogitsLoss
- 输入：预先计算好的 image/text embedding（不涉及原始模态特征）
- 三分支：
  1) v_meta：metadata 表格特征（one-hot + 数值标准化，来自 now_processed.csv）
  2) v_key：第一印象（title_blurb + cover_image embedding 融合；缺失用零向量并计数）
  3) h_seq：正文 content_sequence（不含 title_blurb/cover_image token）

说明：
- 本目录代码可独立运行，不得 import 其它 `src/dl/*` 子模块的代码
- 工程规范完全参考 `docs/dl_guidelines.md`（best checkpoint / 阈值选择 / 产物结构）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ResConfig:
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
    experiment_root: str = "experiments/res"

    # -----------------------------
    # 模式（实验组）
    # -----------------------------
    # 仅修改 baseline_mode 即可复现实验组（其余保持一致）
    baseline_mode: str = "res"  # mlp / res

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
    shuffle_before_split: bool = True

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
    d_model: int = 256
    token_dropout: float = 0.1

    transformer_num_layers: int = 3
    transformer_num_heads: int = 2
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.1

    # MetaEncoder（与 seq 对齐）：metadata 特征 -> FC -> Dropout，输出定长向量
    meta_hidden_dim: int = 64
    meta_dropout: float = 0.45

    # First Impression（title/blurb/cover）分支的 dropout
    # 仅对 baseline_mode=mlp 生效：是否使用“第一印象”分支（v_key）
    # - True：与当前 res/mlp 默认行为一致（使用 v_key）
    # - False：应与 seq 的 trm_pos 且 use_prefix=False 完全一致
    use_first_impression: bool = True
    key_dropout: float = 0.5

    # -----------------------------
    # Head / MLP 超参
    # -----------------------------
    # baseline_mode=mlp：logit = Head( concat(h_seq, meta_enc(v_meta), [v_key]) )（是否包含 v_key 由 use_first_impression 控制）
    # 为了与 seq 模块对齐，分类头超参命名统一为 fusion_*。
    fusion_hidden_dim: int = 768
    fusion_dropout: float = 0.45

    # baseline_mode=res：z_base = MLP_base( concat(LN(h_seq), LN(meta_proj(v_meta))) )
    # 关键：MLP_base 结构需与 src/dl/seq/model.py 的融合头一致
    base_hidden_dim: int = 512
    base_dropout: float = 0.5

    # baseline_mode=res：z_res = MLP_prior( concat(LN(v_key), LN(meta_proj(v_meta)), LN(v_key⊙meta_proj(v_meta))) )
    prior_hidden_dim: int = 256
    prior_dropout: float = 0.5

    # baseline_mode=res：z = z_base + delta_scale * z_res（delta_scale 为可学习标量）
    delta_scale_init: float = 0.0

    # baseline_mode=res：残差抑制（用于缓解“残差拟合过强/过拟合”）
    # - delta_scale 采用 tanh 参数化并限制在 [-delta_scale_max, delta_scale_max]
    # - z_res 采用 tanh 软裁剪并限制在 [-residual_logit_max, residual_logit_max]
    # - 可选：基于 |z_base| 的“置信门控”（base 越自信，残差越小）
    #
    # 经验建议：先保持 head/base/prior 的容量不变，仅通过下面 3 个超参做抑制：
    # - delta_scale_max（建议 0.3~0.8）
    # - residual_logit_max（建议 1.0~3.0）
    # - residual_gate_mode="conf"（默认开启）
    delta_scale_max: float = 0.5
    residual_logit_max: float = 2.0
    residual_gate_mode: str = "conf"  # none / conf
    residual_gate_scale_init: float = 1.0
    residual_gate_bias_init: float = 0.0
    residual_detach_base_in_gate: bool = True

    # -----------------------------
    # 训练超参（与 seq 训练流程对齐）
    # -----------------------------
    alpha: float = 4e-6  # weight_decay（AdamW）
    learning_rate_init: float = 2e-4
    batch_size: int = 256

    max_epochs: int = 80
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    lr_scheduler_min_lr: float = 1e-5

    max_grad_norm: float = 1.0
    random_seed: int = 72
    save_plots: bool = True

    # baseline_mode=res 的残差调试统计开关（写入 history.csv，并打印到 train.log）
    debug_residual_stats: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
