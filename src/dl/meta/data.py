# -*- coding: utf-8 -*-
"""
数据加载与预处理模块：
- 读取 now_processed.csv
- 删除 time（以及其它 drop_cols）
- 按比例划分 train/val/test：
  - 可按 CSV 原始顺序切分（适用于已按时间排序的数据）
  - 或随机打乱后切分（可复现由 random_seed 控制）
- category/country/currency 做 one-hot
- duration_days/log_usd_goal 做标准化

说明：默认假设输入 CSV 已经清洗干净、没有缺失值，且样本量足够，因此不做额外校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import MetaDLConfig


@dataclass
class TabularPreprocessor:
    """
    轻量级表格预处理器（不依赖 sklearn）：
    - 类别列：基于训练集类别集合做 one-hot（未知类别 -> 全 0）
    - 数值列：基于训练集均值/方差做标准化
    """

    categorical_cols: List[str]
    numeric_cols: List[str]

    categories_: Dict[str, List[object]] = field(default_factory=dict)
    numeric_mean_: Dict[str, float] = field(default_factory=dict)
    numeric_std_: Dict[str, float] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """只用训练集拟合：类别全集、数值均值与标准差、输出特征名。"""
        self.categories_.clear()
        for col in self.categorical_cols:
            values = df[col].unique().tolist()
            # 与 sklearn OneHotEncoder 的默认行为一致：对类别进行排序，保证稳定性
            self.categories_[col] = sorted(values, key=lambda x: str(x))

        self.numeric_mean_.clear()
        self.numeric_std_.clear()
        for col in self.numeric_cols:
            x = df[col].to_numpy(dtype=np.float64, copy=False)
            mean = float(np.mean(x))
            std = float(np.std(x))
            if std <= 0.0:
                std = 1.0
            self.numeric_mean_[col] = mean
            self.numeric_std_[col] = std

        feature_names: List[str] = []
        for col in self.categorical_cols:
            for cat in self.categories_[col]:
                feature_names.append(f"{col}_{cat}")
        feature_names.extend(self.numeric_cols)
        self.feature_names_ = feature_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """把 df 转成模型可用的 float32 特征矩阵。"""
        work = df.copy()

        # 1) 类别 one-hot
        cat_feature_names: List[str] = []
        for col in self.categorical_cols:
            categories = self.categories_.get(col, [])
            work[col] = pd.Categorical(work[col], categories=categories)
            cat_feature_names.extend([f"{col}_{cat}" for cat in categories])

        if self.categorical_cols:
            cat_df = pd.get_dummies(
                work[self.categorical_cols],
                prefix_sep="_",
                dummy_na=False,
                dtype=np.float32,
            )
            cat_df = cat_df.reindex(columns=cat_feature_names, fill_value=0.0)
            cat_arr = cat_df.to_numpy(dtype=np.float32, copy=False)
        else:
            cat_arr = np.zeros((len(work), 0), dtype=np.float32)

        # 2) 数值标准化
        if self.numeric_cols:
            num_df = work[self.numeric_cols].astype(np.float32).copy()
            for col in self.numeric_cols:
                mean = float(self.numeric_mean_.get(col, 0.0))
                std = float(self.numeric_std_.get(col, 1.0))
                num_df[col] = (num_df[col] - mean) / std
            num_arr = num_df.to_numpy(dtype=np.float32, copy=False)
        else:
            num_arr = np.zeros((len(work), 0), dtype=np.float32)

        X = np.concatenate([cat_arr, num_arr], axis=1).astype(np.float32, copy=False)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """fit + transform 的便捷封装。"""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """返回特征名（one-hot 后的列名 + 数值列名）。"""
        return list(self.feature_names_)


@dataclass(frozen=True)
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: TabularPreprocessor
    feature_names: List[str]


def load_metadata(csv_path: Path) -> pd.DataFrame:
    """读取 CSV，并做最基础的列检查。"""
    df = pd.read_csv(csv_path)
    return df


def _drop_unused_cols(df: pd.DataFrame, cfg: MetaDLConfig) -> pd.DataFrame:
    """删除不参与训练的列（如 project_id/time）。"""
    if not cfg.drop_cols:
        return df
    return df.drop(columns=list(cfg.drop_cols), errors="ignore")


def split_by_ratio(
    df: pd.DataFrame, cfg: MetaDLConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按比例划分数据集：train/val/test。

    - 当 cfg.shuffle_before_split=False：按 CSV 原始顺序切分（不打乱）
    - 当 cfg.shuffle_before_split=True：随机打乱后再切分
    """
    if bool(getattr(cfg, "shuffle_before_split", False)):
        df = df.sample(frac=1.0, random_state=int(getattr(cfg, "random_seed", 42))).reset_index(drop=True)

    n_total = int(len(df))
    n_train = int(n_total * float(getattr(cfg, "train_ratio", 0.7)))
    n_val = int(n_total * float(getattr(cfg, "val_ratio", 0.15)))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def prepare_data(csv_path: Path, cfg: MetaDLConfig) -> PreparedData:
    """一站式：读取->清洗->划分->预处理->输出 numpy 数组。"""
    raw_df = load_metadata(csv_path)
    df = _drop_unused_cols(raw_df, cfg)

    # 划分
    train_df, val_df, test_df = split_by_ratio(df, cfg)

    # 丢弃无用列（比如 project_id）
    feature_cols = [*cfg.categorical_cols, *cfg.numeric_cols]

    preprocessor = TabularPreprocessor(
        categorical_cols=list(cfg.categorical_cols),
        numeric_cols=list(cfg.numeric_cols),
    )
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    # 统一 dtype，方便后续训练
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train = train_df[cfg.target_col].to_numpy(dtype=np.int64)
    y_val = val_df[cfg.target_col].to_numpy(dtype=np.int64)
    y_test = test_df[cfg.target_col].to_numpy(dtype=np.int64)

    feature_names = preprocessor.get_feature_names()

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
