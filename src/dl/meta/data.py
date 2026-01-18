# -*- coding: utf-8 -*-
"""
数据加载与预处理模块：
- 读取 now_processed.csv
- 按 year 划分：2023/2024 训练，2025 验证与测试
- 删除 project_id
- category/country/currency 做 one-hot
- duration_days/log_usd_goal 做标准化
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
        missing = [c for c in [*self.categorical_cols, *self.numeric_cols] if c not in df.columns]
        if missing:
            raise ValueError(f"预处理器 fit 缺少列：{missing}")

        self.categories_.clear()
        for col in self.categorical_cols:
            values = df[col].dropna().unique().tolist()
            # 与 sklearn OneHotEncoder 的默认行为一致：对类别进行排序，保证稳定性
            self.categories_[col] = sorted(values, key=lambda x: str(x))

        self.numeric_mean_.clear()
        self.numeric_std_.clear()
        for col in self.numeric_cols:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            mean = float(np.nanmean(x))
            std = float(np.nanstd(x))
            if not np.isfinite(mean):
                mean = 0.0
            if not np.isfinite(std) or std <= 0.0:
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
        if not self.feature_names_:
            raise RuntimeError("预处理器尚未 fit，无法 transform。")

        missing = [c for c in [*self.categorical_cols, *self.numeric_cols] if c not in df.columns]
        if missing:
            raise ValueError(f"预处理器 transform 缺少列：{missing}")

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
            num_df = work[self.numeric_cols].copy()
            for col in self.numeric_cols:
                x = pd.to_numeric(num_df[col], errors="coerce").to_numpy(dtype=np.float32)
                mean = float(self.numeric_mean_.get(col, 0.0))
                std = float(self.numeric_std_.get(col, 1.0))
                x = np.where(np.isfinite(x), x, mean).astype(np.float32, copy=False)
                num_df[col] = (x - mean) / std
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


def _validate_and_clean(df: pd.DataFrame, cfg: MetaDLConfig) -> pd.DataFrame:
    """清洗与类型转换：缺失值、类型、目标值范围等。"""
    required_cols = (
        ("year",)
        + cfg.categorical_cols
        + cfg.numeric_cols
        + (cfg.target_col,)
        + cfg.drop_cols
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")

    work = df.copy()

    # 数值列转为 float，无法解析的设为 NaN
    for c in cfg.numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # year/target 转为 int，无法解析的设为 NaN
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work[cfg.target_col] = pd.to_numeric(work[cfg.target_col], errors="coerce")

    # 去掉关键列为空的样本
    key_cols = ["year", *cfg.categorical_cols, *cfg.numeric_cols, cfg.target_col]
    work = work.dropna(subset=key_cols).copy()

    # 明确类型
    work["year"] = work["year"].astype(int)
    work[cfg.target_col] = work[cfg.target_col].astype(int)

    # 只保留二分类标签 0/1
    work = work[work[cfg.target_col].isin([0, 1])].copy()

    return work


def split_by_year(
    df: pd.DataFrame, cfg: MetaDLConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按年划分数据集（2023/2024 -> train，2025 -> val/test）。"""
    train_df = df[df["year"].isin(cfg.train_years)].copy()
    eval_df = df[df["year"] == cfg.eval_year].copy()

    if train_df.empty:
        raise ValueError(f"训练集为空：year in {cfg.train_years}")
    if eval_df.empty:
        raise ValueError(f"验证/测试集为空：year == {cfg.eval_year}")

    if cfg.use_same_eval_for_val_and_test:
        val_df = eval_df
        test_df = eval_df
        return train_df, val_df, test_df

    # 从 2025 再切一刀：验证/测试
    if not (0.0 < cfg.val_ratio_in_eval < 1.0):
        raise ValueError("val_ratio_in_eval 需要在 (0, 1) 之间")

    # 在不依赖 sklearn 的前提下，做一个简单的分层随机划分
    rng = np.random.default_rng(cfg.random_seed)
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label, group in eval_df.groupby(cfg.target_col):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * cfg.val_ratio_in_eval))
        # 尽量保证 val/test 两边都有样本（当某类样本数 >= 2 时）
        if len(idx) >= 2:
            n_val = max(1, min(n_val, len(idx) - 1))
        val_idx.extend(idx[:n_val].tolist())
        test_idx.extend(idx[n_val:].tolist())

    val_df = eval_df.loc[val_idx].sample(frac=1.0, random_state=cfg.random_seed).copy()
    test_df = eval_df.loc[test_idx].sample(frac=1.0, random_state=cfg.random_seed).copy()
    return train_df, val_df, test_df


def prepare_data(csv_path: Path, cfg: MetaDLConfig) -> PreparedData:
    """一站式：读取->清洗->划分->预处理->输出 numpy 数组。"""
    raw_df = load_metadata(csv_path)
    df = _validate_and_clean(raw_df, cfg)

    # 划分
    train_df, val_df, test_df = split_by_year(df, cfg)

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
