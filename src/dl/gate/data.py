# -*- coding: utf-8 -*-
"""
数据加载与特征构建（gate）：

- meta：从 CSV 读取，one-hot + 标准化
- 第一印象：cover_image_{image_type}.npy 与 title_blurb_{text_type}.npy（必需）
- 图文序列：image_{image_type}.npy 与 text_{text_type}.npy（允许缺失，缺失时按空序列处理）

说明：
- 参考 `src/dl/mlp/data.py` 的结构与缓存方式，但本模块是固定三分支，不再需要 use_* 开关。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from config import GateConfig


def _split_by_ratio(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    shuffle: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按比例切分 train/val/test。"""
    if bool(shuffle):
        df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    n_total = int(len(df))
    n_train = int(n_total * float(train_ratio))
    n_val = int(n_total * float(val_ratio))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def _get_split_mode(cfg: GateConfig) -> str:
    mode = str(getattr(cfg, "split_mode", "ratio") or "ratio").strip().lower()
    return mode or "ratio"


def _kfold_split_indices(
    n_total: int,
    n_splits: int,
    shuffle: bool,
    seed: int,
    y: np.ndarray | None = None,
    stratify: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    轻量级 KFold/StratifiedKFold（不依赖 sklearn）。

    返回：[(train_idx, test_idx), ...]，其中 train_idx/test_idx 都是一维 int64 索引数组。
    """
    n_total = int(n_total)
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"k_folds 需要 >= 2，当前={n_splits}")
    if n_total <= 0:
        raise ValueError("数据集为空，无法做 K 折切分。")
    if n_total < n_splits:
        raise ValueError(f"k_folds={n_splits} 大于样本数 {n_total}，无法分折。")

    rng = np.random.RandomState(int(seed))
    all_idx = np.arange(n_total, dtype=np.int64)

    folds: List[np.ndarray] = []
    use_stratify = bool(stratify) and y is not None
    if use_stratify:
        y = np.asarray(y).astype(int).reshape(-1)
        if int(y.shape[0]) != int(n_total):
            raise ValueError(f"y 长度不匹配：{y.shape[0]} vs {n_total}")

        buckets: List[List[int]] = [[] for _ in range(n_splits)]
        for cls in (0, 1):
            cls_idx = all_idx[y == int(cls)].copy()
            if bool(shuffle):
                rng.shuffle(cls_idx)
            for j, idx in enumerate(cls_idx.tolist()):
                buckets[j % n_splits].append(int(idx))

        for b in buckets:
            arr = np.asarray(b, dtype=np.int64)
            if arr.size > 1 and bool(shuffle):
                rng.shuffle(arr)
            folds.append(arr)
    else:
        perm = all_idx.copy()
        if bool(shuffle):
            rng.shuffle(perm)
        folds = [np.asarray(x, dtype=np.int64) for x in np.array_split(perm, n_splits)]

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for test_idx in folds:
        if int(test_idx.size) <= 0:
            continue
        mask = np.ones((n_total,), dtype=bool)
        mask[test_idx] = False
        train_idx = all_idx[mask]
        pairs.append((train_idx.astype(np.int64, copy=False), test_idx.astype(np.int64, copy=False)))

    if len(pairs) != int(n_splits):
        raise RuntimeError(f"K 折切分异常：期望 {n_splits} 折，实际得到 {len(pairs)} 折（可能存在空折）。")
    return pairs


def _normalize_project_id(value: Any) -> str:
    """把 project_id 统一转成用于文件夹名的字符串形式。"""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    return str(value).strip()


def _encode_binary_target(series: pd.Series) -> np.ndarray:
    """将标签编码为 0/1（兼容数值/字符串）。"""
    if pd.api.types.is_numeric_dtype(series):
        s = series.fillna(0)
        try:
            uniq = pd.unique(s)
            uniq_set = {int(v) for v in uniq if pd.notna(v)}
        except Exception:
            uniq_set = set()
        if uniq_set.issubset({0, 1}):
            return s.astype(int).to_numpy(dtype=np.int64)
        return (s.astype(float) > 0).astype(int).to_numpy(dtype=np.int64)
    s = series.fillna("").astype(str).str.strip().str.lower()
    return (s == "successful").astype(int).to_numpy(dtype=np.int64)


def _as_2d_embedding(arr: np.ndarray, name: str) -> np.ndarray:
    """将 (D,) / (1, D) / (N, D) 统一成 (N, D) 的 float32 矩阵。"""
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] <= 0:
        raise ValueError(f"{name} 的形状不合法：{x.shape}，期望 (D,) 或 (N, D)")
    return x.astype(np.float32, copy=False)


def _stable_hash_int(text: str) -> int:
    """把字符串稳定地映射为 int（用于可复现的抽样）。"""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


@dataclass
class TabularPreprocessor:
    """轻量级表格预处理器（不依赖 sklearn）：one-hot + 标准化。"""

    categorical_cols: List[str]
    numeric_cols: List[str]

    categories_: Dict[str, List[object]] = field(default_factory=dict)
    numeric_mean_: Dict[str, float] = field(default_factory=dict)
    numeric_std_: Dict[str, float] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.categories_.clear()
        for col in self.categorical_cols:
            values = df[col].unique().tolist()
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
        work = df.copy()

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
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names_)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "categorical_cols": list(self.categorical_cols),
            "numeric_cols": list(self.numeric_cols),
            "categories_": {k: list(v) for k, v in self.categories_.items()},
            "numeric_mean_": {k: float(v) for k, v in self.numeric_mean_.items()},
            "numeric_std_": {k: float(v) for k, v in self.numeric_std_.items()},
            "feature_names_": list(self.feature_names_),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "TabularPreprocessor":
        pre = cls(
            categorical_cols=[str(x) for x in state.get("categorical_cols", [])],
            numeric_cols=[str(x) for x in state.get("numeric_cols", [])],
        )
        pre.categories_ = {str(k): list(v) for k, v in dict(state.get("categories_", {})).items()}
        pre.numeric_mean_ = {str(k): float(v) for k, v in dict(state.get("numeric_mean_", {})).items()}
        pre.numeric_std_ = {str(k): float(v) for k, v in dict(state.get("numeric_std_", {})).items()}
        pre.feature_names_ = [str(x) for x in state.get("feature_names_", [])]
        return pre


@dataclass(frozen=True)
class PreparedGateData:
    # 标签与 id
    y_train: np.ndarray
    train_project_ids: List[str]
    y_val: np.ndarray
    val_project_ids: List[str]
    y_test: np.ndarray
    test_project_ids: List[str]

    # meta
    X_meta_train: np.ndarray
    X_meta_val: np.ndarray
    X_meta_test: np.ndarray
    meta_dim: int
    preprocessor: TabularPreprocessor
    feature_names: List[str]

    # 第一印象（向量）
    X_cover_train: np.ndarray
    X_cover_val: np.ndarray
    X_cover_test: np.ndarray
    X_title_blurb_train: np.ndarray
    X_title_blurb_val: np.ndarray
    X_title_blurb_test: np.ndarray

    # 图文序列（序列 + lengths）
    X_image_train: np.ndarray
    len_image_train: np.ndarray
    X_image_val: np.ndarray
    len_image_val: np.ndarray
    X_image_test: np.ndarray
    len_image_test: np.ndarray
    image_embedding_dim: int
    max_image_seq_len: int

    X_text_train: np.ndarray
    len_text_train: np.ndarray
    X_text_val: np.ndarray
    len_text_val: np.ndarray
    X_text_test: np.ndarray
    len_text_test: np.ndarray
    text_embedding_dim: int
    max_text_seq_len: int

    stats: Dict[str, int]


_GATE_CACHE_VERSION = 1


def _validate_embedding_types(cfg: GateConfig) -> Tuple[str, str]:
    img_type = (cfg.image_embedding_type or "").strip().lower()
    if img_type not in {"clip", "siglip", "resnet"}:
        raise ValueError(f"不支持的 image_embedding_type={cfg.image_embedding_type!r}，可选：clip/siglip/resnet")
    txt_type = (cfg.text_embedding_type or "").strip().lower()
    if txt_type not in {"bge", "clip", "siglip"}:
        raise ValueError(f"不支持的 text_embedding_type={cfg.text_embedding_type!r}，可选：bge/clip/siglip")
    return img_type, txt_type


def _select_vectors(seq: np.ndarray, max_vec: int, strategy: str, seed: int) -> np.ndarray:
    if int(max_vec) <= 0 or int(seq.shape[0]) <= int(max_vec):
        return seq

    strategy = str(strategy or "first").strip().lower()
    if strategy == "first":
        return seq[: int(max_vec)]
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(int(seq.shape[0]), size=int(max_vec), replace=False)
        idx = np.sort(idx)
        return seq[idx]
    raise ValueError(f"不支持的 select_strategy={strategy!r}，可选：first/random")


def _load_required_impression_vectors(
    project_dir: Path,
    cfg: GateConfig,
) -> Tuple[np.ndarray | None, np.ndarray | None, int, int]:
    """
    读取第一印象的两个“必需”向量：
    - cover_image_{image_type}.npy
    - title_blurb_{text_type}.npy

    返回：
    - cover_vec: (D_img,) 或 None
    - title_vec: (D_txt,) 或 None
    - skipped: 是否跳过（0/1）
    - missing_required: 是否缺失必需文件（0/1）
    """
    img_type, txt_type = _validate_embedding_types(cfg)
    cover_path = project_dir / f"cover_image_{img_type}.npy"
    tb_path = project_dir / f"title_blurb_{txt_type}.npy"

    missing_strategy = (cfg.missing_strategy or "").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={cfg.missing_strategy!r}，可选：skip/error")

    if not cover_path.exists() or not tb_path.exists():
        if missing_strategy == "skip":
            return None, None, 1, 1
        missing = []
        if not cover_path.exists():
            missing.append(cover_path.name)
        if not tb_path.exists():
            missing.append(tb_path.name)
        raise FileNotFoundError(f"项目 {project_dir.name} 缺少必需文件：{', '.join(missing)}")

    cover = _as_2d_embedding(np.load(cover_path), cover_path.name)
    tb = _as_2d_embedding(np.load(tb_path), tb_path.name)

    cover_vec = np.mean(cover, axis=0).astype(np.float32, copy=False)
    title_vec = np.mean(tb, axis=0).astype(np.float32, copy=False)
    return cover_vec, title_vec, 0, 0


def _load_optional_sequence(
    project_dir: Path,
    filename: str,
    expected_dim: int,
    max_vec: int,
    select_strategy: str,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """
    读取“可缺失”的序列向量文件；缺失时返回空序列。
    返回：
    - seq: (L, D)
    - missing: 该文件是否缺失（0/1）
    """
    path = project_dir / filename
    if not path.exists():
        return np.zeros((0, int(expected_dim)), dtype=np.float32), 1

    seq = _as_2d_embedding(np.load(path), path.name)
    if int(seq.shape[1]) != int(expected_dim):
        raise ValueError(
            f"项目 {project_dir.name} 的 {path.name} 维度不一致：期望 {expected_dim}，但得到 {seq.shape[1]}"
        )

    seq = _select_vectors(seq, max_vec=int(max_vec), strategy=str(select_strategy), seed=int(seed))
    return seq.astype(np.float32, copy=False), 0


def _build_features_for_split(
    df_split: pd.DataFrame,
    X_meta_all: np.ndarray,
    projects_root: Path,
    cfg: GateConfig,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, int],
    int,
    int,
    int,
    int,
]:
    project_ids: List[str] = [_normalize_project_id(v) for v in df_split[cfg.id_col].tolist()]
    y_all = _encode_binary_target(df_split[cfg.target_col])

    kept_idx: List[int] = []
    kept_ids: List[str] = []
    kept_labels: List[int] = []

    cover_vecs: List[np.ndarray] = []
    title_vecs: List[np.ndarray] = []

    img_seqs: List[np.ndarray] = []
    img_lens: List[int] = []
    txt_seqs: List[np.ndarray] = []
    txt_lens: List[int] = []

    missing_image_total = 0
    missing_text_total = 0
    missing_required_total = 0
    skipped_total = 0

    image_dim = 0
    text_dim = 0

    img_type, txt_type = _validate_embedding_types(cfg)
    img_seq_filename = f"image_{img_type}.npy"
    txt_seq_filename = f"text_{txt_type}.npy"

    for i, (pid, label) in enumerate(zip(project_ids, y_all)):
        if not pid:
            skipped_total += 1
            continue

        project_dir = projects_root / pid
        if not project_dir.exists():
            if (cfg.missing_strategy or "").strip().lower() == "skip":
                skipped_total += 1
                continue
            raise FileNotFoundError(f"找不到项目目录：{project_dir}")

        cover_vec, title_vec, skipped, missing_required = _load_required_impression_vectors(project_dir, cfg)
        if skipped:
            skipped_total += 1
            missing_required_total += int(missing_required)
            continue
        if cover_vec is None or title_vec is None:
            skipped_total += 1
            missing_required_total += 1
            continue

        if image_dim <= 0:
            image_dim = int(cover_vec.shape[0])
        if text_dim <= 0:
            text_dim = int(title_vec.shape[0])
        if int(cover_vec.shape[0]) != int(image_dim):
            raise ValueError(f"image_embedding_dim 不一致：期望 {image_dim}，但 {project_dir.name} 为 {cover_vec.shape[0]}")
        if int(title_vec.shape[0]) != int(text_dim):
            raise ValueError(f"text_embedding_dim 不一致：期望 {text_dim}，但 {project_dir.name} 为 {title_vec.shape[0]}")

        seed_base = int(getattr(cfg, "random_seed", 42)) + _stable_hash_int(str(project_dir.name))
        img_seq, missing_img = _load_optional_sequence(
            project_dir,
            filename=img_seq_filename,
            expected_dim=int(image_dim),
            max_vec=int(getattr(cfg, "max_image_vectors", 0)),
            select_strategy=str(getattr(cfg, "image_select_strategy", "first")),
            seed=seed_base,
        )
        txt_seq, missing_txt = _load_optional_sequence(
            project_dir,
            filename=txt_seq_filename,
            expected_dim=int(text_dim),
            max_vec=int(getattr(cfg, "max_text_vectors", 0)),
            select_strategy=str(getattr(cfg, "text_select_strategy", "first")),
            seed=seed_base + 17,
        )

        cover_vecs.append(cover_vec.reshape(1, -1))
        title_vecs.append(title_vec.reshape(1, -1))

        img_seqs.append(img_seq)
        img_lens.append(int(img_seq.shape[0]))
        txt_seqs.append(txt_seq)
        txt_lens.append(int(txt_seq.shape[0]))

        kept_idx.append(int(i))
        kept_ids.append(pid)
        kept_labels.append(int(label))

        missing_image_total += int(missing_img)
        missing_text_total += int(missing_txt)
        missing_required_total += int(missing_required)

    if not kept_ids:
        raise RuntimeError("该数据切分中没有可用样本（可能都被 skip 了或缺少必需嵌入文件）")

    idx_arr = np.asarray(kept_idx, dtype=np.int64)
    X_meta = np.asarray(X_meta_all[idx_arr], dtype=np.float32)

    X_cover = np.concatenate(cover_vecs, axis=0).astype(np.float32, copy=False)
    X_title = np.concatenate(title_vecs, axis=0).astype(np.float32, copy=False)

    max_img_len = int(max(img_lens)) if img_lens else 0
    max_txt_len = int(max(txt_lens)) if txt_lens else 0
    max_img_len = max(1, max_img_len)
    max_txt_len = max(1, max_txt_len)

    X_image = np.zeros((len(img_seqs), max_img_len, int(image_dim)), dtype=np.float32)
    for j, seq in enumerate(img_seqs):
        L = int(seq.shape[0])
        if L > 0:
            X_image[j, :L, :] = seq
    len_image = np.asarray(img_lens, dtype=np.int64)

    X_text = np.zeros((len(txt_seqs), max_txt_len, int(text_dim)), dtype=np.float32)
    for j, seq in enumerate(txt_seqs):
        L = int(seq.shape[0])
        if L > 0:
            X_text[j, :L, :] = seq
    len_text = np.asarray(txt_lens, dtype=np.int64)

    y_arr = np.asarray(kept_labels, dtype=np.int64)
    stats = {
        "skipped_samples": int(skipped_total),
        "missing_required_files": int(missing_required_total),
        "missing_image_sequence_files": int(missing_image_total),
        "missing_text_sequence_files": int(missing_text_total),
    }
    return (
        X_meta,
        X_cover,
        X_title,
        X_image,
        len_image,
        X_text,
        len_text,
        y_arr,
        kept_ids,
        stats,
        int(image_dim),
        int(text_dim),
        int(max_img_len),
        int(max_txt_len),
    )


def _prepare_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    projects_root: Path,
    cfg: GateConfig,
) -> PreparedGateData:
    feature_cols = [*cfg.categorical_cols, *cfg.numeric_cols]
    for col in [cfg.id_col, cfg.target_col, *feature_cols]:
        if col not in train_df.columns:
            raise ValueError(f"CSV 缺少列：{col!r}")

    preprocessor = TabularPreprocessor(
        categorical_cols=list(cfg.categorical_cols),
        numeric_cols=list(cfg.numeric_cols),
    )

    X_meta_train_all = preprocessor.fit_transform(train_df[feature_cols])
    X_meta_val_all = preprocessor.transform(val_df[feature_cols])
    X_meta_test_all = preprocessor.transform(test_df[feature_cols])
    feature_names = preprocessor.get_feature_names()
    meta_dim = int(X_meta_train_all.shape[1])

    (
        X_meta_train,
        X_cover_train,
        X_title_train,
        X_img_train,
        len_img_train,
        X_txt_train,
        len_txt_train,
        y_train,
        train_ids,
        train_stats,
        img_dim,
        txt_dim,
        max_img_train,
        max_txt_train,
    ) = _build_features_for_split(train_df, X_meta_train_all, projects_root, cfg)
    (
        X_meta_val,
        X_cover_val,
        X_title_val,
        X_img_val,
        len_img_val,
        X_txt_val,
        len_txt_val,
        y_val,
        val_ids,
        val_stats,
        img_dim_val,
        txt_dim_val,
        max_img_val,
        max_txt_val,
    ) = _build_features_for_split(val_df, X_meta_val_all, projects_root, cfg)
    (
        X_meta_test,
        X_cover_test,
        X_title_test,
        X_img_test,
        len_img_test,
        X_txt_test,
        len_txt_test,
        y_test,
        test_ids,
        test_stats,
        img_dim_test,
        txt_dim_test,
        max_img_test,
        max_txt_test,
    ) = _build_features_for_split(test_df, X_meta_test_all, projects_root, cfg)

    if int(img_dim_val) != int(img_dim) or int(img_dim_test) != int(img_dim):
        raise ValueError(f"image_embedding_dim 不一致：train={img_dim} val={img_dim_val} test={img_dim_test}")
    if int(txt_dim_val) != int(txt_dim) or int(txt_dim_test) != int(txt_dim):
        raise ValueError(f"text_embedding_dim 不一致：train={txt_dim} val={txt_dim_val} test={txt_dim_test}")

    stats: Dict[str, int] = {}
    for s in (train_stats, val_stats, test_stats):
        for k, v in s.items():
            stats[k] = int(stats.get(k, 0) + int(v))

    return PreparedGateData(
        y_train=y_train,
        train_project_ids=train_ids,
        y_val=y_val,
        val_project_ids=val_ids,
        y_test=y_test,
        test_project_ids=test_ids,
        X_meta_train=X_meta_train,
        X_meta_val=X_meta_val,
        X_meta_test=X_meta_test,
        meta_dim=int(meta_dim),
        preprocessor=preprocessor,
        feature_names=feature_names,
        X_cover_train=X_cover_train,
        X_cover_val=X_cover_val,
        X_cover_test=X_cover_test,
        X_title_blurb_train=X_title_train,
        X_title_blurb_val=X_title_val,
        X_title_blurb_test=X_title_test,
        X_image_train=X_img_train,
        len_image_train=len_img_train,
        X_image_val=X_img_val,
        len_image_val=len_img_val,
        X_image_test=X_img_test,
        len_image_test=len_img_test,
        image_embedding_dim=int(img_dim),
        max_image_seq_len=int(max(max_img_train, max_img_val, max_img_test)),
        X_text_train=X_txt_train,
        len_text_train=len_txt_train,
        X_text_val=X_txt_val,
        len_text_val=len_txt_val,
        X_text_test=X_txt_test,
        len_text_test=len_txt_test,
        text_embedding_dim=int(txt_dim),
        max_text_seq_len=int(max(max_txt_train, max_txt_val, max_txt_test)),
        stats=stats,
    )


def _make_cache_key(csv_path: Path, projects_root: Path, cfg: GateConfig, fold_index: int | None = None) -> str:
    stat = csv_path.stat()
    payload = {
        "cache_version": _GATE_CACHE_VERSION,
        "data_csv": str(csv_path.as_posix()),
        "csv_mtime": float(stat.st_mtime),
        "csv_size": int(stat.st_size),
        "projects_root": str(projects_root.as_posix()),
        "split": {
            "split_mode": _get_split_mode(cfg),
            "train_ratio": float(getattr(cfg, "train_ratio", 0.7)),
            "val_ratio": float(getattr(cfg, "val_ratio", 0.15)),
            "test_ratio": float(getattr(cfg, "test_ratio", 0.15)),
            "shuffle_before_split": bool(getattr(cfg, "shuffle_before_split", False)),
            "random_seed": int(getattr(cfg, "random_seed", 42)),
            "k_folds": int(getattr(cfg, "k_folds", 5)),
            "kfold_shuffle": bool(getattr(cfg, "kfold_shuffle", True)),
            "kfold_stratify": bool(getattr(cfg, "kfold_stratify", True)),
            "fold_index": None if fold_index is None else int(fold_index),
        },
        "embeddings": {
            "image_embedding_type": str(getattr(cfg, "image_embedding_type", "")),
            "text_embedding_type": str(getattr(cfg, "text_embedding_type", "")),
            "max_image_vectors": int(getattr(cfg, "max_image_vectors", 0)),
            "image_select_strategy": str(getattr(cfg, "image_select_strategy", "first")),
            "max_text_vectors": int(getattr(cfg, "max_text_vectors", 0)),
            "text_select_strategy": str(getattr(cfg, "text_select_strategy", "first")),
        },
        "missing_strategy": str(getattr(cfg, "missing_strategy", "")),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:16]
    return f"gate_{h}"


def _save_cache(cache_path: Path, prepared: PreparedGateData, meta: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    stats_json = json.dumps(prepared.stats, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    preprocessor_json = json.dumps(prepared.preprocessor.to_state_dict(), ensure_ascii=False, sort_keys=True)

    arrays: Dict[str, Any] = {
        "y_train": prepared.y_train.astype(np.int64, copy=False),
        "train_project_ids": np.asarray(prepared.train_project_ids, dtype=str),
        "y_val": prepared.y_val.astype(np.int64, copy=False),
        "val_project_ids": np.asarray(prepared.val_project_ids, dtype=str),
        "y_test": prepared.y_test.astype(np.int64, copy=False),
        "test_project_ids": np.asarray(prepared.test_project_ids, dtype=str),
        "meta_dim": np.asarray(int(prepared.meta_dim), dtype=np.int64),
        "image_embedding_dim": np.asarray(int(prepared.image_embedding_dim), dtype=np.int64),
        "text_embedding_dim": np.asarray(int(prepared.text_embedding_dim), dtype=np.int64),
        "max_image_seq_len": np.asarray(int(prepared.max_image_seq_len), dtype=np.int64),
        "max_text_seq_len": np.asarray(int(prepared.max_text_seq_len), dtype=np.int64),
        "feature_names": np.asarray(prepared.feature_names, dtype=str),
        "stats_json": np.asarray(stats_json, dtype=str),
        "meta_json": np.asarray(meta_json, dtype=str),
        "preprocessor_json": np.asarray(preprocessor_json, dtype=str),
        "X_meta_train": prepared.X_meta_train.astype(np.float32, copy=False),
        "X_meta_val": prepared.X_meta_val.astype(np.float32, copy=False),
        "X_meta_test": prepared.X_meta_test.astype(np.float32, copy=False),
        "X_cover_train": prepared.X_cover_train.astype(np.float32, copy=False),
        "X_cover_val": prepared.X_cover_val.astype(np.float32, copy=False),
        "X_cover_test": prepared.X_cover_test.astype(np.float32, copy=False),
        "X_title_blurb_train": prepared.X_title_blurb_train.astype(np.float32, copy=False),
        "X_title_blurb_val": prepared.X_title_blurb_val.astype(np.float32, copy=False),
        "X_title_blurb_test": prepared.X_title_blurb_test.astype(np.float32, copy=False),
        "X_image_train": prepared.X_image_train.astype(np.float32, copy=False),
        "len_image_train": prepared.len_image_train.astype(np.int64, copy=False),
        "X_image_val": prepared.X_image_val.astype(np.float32, copy=False),
        "len_image_val": prepared.len_image_val.astype(np.int64, copy=False),
        "X_image_test": prepared.X_image_test.astype(np.float32, copy=False),
        "len_image_test": prepared.len_image_test.astype(np.int64, copy=False),
        "X_text_train": prepared.X_text_train.astype(np.float32, copy=False),
        "len_text_train": prepared.len_text_train.astype(np.int64, copy=False),
        "X_text_val": prepared.X_text_val.astype(np.float32, copy=False),
        "len_text_val": prepared.len_text_val.astype(np.int64, copy=False),
        "X_text_test": prepared.X_text_test.astype(np.float32, copy=False),
        "len_text_test": prepared.len_text_test.astype(np.int64, copy=False),
    }

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **arrays)
    tmp_path.replace(cache_path)


def _load_cache(cache_path: Path) -> PreparedGateData:
    with np.load(cache_path, allow_pickle=False) as z:
        stats_json = z.get("stats_json", np.array("{}", dtype=str)).item()
        stats = json.loads(stats_json) if isinstance(stats_json, str) else {}

        feature_names = z.get("feature_names", np.asarray([], dtype=str)).astype(str).tolist()

        pre_json = z.get("preprocessor_json", np.array("{}", dtype=str)).item()
        pre_state = json.loads(pre_json) if isinstance(pre_json, str) else {}
        preprocessor = TabularPreprocessor.from_state_dict(pre_state)

        return PreparedGateData(
            y_train=z["y_train"].astype(np.int64, copy=False),
            train_project_ids=z["train_project_ids"].astype(str).tolist(),
            y_val=z["y_val"].astype(np.int64, copy=False),
            val_project_ids=z["val_project_ids"].astype(str).tolist(),
            y_test=z["y_test"].astype(np.int64, copy=False),
            test_project_ids=z["test_project_ids"].astype(str).tolist(),
            X_meta_train=z["X_meta_train"].astype(np.float32, copy=False),
            X_meta_val=z["X_meta_val"].astype(np.float32, copy=False),
            X_meta_test=z["X_meta_test"].astype(np.float32, copy=False),
            meta_dim=int(z.get("meta_dim", np.asarray(0, dtype=np.int64)).item()),
            preprocessor=preprocessor,
            feature_names=[str(x) for x in feature_names],
            X_cover_train=z["X_cover_train"].astype(np.float32, copy=False),
            X_cover_val=z["X_cover_val"].astype(np.float32, copy=False),
            X_cover_test=z["X_cover_test"].astype(np.float32, copy=False),
            X_title_blurb_train=z["X_title_blurb_train"].astype(np.float32, copy=False),
            X_title_blurb_val=z["X_title_blurb_val"].astype(np.float32, copy=False),
            X_title_blurb_test=z["X_title_blurb_test"].astype(np.float32, copy=False),
            X_image_train=z["X_image_train"].astype(np.float32, copy=False),
            len_image_train=z["len_image_train"].astype(np.int64, copy=False),
            X_image_val=z["X_image_val"].astype(np.float32, copy=False),
            len_image_val=z["len_image_val"].astype(np.int64, copy=False),
            X_image_test=z["X_image_test"].astype(np.float32, copy=False),
            len_image_test=z["len_image_test"].astype(np.int64, copy=False),
            image_embedding_dim=int(z.get("image_embedding_dim", np.asarray(0, dtype=np.int64)).item()),
            max_image_seq_len=int(z.get("max_image_seq_len", np.asarray(0, dtype=np.int64)).item()),
            X_text_train=z["X_text_train"].astype(np.float32, copy=False),
            len_text_train=z["len_text_train"].astype(np.int64, copy=False),
            X_text_val=z["X_text_val"].astype(np.float32, copy=False),
            len_text_val=z["len_text_val"].astype(np.int64, copy=False),
            X_text_test=z["X_text_test"].astype(np.float32, copy=False),
            len_text_test=z["len_text_test"].astype(np.int64, copy=False),
            text_embedding_dim=int(z.get("text_embedding_dim", np.asarray(0, dtype=np.int64)).item()),
            max_text_seq_len=int(z.get("max_text_seq_len", np.asarray(0, dtype=np.int64)).item()),
            stats={str(k): int(v) for k, v in dict(stats).items()},
        )


def prepare_gate_data(
    csv_path: Path,
    projects_root: Path,
    cfg: GateConfig,
    cache_dir: Path | None = None,
    logger=None,
) -> PreparedGateData:
    """ratio：读 CSV -> 切分 -> 构建三分支特征 -> 返回 numpy 数组。支持缓存。"""
    mode = _get_split_mode(cfg)
    if mode != "ratio":
        raise ValueError(f"prepare_gate_data 仅支持 split_mode=ratio，当前={mode!r}")

    use_cache = bool(getattr(cfg, "use_cache", False)) and cache_dir is not None
    cache_path: Path | None = None
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _make_cache_key(csv_path, projects_root, cfg)
        cache_path = cache_dir / f"{cache_key}.npz"
        if cache_path.exists():
            try:
                prepared = _load_cache(cache_path)
                if logger is not None:
                    logger.info("使用数据缓存：%s", str(cache_path))
                return prepared
            except Exception as e:
                if logger is not None:
                    logger.warning("读取缓存失败，将重新构建：%s", e)

    raw_df = pd.read_csv(csv_path)
    if cfg.id_col not in raw_df.columns:
        raise ValueError(f"CSV 缺少 id_col={cfg.id_col!r}")
    if cfg.target_col not in raw_df.columns:
        raise ValueError(f"CSV 缺少 target_col={cfg.target_col!r}")

    df = raw_df.drop(columns=list(cfg.drop_cols), errors="ignore") if cfg.drop_cols else raw_df
    train_df, val_df, test_df = _split_by_ratio(
        df,
        train_ratio=float(getattr(cfg, "train_ratio", 0.7)),
        val_ratio=float(getattr(cfg, "val_ratio", 0.15)),
        shuffle=bool(getattr(cfg, "shuffle_before_split", False)),
        seed=int(getattr(cfg, "random_seed", 42)),
    )

    prepared = _prepare_from_splits(train_df=train_df, val_df=val_df, test_df=test_df, projects_root=projects_root, cfg=cfg)

    if use_cache and cache_path is not None:
        try:
            meta = {
                "cache_version": _GATE_CACHE_VERSION,
                "cache_key": cache_path.stem,
                "csv_path": str(csv_path.as_posix()),
                "projects_root": str(projects_root.as_posix()),
                "split_mode": "ratio",
                "config": cfg.to_dict(),
            }
            _save_cache(cache_path, prepared, meta=meta)
            if logger is not None:
                logger.info("已写入数据缓存：%s", str(cache_path))
        except Exception as e:
            if logger is not None:
                logger.warning("写入缓存失败：%s", e)

    return prepared


def iter_gate_kfold_data(
    csv_path: Path,
    projects_root: Path,
    cfg: GateConfig,
    cache_dir: Path | None = None,
    logger=None,
) -> Iterator[Tuple[int, PreparedGateData]]:
    """
    K 折交叉验证数据迭代器（每折仅切分 train/test；val 复用 train）。

    - cfg.split_mode 必须是 kfold
    - cfg.k_fold_index=-1：遍历全部折；>=0：仅返回指定折
    """
    mode = _get_split_mode(cfg)
    if mode != "kfold":
        raise ValueError(f"iter_gate_kfold_data 仅支持 split_mode=kfold，当前={mode!r}")

    raw_df = pd.read_csv(csv_path)
    if cfg.id_col not in raw_df.columns:
        raise ValueError(f"CSV 缺少 id_col={cfg.id_col!r}")
    if cfg.target_col not in raw_df.columns:
        raise ValueError(f"CSV 缺少 target_col={cfg.target_col!r}")

    y = None
    if bool(getattr(cfg, "kfold_stratify", True)):
        y = _encode_binary_target(raw_df[cfg.target_col])

    pairs = _kfold_split_indices(
        n_total=int(len(raw_df)),
        n_splits=int(getattr(cfg, "k_folds", 5)),
        shuffle=bool(getattr(cfg, "kfold_shuffle", True)),
        seed=int(getattr(cfg, "random_seed", 42)),
        y=y,
        stratify=bool(getattr(cfg, "kfold_stratify", True)),
    )

    only_fold = int(getattr(cfg, "k_fold_index", -1))
    selected = range(len(pairs)) if only_fold < 0 else [only_fold]
    for fold_idx in selected:
        if fold_idx < 0 or fold_idx >= len(pairs):
            raise ValueError(f"k_fold_index 越界：{fold_idx}，有效范围 0..{len(pairs)-1}")

        train_idx, test_idx = pairs[int(fold_idx)]
        train_df = raw_df.iloc[train_idx].copy()
        test_df = raw_df.iloc[test_idx].copy()
        val_df = train_df

        use_cache = bool(getattr(cfg, "use_cache", False)) and cache_dir is not None
        cache_path: Path | None = None
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = _make_cache_key(csv_path, projects_root, cfg, fold_index=int(fold_idx))
            cache_path = cache_dir / f"{cache_key}.npz"
            if cache_path.exists():
                try:
                    prepared = _load_cache(cache_path)
                    if logger is not None:
                        logger.info("fold=%d 使用数据缓存：%s", int(fold_idx), str(cache_path))
                    yield int(fold_idx), prepared
                    continue
                except Exception as e:
                    if logger is not None:
                        logger.warning("fold=%d 读取缓存失败，将重新构建：%s", int(fold_idx), e)

        prepared = _prepare_from_splits(train_df=train_df, val_df=val_df, test_df=test_df, projects_root=projects_root, cfg=cfg)

        if use_cache and cache_path is not None:
            try:
                meta = {
                    "cache_version": _GATE_CACHE_VERSION,
                    "cache_key": cache_path.stem,
                    "csv_path": str(csv_path.as_posix()),
                    "projects_root": str(projects_root.as_posix()),
                    "split_mode": "kfold",
                    "fold_index": int(fold_idx),
                    "k_folds": int(getattr(cfg, "k_folds", 5)),
                    "config": cfg.to_dict(),
                }
                _save_cache(cache_path, prepared, meta=meta)
                if logger is not None:
                    logger.info("fold=%d 已写入数据缓存：%s", int(fold_idx), str(cache_path))
            except Exception as e:
                if logger is not None:
                    logger.warning("fold=%d 写入缓存失败：%s", int(fold_idx), e)

        yield int(fold_idx), prepared
