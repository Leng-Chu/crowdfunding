# -*- coding: utf-8 -*-
"""
数据加载与特征构建（late）：

- 读取项目目录下的 content.json，并根据 content_sequence 做“统一序列截断”
- 根据截断后的内容块，映射回 image_{emb}.npy / text_{emb}.npy 的子集（不做顺序建模）
- 输出：
  - image 集合张量：X_image=[N, L_img, D_img]，len_image=[N]
  - text 集合张量：X_text=[N, L_txt, D_txt]，len_text=[N]
  - 可选 meta 特征：X_meta=[N, F]
- 支持 .npz 缓存（cache key 必须包含 embedding type / max_seq_len / 截断策略）
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import LateConfig


def _split_by_ratio(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按比例切分 train/val/test。"""
    if bool(shuffle):
        df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    test_ratio = float(test_ratio)
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("train_ratio/val_ratio/test_ratio 不能为负数。")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("train_ratio/val_ratio/test_ratio 之和必须大于 0。")

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum

    n_total = int(len(df))
    n_train = int(n_total * float(train_ratio))
    n_val = int(n_total * float(val_ratio))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


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
    """把字符串稳定地映射为 int（用于可复现的抽样/截断）。"""
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


_LATE_CACHE_VERSION = 1


@dataclass(frozen=True)
class PreparedLateData:
    # labels + ids
    y_train: np.ndarray
    train_project_ids: List[str]
    y_val: np.ndarray
    val_project_ids: List[str]
    y_test: np.ndarray
    test_project_ids: List[str]

    # meta（可选）
    X_meta_train: Optional[np.ndarray]
    X_meta_val: Optional[np.ndarray]
    X_meta_test: Optional[np.ndarray]
    meta_dim: int
    preprocessor: Optional[TabularPreprocessor]
    feature_names: List[str]

    # image（必选但允许空集合）
    X_image_train: np.ndarray
    len_image_train: np.ndarray
    X_image_val: np.ndarray
    len_image_val: np.ndarray
    X_image_test: np.ndarray
    len_image_test: np.ndarray
    image_embedding_dim: int
    max_image_keep_len: int

    # text（必选但允许空集合）
    X_text_train: np.ndarray
    len_text_train: np.ndarray
    X_text_val: np.ndarray
    len_text_val: np.ndarray
    X_text_test: np.ndarray
    len_text_test: np.ndarray
    text_embedding_dim: int
    max_text_keep_len: int

    # 统一截断配置（用于复现与对齐）
    max_seq_len: int

    stats: Dict[str, int]


def _late_load_dataframe(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _late_make_cache_key(
    csv_path: Path,
    projects_root: Path,
    cfg: LateConfig,
    use_meta: bool,
) -> str:
    """
    cache key 必须包含：
    - embedding type（image/text）
    - max_seq_len
    - truncation_strategy
    """
    stat = csv_path.stat()
    payload = {
        "cache_version": _LATE_CACHE_VERSION,
        "data_csv": str(csv_path.as_posix()),
        "csv_mtime": float(stat.st_mtime),
        "csv_size": int(stat.st_size),
        "projects_root": str(projects_root.as_posix()),
        "use_meta": bool(use_meta),
        "columns": {
            "id_col": str(cfg.id_col),
            "target_col": str(cfg.target_col),
            "categorical_cols": list(cfg.categorical_cols) if use_meta else [],
            "numeric_cols": list(cfg.numeric_cols) if use_meta else [],
        },
        "split": {
            "train_ratio": float(getattr(cfg, "train_ratio", 0.7)),
            "val_ratio": float(getattr(cfg, "val_ratio", 0.15)),
            "test_ratio": float(getattr(cfg, "test_ratio", 0.15)),
            "shuffle_before_split": bool(getattr(cfg, "shuffle_before_split", False)),
            "random_seed": int(getattr(cfg, "random_seed", 42)),
        },
        "sequence": {
            "max_seq_len": int(getattr(cfg, "max_seq_len", 0)),
            "truncation_strategy": str(getattr(cfg, "truncation_strategy", "first")),
        },
        "image": {"embedding_type": str(getattr(cfg, "image_embedding_type", ""))},
        "text": {"embedding_type": str(getattr(cfg, "text_embedding_type", ""))},
        "missing_strategy": str(getattr(cfg, "missing_strategy", "")),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:16]
    return f"late_{h}"


def _late_save_cache(cache_path: Path, prepared: PreparedLateData, meta: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats_json = json.dumps(prepared.stats, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)

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
        "max_image_keep_len": np.asarray(int(prepared.max_image_keep_len), dtype=np.int64),
        "max_text_keep_len": np.asarray(int(prepared.max_text_keep_len), dtype=np.int64),
        "max_seq_len": np.asarray(int(prepared.max_seq_len), dtype=np.int64),
        "feature_names": np.asarray(prepared.feature_names, dtype=str),
        "stats_json": np.asarray(stats_json, dtype=str),
        "meta_json": np.asarray(meta_json, dtype=str),
        # image
        "X_image_train": prepared.X_image_train.astype(np.float32, copy=False),
        "len_image_train": prepared.len_image_train.astype(np.int64, copy=False),
        "X_image_val": prepared.X_image_val.astype(np.float32, copy=False),
        "len_image_val": prepared.len_image_val.astype(np.int64, copy=False),
        "X_image_test": prepared.X_image_test.astype(np.float32, copy=False),
        "len_image_test": prepared.len_image_test.astype(np.int64, copy=False),
        # text
        "X_text_train": prepared.X_text_train.astype(np.float32, copy=False),
        "len_text_train": prepared.len_text_train.astype(np.int64, copy=False),
        "X_text_val": prepared.X_text_val.astype(np.float32, copy=False),
        "len_text_val": prepared.len_text_val.astype(np.int64, copy=False),
        "X_text_test": prepared.X_text_test.astype(np.float32, copy=False),
        "len_text_test": prepared.len_text_test.astype(np.int64, copy=False),
    }

    if prepared.X_meta_train is not None:
        arrays["X_meta_train"] = prepared.X_meta_train.astype(np.float32, copy=False)
        arrays["X_meta_val"] = prepared.X_meta_val.astype(np.float32, copy=False)
        arrays["X_meta_test"] = prepared.X_meta_test.astype(np.float32, copy=False)
        preprocessor_json = json.dumps(
            prepared.preprocessor.to_state_dict() if prepared.preprocessor else {},
            ensure_ascii=False,
            sort_keys=True,
        )
        arrays["preprocessor_json"] = np.asarray(preprocessor_json, dtype=str)

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(cache_path)


def _late_load_cache(cache_path: Path) -> PreparedLateData:
    with np.load(cache_path, allow_pickle=False) as z:
        stats = json.loads(str(z.get("stats_json", np.asarray("{}", dtype=str)).item()))

        preprocessor = None
        if "preprocessor_json" in z:
            pre_state = json.loads(str(z["preprocessor_json"].item()))
            preprocessor = TabularPreprocessor.from_state_dict(pre_state)

        X_meta_train = z["X_meta_train"].astype(np.float32, copy=False) if "X_meta_train" in z else None
        X_meta_val = z["X_meta_val"].astype(np.float32, copy=False) if "X_meta_val" in z else None
        X_meta_test = z["X_meta_test"].astype(np.float32, copy=False) if "X_meta_test" in z else None

        return PreparedLateData(
            y_train=z["y_train"].astype(np.int64, copy=False),
            train_project_ids=[str(x) for x in z["train_project_ids"].tolist()],
            y_val=z["y_val"].astype(np.int64, copy=False),
            val_project_ids=[str(x) for x in z["val_project_ids"].tolist()],
            y_test=z["y_test"].astype(np.int64, copy=False),
            test_project_ids=[str(x) for x in z["test_project_ids"].tolist()],
            X_meta_train=X_meta_train,
            X_meta_val=X_meta_val,
            X_meta_test=X_meta_test,
            meta_dim=int(z.get("meta_dim", np.asarray(0, dtype=np.int64)).item()),
            preprocessor=preprocessor,
            feature_names=[str(x) for x in z.get("feature_names", np.asarray([], dtype=str)).tolist()],
            X_image_train=z["X_image_train"].astype(np.float32, copy=False),
            len_image_train=z["len_image_train"].astype(np.int64, copy=False),
            X_image_val=z["X_image_val"].astype(np.float32, copy=False),
            len_image_val=z["len_image_val"].astype(np.int64, copy=False),
            X_image_test=z["X_image_test"].astype(np.float32, copy=False),
            len_image_test=z["len_image_test"].astype(np.int64, copy=False),
            image_embedding_dim=int(z.get("image_embedding_dim", np.asarray(0, dtype=np.int64)).item()),
            max_image_keep_len=int(z.get("max_image_keep_len", np.asarray(0, dtype=np.int64)).item()),
            X_text_train=z["X_text_train"].astype(np.float32, copy=False),
            len_text_train=z["len_text_train"].astype(np.int64, copy=False),
            X_text_val=z["X_text_val"].astype(np.float32, copy=False),
            len_text_val=z["len_text_val"].astype(np.int64, copy=False),
            X_text_test=z["X_text_test"].astype(np.float32, copy=False),
            len_text_test=z["len_text_test"].astype(np.int64, copy=False),
            text_embedding_dim=int(z.get("text_embedding_dim", np.asarray(0, dtype=np.int64)).item()),
            max_text_keep_len=int(z.get("max_text_keep_len", np.asarray(0, dtype=np.int64)).item()),
            max_seq_len=int(z.get("max_seq_len", np.asarray(0, dtype=np.int64)).item()),
            stats={k: int(v) for k, v in dict(stats).items()},
        )


def _read_content_sequence(content_json_path: Path) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(content_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"读取 content.json 失败：{content_json_path} | {type(e).__name__}: {e}") from e

    seq = obj.get("content_sequence", None)
    if not isinstance(seq, list):
        raise ValueError(f"content_sequence 不存在或不是 list：{content_json_path}")
    return list(seq)


def _truncate_window(seq_len: int, max_seq_len: int, strategy: str, seed: int) -> Tuple[int, int]:
    """
    统一序列截断（仅用于控制样本使用的内容块数量，不用于顺序建模）。
    - first：取 [0, max_seq_len)
    - random：在所有长度为 max_seq_len 的窗口中随机选一个（可复现）
    """
    L = int(seq_len)
    m = int(max_seq_len)
    if m <= 0:
        raise ValueError("max_seq_len 需要 > 0。")
    if L <= 0:
        return 0, 0
    if L <= m:
        return 0, L

    strategy = str(strategy or "first").strip().lower()
    if strategy == "first":
        return 0, m
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        start = int(rng.integers(0, L - m + 1))
        return start, start + m
    raise ValueError(f"不支持的 truncation_strategy={strategy!r}，可选：first/random")


def _late_load_project_keep_sets(
    project_dir: Path,
    project_id: str,
    cfg: LateConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int, int, int, int]:
    """
    返回：img_keep, txt_keep, img_dim, txt_dim, len_img, len_txt, skipped(0/1)
    - img_keep/txt_keep 可能为空集合（len=0）
    - 当必须文件缺失且 missing_strategy=skip 时，skipped=1
    """
    missing_strategy = str(getattr(cfg, "missing_strategy", "error") or "error").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={cfg.missing_strategy!r}，可选：skip/error")

    content_json_path = project_dir / "content.json"
    if not content_json_path.exists():
        if missing_strategy == "skip":
            return None, None, 0, 0, 0, 0, 1
        raise FileNotFoundError(f"缺少 content.json：{content_json_path}")

    seq = _read_content_sequence(content_json_path)
    if not seq:
        if missing_strategy == "skip":
            return None, None, 0, 0, 0, 0, 1
        raise ValueError(f"项目 {project_id} content_sequence 为空，无法训练。")

    n_img_expected = 0
    n_txt_expected = 0
    for pos, item in enumerate(seq):
        t = (item.get("type", None) or "").strip().lower()
        if t == "image":
            n_img_expected += 1
        elif t == "text":
            n_txt_expected += 1
        else:
            raise ValueError(f"项目 {project_id} content_sequence type 不支持：{t!r}（pos={pos}）")

    img_type = str(getattr(cfg, "image_embedding_type", "") or "").strip().lower()
    if img_type not in {"clip", "siglip", "resnet"}:
        raise ValueError(f"不支持的 image_embedding_type={cfg.image_embedding_type!r}，可选：clip/siglip/resnet")
    txt_type = str(getattr(cfg, "text_embedding_type", "") or "").strip().lower()
    if txt_type not in {"bge", "clip", "siglip"}:
        raise ValueError(f"不支持的 text_embedding_type={cfg.text_embedding_type!r}，可选：bge/clip/siglip")

    image_emb_path = project_dir / f"image_{img_type}.npy"
    text_emb_path = project_dir / f"text_{txt_type}.npy"

    img_emb = None
    img_dim = 0
    if image_emb_path.exists():
        img_emb = _as_2d_embedding(np.load(image_emb_path), image_emb_path.name)
        img_dim = int(img_emb.shape[1])
        if int(img_emb.shape[0]) != int(n_img_expected):
            raise ValueError(
                f"项目 {project_id} image 数量不一致：content_sequence={n_img_expected} vs {image_emb_path.name}={img_emb.shape}"
            )
    else:
        # 注意：当 content_sequence 中 image 数量为 0 时，允许缺少 image_*.npy
        if int(n_img_expected) != 0:
            if missing_strategy == "skip":
                return None, None, 0, 0, 0, 0, 1
            raise FileNotFoundError(f"缺少 image 嵌入文件：{image_emb_path}")

    txt_emb = None
    txt_dim = 0
    if text_emb_path.exists():
        txt_emb = _as_2d_embedding(np.load(text_emb_path), text_emb_path.name)
        txt_dim = int(txt_emb.shape[1])
        if int(txt_emb.shape[0]) != int(n_txt_expected):
            raise ValueError(
                f"项目 {project_id} text 数量不一致：content_sequence={n_txt_expected} vs {text_emb_path.name}={txt_emb.shape}"
            )
    else:
        # 注意：当 content_sequence 中 text 数量为 0 时，允许缺少 text_*.npy
        if int(n_txt_expected) != 0:
            if missing_strategy == "skip":
                return None, None, 0, 0, 0, 0, 1
            raise FileNotFoundError(f"缺少 text 嵌入文件：{text_emb_path}")

    # 统一序列截断：决定窗口 [start, end)
    max_seq_len = int(getattr(cfg, "max_seq_len", 0))
    if max_seq_len <= 0:
        raise ValueError("max_seq_len 需要 > 0。")
    truncation_strategy = str(getattr(cfg, "truncation_strategy", "first"))
    seed = int(getattr(cfg, "random_seed", 42)) + _stable_hash_int(str(project_id))
    start, end = _truncate_window(len(seq), max_seq_len=max_seq_len, strategy=truncation_strategy, seed=seed)

    keep_img_indices: List[int] = []
    keep_txt_indices: List[int] = []
    img_idx = 0
    txt_idx = 0

    for pos, item in enumerate(seq):
        t = (item.get("type", None) or "").strip().lower()
        in_window = int(start) <= int(pos) < int(end)
        if t == "image":
            if in_window:
                keep_img_indices.append(int(img_idx))
            img_idx += 1
        elif t == "text":
            if in_window:
                keep_txt_indices.append(int(txt_idx))
            txt_idx += 1
        else:
            raise ValueError(f"项目 {project_id} content_sequence type 不支持：{t!r}（pos={pos}）")

    if int(img_idx) != int(n_img_expected):
        raise ValueError(f"项目 {project_id} image 计数器不一致：{img_idx} vs expected {n_img_expected}")
    if int(txt_idx) != int(n_txt_expected):
        raise ValueError(f"项目 {project_id} text 计数器不一致：{txt_idx} vs expected {n_txt_expected}")

    img_keep = None
    len_img = 0
    if img_emb is not None:
        if keep_img_indices:
            img_keep = img_emb[np.asarray(keep_img_indices, dtype=np.int64)]
        else:
            img_keep = img_emb[:0]
        len_img = int(img_keep.shape[0])

    txt_keep = None
    len_txt = 0
    if txt_emb is not None:
        if keep_txt_indices:
            txt_keep = txt_emb[np.asarray(keep_txt_indices, dtype=np.int64)]
        else:
            txt_keep = txt_emb[:0]
        len_txt = int(txt_keep.shape[0])

    return img_keep, txt_keep, int(img_dim), int(txt_dim), int(len_img), int(len_txt), 0


def _infer_embedding_dims(projects_root: Path, project_ids: List[str], cfg: LateConfig) -> Tuple[int, int]:
    """
    为了支持“某个 split 内全为空集合”的极端情况，提前从全量项目中推断 D_img / D_txt。
    - 只要找到任意一个存在的 embedding 文件即可
    - 若始终无法推断，将报错（因为模型投影层需要明确输入维度）
    """
    img_type = str(getattr(cfg, "image_embedding_type", "") or "").strip().lower()
    txt_type = str(getattr(cfg, "text_embedding_type", "") or "").strip().lower()

    image_dim = 0
    text_dim = 0

    for pid in project_ids:
        pid = _normalize_project_id(pid)
        if not pid:
            continue
        project_dir = projects_root / pid
        if not project_dir.exists():
            continue

        if image_dim <= 0:
            p = project_dir / f"image_{img_type}.npy"
            if p.exists():
                try:
                    arr = _as_2d_embedding(np.load(p), p.name)
                    image_dim = int(arr.shape[1])
                except Exception:
                    pass

        if text_dim <= 0:
            p = project_dir / f"text_{txt_type}.npy"
            if p.exists():
                try:
                    arr = _as_2d_embedding(np.load(p), p.name)
                    text_dim = int(arr.shape[1])
                except Exception:
                    pass

        if int(image_dim) > 0 and int(text_dim) > 0:
            break

    return int(image_dim), int(text_dim)


def _late_build_features_for_split(
    df_split: pd.DataFrame,
    X_meta_all: Optional[np.ndarray],
    projects_root: Path,
    cfg: LateConfig,
    use_meta: bool,
    fallback_image_dim: int,
    fallback_text_dim: int,
) -> Tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, int], int, int, int, int]:
    """为一个 split 构建 X/len/y。"""
    if use_meta:
        if X_meta_all is None:
            raise ValueError("use_meta=True 时，X_meta_all 不能为空。")
        if int(X_meta_all.shape[0]) != int(len(df_split)):
            raise ValueError(f"X_meta_all 与 df_split 样本数不一致：{X_meta_all.shape[0]} vs {len(df_split)}")

    project_ids: List[str] = [_normalize_project_id(v) for v in df_split[cfg.id_col].tolist()]
    y_all = _encode_binary_target(df_split[cfg.target_col])

    kept_ids: List[str] = []
    kept_indices: List[int] = []

    img_seqs: List[Optional[np.ndarray]] = []
    img_lens: List[int] = []
    txt_seqs: List[Optional[np.ndarray]] = []
    txt_lens: List[int] = []

    skipped_total = 0
    missing_project_dir = 0
    image_dim = int(fallback_image_dim)
    text_dim = int(fallback_text_dim)

    missing_strategy = str(getattr(cfg, "missing_strategy", "error") or "error").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={cfg.missing_strategy!r}，可选：skip/error")

    for i, (pid, _) in enumerate(zip(project_ids, y_all)):
        if not pid:
            skipped_total += 1
            continue

        project_dir = projects_root / pid
        if not project_dir.exists():
            missing_project_dir += 1
            if missing_strategy == "skip":
                skipped_total += 1
                continue
            raise FileNotFoundError(f"找不到项目目录：{project_dir}")

        img_keep, txt_keep, img_dim_i, txt_dim_i, len_img, len_txt, skipped = _late_load_project_keep_sets(
            project_dir, project_id=pid, cfg=cfg
        )
        if skipped:
            skipped_total += 1
            continue

        if int(img_dim_i) > 0:
            if int(image_dim) <= 0:
                image_dim = int(img_dim_i)
            if int(img_dim_i) != int(image_dim):
                raise ValueError(f"image_embedding_dim 不一致：期望 {image_dim}，但 {pid} 为 {img_dim_i}")

        if int(txt_dim_i) > 0:
            if int(text_dim) <= 0:
                text_dim = int(txt_dim_i)
            if int(txt_dim_i) != int(text_dim):
                raise ValueError(f"text_embedding_dim 不一致：期望 {text_dim}，但 {pid} 为 {txt_dim_i}")

        img_seqs.append(img_keep)
        img_lens.append(int(len_img))
        txt_seqs.append(txt_keep)
        txt_lens.append(int(len_txt))

        kept_ids.append(pid)
        kept_indices.append(i)

    if not kept_indices:
        raise RuntimeError("该数据切分中没有可用样本（可能都被 skip 了或缺少必须文件）。")

    if int(image_dim) <= 0:
        raise RuntimeError("无法确定 image_embedding_dim：可能所有样本都没有 image 向量。")
    if int(text_dim) <= 0:
        raise RuntimeError("无法确定 text_embedding_dim：可能所有样本都没有 text 向量。")

    idx_arr = np.asarray(kept_indices, dtype=np.int64)
    y_arr = np.asarray(y_all[idx_arr], dtype=np.int64)

    X_meta = np.asarray(X_meta_all[idx_arr], dtype=np.float32) if use_meta else None

    max_img_len = int(max(img_lens)) if img_lens else 0
    X_image = np.zeros((len(img_seqs), max_img_len, int(image_dim)), dtype=np.float32)
    for j, seq in enumerate(img_seqs):
        if seq is None or int(seq.shape[0]) <= 0:
            continue
        if int(seq.shape[1]) != int(image_dim):
            raise ValueError(f"image_embedding_dim 不一致：期望 {image_dim}，但样本 {kept_ids[j]} 为 {seq.shape[1]}")
        L = int(seq.shape[0])
        X_image[j, :L, :] = seq
    len_image = np.asarray(img_lens, dtype=np.int64)

    max_txt_len = int(max(txt_lens)) if txt_lens else 0
    X_text = np.zeros((len(txt_seqs), max_txt_len, int(text_dim)), dtype=np.float32)
    for j, seq in enumerate(txt_seqs):
        if seq is None or int(seq.shape[0]) <= 0:
            continue
        if int(seq.shape[1]) != int(text_dim):
            raise ValueError(f"text_embedding_dim 不一致：期望 {text_dim}，但样本 {kept_ids[j]} 为 {seq.shape[1]}")
        L = int(seq.shape[0])
        X_text[j, :L, :] = seq
    len_text = np.asarray(txt_lens, dtype=np.int64)

    stats: Dict[str, int] = {
        "skipped_samples": int(skipped_total),
        "missing_project_dir": int(missing_project_dir),
    }

    return (
        X_meta,
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


def prepare_late_data(
    csv_path: Path,
    projects_root: Path,
    cfg: LateConfig,
    use_meta: bool,
    cache_dir: Path | None = None,
    logger=None,
) -> PreparedLateData:
    """读 CSV -> 切分 -> 构建特征（image/text + 可选 meta）-> 返回 numpy 数组。支持缓存。"""
    use_cache = bool(getattr(cfg, "use_cache", False)) and cache_dir is not None

    cache_path: Path | None = None
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _late_make_cache_key(csv_path, projects_root, cfg, use_meta=use_meta)
        cache_path = cache_dir / f"{cache_key}.npz"
        if cache_path.exists():
            try:
                prepared = _late_load_cache(cache_path)
                if logger is not None:
                    logger.info("使用数据缓存：%s", str(cache_path))
                return prepared
            except Exception as e:
                if logger is not None:
                    logger.warning("读取缓存失败，将重新构建：%s", e)

    raw_df = _late_load_dataframe(csv_path)
    if cfg.id_col not in raw_df.columns:
        raise ValueError(f"CSV 缺少 id_col={cfg.id_col!r}")
    if cfg.target_col not in raw_df.columns:
        raise ValueError(f"CSV 缺少 target_col={cfg.target_col!r}")

    if use_meta:
        feature_cols = [*cfg.categorical_cols, *cfg.numeric_cols]
        for col in feature_cols:
            if col not in raw_df.columns:
                raise ValueError(f"CSV 缺少特征列：{col!r}")

    train_df, val_df, test_df = _split_by_ratio(
        raw_df,
        train_ratio=float(getattr(cfg, "train_ratio", 0.7)),
        val_ratio=float(getattr(cfg, "val_ratio", 0.15)),
        test_ratio=float(getattr(cfg, "test_ratio", 0.15)),
        shuffle=bool(getattr(cfg, "shuffle_before_split", False)),
        seed=int(getattr(cfg, "random_seed", 42)),
    )

    all_project_ids: List[str] = [_normalize_project_id(v) for v in raw_df[cfg.id_col].tolist()]
    fallback_image_dim, fallback_text_dim = _infer_embedding_dims(projects_root, all_project_ids, cfg)
    if int(fallback_image_dim) <= 0:
        raise RuntimeError("无法推断 image_embedding_dim：可能所有项目都缺少 image_*.npy。")
    if int(fallback_text_dim) <= 0:
        raise RuntimeError("无法推断 text_embedding_dim：可能所有项目都缺少 text_*.npy。")

    preprocessor = None
    feature_names: List[str] = []
    meta_dim = 0
    X_meta_train_all = None
    X_meta_val_all = None
    X_meta_test_all = None
    if use_meta:
        preprocessor = TabularPreprocessor(
            categorical_cols=list(cfg.categorical_cols),
            numeric_cols=list(cfg.numeric_cols),
        )
        feature_cols = [*cfg.categorical_cols, *cfg.numeric_cols]
        X_meta_train_all = preprocessor.fit_transform(train_df[feature_cols])
        X_meta_val_all = preprocessor.transform(val_df[feature_cols])
        X_meta_test_all = preprocessor.transform(test_df[feature_cols])
        feature_names = preprocessor.get_feature_names()
        meta_dim = int(X_meta_train_all.shape[1])

    (
        X_meta_train,
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
    ) = _late_build_features_for_split(
        train_df,
        X_meta_train_all,
        projects_root,
        cfg,
        use_meta,
        fallback_image_dim=fallback_image_dim,
        fallback_text_dim=fallback_text_dim,
    )
    (
        X_meta_val,
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
    ) = _late_build_features_for_split(
        val_df,
        X_meta_val_all,
        projects_root,
        cfg,
        use_meta,
        fallback_image_dim=fallback_image_dim,
        fallback_text_dim=fallback_text_dim,
    )
    (
        X_meta_test,
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
    ) = _late_build_features_for_split(
        test_df,
        X_meta_test_all,
        projects_root,
        cfg,
        use_meta,
        fallback_image_dim=fallback_image_dim,
        fallback_text_dim=fallback_text_dim,
    )

    if int(img_dim_val) != int(img_dim) or int(img_dim_test) != int(img_dim):
        raise ValueError(f"image_embedding_dim 不一致：train={img_dim} val={img_dim_val} test={img_dim_test}")
    if int(txt_dim_val) != int(txt_dim) or int(txt_dim_test) != int(txt_dim):
        raise ValueError(f"text_embedding_dim 不一致：train={txt_dim} val={txt_dim_val} test={txt_dim_test}")

    stats: Dict[str, int] = {}
    for s in (train_stats, val_stats, test_stats):
        for k, v in s.items():
            stats[k] = int(stats.get(k, 0) + int(v))

    prepared = PreparedLateData(
        y_train=y_train,
        train_project_ids=train_ids,
        y_val=y_val,
        val_project_ids=val_ids,
        y_test=y_test,
        test_project_ids=test_ids,
        X_meta_train=X_meta_train,
        X_meta_val=X_meta_val,
        X_meta_test=X_meta_test,
        meta_dim=int(meta_dim) if use_meta else 0,
        preprocessor=preprocessor,
        feature_names=feature_names,
        X_image_train=X_img_train,
        len_image_train=len_img_train,
        X_image_val=X_img_val,
        len_image_val=len_img_val,
        X_image_test=X_img_test,
        len_image_test=len_img_test,
        image_embedding_dim=int(img_dim),
        max_image_keep_len=int(max(max_img_train, max_img_val, max_img_test)),
        X_text_train=X_txt_train,
        len_text_train=len_txt_train,
        X_text_val=X_txt_val,
        len_text_val=len_txt_val,
        X_text_test=X_txt_test,
        len_text_test=len_txt_test,
        text_embedding_dim=int(txt_dim),
        max_text_keep_len=int(max(max_txt_train, max_txt_val, max_txt_test)),
        max_seq_len=int(getattr(cfg, "max_seq_len", 0)),
        stats=stats,
    )

    if use_cache and cache_path is not None:
        try:
            meta = {
                "cache_version": _LATE_CACHE_VERSION,
                "cache_key": cache_path.stem,
                "csv_path": str(csv_path.as_posix()),
                "projects_root": str(projects_root.as_posix()),
                "use_meta": bool(use_meta),
                "config": cfg.to_dict(),
            }
            _late_save_cache(cache_path, prepared, meta=meta)
            if logger is not None:
                logger.info("已写入数据缓存：%s", str(cache_path))
        except Exception as e:
            if logger is not None:
                logger.warning("写入缓存失败：%s", e)

    return prepared
