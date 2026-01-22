# -*- coding: utf-8 -*-
"""
数据加载与特征构建（seq）：

- 读取 metadata CSV（与 mlp baseline 对齐的列约定）
- 读取项目目录下的 content.json，并根据 content_sequence 构造“图文交替统一序列”
- 读取预计算 embedding：image_{emb_type}.npy / text_{emb_type}.npy
- 计算每个内容块的属性：文本长度 / 图片面积（直接读取 content.json 中预处理好的 content_length/width/height）
- 支持按 max_seq_len 截断（first/random）并输出 seq_mask
- 支持 .npz 缓存（cache key 包含 embedding type / max_seq_len / 截断策略）

注意：
- 本模块不复用 `src/dl/mlp` 的代码，但工程行为需要对齐以便横向对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import SeqConfig


def _split_by_ratio(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按比例切分 train/val/test（与 mlp baseline 一致）。"""
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
    """将标签编码为 0/1（兼容数值/字符串；与 mlp baseline 一致）。"""
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
    """把字符串稳定地映射为 int（用于可复现的随机截断/打乱）。"""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


@dataclass
class TabularPreprocessor:
    """轻量级表格预处理器：one-hot + 标准化（结构与 mlp baseline 对齐）。"""

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


_SEQ_CACHE_VERSION = 2


@dataclass(frozen=True)
class PreparedSeqData:
    # labels + ids
    y_train: np.ndarray
    train_project_ids: List[str]
    y_val: np.ndarray
    val_project_ids: List[str]
    y_test: np.ndarray
    test_project_ids: List[str]

    # seq（必选）
    X_img_train: np.ndarray
    X_txt_train: np.ndarray
    seq_type_train: np.ndarray
    seq_attr_train: np.ndarray
    seq_mask_train: np.ndarray

    X_img_val: np.ndarray
    X_txt_val: np.ndarray
    seq_type_val: np.ndarray
    seq_attr_val: np.ndarray
    seq_mask_val: np.ndarray

    X_img_test: np.ndarray
    X_txt_test: np.ndarray
    seq_type_test: np.ndarray
    seq_attr_test: np.ndarray
    seq_mask_test: np.ndarray

    image_embedding_dim: int
    text_embedding_dim: int
    max_seq_len: int

    # meta（可选）
    X_meta_train: Optional[np.ndarray]
    X_meta_val: Optional[np.ndarray]
    X_meta_test: Optional[np.ndarray]
    meta_dim: int
    preprocessor: Optional[TabularPreprocessor]
    feature_names: List[str]

    stats: Dict[str, int]


def _seq_make_cache_key(
    csv_path: Path,
    projects_root: Path,
    cfg: SeqConfig,
    use_meta: bool,
) -> str:
    stat = csv_path.stat()
    payload = {
        "cache_version": _SEQ_CACHE_VERSION,
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
        "embedding": {
            "image_embedding_type": str(getattr(cfg, "image_embedding_type", "")),
            "text_embedding_type": str(getattr(cfg, "text_embedding_type", "")),
        },
        "sequence": {
            "max_seq_len": int(getattr(cfg, "max_seq_len", 0)),
            "truncation_strategy": str(getattr(cfg, "truncation_strategy", "first")),
        },
        "missing_strategy": str(getattr(cfg, "missing_strategy", "")),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:16]
    return f"seq_{h}"


def _seq_save_cache(cache_path: Path, prepared: PreparedSeqData, meta: Dict[str, Any]) -> None:
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
        "image_embedding_dim": np.asarray(int(prepared.image_embedding_dim), dtype=np.int64),
        "text_embedding_dim": np.asarray(int(prepared.text_embedding_dim), dtype=np.int64),
        "max_seq_len": np.asarray(int(prepared.max_seq_len), dtype=np.int64),
        "feature_names": np.asarray(prepared.feature_names, dtype=str),
        "meta_dim": np.asarray(int(prepared.meta_dim), dtype=np.int64),
        "stats_json": np.asarray(stats_json, dtype=str),
        "meta_json": np.asarray(meta_json, dtype=str),
        # train
        "X_img_train": prepared.X_img_train.astype(np.float32, copy=False),
        "X_txt_train": prepared.X_txt_train.astype(np.float32, copy=False),
        "seq_type_train": prepared.seq_type_train.astype(np.int64, copy=False),
        "seq_attr_train": prepared.seq_attr_train.astype(np.float32, copy=False),
        "seq_mask_train": prepared.seq_mask_train.astype(bool, copy=False),
        # val
        "X_img_val": prepared.X_img_val.astype(np.float32, copy=False),
        "X_txt_val": prepared.X_txt_val.astype(np.float32, copy=False),
        "seq_type_val": prepared.seq_type_val.astype(np.int64, copy=False),
        "seq_attr_val": prepared.seq_attr_val.astype(np.float32, copy=False),
        "seq_mask_val": prepared.seq_mask_val.astype(bool, copy=False),
        # test
        "X_img_test": prepared.X_img_test.astype(np.float32, copy=False),
        "X_txt_test": prepared.X_txt_test.astype(np.float32, copy=False),
        "seq_type_test": prepared.seq_type_test.astype(np.int64, copy=False),
        "seq_attr_test": prepared.seq_attr_test.astype(np.float32, copy=False),
        "seq_mask_test": prepared.seq_mask_test.astype(bool, copy=False),
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

    # numpy.savez 会在文件名不是以 ".npz" 结尾时自动追加扩展名；
    # 因此临时文件必须以 ".npz" 结尾，否则 replace 会找不到实际写出的文件。
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(cache_path)


def _seq_load_cache(cache_path: Path) -> PreparedSeqData:
    with np.load(cache_path, allow_pickle=False) as z:
        stats = json.loads(str(z["stats_json"].item())) if "stats_json" in z else {}
        preprocessor = None
        feature_names: List[str] = []
        meta_dim = int(z["meta_dim"].item()) if "meta_dim" in z else 0
        if "preprocessor_json" in z:
            preprocessor_state = json.loads(str(z["preprocessor_json"].item()))
            preprocessor = TabularPreprocessor.from_state_dict(preprocessor_state)
        if "feature_names" in z:
            feature_names = [str(x) for x in z["feature_names"].tolist()]

        X_meta_train = z["X_meta_train"].astype(np.float32) if "X_meta_train" in z else None
        X_meta_val = z["X_meta_val"].astype(np.float32) if "X_meta_val" in z else None
        X_meta_test = z["X_meta_test"].astype(np.float32) if "X_meta_test" in z else None

        return PreparedSeqData(
            y_train=z["y_train"].astype(np.int64),
            train_project_ids=[str(x) for x in z["train_project_ids"].tolist()],
            y_val=z["y_val"].astype(np.int64),
            val_project_ids=[str(x) for x in z["val_project_ids"].tolist()],
            y_test=z["y_test"].astype(np.int64),
            test_project_ids=[str(x) for x in z["test_project_ids"].tolist()],
            X_img_train=z["X_img_train"].astype(np.float32),
            X_txt_train=z["X_txt_train"].astype(np.float32),
            seq_type_train=z["seq_type_train"].astype(np.int64),
            seq_attr_train=z["seq_attr_train"].astype(np.float32),
            seq_mask_train=z["seq_mask_train"].astype(bool),
            X_img_val=z["X_img_val"].astype(np.float32),
            X_txt_val=z["X_txt_val"].astype(np.float32),
            seq_type_val=z["seq_type_val"].astype(np.int64),
            seq_attr_val=z["seq_attr_val"].astype(np.float32),
            seq_mask_val=z["seq_mask_val"].astype(bool),
            X_img_test=z["X_img_test"].astype(np.float32),
            X_txt_test=z["X_txt_test"].astype(np.float32),
            seq_type_test=z["seq_type_test"].astype(np.int64),
            seq_attr_test=z["seq_attr_test"].astype(np.float32),
            seq_mask_test=z["seq_mask_test"].astype(bool),
            image_embedding_dim=int(z["image_embedding_dim"].item()),
            text_embedding_dim=int(z["text_embedding_dim"].item()),
            max_seq_len=int(z["max_seq_len"].item()),
            X_meta_train=X_meta_train,
            X_meta_val=X_meta_val,
            X_meta_test=X_meta_test,
            meta_dim=int(meta_dim),
            preprocessor=preprocessor,
            feature_names=feature_names,
            stats={k: int(v) for k, v in dict(stats).items()},
        )


def _infer_embedding_dim(projects_root: Path, project_ids: List[str], filename: str) -> int:
    """扫描样本，读取第一个存在的 embedding 文件以推断维度。找不到则返回 0。"""
    for pid in project_ids:
        pdir = projects_root / str(pid)
        f = pdir / filename
        if not f.exists():
            continue
        try:
            arr = _as_2d_embedding(np.load(f), f.name)
            return int(arr.shape[1])
        except Exception:
            continue
    return 0


def _read_content_sequence(content_json_path: Path) -> List[Dict[str, Any]]:
    with content_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    seq = obj.get("content_sequence", None)
    if not isinstance(seq, list):
        raise ValueError(f"content_sequence 不存在或不是 list：{content_json_path}")
    return [dict(x) for x in seq]

def _require_int_field(item: Dict[str, Any], key: str, project_id: str, pos: int) -> int:
    if key not in item:
        raise ValueError(f"项目 {project_id} 的 content_sequence 缺少字段 {key!r}（pos={pos}）")
    v = item.get(key)
    if v is None:
        raise ValueError(f"项目 {project_id} 的 content_sequence 字段 {key!r} 为空（pos={pos}）")
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"项目 {project_id} 的字段 {key!r} 不是整数：{v!r}（pos={pos}）") from e


def _truncate_window(length: int, max_len: int, strategy: str, seed: int) -> Tuple[int, int]:
    """返回截断窗口 [start, end)（保序截断）。"""
    if max_len <= 0 or length <= max_len:
        return 0, int(length)

    strategy = str(strategy or "first").strip().lower()
    if strategy == "first":
        return 0, int(max_len)
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        start = int(rng.integers(low=0, high=int(length - max_len + 1)))
        return start, int(start + max_len)
    raise ValueError(f"不支持的 truncation_strategy={strategy!r}，可选：first/random")


def _build_one_project_sequence(
    project_id: str,
    content_sequence: List[Dict[str, Any]],
    image_emb_path: Path,
    text_emb_path: Path,
    image_dim: int,
    text_dim: int,
    max_seq_len: int,
    truncation_strategy: str,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造单项目的统一序列特征（已截断 + padding 到 max_seq_len）：
    - X_img: [max_seq_len, D_img]（非 image token 为 0）
    - X_txt: [max_seq_len, D_txt]（非 text token 为 0）
    - seq_type: [max_seq_len]（0=text，1=image）
    - seq_attr: [max_seq_len]（log(length/area)，padding 为 0）
    - seq_mask: [max_seq_len]（True=有效）
    """
    seq = list(content_sequence)
    n_img_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "image"))
    n_txt_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "text"))

    img_emb = np.zeros((0, int(image_dim)), dtype=np.float32)
    if n_img_expected > 0:
        img_emb = _as_2d_embedding(np.load(image_emb_path), image_emb_path.name)
        if int(img_emb.shape[0]) != int(n_img_expected):
            raise ValueError(
                f"项目 {project_id} image 数量不一致：content_sequence={n_img_expected} vs {image_emb_path.name}={img_emb.shape}"
            )
        if int(img_emb.shape[1]) != int(image_dim):
            raise ValueError(f"项目 {project_id} image dim 不一致：期望 {image_dim}，但得到 {img_emb.shape[1]}")
    else:
        if image_emb_path.exists():
            img_emb = _as_2d_embedding(np.load(image_emb_path), image_emb_path.name)
            if int(img_emb.shape[0]) != 0:
                raise ValueError(
                    f"项目 {project_id} image 数量不一致：content_sequence=0 vs {image_emb_path.name}={img_emb.shape}"
                )

    txt_emb = np.zeros((0, int(text_dim)), dtype=np.float32)
    if n_txt_expected > 0:
        txt_emb = _as_2d_embedding(np.load(text_emb_path), text_emb_path.name)
        if int(txt_emb.shape[0]) != int(n_txt_expected):
            raise ValueError(
                f"项目 {project_id} text 数量不一致：content_sequence={n_txt_expected} vs {text_emb_path.name}={txt_emb.shape}"
            )
        if int(txt_emb.shape[1]) != int(text_dim):
            raise ValueError(f"项目 {project_id} text dim 不一致：期望 {text_dim}，但得到 {txt_emb.shape[1]}")
    else:
        if text_emb_path.exists():
            txt_emb = _as_2d_embedding(np.load(text_emb_path), text_emb_path.name)
            if int(txt_emb.shape[0]) != 0:
                raise ValueError(
                    f"项目 {project_id} text 数量不一致：content_sequence=0 vs {text_emb_path.name}={txt_emb.shape}"
                )

    seq_len = int(len(seq))
    img_seq = np.zeros((seq_len, int(image_dim)), dtype=np.float32)
    txt_seq = np.zeros((seq_len, int(text_dim)), dtype=np.float32)
    types: List[int] = []
    attrs: List[float] = []

    img_idx = 0
    txt_idx = 0

    for pos, item in enumerate(seq):
        t = str(item.get("type", "")).strip().lower()
        if t == "image":
            if img_idx >= int(img_emb.shape[0]):
                raise ValueError(f"项目 {project_id} image 指针越界：img_idx={img_idx} img_emb={img_emb.shape}")
            img_seq[pos] = img_emb[img_idx]
            img_idx += 1

            # 图片尺寸已在 content.json 中预处理好：width/height
            # 不再读取本地图片文件，避免数据加载极慢。
            w = _require_int_field(item, "width", project_id=project_id, pos=pos)
            h = _require_int_field(item, "height", project_id=project_id, pos=pos)
            if int(w) <= 0 or int(h) <= 0:
                raise ValueError(f"项目 {project_id} 的图片尺寸不合法：width={w} height={h}（pos={pos}）")
            area = int(w) * int(h)

            types.append(1)
            attrs.append(float(np.log(max(1.0, float(area)))))
        elif t == "text":
            if txt_idx >= int(txt_emb.shape[0]):
                raise ValueError(f"项目 {project_id} text 指针越界：txt_idx={txt_idx} txt_emb={txt_emb.shape}")
            txt_seq[pos] = txt_emb[txt_idx]
            txt_idx += 1

            # 文本长度已在 content.json 中预处理好：content_length
            length = _require_int_field(item, "content_length", project_id=project_id, pos=pos)
            if int(length) < 0:
                raise ValueError(f"项目 {project_id} 的文本长度不合法：content_length={length}（pos={pos}）")
            types.append(0)
            attrs.append(float(np.log(max(1.0, float(length)))))
        else:
            raise ValueError(f"项目 {project_id} content_sequence type 不支持：{t!r}（pos={pos}）")

    if img_idx != int(img_emb.shape[0]) or txt_idx != int(txt_emb.shape[0]):
        raise ValueError(
            f"项目 {project_id} 计数不一致：img_used={img_idx}/{img_emb.shape[0]} txt_used={txt_idx}/{txt_emb.shape[0]}"
        )

    L = int(len(types))
    if L <= 0:
        raise ValueError(f"项目 {project_id} content_sequence 为空，无法训练。")

    seed = int(random_seed) + _stable_hash_int(str(project_id))
    start, end = _truncate_window(L, int(max_seq_len), str(truncation_strategy), seed=seed)

    img_seq = img_seq[start:end].astype(np.float32, copy=False)
    txt_seq = txt_seq[start:end].astype(np.float32, copy=False)
    type_arr = np.asarray(types[start:end], dtype=np.int64)
    attr_arr = np.asarray(attrs[start:end], dtype=np.float32)

    L2 = int(type_arr.shape[0])
    pad_img = np.zeros((int(max_seq_len), int(image_dim)), dtype=np.float32)
    pad_txt = np.zeros((int(max_seq_len), int(text_dim)), dtype=np.float32)
    pad_type = np.zeros((int(max_seq_len),), dtype=np.int64)
    pad_attr = np.zeros((int(max_seq_len),), dtype=np.float32)
    pad_mask = np.zeros((int(max_seq_len),), dtype=bool)

    pad_img[:L2] = img_seq
    pad_txt[:L2] = txt_seq
    pad_type[:L2] = type_arr
    pad_attr[:L2] = attr_arr
    pad_mask[:L2] = True
    return pad_img, pad_txt, pad_type, pad_attr, pad_mask


def _build_features_for_split(
    df_split: pd.DataFrame,
    X_meta_all: Optional[np.ndarray],
    projects_root: Path,
    cfg: SeqConfig,
    use_meta: bool,
    image_dim: int,
    text_dim: int,
    logger=None,
) -> Tuple[
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, int],
]:
    missing_strategy = str(getattr(cfg, "missing_strategy", "error") or "error").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={missing_strategy!r}，可选：skip/error")

    emb_img = str(getattr(cfg, "image_embedding_type", "")).strip().lower()
    emb_txt = str(getattr(cfg, "text_embedding_type", "")).strip().lower()
    if emb_img not in {"clip", "siglip", "resnet"}:
        raise ValueError(f"不支持的 image_embedding_type={emb_img!r}，可选：clip/siglip/resnet")
    if emb_txt not in {"bge", "clip", "siglip"}:
        raise ValueError(f"不支持的 text_embedding_type={emb_txt!r}，可选：bge/clip/siglip")

    max_seq_len = int(getattr(cfg, "max_seq_len", 0))
    if max_seq_len <= 0:
        raise ValueError("max_seq_len 需要 > 0。")

    y_arr = _encode_binary_target(df_split[cfg.target_col])
    ids = [_normalize_project_id(x) for x in df_split[cfg.id_col].tolist()]

    kept_ids: List[str] = []
    kept_y: List[int] = []
    kept_meta_idx: List[int] = []

    X_img_list: List[np.ndarray] = []
    X_txt_list: List[np.ndarray] = []
    seq_type_list: List[np.ndarray] = []
    seq_attr_list: List[np.ndarray] = []
    seq_mask_list: List[np.ndarray] = []

    stats: Dict[str, int] = {
        "skipped_samples": 0,
        "missing_project_dir": 0,
        "missing_content_json": 0,
        "missing_image_embedding": 0,
        "missing_text_embedding": 0,
        "bad_sequence": 0,
    }

    for i, pid in enumerate(ids):
        pid = str(pid)
        if not pid:
            stats["skipped_samples"] += 1
            continue

        project_dir = projects_root / pid
        if not project_dir.exists():
            stats["missing_project_dir"] += 1
            if missing_strategy == "skip":
                stats["skipped_samples"] += 1
                continue
            raise FileNotFoundError(f"项目目录不存在：{project_dir}")

        content_json_path = project_dir / "content.json"
        if not content_json_path.exists():
            stats["missing_content_json"] += 1
            if missing_strategy == "skip":
                stats["skipped_samples"] += 1
                continue
            raise FileNotFoundError(f"缺少 content.json：{content_json_path}")

        image_emb_path = project_dir / f"image_{emb_img}.npy"
        text_emb_path = project_dir / f"text_{emb_txt}.npy"

        try:
            seq = _read_content_sequence(content_json_path)
            n_img_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "image"))
            n_txt_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "text"))

            if n_img_expected > 0 and not image_emb_path.exists():
                stats["missing_image_embedding"] += 1
                if missing_strategy == "skip":
                    stats["skipped_samples"] += 1
                    if logger is not None:
                        logger.warning("跳过项目 %s：缺少 %s", pid, str(image_emb_path))
                    continue
                raise FileNotFoundError(f"缺少 image embedding：{image_emb_path}")

            if n_txt_expected > 0 and not text_emb_path.exists():
                stats["missing_text_embedding"] += 1
                if missing_strategy == "skip":
                    stats["skipped_samples"] += 1
                    if logger is not None:
                        logger.warning("跳过项目 %s：缺少 %s", pid, str(text_emb_path))
                    continue
                raise FileNotFoundError(f"缺少 text embedding：{text_emb_path}")

            pad_img, pad_txt, pad_type, pad_attr, pad_mask = _build_one_project_sequence(
                project_id=pid,
                content_sequence=seq,
                image_emb_path=image_emb_path,
                text_emb_path=text_emb_path,
                image_dim=int(image_dim),
                text_dim=int(text_dim),
                max_seq_len=int(max_seq_len),
                truncation_strategy=str(getattr(cfg, "truncation_strategy", "first")),
                random_seed=int(getattr(cfg, "random_seed", 42)),
            )
        except Exception as e:
            stats["bad_sequence"] += 1
            if missing_strategy == "skip":
                stats["skipped_samples"] += 1
                if logger is not None:
                    logger.warning("跳过项目 %s：序列构造失败：%s", pid, e)
                continue
            raise

        kept_ids.append(pid)
        kept_y.append(int(y_arr[i]))
        if use_meta:
            kept_meta_idx.append(int(i))

        X_img_list.append(pad_img)
        X_txt_list.append(pad_txt)
        seq_type_list.append(pad_type)
        seq_attr_list.append(pad_attr)
        seq_mask_list.append(pad_mask)

    if not kept_ids:
        raise RuntimeError("该 split 没有可用样本（可能全部被 missing_strategy=skip 跳过）。")

    X_img = np.stack(X_img_list, axis=0).astype(np.float32, copy=False)
    X_txt = np.stack(X_txt_list, axis=0).astype(np.float32, copy=False)
    seq_type = np.stack(seq_type_list, axis=0).astype(np.int64, copy=False)
    seq_attr = np.stack(seq_attr_list, axis=0).astype(np.float32, copy=False)
    seq_mask = np.stack(seq_mask_list, axis=0).astype(bool, copy=False)
    y = np.asarray(kept_y, dtype=np.int64)

    X_meta = None
    if use_meta:
        if X_meta_all is None:
            raise ValueError("use_meta=True 时，X_meta_all 不能为空。")
        X_meta = X_meta_all[np.asarray(kept_meta_idx, dtype=np.int64)].astype(np.float32, copy=False)

    return X_meta, X_img, X_txt, seq_type, seq_attr, seq_mask, y, kept_ids, stats


def prepare_seq_data(
    csv_path: Path,
    projects_root: Path,
    cfg: SeqConfig,
    use_meta: bool,
    cache_dir: Path | None = None,
    logger=None,
) -> PreparedSeqData:
    """读 CSV -> 切分 -> 构建 seq（+可选 meta）特征 -> 返回 numpy 数组。支持缓存。"""
    use_cache = bool(getattr(cfg, "use_cache", False)) and cache_dir is not None

    cache_path: Path | None = None
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _seq_make_cache_key(csv_path, projects_root, cfg, use_meta=use_meta)
        cache_path = cache_dir / f"{cache_key}.npz"
        if cache_path.exists():
            try:
                prepared = _seq_load_cache(cache_path)
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

    emb_img = str(getattr(cfg, "image_embedding_type", "")).strip().lower()
    emb_txt = str(getattr(cfg, "text_embedding_type", "")).strip().lower()
    img_name = f"image_{emb_img}.npy"
    txt_name = f"text_{emb_txt}.npy"

    all_ids = [_normalize_project_id(x) for x in raw_df[cfg.id_col].tolist()]
    image_dim = _infer_embedding_dim(projects_root, all_ids, img_name)
    text_dim = _infer_embedding_dim(projects_root, all_ids, txt_name)
    if int(image_dim) <= 0:
        raise RuntimeError(f"无法推断 image embedding dim：未找到任何 {img_name}")
    if int(text_dim) <= 0:
        raise RuntimeError(f"无法推断 text embedding dim：未找到任何 {txt_name}")

    (
        X_meta_train,
        X_img_train,
        X_txt_train,
        seq_type_train,
        seq_attr_train,
        seq_mask_train,
        y_train,
        train_ids,
        train_stats,
    ) = _build_features_for_split(
        train_df,
        X_meta_train_all,
        projects_root,
        cfg,
        use_meta=use_meta,
        image_dim=int(image_dim),
        text_dim=int(text_dim),
        logger=logger,
    )
    (
        X_meta_val,
        X_img_val,
        X_txt_val,
        seq_type_val,
        seq_attr_val,
        seq_mask_val,
        y_val,
        val_ids,
        val_stats,
    ) = _build_features_for_split(
        val_df,
        X_meta_val_all,
        projects_root,
        cfg,
        use_meta=use_meta,
        image_dim=int(image_dim),
        text_dim=int(text_dim),
        logger=logger,
    )
    (
        X_meta_test,
        X_img_test,
        X_txt_test,
        seq_type_test,
        seq_attr_test,
        seq_mask_test,
        y_test,
        test_ids,
        test_stats,
    ) = _build_features_for_split(
        test_df,
        X_meta_test_all,
        projects_root,
        cfg,
        use_meta=use_meta,
        image_dim=int(image_dim),
        text_dim=int(text_dim),
        logger=logger,
    )

    stats: Dict[str, int] = {}
    for s in (train_stats, val_stats, test_stats):
        for k, v in s.items():
            stats[k] = int(stats.get(k, 0) + int(v))

    prepared = PreparedSeqData(
        y_train=y_train,
        train_project_ids=train_ids,
        y_val=y_val,
        val_project_ids=val_ids,
        y_test=y_test,
        test_project_ids=test_ids,
        X_img_train=X_img_train,
        X_txt_train=X_txt_train,
        seq_type_train=seq_type_train,
        seq_attr_train=seq_attr_train,
        seq_mask_train=seq_mask_train,
        X_img_val=X_img_val,
        X_txt_val=X_txt_val,
        seq_type_val=seq_type_val,
        seq_attr_val=seq_attr_val,
        seq_mask_val=seq_mask_val,
        X_img_test=X_img_test,
        X_txt_test=X_txt_test,
        seq_type_test=seq_type_test,
        seq_attr_test=seq_attr_test,
        seq_mask_test=seq_mask_test,
        image_embedding_dim=int(image_dim),
        text_embedding_dim=int(text_dim),
        max_seq_len=int(getattr(cfg, "max_seq_len", 0)),
        X_meta_train=X_meta_train,
        X_meta_val=X_meta_val,
        X_meta_test=X_meta_test,
        meta_dim=int(meta_dim) if use_meta else 0,
        preprocessor=preprocessor,
        feature_names=feature_names,
        stats=stats,
    )

    if use_cache and cache_path is not None:
        try:
            meta = {
                "cache_version": _SEQ_CACHE_VERSION,
                "cache_key": cache_path.stem,
                "csv_path": str(csv_path.as_posix()),
                "projects_root": str(projects_root.as_posix()),
                "use_meta": bool(use_meta),
                "config": cfg.to_dict(),
            }
            _seq_save_cache(cache_path, prepared, meta=meta)
            if logger is not None:
                logger.info("已写入数据缓存：%s", str(cache_path))
        except Exception as e:
            if logger is not None:
                logger.warning("写入缓存失败：%s", e)

    return prepared
