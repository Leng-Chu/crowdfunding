# -*- coding: utf-8 -*-
"""
数据加载与特征构建（mdl）：

- 读取项目目录下的 content.json，并根据 content_sequence 做“统一序列截断”
- 根据截断后的内容块，映射回 image_{emb}.npy / text_{emb}.npy 的子集
- 并将 cover_image_{emb}.npy / title_blurb_{emb}.npy 分别拼接到 image/text 序列最前面
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MdlConfig


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
    """
    将标签编码为 0/1。

    约定：target_col 为数值标签，取值只能是 0/1。
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("target_col 需要为 0/1 的数值标签。")

    s = series.fillna(0)
    uniq = pd.unique(s)
    try:
        uniq_set = {int(v) for v in uniq if pd.notna(v)}
    except Exception as e:
        raise ValueError(f"target_col 无法解析为整数标签：{e}") from e

    if not uniq_set.issubset({0, 1}):
        raise ValueError(f"target_col 取值必须为 0/1，但得到：{sorted(uniq_set)}")

    return s.astype(int).to_numpy(dtype=np.int64)


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
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    v = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(v & 0x7FFFFFFFFFFFFFFF)


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


# -----------------------------
# multimodal
# -----------------------------

@dataclass(frozen=True)
class PreparedMultiModalData:
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

    # image（可选）
    X_image_train: Optional[np.ndarray]
    len_image_train: Optional[np.ndarray]
    X_image_val: Optional[np.ndarray]
    len_image_val: Optional[np.ndarray]
    X_image_test: Optional[np.ndarray]
    len_image_test: Optional[np.ndarray]
    image_embedding_dim: int
    max_image_seq_len: int

    # text（可选）
    X_text_train: Optional[np.ndarray]
    len_text_train: Optional[np.ndarray]
    X_text_val: Optional[np.ndarray]
    len_text_val: Optional[np.ndarray]
    X_text_test: Optional[np.ndarray]
    len_text_test: Optional[np.ndarray]
    text_embedding_dim: int
    max_text_seq_len: int

    # 统一截断配置（用于复现）
    max_seq_len: int

    stats: Dict[str, int]


def _mm_load_dataframe(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _read_content_json(content_json_path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(content_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"读取 content.json 失败：{content_json_path} | {type(e).__name__}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"content.json 顶层不是 dict：{content_json_path}")
    return dict(obj)


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


def _mdl_load_project_keep_seqs(
    project_dir: Path,
    project_id: str,
    cfg: MdlConfig,
    use_image: bool,
    use_text: bool,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
    int,
    int,
    int,
    int,
]:
    """
    返回：img_keep, txt_keep, img_dim, txt_dim, len_img, len_txt, skipped(0/1)

    说明：
    - img_keep/txt_keep 可能为“仅包含 cover/title_blurb”的序列；
    - 当必须文件缺失且 missing_strategy=skip 时，skipped=1。
    """
    missing_strategy = str(getattr(cfg, "missing_strategy", "error") or "error").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={cfg.missing_strategy!r}，可选：skip/error")

    if not (bool(use_image) or bool(use_text)):
        return None, None, 0, 0, 0, 0, 0

    content_json_path = project_dir / "content.json"
    if not content_json_path.exists():
        if missing_strategy == "skip":
            return None, None, 0, 0, 0, 0, 1
        raise FileNotFoundError(f"缺少 content.json：{content_json_path}")

    content_obj = _read_content_json(content_json_path)
    seq = content_obj.get("content_sequence", None)
    if not isinstance(seq, list):
        raise ValueError(f"content_sequence 不存在或不是 list：{content_json_path}")
    seq = list(seq)
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

    img_keep = None
    txt_keep = None
    img_dim = 0
    txt_dim = 0
    len_img = 0
    len_txt = 0

    img_emb = None
    txt_emb = None

    if bool(use_image):
        img_type = str(getattr(cfg, "image_embedding_type", "") or "").strip().lower()
        if img_type not in {"clip", "siglip", "resnet"}:
            raise ValueError(f"不支持的 image_embedding_type={cfg.image_embedding_type!r}，可选：clip/siglip/resnet")

        cover_emb_path = project_dir / f"cover_image_{img_type}.npy"
        if not cover_emb_path.exists():
            if missing_strategy == "skip":
                return None, None, 0, 0, 0, 0, 1
            raise FileNotFoundError(f"缺少 cover_image 嵌入文件：{cover_emb_path}")

        cover_emb = _as_2d_embedding(np.load(cover_emb_path), cover_emb_path.name)
        if int(cover_emb.shape[0]) != 1:
            raise ValueError(f"项目 {project_id} cover_image 行数不合法：期望 1，但得到 {cover_emb.shape}")
        img_dim = int(cover_emb.shape[1])

        image_emb_path = project_dir / f"image_{img_type}.npy"
        if image_emb_path.exists():
            img_emb = _as_2d_embedding(np.load(image_emb_path), image_emb_path.name)
            if int(img_emb.shape[1]) != int(img_dim):
                raise ValueError(f"项目 {project_id} image dim 不一致：cover={img_dim} vs story={img_emb.shape[1]}")
            if int(img_emb.shape[0]) != int(n_img_expected):
                raise ValueError(
                    f"项目 {project_id} image 数量不一致：content_sequence={n_img_expected} vs {image_emb_path.name}={img_emb.shape}"
                )
        else:
            if int(n_img_expected) != 0:
                if missing_strategy == "skip":
                    return None, None, 0, 0, 0, 0, 1
                raise FileNotFoundError(f"缺少 image 嵌入文件：{image_emb_path}")

    if bool(use_text):
        txt_type = str(getattr(cfg, "text_embedding_type", "") or "").strip().lower()
        if txt_type not in {"bge", "clip", "siglip"}:
            raise ValueError(f"不支持的 text_embedding_type={cfg.text_embedding_type!r}，可选：bge/clip/siglip")

        title_blurb_emb_path = project_dir / f"title_blurb_{txt_type}.npy"
        if not title_blurb_emb_path.exists():
            if missing_strategy == "skip":
                return None, None, 0, 0, 0, 0, 1
            raise FileNotFoundError(f"缺少 title_blurb 嵌入文件：{title_blurb_emb_path}")

        title_blurb_emb = _as_2d_embedding(np.load(title_blurb_emb_path), title_blurb_emb_path.name)
        if int(title_blurb_emb.shape[0]) <= 0:
            raise ValueError(f"项目 {project_id} title_blurb 为空：{title_blurb_emb_path}")
        txt_dim = int(title_blurb_emb.shape[1])

        text_emb_path = project_dir / f"text_{txt_type}.npy"
        if text_emb_path.exists():
            txt_emb = _as_2d_embedding(np.load(text_emb_path), text_emb_path.name)
            if int(txt_emb.shape[1]) != int(txt_dim):
                raise ValueError(f"项目 {project_id} text dim 不一致：title_blurb={txt_dim} vs story={txt_emb.shape[1]}")
            if int(txt_emb.shape[0]) != int(n_txt_expected):
                raise ValueError(
                    f"项目 {project_id} text 数量不一致：content_sequence={n_txt_expected} vs {text_emb_path.name}={txt_emb.shape}"
                )
        else:
            if int(n_txt_expected) != 0:
                if missing_strategy == "skip":
                    return None, None, 0, 0, 0, 0, 1
                raise FileNotFoundError(f"缺少 text 嵌入文件：{text_emb_path}")

    # 统一序列截断：决定窗口 [start, end)
    max_seq_len = int(getattr(cfg, "max_seq_len", 0))
    if max_seq_len <= 0:
        raise ValueError("max_seq_len 需要 > 0。")
    truncation_strategy = str(getattr(cfg, "truncation_strategy", "first"))
    seed = (int(getattr(cfg, "random_seed", 42)) + _stable_hash_int(str(project_id))) & 0x7FFFFFFFFFFFFFFF
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

    if bool(use_image):
        if img_emb is not None:
            if keep_img_indices:
                img_keep_story = img_emb[np.asarray(keep_img_indices, dtype=np.int64)]
            else:
                img_keep_story = img_emb[:0]
        else:
            img_keep_story = cover_emb[:0]
        img_keep = np.concatenate([cover_emb, img_keep_story], axis=0).astype(np.float32, copy=False)
        len_img = int(img_keep.shape[0])

    if bool(use_text):
        if txt_emb is not None:
            if keep_txt_indices:
                txt_keep_story = txt_emb[np.asarray(keep_txt_indices, dtype=np.int64)]
            else:
                txt_keep_story = txt_emb[:0]
        else:
            txt_keep_story = title_blurb_emb[:0]
        txt_keep = np.concatenate([title_blurb_emb, txt_keep_story], axis=0).astype(np.float32, copy=False)
        len_txt = int(txt_keep.shape[0])

    return img_keep, txt_keep, int(img_dim), int(txt_dim), int(len_img), int(len_txt), 0


def _mm_build_features_for_split(
    df_split: pd.DataFrame,
    X_meta_all: Optional[np.ndarray],
    projects_root: Path,
    cfg: MdlConfig,
    use_meta: bool,
    use_image: bool,
    use_text: bool,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    List[str],
    Dict[str, int],
    int,
    int,
    int,
    int,
]:
    """为一个 split 构建所选分支的 X/len/y。"""
    if use_meta:
        if X_meta_all is None:
            raise ValueError("use_meta=True 时，X_meta_all 不能为空。")
        if int(X_meta_all.shape[0]) != int(len(df_split)):
            raise ValueError(f"X_meta_all 与 df_split 样本数不一致：{X_meta_all.shape[0]} vs {len(df_split)}")

    project_ids: List[str] = [_normalize_project_id(v) for v in df_split[cfg.id_col].tolist()]
    y_all = _encode_binary_target(df_split[cfg.target_col])

    kept_ids: List[str] = []
    kept_indices: List[int] = []

    img_seqs: List[np.ndarray] = []
    img_lens: List[int] = []
    txt_seqs: List[np.ndarray] = []
    txt_lens: List[int] = []

    skipped_total = 0
    missing_project_dir = 0
    image_dim = 0
    text_dim = 0

    need_project_dir = bool(use_image or use_text)
    missing_strategy = str(getattr(cfg, "missing_strategy", "error") or "error").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={cfg.missing_strategy!r}，可选：skip/error")

    for i, (pid, _) in enumerate(zip(project_ids, y_all)):
        if not pid:
            skipped_total += 1
            continue

        project_dir = projects_root / pid if need_project_dir else None
        if need_project_dir and project_dir is not None and not project_dir.exists():
            missing_project_dir += 1
            if missing_strategy == "skip":
                skipped_total += 1
                continue
            raise FileNotFoundError(f"找不到项目目录：{project_dir}")

        img_seq = None
        txt_seq = None
        img_dim_one = 0
        txt_dim_one = 0
        len_img = 0
        len_txt = 0
        skipped = 0
        if need_project_dir and project_dir is not None:
            img_seq, txt_seq, img_dim_one, txt_dim_one, len_img, len_txt, skipped = _mdl_load_project_keep_seqs(
                project_dir,
                project_id=str(pid),
                cfg=cfg,
                use_image=bool(use_image),
                use_text=bool(use_text),
            )
        if int(skipped) != 0:
            skipped_total += 1
            continue

        if bool(use_image):
            if img_seq is None or int(len_img) <= 0:
                raise RuntimeError(f"加载 image 失败但未被 skip：{pid}")
            if int(image_dim) <= 0:
                image_dim = int(img_dim_one)
            if int(img_dim_one) != int(image_dim):
                raise ValueError(f"image_embedding_dim 不一致：期望 {image_dim}，但 {pid} 为 {img_dim_one}")
            img_seqs.append(img_seq)
            img_lens.append(int(len_img))

        if bool(use_text):
            if txt_seq is None or int(len_txt) <= 0:
                raise RuntimeError(f"加载 text 失败但未被 skip：{pid}")
            if int(text_dim) <= 0:
                text_dim = int(txt_dim_one)
            if int(txt_dim_one) != int(text_dim):
                raise ValueError(f"text_embedding_dim 不一致：期望 {text_dim}，但 {pid} 为 {txt_dim_one}")
            txt_seqs.append(txt_seq)
            txt_lens.append(int(len_txt))

        kept_ids.append(pid)
        kept_indices.append(i)

    if not kept_indices:
        raise RuntimeError("该数据切分中没有可用样本（可能都被 skip 了或缺少嵌入文件）。")

    idx_arr = np.asarray(kept_indices, dtype=np.int64)
    y_arr = np.asarray(y_all[idx_arr], dtype=np.int64)

    X_meta = np.asarray(X_meta_all[idx_arr], dtype=np.float32) if use_meta else None

    X_image = None
    len_image = None
    max_img_len = 0
    if use_image:
        max_img_len = int(max(img_lens)) if img_lens else 0
        X_image = np.zeros((len(img_seqs), max_img_len, int(image_dim)), dtype=np.float32)
        for j, seq in enumerate(img_seqs):
            L = int(seq.shape[0])
            X_image[j, :L, :] = seq
        len_image = np.asarray(img_lens, dtype=np.int64)

    X_text = None
    len_text = None
    max_txt_len = 0
    if use_text:
        max_txt_len = int(max(txt_lens)) if txt_lens else 0
        X_text = np.zeros((len(txt_seqs), max_txt_len, int(text_dim)), dtype=np.float32)
        for j, seq in enumerate(txt_seqs):
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


def prepare_multimodal_data(
    csv_path: Path,
    projects_root: Path,
    cfg: MdlConfig,
    use_meta: bool,
    use_image: bool,
    use_text: bool,
    logger=None,
) -> PreparedMultiModalData:
    """两路/三路：读 CSV -> 切分 -> 构建所选分支特征 -> 返回 numpy 数组。"""

    raw_df = _mm_load_dataframe(csv_path)
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
    ) = _mm_build_features_for_split(train_df, X_meta_train_all, projects_root, cfg, use_meta, use_image, use_text)
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
    ) = _mm_build_features_for_split(val_df, X_meta_val_all, projects_root, cfg, use_meta, use_image, use_text)
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
    ) = _mm_build_features_for_split(test_df, X_meta_test_all, projects_root, cfg, use_meta, use_image, use_text)

    if use_image and (int(img_dim_val) != int(img_dim) or int(img_dim_test) != int(img_dim)):
        raise ValueError(f"image_embedding_dim 不一致：train={img_dim} val={img_dim_val} test={img_dim_test}")
    if use_text and (int(txt_dim_val) != int(txt_dim) or int(txt_dim_test) != int(txt_dim)):
        raise ValueError(f"text_embedding_dim 不一致：train={txt_dim} val={txt_dim_val} test={txt_dim_test}")

    stats: Dict[str, int] = {}
    for s in (train_stats, val_stats, test_stats):
        for k, v in s.items():
            stats[k] = int(stats.get(k, 0) + int(v))

    prepared = PreparedMultiModalData(
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
        image_embedding_dim=int(img_dim) if use_image else 0,
        max_image_seq_len=int(max(max_img_train, max_img_val, max_img_test)) if use_image else 0,
        X_text_train=X_txt_train,
        len_text_train=len_txt_train,
        X_text_val=X_txt_val,
        len_text_val=len_txt_val,
        X_text_test=X_txt_test,
        len_text_test=len_txt_test,
        text_embedding_dim=int(txt_dim) if use_text else 0,
        max_text_seq_len=int(max(max_txt_train, max_txt_val, max_txt_test)) if use_text else 0,
        max_seq_len=int(getattr(cfg, "max_seq_len", 0)),
        stats=stats,
    )

    return prepared
