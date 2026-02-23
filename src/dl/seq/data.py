# -*- coding: utf-8 -*-
"""
数据加载与特征构建（seq）：

- 读取 metadata CSV（样本 ID、标签、可选 meta 特征列）
- 读取项目目录下的 content.json，并根据 title/blurb/cover_image + content_sequence 构造“图文交替统一序列”
- 读取预计算 embedding：cover_image_{emb_type}.npy / title_blurb_{emb_type}.npy / image_{emb_type}.npy / text_{emb_type}.npy
- 可选计算每个内容块的属性：文本长度 / 图片面积（`use_seq_attr=True` 时读取 content_length/width/height）
- 支持按 max_seq_len 截断（first/random）并输出 seq_mask
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
    """把字符串稳定地映射为 int（用于可复现的随机截断/打乱）。"""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    v = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(v & 0x7FFFFFFFFFFFFFFF)


@dataclass
class TabularPreprocessor:
    """轻量级表格预处理器：one-hot + 数值标准化（mean/std）。"""

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


def _read_content_json(content_json_path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(content_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"读取 content.json 失败：{content_json_path} | {type(e).__name__}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"content.json 顶层不是 dict：{content_json_path}")
    return dict(obj)


def _read_content_sequence(content_json_path: Path) -> List[Dict[str, Any]]:
    obj = _read_content_json(content_json_path)
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


def _extract_title_blurb_lengths(content_obj: Dict[str, Any]) -> List[int]:
    """
    读取 content.json 中的 title/blurb，并按 [title, blurb] 的顺序返回长度列表。

    说明：title_blurb_{model}.npy 的行顺序来自向量化脚本 `vectorize_csv_data.py`：
    - 若 title 和 blurb 都存在：保存 [title, blurb] 两行
    - 若只有一个存在：只保存该一个
    """
    def _len_from_obj(v: Any) -> int:
        if v is None:
            return 0
        if not isinstance(v, dict):
            raise ValueError("content.json 的 title/blurb 需要是 dict（包含 content/content_length）。")
        content = str(v.get("content", "") or "").strip()
        if not content:
            return 0
        if "content_length" in v:
            try:
                return int(v.get("content_length") or 0)
            except Exception:
                pass
        return int(len(content))

    title_len = _len_from_obj(content_obj.get("title", None))
    blurb_len = _len_from_obj(content_obj.get("blurb", None))

    lengths: List[int] = []
    if int(title_len) > 0:
        lengths.append(int(title_len))
    if int(blurb_len) > 0:
        lengths.append(int(blurb_len))
    return lengths


def _extract_cover_area(content_obj: Dict[str, Any], project_id: str) -> int:
    """
    从 content.json 的 cover_image 字段读取 width/height，并返回 area=width*height。
    """
    cover = content_obj.get("cover_image", None)
    if not isinstance(cover, dict):
        raise ValueError(f"项目 {project_id} cover_image 字段不合法：期望 dict，但得到 {type(cover).__name__}")

    w = cover.get("width", None)
    h = cover.get("height", None)
    try:
        w = int(w)
        h = int(h)
    except Exception as e:
        raise ValueError(f"项目 {project_id} cover_image 的 width/height 不是整数：{w!r}/{h!r}") from e
    if int(w) <= 0 or int(h) <= 0:
        raise ValueError(f"项目 {project_id} cover_image 尺寸不合法：width={w} height={h}")
    return int(w) * int(h)


def _build_one_project_sequence(
    project_id: str,
    content_obj: Dict[str, Any],
    image_emb_path: Path,
    text_emb_path: Path,
    cover_emb_path: Path,
    title_blurb_emb_path: Path,
    image_dim: int,
    text_dim: int,
    max_seq_len: int,
    truncation_strategy: str,
    random_seed: int,
    use_prefix: bool = True,
    use_seq_attr: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造单项目的统一序列特征（已截断 + padding 到 max_seq_len）：
    - X_img: [max_seq_len, D_img]（非 image token 为 0）
    - X_txt: [max_seq_len, D_txt]（非 text token 为 0）
    - seq_type: [max_seq_len]（0=text，1=image）
    - seq_attr: [max_seq_len]（use_seq_attr=True 时为 log(length/area)，否则全 0）
    - seq_mask: [max_seq_len]（True=有效）
    """
    use_prefix = bool(use_prefix)
    use_seq_attr = bool(use_seq_attr)

    content_sequence = content_obj.get("content_sequence", None)
    if not isinstance(content_sequence, list):
        raise ValueError(f"项目 {project_id} content_sequence 不存在或不是 list")
    seq = [dict(x) for x in content_sequence]
    n_img_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "image"))
    n_txt_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "text"))

    # 根据 use_prefix 决定是否加载和使用 prefix tokens
    cover_emb = None
    title_blurb_emb = None
    title_blurb_lengths = []
    prefix_len = 0
    
    if use_prefix:
        if not cover_emb_path.exists():
            raise FileNotFoundError(f"缺少 cover_image embedding：{cover_emb_path}")
        cover_emb = _as_2d_embedding(np.load(cover_emb_path), cover_emb_path.name)
        if int(cover_emb.shape[0]) != 1:
            raise ValueError(f"项目 {project_id} cover_image 行数不合法：期望 1，但得到 {cover_emb.shape}")
        if int(cover_emb.shape[1]) != int(image_dim):
            raise ValueError(f"项目 {project_id} cover_image dim 不一致：期望 {image_dim}，但得到 {cover_emb.shape[1]}")

        if not title_blurb_emb_path.exists():
            raise FileNotFoundError(f"缺少 title_blurb embedding：{title_blurb_emb_path}")
        title_blurb_emb = _as_2d_embedding(np.load(title_blurb_emb_path), title_blurb_emb_path.name)
        title_blurb_lengths = _extract_title_blurb_lengths(content_obj)
        if int(title_blurb_emb.shape[0]) != int(len(title_blurb_lengths)):
            raise ValueError(
                f"项目 {project_id} title_blurb 数量不一致：content.json(title/blurb)={len(title_blurb_lengths)} "
                f"vs {title_blurb_emb_path.name}={title_blurb_emb.shape}"
            )
        if int(title_blurb_emb.shape[0]) <= 0:
            raise ValueError(f"项目 {project_id} title_blurb 为空：{title_blurb_emb_path}")
        if int(title_blurb_emb.shape[1]) != int(text_dim):
            raise ValueError(f"项目 {project_id} title_blurb dim 不一致：期望 {text_dim}，但得到 {title_blurb_emb.shape[1]}")
        
        prefix_len = int(title_blurb_emb.shape[0]) + 1  # +1 = cover_image
    else:
        # 当 use_prefix=False 时，跳过 cover_image 和 title_blurb 的验证与加载
        prefix_len = 0

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

    seq_len = int(prefix_len + len(seq))
    img_seq = np.zeros((seq_len, int(image_dim)), dtype=np.float32)
    txt_seq = np.zeros((seq_len, int(text_dim)), dtype=np.float32)
    types: List[int] = []
    attrs: List[float] = []

    img_idx = 0
    txt_idx = 0

    # 如果 use_prefix=True，则添加 prefix：title -> blurb -> cover_image
    if use_prefix:
        prefix_pos = 0
        for i in range(int(title_blurb_emb.shape[0])):
            txt_seq[prefix_pos] = title_blurb_emb[i]
            types.append(0)
            if use_seq_attr:
                attrs.append(float(np.log(max(1.0, float(title_blurb_lengths[i])))))
            else:
                attrs.append(0.0)
            prefix_pos += 1

        img_seq[prefix_pos] = cover_emb[0]
        types.append(1)
        if use_seq_attr:
            cover_area = _extract_cover_area(content_obj, project_id=project_id)
            attrs.append(float(np.log(max(1.0, float(cover_area)))))
        else:
            attrs.append(0.0)
        prefix_pos += 1
    else:
        # 如果 use_prefix=False，则不添加任何 prefix
        prefix_pos = 0

    for pos, item in enumerate(seq):
        pos2 = int(prefix_pos + pos)
        t = str(item.get("type", "")).strip().lower()
        if t == "image":
            if img_idx >= int(img_emb.shape[0]):
                raise ValueError(f"项目 {project_id} image 指针越界：img_idx={img_idx} img_emb={img_emb.shape}")
            img_seq[pos2] = img_emb[img_idx]
            img_idx += 1

            types.append(1)
            if use_seq_attr:
                # 图片尺寸已在 content.json 中预处理好：width/height
                # 不读取本地图片文件（仅使用预计算 embedding）。
                w = _require_int_field(item, "width", project_id=project_id, pos=pos)
                h = _require_int_field(item, "height", project_id=project_id, pos=pos)
                if int(w) <= 0 or int(h) <= 0:
                    raise ValueError(f"项目 {project_id} 的图片尺寸不合法：width={w} height={h}（pos={pos}）")
                area = int(w) * int(h)
                attrs.append(float(np.log(max(1.0, float(area)))))
            else:
                attrs.append(0.0)
        elif t == "text":
            if txt_idx >= int(txt_emb.shape[0]):
                raise ValueError(f"项目 {project_id} text 指针越界：txt_idx={txt_idx} txt_emb={txt_emb.shape}")
            txt_seq[pos2] = txt_emb[txt_idx]
            txt_idx += 1

            types.append(0)
            if use_seq_attr:
                # 文本长度已在 content.json 中预处理好：content_length
                length = _require_int_field(item, "content_length", project_id=project_id, pos=pos)
                if int(length) < 0:
                    raise ValueError(f"项目 {project_id} 的文本长度不合法：content_length={length}（pos={pos}）")
                attrs.append(float(np.log(max(1.0, float(length)))))
            else:
                attrs.append(0.0)
        else:
            raise ValueError(f"项目 {project_id} content_sequence type 不支持：{t!r}（pos={pos}）")

    if img_idx != int(img_emb.shape[0]) or txt_idx != int(txt_emb.shape[0]):
        raise ValueError(
            f"项目 {project_id} 计数不一致：img_used={img_idx}/{img_emb.shape[0]} txt_used={txt_idx}/{txt_emb.shape[0]}"
        )

    L = int(len(types))
    if L <= 0:
        raise ValueError(f"项目 {project_id} content_sequence 为空，无法训练。")

    seed = (int(random_seed) + _stable_hash_int(str(project_id))) & 0x7FFFFFFFFFFFFFFF
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

    # 获取开关配置
    use_prefix = getattr(cfg, "use_prefix", True)
    use_seq_attr = getattr(cfg, "use_seq_attr", True)

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
        "missing_cover_image_embedding": 0,
        "missing_title_blurb_embedding": 0,
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
        cover_emb_path = project_dir / f"cover_image_{emb_img}.npy"
        title_blurb_emb_path = project_dir / f"title_blurb_{emb_txt}.npy"

        try:
            content_obj = _read_content_json(content_json_path)
            seq = content_obj.get("content_sequence", None)
            if not isinstance(seq, list):
                raise ValueError(f"content_sequence 不存在或不是 list：{content_json_path}")
            seq = [dict(x) for x in seq]
            n_img_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "image"))
            n_txt_expected = int(sum(1 for x in seq if str(x.get("type", "")).strip().lower() == "text"))

            # 根据use_prefix决定是否检查cover和title_blurb embedding文件
            if use_prefix:
                if not cover_emb_path.exists():
                    stats["missing_cover_image_embedding"] += 1
                    if missing_strategy == "skip":
                        stats["skipped_samples"] += 1
                        if logger is not None:
                            logger.warning("跳过项目 %s：缺少 %s", pid, str(cover_emb_path))
                        continue
                    raise FileNotFoundError(f"缺少 cover_image embedding：{cover_emb_path}")

                if not title_blurb_emb_path.exists():
                    stats["missing_title_blurb_embedding"] += 1
                    if missing_strategy == "skip":
                        stats["skipped_samples"] += 1
                        if logger is not None:
                            logger.warning("跳过项目 %s：缺少 %s", pid, str(title_blurb_emb_path))
                        continue
                    raise FileNotFoundError(f"缺少 title_blurb embedding：{title_blurb_emb_path}")

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
                content_obj=content_obj,
                image_emb_path=image_emb_path,
                text_emb_path=text_emb_path,
                cover_emb_path=cover_emb_path,
                title_blurb_emb_path=title_blurb_emb_path,
                image_dim=int(image_dim),
                text_dim=int(text_dim),
                max_seq_len=int(max_seq_len),
                truncation_strategy=str(getattr(cfg, "truncation_strategy", "first")),
                random_seed=int(getattr(cfg, "random_seed", 42)),
                use_prefix=use_prefix,
                use_seq_attr=use_seq_attr,
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
    logger=None,
) -> PreparedSeqData:
    """读 CSV -> 切分 -> 构建 seq（+可选 meta）特征 -> 返回 numpy 数组。"""

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
    cover_name = f"cover_image_{emb_img}.npy"
    title_blurb_name = f"title_blurb_{emb_txt}.npy"

    all_ids = [_normalize_project_id(x) for x in raw_df[cfg.id_col].tolist()]
    
    # 根据use_prefix参数决定是否推断prefix embeddings的维度
    use_prefix = getattr(cfg, "use_prefix", True)
    
    # 推断image embedding维度
    image_dim = _infer_embedding_dim(projects_root, all_ids, img_name)
    if int(image_dim) <= 0 and use_prefix:
        image_dim = _infer_embedding_dim(projects_root, all_ids, cover_name)

    # 推断text embedding维度
    text_dim = _infer_embedding_dim(projects_root, all_ids, txt_name)
    if int(text_dim) <= 0 and use_prefix:
        text_dim = _infer_embedding_dim(projects_root, all_ids, title_blurb_name)
    
    if int(image_dim) <= 0:
        raise RuntimeError(f"无法推断 image embedding dim：未找到任何 {img_name} 或 {cover_name}")
    if int(text_dim) <= 0:
        raise RuntimeError(f"无法推断 text embedding dim：未找到任何 {txt_name} 或 {title_blurb_name}")

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

    return prepared
