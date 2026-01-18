# -*- coding: utf-8 -*-
"""
数据加载与特征构建：
- 读取 CSV（只用 project_id + state）
- 从 data/projects/.../<project_id>/ 读取文本嵌入 npy
- 将 title_blurb_{type}.npy 与 text_{type}.npy 堆叠为一个“向量集合”（形状为 (L, D)）

说明：
- title_blurb_*.npy 通常是 (1, D) 或 (2, D)
- text_*.npy 通常是 (N, D)，允许缺失；缺失时只使用 title_blurb
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import TextDLConfig

_CACHE_VERSION = 1


@dataclass(frozen=True)
class PreparedTextData:
    # X_*: (N, L_max, D)，len_*: (N,)
    X_train: np.ndarray
    len_train: np.ndarray
    y_train: np.ndarray
    train_project_ids: List[str]

    X_val: np.ndarray
    len_val: np.ndarray
    y_val: np.ndarray
    val_project_ids: List[str]

    X_test: np.ndarray
    len_test: np.ndarray
    y_test: np.ndarray
    test_project_ids: List[str]

    embedding_dim: int
    max_seq_len: int
    stats: Dict[str, int]


def load_labels(csv_path: Path, cfg: TextDLConfig) -> pd.DataFrame:
    """读取 CSV，只保留 project_id 与标签列。"""
    return pd.read_csv(csv_path, usecols=[cfg.id_col, cfg.target_col])


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
    将标签编码为 0/1：
    - 若已是数值：直接转 int
    - 若是字符串：仅把 successful 视为 1，其它视为 0
    """
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


def split_by_ratio(
    df: pd.DataFrame, cfg: TextDLConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按比例切分 train/val/test。"""
    if bool(getattr(cfg, "shuffle_before_split", False)):
        df = df.sample(frac=1.0, random_state=int(getattr(cfg, "random_seed", 42))).reset_index(drop=True)

    n_total = int(len(df))
    n_train = int(n_total * float(getattr(cfg, "train_ratio", 0.7)))
    n_val = int(n_total * float(getattr(cfg, "val_ratio", 0.15)))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def _as_2d_embedding(arr: np.ndarray, name: str) -> np.ndarray:
    """把 (D,) / (1, D) / (N, D) 统一成 (N, D) 的 float32 矩阵。"""
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


def _load_text_embedding_stack(project_dir: Path, cfg: TextDLConfig) -> Tuple[Optional[np.ndarray], int, int]:
    """
    读取并堆叠 title_blurb + text 的向量集合：
    - title_blurb 必须存在
    - text 可缺失；缺失时只返回 title_blurb
    返回：
    - seq: (L, D) 或 None
    - missing_text: 该样本是否缺失 text 文件（0/1）
    - skipped: 是否跳过（0/1）
    """
    emb_type = (cfg.embedding_type or "").strip().lower()
    if emb_type not in {"bge", "clip", "siglip"}:
        raise ValueError(f"不支持的 embedding_type={cfg.embedding_type!r}，可选：bge/clip/siglip")

    title_blurb_path = project_dir / f"title_blurb_{emb_type}.npy"
    text_path = project_dir / f"text_{emb_type}.npy"

    missing_strategy = (cfg.missing_strategy or "").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={cfg.missing_strategy!r}，可选：skip/error")

    if not title_blurb_path.exists():
        if missing_strategy == "skip":
            return None, 1, 1
        raise FileNotFoundError(f"缺少 title_blurb 嵌入文件：{title_blurb_path}")

    title_blurb = _as_2d_embedding(np.load(title_blurb_path), name=title_blurb_path.name)
    parts: List[np.ndarray] = [title_blurb]

    missing_text = 0
    if text_path.exists():
        text_arr = _as_2d_embedding(np.load(text_path), name=text_path.name)
        max_text_vectors = int(getattr(cfg, "max_text_vectors", 0))
        if max_text_vectors > 0 and int(text_arr.shape[0]) > max_text_vectors:
            strategy = (getattr(cfg, "text_select_strategy", "first") or "").strip().lower()
            if strategy not in {"first", "random"}:
                raise ValueError("text_select_strategy 仅支持 first/random。")
            if strategy == "first":
                text_arr = text_arr[:max_text_vectors]
            else:
                seed = int(getattr(cfg, "random_seed", 42))
                h = _stable_hash_int(f"{seed}_{project_dir.name}")
                rng = np.random.default_rng(h)
                idx = rng.choice(int(text_arr.shape[0]), size=max_text_vectors, replace=False)
                idx = np.sort(idx)
                text_arr = text_arr[idx]
        parts.append(text_arr)
    else:
        missing_text = 1

    seq = np.concatenate(parts, axis=0)
    return seq.astype(np.float32, copy=False), int(missing_text), 0


def _build_features_for_split(
    df: pd.DataFrame,
    projects_root: Path,
    cfg: TextDLConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, int], int, int]:
    """
    将一个 split 的样本转为 numpy：
    - X: (N, Lmax, D)
    - len: (N,)
    - y: (N,)
    - project_ids: List[str]
    - stats: 统计信息
    - embedding_dim, max_seq_len
    """
    ids = [_normalize_project_id(v) for v in df[cfg.id_col].tolist()]
    y = _encode_binary_target(df[cfg.target_col])

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    kept_ids: List[str] = []
    kept_labels: List[int] = []

    embedding_dim = 0
    missing_text_total = 0
    skipped_total = 0

    for pid, label in zip(ids, y):
        if not pid:
            skipped_total += 1
            continue
        project_dir = projects_root / pid
        if not project_dir.exists():
            if (cfg.missing_strategy or "").strip().lower() == "skip":
                skipped_total += 1
                continue
            raise FileNotFoundError(f"找不到项目目录：{project_dir}")

        seq, missing_text, skipped = _load_text_embedding_stack(project_dir, cfg)
        if skipped:
            skipped_total += skipped
            continue
        if seq is None:
            skipped_total += 1
            continue

        if embedding_dim <= 0:
            embedding_dim = int(seq.shape[1])
        if int(seq.shape[1]) != int(embedding_dim):
            raise ValueError(
                f"embedding_dim 不一致：期望 {embedding_dim}，但 {project_dir.name} 中为 {seq.shape[1]}"
            )

        sequences.append(seq)
        lengths.append(int(seq.shape[0]))
        kept_ids.append(pid)
        kept_labels.append(int(label))
        missing_text_total += int(missing_text)

    if not sequences:
        raise RuntimeError("该数据切分中没有可用样本（可能都被 skip 了或缺少嵌入文件）。")

    max_seq_len = int(max(lengths))
    X = np.zeros((len(sequences), max_seq_len, int(embedding_dim)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        L = int(seq.shape[0])
        X[i, :L, :] = seq

    y_arr = np.asarray(kept_labels, dtype=np.int64)
    len_arr = np.asarray(lengths, dtype=np.int64)
    stats = {"missing_text_files": int(missing_text_total), "skipped_samples": int(skipped_total)}
    return X, len_arr, y_arr, kept_ids, stats, int(embedding_dim), int(max_seq_len)


def _make_cache_key(csv_path: Path, projects_root: Path, cfg: TextDLConfig) -> str:
    payload = {
        "cache_version": _CACHE_VERSION,
        "csv": str(csv_path.as_posix()),
        "projects_root": str(projects_root.as_posix()),
        "embedding_type": str(getattr(cfg, "embedding_type", "")),
        "max_text_vectors": int(getattr(cfg, "max_text_vectors", 0)),
        "text_select_strategy": str(getattr(cfg, "text_select_strategy", "")),
        "missing_strategy": str(getattr(cfg, "missing_strategy", "")),
        "split": {
            "train_ratio": float(getattr(cfg, "train_ratio", 0.7)),
            "val_ratio": float(getattr(cfg, "val_ratio", 0.15)),
            "test_ratio": float(getattr(cfg, "test_ratio", 0.15)),
            "shuffle_before_split": bool(getattr(cfg, "shuffle_before_split", False)),
            "random_seed": int(getattr(cfg, "random_seed", 42)),
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _save_cache(path: Path, prepared: PreparedTextData, meta: Dict[str, Any], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savez = np.savez_compressed if compress else np.savez
    savez(
        path,
        X_train=prepared.X_train,
        len_train=prepared.len_train,
        y_train=prepared.y_train,
        train_project_ids=np.asarray(prepared.train_project_ids, dtype=object),
        X_val=prepared.X_val,
        len_val=prepared.len_val,
        y_val=prepared.y_val,
        val_project_ids=np.asarray(prepared.val_project_ids, dtype=object),
        X_test=prepared.X_test,
        len_test=prepared.len_test,
        y_test=prepared.y_test,
        test_project_ids=np.asarray(prepared.test_project_ids, dtype=object),
        embedding_dim=np.asarray([prepared.embedding_dim], dtype=np.int64),
        max_seq_len=np.asarray([prepared.max_seq_len], dtype=np.int64),
        stats_json=np.asarray([json.dumps(prepared.stats, ensure_ascii=False)], dtype=object),
        meta_json=np.asarray([json.dumps(meta, ensure_ascii=False)], dtype=object),
    )


def _load_cache(path: Path) -> PreparedTextData:
    with np.load(path, allow_pickle=True) as z:
        stats = json.loads(str(z["stats_json"].reshape(-1)[0]))
        return PreparedTextData(
            X_train=np.asarray(z["X_train"], dtype=np.float32),
            len_train=np.asarray(z["len_train"], dtype=np.int64),
            y_train=np.asarray(z["y_train"], dtype=np.int64),
            train_project_ids=[str(x) for x in z["train_project_ids"].tolist()],
            X_val=np.asarray(z["X_val"], dtype=np.float32),
            len_val=np.asarray(z["len_val"], dtype=np.int64),
            y_val=np.asarray(z["y_val"], dtype=np.int64),
            val_project_ids=[str(x) for x in z["val_project_ids"].tolist()],
            X_test=np.asarray(z["X_test"], dtype=np.float32),
            len_test=np.asarray(z["len_test"], dtype=np.int64),
            y_test=np.asarray(z["y_test"], dtype=np.int64),
            test_project_ids=[str(x) for x in z["test_project_ids"].tolist()],
            embedding_dim=int(np.asarray(z["embedding_dim"]).reshape(-1)[0]),
            max_seq_len=int(np.asarray(z["max_seq_len"]).reshape(-1)[0]),
            stats={k: int(v) for k, v in stats.items()},
        )


def prepare_data(
    csv_path: Path,
    projects_root: Path,
    cfg: TextDLConfig,
    cache_dir: Path | None = None,
    logger=None,
) -> PreparedTextData:
    """一站式：读 CSV -> 切分 -> 构建文本嵌入特征 -> 返回 numpy 数组。支持缓存。"""
    use_cache = bool(getattr(cfg, "use_cache", False)) and cache_dir is not None
    refresh_cache = bool(getattr(cfg, "refresh_cache", False))
    compress = bool(getattr(cfg, "cache_compress", False))

    cache_path: Path | None = None
    cache_key: str | None = None
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _make_cache_key(csv_path, projects_root, cfg)
        cache_path = cache_dir / f"{cache_key}.npz"
        if cache_path.exists() and not refresh_cache:
            try:
                prepared = _load_cache(cache_path)
                if logger is not None:
                    logger.info("使用数据缓存：%s", str(cache_path))
                return prepared
            except Exception as e:
                if logger is not None:
                    logger.warning("读取缓存失败，将重新构建：%s", e)

    raw_df = load_labels(csv_path, cfg)
    train_df, val_df, test_df = split_by_ratio(raw_df, cfg)

    X_train, len_train, y_train, train_ids, train_stats, emb_dim, max_len_train = _build_features_for_split(
        train_df, projects_root, cfg
    )
    X_val, len_val, y_val, val_ids, val_stats, emb_dim_val, max_len_val = _build_features_for_split(
        val_df, projects_root, cfg
    )
    X_test, len_test, y_test, test_ids, test_stats, emb_dim_test, max_len_test = _build_features_for_split(
        test_df, projects_root, cfg
    )

    if int(emb_dim_val) != int(emb_dim) or int(emb_dim_test) != int(emb_dim):
        raise ValueError(f"embedding_dim 不一致：train={emb_dim} val={emb_dim_val} test={emb_dim_test}")

    stats: Dict[str, int] = {"missing_text_files": 0, "skipped_samples": 0}
    for s in (train_stats, val_stats, test_stats):
        for k, v in s.items():
            stats[k] = int(stats.get(k, 0) + int(v))

    prepared = PreparedTextData(
        X_train=X_train,
        len_train=len_train,
        y_train=y_train,
        train_project_ids=train_ids,
        X_val=X_val,
        len_val=len_val,
        y_val=y_val,
        val_project_ids=val_ids,
        X_test=X_test,
        len_test=len_test,
        y_test=y_test,
        test_project_ids=test_ids,
        embedding_dim=int(emb_dim),
        max_seq_len=int(max(max_len_train, max_len_val, max_len_test)),
        stats=stats,
    )

    if use_cache and cache_path is not None:
        try:
            meta = {
                "cache_version": _CACHE_VERSION,
                "cache_key": cache_key,
                "csv_path": str(csv_path.as_posix()),
                "projects_root": str(projects_root.as_posix()),
                "config": cfg.to_dict(),
            }
            _save_cache(cache_path, prepared, meta=meta, compress=compress)
            if logger is not None:
                logger.info("已写入数据缓存：%s", str(cache_path))
        except Exception as e:
            if logger is not None:
                logger.warning("写入缓存失败：%s", e)

    return prepared

