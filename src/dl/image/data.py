# -*- coding: utf-8 -*-
"""
数据加载与特征构建：
- 读取 CSV（只用 project_id + state）
- 从 data/projects/.../<project_id>/ 读取图片嵌入 npy
- 将 cover_image_{type}.npy 与 image_{type}.npy 堆叠为一个“向量集合”（形状为 (L, D)）

说明：
- cover_image_*.npy 通常是 (1, D)
- image_*.npy 通常是 (N, D)，允许缺失；若缺失则只使用 cover_image
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ImageDLConfig

_CACHE_VERSION = 1


@dataclass(frozen=True)
class PreparedImageData:
    # X_*: (N, L_max, D)；len_*: (N,)
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


def load_labels(csv_path: Path, cfg: ImageDLConfig) -> pd.DataFrame:
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
    - 若是字符串：仅把 successful 视为 1，其余视为 0
    """
    if pd.api.types.is_numeric_dtype(series):
        s = series.fillna(0)
        # 尽量兼容 0/1 之外的编码：若不是严格的 {0,1}，则把 >0 视为 1
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
    df: pd.DataFrame, cfg: ImageDLConfig
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


def _load_image_embedding_stack(
    project_dir: Path, cfg: ImageDLConfig
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    读取并堆叠 cover + image 的向量集合：
    - cover 必须存在
    - image 可缺失；缺失时只返回 cover
    返回：
    - seq: (L, D) 或 None
    - missing_image: 该样本是否缺失 image 文件（0/1）
    - skipped: 是否跳过（0/1）
    """
    emb_type = (cfg.embedding_type or "").strip().lower()
    if emb_type not in {"clip", "siglip", "resnet"}:
        raise ValueError(f"不支持的 embedding_type={cfg.embedding_type!r}，可选：clip/siglip/resnet")

    cover_path = project_dir / f"cover_image_{emb_type}.npy"
    image_path = project_dir / f"image_{emb_type}.npy"

    missing_strategy = (cfg.missing_strategy or "").strip().lower()
    if missing_strategy not in {"skip", "error"}:
        raise ValueError(f"不支持的 missing_strategy={cfg.missing_strategy!r}，可选：skip/error")

    if not cover_path.exists():
        if missing_strategy == "skip":
            return None, 0, 1
        raise FileNotFoundError(f"项目 {project_dir.name} 缺少文件：{cover_path.name}")

    cover = _as_2d_embedding(np.load(cover_path), cover_path.name)

    missing_image = 0
    if image_path.exists():
        image = _as_2d_embedding(np.load(image_path), image_path.name)
        # 可选：限制每个项目使用的 image 向量数量，降低过拟合并减少 padding
        max_vec = int(getattr(cfg, "max_image_vectors", 0))
        if max_vec > 0 and int(image.shape[0]) > max_vec:
            strategy = str(getattr(cfg, "image_select_strategy", "first")).strip().lower()
            if strategy == "first":
                image = image[:max_vec]
            elif strategy == "random":
                seed = int(getattr(cfg, "random_seed", 42)) + _stable_hash_int(str(project_dir.name))
                rng = np.random.default_rng(seed)
                idx = rng.choice(int(image.shape[0]), size=max_vec, replace=False)
                idx = np.sort(idx)
                image = image[idx]
            else:
                raise ValueError(f"不支持的 image_select_strategy={strategy!r}，可选：first/random")
        if int(image.shape[0]) > 0:
            if int(image.shape[1]) != int(cover.shape[1]):
                raise ValueError(
                    f"项目 {project_dir.name} 的 cover/image 维度不一致：{cover.shape} vs {image.shape}"
                )
            seq = np.concatenate([cover, image], axis=0).astype(np.float32, copy=False)
        else:
            seq = cover
            missing_image = 1
    else:
        seq = cover
        missing_image = 1

    return seq, missing_image, 0


def _make_cache_key(csv_path: Path, projects_root: Path, cfg: ImageDLConfig) -> str:
    """根据数据与关键配置生成缓存 key（用于命名 .npz 文件）。"""
    stat = csv_path.stat()
    payload = {
        "cache_version": _CACHE_VERSION,
        "data_csv": str(csv_path.as_posix()),
        "csv_mtime": float(stat.st_mtime),
        "csv_size": int(stat.st_size),
        "projects_root": str(projects_root.as_posix()),
        "embedding_type": str(getattr(cfg, "embedding_type", "")),
        "missing_strategy": str(getattr(cfg, "missing_strategy", "")),
        "train_ratio": float(getattr(cfg, "train_ratio", 0.7)),
        "val_ratio": float(getattr(cfg, "val_ratio", 0.15)),
        "test_ratio": float(getattr(cfg, "test_ratio", 0.15)),
        "shuffle_before_split": bool(getattr(cfg, "shuffle_before_split", False)),
        "random_seed": int(getattr(cfg, "random_seed", 42)),
        "max_image_vectors": int(getattr(cfg, "max_image_vectors", 0)),
        "image_select_strategy": str(getattr(cfg, "image_select_strategy", "first")),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    return f"image_dl_{h}"


def _load_cache(cache_path: Path) -> PreparedImageData:
    """从 .npz 缓存加载 PreparedImageData。"""
    with np.load(cache_path, allow_pickle=False) as z:
        stats_json = z.get("stats_json", np.array("{}", dtype=str)).item()
        stats = json.loads(stats_json) if isinstance(stats_json, str) else {}

        return PreparedImageData(
            X_train=z["X_train"].astype(np.float32, copy=False),
            len_train=z["len_train"].astype(np.int64, copy=False),
            y_train=z["y_train"].astype(np.int64, copy=False),
            train_project_ids=z["train_project_ids"].astype(str).tolist(),
            X_val=z["X_val"].astype(np.float32, copy=False),
            len_val=z["len_val"].astype(np.int64, copy=False),
            y_val=z["y_val"].astype(np.int64, copy=False),
            val_project_ids=z["val_project_ids"].astype(str).tolist(),
            X_test=z["X_test"].astype(np.float32, copy=False),
            len_test=z["len_test"].astype(np.int64, copy=False),
            y_test=z["y_test"].astype(np.int64, copy=False),
            test_project_ids=z["test_project_ids"].astype(str).tolist(),
            embedding_dim=int(z["embedding_dim"].item()),
            max_seq_len=int(z["max_seq_len"].item()),
            stats={k: int(v) for k, v in dict(stats).items()},
        )


def _save_cache(cache_path: Path, prepared: PreparedImageData, meta: Dict[str, Any], compress: bool) -> None:
    """保存 PreparedImageData 到 .npz 缓存。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats_json = json.dumps(prepared.stats, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)

    arrays = {
        "X_train": prepared.X_train.astype(np.float32, copy=False),
        "len_train": prepared.len_train.astype(np.int64, copy=False),
        "y_train": prepared.y_train.astype(np.int64, copy=False),
        "train_project_ids": np.asarray(prepared.train_project_ids, dtype=str),
        "X_val": prepared.X_val.astype(np.float32, copy=False),
        "len_val": prepared.len_val.astype(np.int64, copy=False),
        "y_val": prepared.y_val.astype(np.int64, copy=False),
        "val_project_ids": np.asarray(prepared.val_project_ids, dtype=str),
        "X_test": prepared.X_test.astype(np.float32, copy=False),
        "len_test": prepared.len_test.astype(np.int64, copy=False),
        "y_test": prepared.y_test.astype(np.int64, copy=False),
        "test_project_ids": np.asarray(prepared.test_project_ids, dtype=str),
        "embedding_dim": np.asarray(int(prepared.embedding_dim), dtype=np.int64),
        "max_seq_len": np.asarray(int(prepared.max_seq_len), dtype=np.int64),
        "stats_json": np.asarray(stats_json, dtype=str),
        "meta_json": np.asarray(meta_json, dtype=str),
    }

    # numpy.savez 会在路径不以 ".npz" 结尾时自动追加后缀，导致临时文件名错位；
    # 这里用文件句柄写入，保证临时文件路径可控。
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with tmp_path.open("wb") as f:
        if compress:
            np.savez_compressed(f, **arrays)
        else:
            np.savez(f, **arrays)
    tmp_path.replace(cache_path)


def _build_features_for_split(
    df_split: pd.DataFrame,
    projects_root: Path,
    cfg: ImageDLConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, int], int, int]:
    """为一个 split 构建 X/len/y，并返回缺失统计、embedding_dim 与 max_seq_len。"""
    project_ids: List[str] = [_normalize_project_id(v) for v in df_split[cfg.id_col].tolist()]
    y_all = _encode_binary_target(df_split[cfg.target_col])

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    kept_ids: List[str] = []
    kept_labels: List[int] = []

    missing_image_total = 0
    skipped_total = 0
    embedding_dim: int = 0

    for pid, label in zip(project_ids, y_all):
        if not pid:
            skipped_total += 1
            continue

        project_dir = projects_root / pid
        if not project_dir.exists():
            if (cfg.missing_strategy or "").strip().lower() == "skip":
                skipped_total += 1
                continue
            raise FileNotFoundError(f"找不到项目目录：{project_dir}")

        seq, missing_image, skipped = _load_image_embedding_stack(project_dir, cfg)
        if skipped:
            skipped_total += skipped
            continue
        if seq is None:
            skipped_total += 1
            continue

        if embedding_dim <= 0:
            embedding_dim = int(seq.shape[1])
        if int(seq.shape[1]) != int(embedding_dim):
            raise ValueError(f"embedding_dim 不一致：期望 {embedding_dim}，但 {project_dir.name} 为 {seq.shape[1]}")

        sequences.append(seq)
        lengths.append(int(seq.shape[0]))
        kept_ids.append(pid)
        kept_labels.append(int(label))
        missing_image_total += int(missing_image)

    if not sequences:
        raise RuntimeError("该数据切分中没有可用样本（可能都被 skip 了或缺少嵌入文件）")

    max_seq_len = int(max(lengths))
    X = np.zeros((len(sequences), max_seq_len, int(embedding_dim)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        L = int(seq.shape[0])
        X[i, :L, :] = seq
    y = np.asarray(kept_labels, dtype=np.int64)
    len_arr = np.asarray(lengths, dtype=np.int64)

    stats = {"missing_image_files": int(missing_image_total), "skipped_samples": int(skipped_total)}
    return X, len_arr, y, kept_ids, stats, int(embedding_dim), int(max_seq_len)


def prepare_data(
    csv_path: Path,
    projects_root: Path,
    cfg: ImageDLConfig,
    cache_dir: Path | None = None,
    logger=None,
) -> PreparedImageData:
    """一站式：读 CSV -> 切分 -> 构建图片嵌入特征 -> 返回 numpy 数组。支持缓存。"""
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

    stats: Dict[str, int] = {"missing_image_files": 0, "skipped_samples": 0}
    for s in (train_stats, val_stats, test_stats):
        for k, v in s.items():
            stats[k] = int(stats.get(k, 0) + int(v))

    prepared = PreparedImageData(
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
