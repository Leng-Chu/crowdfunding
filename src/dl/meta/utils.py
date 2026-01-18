# -*- coding: utf-8 -*-
"""
工具与实用函数模块：
- 随机种子
- 日志
- 指标计算与绘图
- JSON/文本保存
"""

from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def set_global_seed(seed: int) -> None:
    """设置全局随机种子（numpy/random）。"""
    random.seed(seed)
    np.random.seed(seed)
    # torch 不是强依赖，但在本项目的 meta_dl 里会用到；这里做个兼容设置
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _sanitize_name(name: str) -> str:
    """把 run_name 处理成适合当文件夹名的形式。"""
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name, flags=re.UNICODE)
    return name[:80] if name else "run"


def make_run_dirs(experiment_root: Path, run_name: Optional[str] = None) -> Tuple[str, Path, Path]:
    """
    在 experiments/meta_dl 下创建本次实验的：
    - <run_id>/model：保存模型、预处理器、指标、图等产物
    - <run_id>/log：保存日志文件
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_name(run_name) if run_name else ""
    base_run_id = f"{ts}_{suffix}" if suffix else ts

    for i in range(1000):
        run_id = base_run_id if i == 0 else f"{base_run_id}_{i}"
        run_dir = experiment_root / run_id
        model_dir = run_dir / "model"
        log_dir = run_dir / "log"

        # 用 run_dir 是否存在来判重；避免出现只创建了半边目录的边界情况
        if not run_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=False)
            log_dir.mkdir(parents=True, exist_ok=False)
            return run_id, model_dir, log_dir

    raise RuntimeError("创建 run 目录失败：可能存在大量同名实验目录")


def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """同时输出到控制台与文件的 logger（utf-8）。"""
    logger = logging.getLogger("meta_dl")
    logger.setLevel(level)
    logger.propagate = False

    # 避免重复添加 handler（例如在 notebook/多次运行时）
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(data: Dict[str, Any], path: Path) -> None:
    """保存 JSON（utf-8，中文不转义）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(lines: List[str], path: Path) -> None:
    """逐行保存文本（utf-8）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """计算二分类混淆矩阵：tn, fp, fn, tp。"""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred 形状不一致：{y_true.shape} vs {y_pred.shape}")

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 ROC 曲线（不依赖 sklearn）。

    返回：
    - fpr：假阳率
    - tpr：真阳率
    - thresholds：对应阈值（从高到低）
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true/y_score 形状不一致：{y_true.shape} vs {y_score.shape}")

    # ROC/AUC 需要同时存在正负类
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC 曲线需要同时包含正类与负类样本")

    # 按 score 降序排列；mergesort 稳定，便于处理同分数的边界
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # 找到 score 变化的位置（不同阈值点）
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # 加上起点 (0,0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score_sorted[threshold_idxs]]

    tpr = tps / float(n_pos)
    fpr = fps / float(n_neg)
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """用梯形法则计算曲线下面积。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.trapz(y, x))


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """二分类指标：acc/precision/recall/f1/auc/logloss/confusion。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    y_true_1d = y_true.reshape(-1)
    y_pred_1d = y_pred.reshape(-1)
    y_prob_1d = y_prob.reshape(-1)
    if y_true_1d.shape != y_prob_1d.shape:
        raise ValueError(f"y_true/y_prob 形状不一致：{y_true_1d.shape} vs {y_prob_1d.shape}")

    tn, fp, fn, tp = _binary_confusion(y_true_1d, y_pred_1d)
    n = int(y_true_1d.size)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 对数损失（交叉熵）
    logloss = float(
        -np.mean(y_true_1d * np.log(y_prob_1d) + (1 - y_true_1d) * np.log(1 - y_prob_1d))
    ) if n > 0 else 0.0

    # AUC：当只有一个类别时返回 None
    try:
        fpr, tpr, _ = _roc_curve(y_true_1d, y_prob_1d)
        roc_auc = _auc_trapz(fpr, tpr)
    except Exception:
        roc_auc = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "log_loss": float(logloss),
        "threshold": float(threshold),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_history(history: List[Dict[str, Any]], save_path: Path) -> None:
    """绘制训练曲线（loss/auc）。"""
    if not history:
        return

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    if "epoch" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # loss
    if "train_log_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train_log_loss"], label="train")
    if "val_log_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_log_loss"], label="val")
    axes[0].set_title("Log Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # auc
    if "train_roc_auc" in df.columns:
        axes[1].plot(df["epoch"], df["train_roc_auc"], label="train")
    if "val_roc_auc" in df.columns:
        axes[1].plot(df["epoch"], df["val_roc_auc"], label="val")
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """绘制 ROC 曲线。"""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    try:
        fpr, tpr, _ = _roc_curve(y_true, y_prob)
        roc_auc = _auc_trapz(fpr, tpr)
    except Exception:
        # 只有一个类别时无法画 ROC，这里直接返回
        return

    fig = plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
