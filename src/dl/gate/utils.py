# -*- coding: utf-8 -*-
"""
工具与实用函数模块（gate）：
- 随机种子
- 日志
- 指标计算与绘图
- JSON/文本保存

注意：作图时图里不要有中文，因此图标题/坐标轴保持英文。
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def baseline_uses_meta(baseline_mode: str) -> bool:
    """判断某个 baseline_mode 是否需要 meta 分支。"""
    m = str(baseline_mode or "").strip().lower()
    return m not in {"seq_only", "key_only"}


def set_global_seed(seed: int) -> None:
    """设置全局随机种子（numpy/random/torch）。"""
    random.seed(seed)
    np.random.seed(seed)
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


def make_run_dirs(experiment_root: Path, run_name: Optional[str] = None) -> Tuple[str, Path, Path, Path]:
    """
    在 experiment_root 下创建本次实验的：
    - <run_id>/artifacts：模型权重、预处理器等可复现产物
    - <run_id>/reports：日志/指标/配置等报告产物
    - <run_id>/plots：训练曲线、ROC 等图片
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_name(run_name) if run_name else ""
    base_run_id = f"{ts}_{suffix}" if suffix else ts

    for i in range(1000):
        run_id = base_run_id if i == 0 else f"{base_run_id}_{i}"
        run_dir = experiment_root / run_id
        artifacts_dir = run_dir / "artifacts"
        reports_dir = run_dir / "reports"
        plots_dir = run_dir / "plots"

        if not run_dir.exists():
            artifacts_dir.mkdir(parents=True, exist_ok=False)
            reports_dir.mkdir(parents=True, exist_ok=False)
            plots_dir.mkdir(parents=True, exist_ok=False)
            return run_id, artifacts_dir, reports_dir, plots_dir

    raise RuntimeError("创建 run 目录失败：可能存在大量同名实验目录。")


def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """同时输出到控制台与文件的 logger（utf-8）。"""
    logger = logging.getLogger("gate")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
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
    """计算 ROC 曲线（不依赖 sklearn）。"""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true/y_score 形状不一致：{y_true.shape} vs {y_score.shape}")

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC 需要同时包含正负样本。")

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    distinct = np.where(np.diff(y_score_sorted))[0]
    thr_idx = np.r_[distinct, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[thr_idx]
    fps = 1 + thr_idx - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score_sorted[thr_idx]]

    fpr = fps / max(1, n_neg)
    tpr = tps / max(1, n_pos)
    return fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)


def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """梯形法则计算 AUC。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return 0.0
    return float(np.trapz(y, x))


def _roc_auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """秩统计方式计算 AUC（需要同时包含正负样本）。"""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC 需要同时包含正负样本。")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """计算 accuracy/precision/recall/f1/auc/logloss/confusion。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= float(threshold)).astype(int)

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

    logloss = (
        float(-np.mean(y_true_1d * np.log(y_prob_1d) + (1 - y_true_1d) * np.log(1 - y_prob_1d))) if n > 0 else 0.0
    )

    try:
        fpr, tpr, _ = _roc_curve(y_true_1d, y_prob_1d)
        roc_auc = _auc_trapz(fpr, tpr)
        roc_auc_error: Optional[str] = None
    except Exception as e:
        roc_auc = None
        roc_auc_error = f"{type(e).__name__}: {e}"
        try:
            mask = np.isfinite(y_prob_1d)
            if int(np.sum(mask)) < int(y_prob_1d.size):
                roc_auc = _roc_auc_rank(y_true_1d[mask], y_prob_1d[mask])
            else:
                roc_auc = _roc_auc_rank(y_true_1d, y_prob_1d)
            roc_auc_error = None
        except Exception as e2:
            roc_auc = None
            if roc_auc_error is None:
                roc_auc_error = f"{type(e2).__name__}: {e2}"

    out: Dict[str, Any] = {
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
    if roc_auc is None and roc_auc_error:
        out["roc_auc_error"] = str(roc_auc_error)
    return out


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    在验证集上选择阈值：最大化 F1（预测规则：y_pred = y_prob >= threshold）。

    说明：
    - 为保证可复现与效率，使用排序 + 累积统计的方式遍历所有“分数变化点”的阈值候选。
    - 若出现多个阈值 F1 相同，选择“阈值更大”的那个（更保守，减少误报），保证确定性。
    - 若验证集中没有正样本或没有负样本，返回 0.5（避免异常）。
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true/y_prob 形状不一致：{y_true.shape} vs {y_prob.shape}")

    n = int(y_true.size)
    if n == 0:
        return 0.5

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    mask = np.isfinite(y_prob)
    if int(np.sum(mask)) != n:
        y_true = y_true[mask]
        y_prob = y_prob[mask]
        n = int(y_true.size)
        if n == 0:
            return 0.5
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        if n_pos == 0 or n_neg == 0:
            return 0.5

    order = np.argsort(-y_prob, kind="mergesort")
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    distinct = np.where(np.diff(y_prob_sorted))[0]
    thr_idx = np.r_[distinct, y_true_sorted.size - 1]

    cum_tp = np.cumsum(y_true_sorted == 1)[thr_idx]
    cum_fp = (1 + thr_idx) - cum_tp

    best_f1 = -1.0
    best_thr = 0.5

    for tp, fp, idx in zip(cum_tp, cum_fp, thr_idx):
        tp = int(tp)
        fp = int(fp)
        fn = int(n_pos - tp)
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        thr = float(y_prob_sorted[int(idx)])

        if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-15 and thr > best_thr):
            best_f1 = float(f1)
            best_thr = float(thr)

    if not np.isfinite(best_thr):
        return 0.5
    return float(np.clip(best_thr, 0.0, 1.0))


def plot_history(history: List[Dict[str, Any]], save_path: Path) -> None:
    """绘制训练曲线（logloss/auc）。"""
    if not history:
        return

    import matplotlib.pyplot as plt

    def _extract_xy(metric_key: str) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for row in history:
            if "epoch" not in row:
                continue
            y = row.get(metric_key)
            if y is None:
                continue
            xs.append(float(row["epoch"]))
            ys.append(float(y))
        return xs, ys

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_x, train_y = _extract_xy("train_log_loss")
    val_x, val_y = _extract_xy("val_log_loss")
    if train_y:
        axes[0].plot(train_x, train_y, label="train")
    if val_y:
        axes[0].plot(val_x, val_y, label="val")
    axes[0].set_title("Log Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    train_x, train_y = _extract_xy("train_roc_auc")
    val_x, val_y = _extract_xy("val_roc_auc")
    if train_y:
        axes[1].plot(train_x, train_y, label="train")
    if val_y:
        axes[1].plot(val_x, val_y, label="val")
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    if axes[1].lines:
        axes[1].legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """绘制 ROC 曲线。"""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    if y_true.size == 0:
        return

    try:
        fpr, tpr, _ = _roc_curve(y_true, y_prob)
        auc_val = _auc_trapz(fpr, tpr)
    except Exception:
        return

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
