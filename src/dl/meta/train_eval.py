# -*- coding: utf-8 -*-
"""
训练与评估模块：
- 按 epoch 训练（PyTorch 手写 MLP）
- 在验证集上做 early stopping
- 输出训练曲线与最终测试指标
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MetaDLConfig
from utils import compute_binary_metrics


def _get_device() -> torch.device:
    """优先使用 GPU（若可用），否则使用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _positive_proba(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    """返回正类概率（shape: [N]）。"""
    model.eval()
    x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )

    probs: List[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob.astype(np.float64, copy=False))
    if not probs:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(probs, axis=0).reshape(-1)


def train_with_early_stopping(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MetaDLConfig,
    logger,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    """
    返回：
    - best_model：验证集上最优的模型（会把 best 权重 load 回 model）
    - history：每个 epoch 的指标记录
    - best_info：最佳 epoch 与其验证指标
    """
    device = _get_device()
    model = model.to(device)

    # 数据
    x_train = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    train_loader = DataLoader(
        TensorDataset(x_train, y_train_t),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
    )

    # 损失与优化器：二分类用 logits + BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.learning_rate_init),
        weight_decay=float(cfg.alpha),
    )

    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0

    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        # 1) 训练一个 epoch
        model.train()
        running_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(xb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        # 2) 评估（全量算指标，便于对齐旧日志/曲线）
        train_prob = _positive_proba(model, X_train, device=device, batch_size=cfg.batch_size)
        val_prob = _positive_proba(model, X_val, device=device, batch_size=cfg.batch_size)

        train_metrics = compute_binary_metrics(y_train, train_prob, threshold=cfg.threshold)
        val_metrics = compute_binary_metrics(y_val, val_prob, threshold=cfg.threshold)

        row: Dict[str, Any] = {"epoch": epoch}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        history.append(row)

        # 选择用于 early-stopping 的指标
        if cfg.metric_for_best == "val_loss":
            score = -float(val_metrics["log_loss"])
        else:
            # 默认用 AUC，越大越好；如果算不出来则回退到 loss
            score = val_metrics.get("roc_auc")
            if score is None:
                score = -float(val_metrics["log_loss"])
            else:
                score = float(score)

        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | train_auc=%s val_auc=%s | best_epoch=%d bad=%d",
            epoch,
            cfg.max_epochs,
            train_metrics["log_loss"],
            val_metrics["log_loss"],
            f"{train_metrics.get('roc_auc'):.4f}" if train_metrics.get("roc_auc") is not None else "NA",
            f"{val_metrics.get('roc_auc'):.4f}" if val_metrics.get("roc_auc") is not None else "NA",
            best_epoch,
            bad_epochs,
        )

        # 可选：把当前 epoch 的平均训练损失也写进 history，便于排查
        history[-1]["train_epoch_loss"] = float(train_loss)

        if bad_epochs >= cfg.early_stop_patience:
            logger.info("触发 early stopping：连续 %d 个 epoch 无提升。", bad_epochs)
            break

    if best_state is None:
        # 理论上不应发生，但兜底一下
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history) if history else 0
    model.load_state_dict(best_state)

    best_val = None
    if history:
        best_val = next((h for h in history if h["epoch"] == best_epoch), None)

    best_info = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric_for_best": cfg.metric_for_best,
        "best_val_row": best_val,
    }
    return model, history, best_info


def evaluate_split(
    model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: MetaDLConfig
) -> Dict[str, Any]:
    """评估一个数据集切分，并返回指标与预测概率。"""
    device = _get_device()
    model = model.to(device)
    prob = _positive_proba(model, X, device=device, batch_size=cfg.batch_size)
    metrics = compute_binary_metrics(y, prob, threshold=cfg.threshold)
    return {"metrics": metrics, "prob": prob}
