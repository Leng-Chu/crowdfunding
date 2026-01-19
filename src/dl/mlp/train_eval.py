# -*- coding: utf-8 -*-
"""
训练与评估模块：
- 按 epoch 训练（多模态三路融合网络）
- 在验证集上早停
- 输出训练曲线与测试指标
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MlpDLConfig
from utils import compute_binary_metrics


def _get_device() -> torch.device:
    """优先使用 GPU（若可用），否则使用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _positive_proba(
    model: nn.Module,
    X_meta: np.ndarray,
    X_image: np.ndarray,
    len_image: np.ndarray,
    X_text: np.ndarray,
    len_text: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """返回正类概率（shape: [N]）。"""
    model.eval()

    meta_t = torch.from_numpy(np.asarray(X_meta, dtype=np.float32))
    img_t = torch.from_numpy(np.asarray(X_image, dtype=np.float32))
    len_img_t = torch.from_numpy(np.asarray(len_image, dtype=np.int64))
    txt_t = torch.from_numpy(np.asarray(X_text, dtype=np.float32))
    len_txt_t = torch.from_numpy(np.asarray(len_text, dtype=np.int64))

    loader = DataLoader(
        TensorDataset(meta_t, img_t, len_img_t, txt_t, len_txt_t),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )

    probs: List[np.ndarray] = []
    for mb, ib, lib, tb, ltb in loader:
        mb = mb.to(device)
        ib = ib.to(device)
        lib = lib.to(device)
        tb = tb.to(device)
        ltb = ltb.to(device)
        logits = model(mb, ib, lib, tb, ltb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob.astype(np.float64, copy=False))

    if not probs:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(probs, axis=0).reshape(-1)


def train_with_early_stopping(
    model: nn.Module,
    X_meta_train: np.ndarray,
    X_image_train: np.ndarray,
    len_image_train: np.ndarray,
    X_text_train: np.ndarray,
    len_text_train: np.ndarray,
    y_train: np.ndarray,
    X_meta_val: np.ndarray,
    X_image_val: np.ndarray,
    len_image_val: np.ndarray,
    X_text_val: np.ndarray,
    len_text_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MlpDLConfig,
    logger,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    """
    返回：
    - best_model：验证集上最优的模型（已加载 best 权重）
    - history：每个 epoch 的指标记录
    - best_info：最优 epoch 与其验证指标
    """
    device = _get_device()
    model = model.to(device)

    meta_t = torch.from_numpy(np.asarray(X_meta_train, dtype=np.float32))
    img_t = torch.from_numpy(np.asarray(X_image_train, dtype=np.float32))
    len_img_t = torch.from_numpy(np.asarray(len_image_train, dtype=np.int64))
    txt_t = torch.from_numpy(np.asarray(X_text_train, dtype=np.float32))
    len_txt_t = torch.from_numpy(np.asarray(len_text_train, dtype=np.int64))
    y_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))

    train_loader = DataLoader(
        TensorDataset(meta_t, img_t, len_img_t, txt_t, len_txt_t, y_t),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate_init),
        weight_decay=float(cfg.alpha),
    )

    scheduler = None
    if getattr(cfg, "use_lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(cfg, "lr_scheduler_factor", 0.5)),
            patience=int(getattr(cfg, "lr_scheduler_patience", 3)),
            min_lr=float(getattr(cfg, "lr_scheduler_min_lr", 1e-6)),
        )

    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0

    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(cfg.max_epochs) + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for mb, ib, lib, tb, ltb, yb in train_loader:
            mb = mb.to(device)
            ib = ib.to(device)
            lib = lib.to(device)
            tb = tb.to(device)
            ltb = ltb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(mb, ib, lib, tb, ltb)
            loss = criterion(logits, yb)
            loss.backward()

            max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            bs = int(mb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        train_prob = _positive_proba(
            model,
            X_meta_train,
            X_image_train,
            len_image_train,
            X_text_train,
            len_text_train,
            device=device,
            batch_size=cfg.batch_size,
        )
        val_prob = _positive_proba(
            model,
            X_meta_val,
            X_image_val,
            len_image_val,
            X_text_val,
            len_text_val,
            device=device,
            batch_size=cfg.batch_size,
        )

        train_metrics = compute_binary_metrics(y_train, train_prob, threshold=cfg.threshold)
        val_metrics = compute_binary_metrics(y_val, val_prob, threshold=cfg.threshold)

        row: Dict[str, Any] = {"epoch": epoch}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        row["train_epoch_loss"] = float(train_loss)
        history.append(row)

        metric_for_best = str(getattr(cfg, "metric_for_best", "val_accuracy")).strip().lower()
        if metric_for_best in {"val_loss", "loss"}:
            score = -float(val_metrics["log_loss"])
        elif metric_for_best in {"val_accuracy", "val_acc", "accuracy"}:
            score = float(val_metrics["accuracy"])
        else:
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

        lr_now = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %d/%d | lr=%.6g | train_loss=%.4f val_loss=%.4f | train_acc=%.4f val_acc=%.4f | best_epoch=%d bad=%d",
            epoch,
            cfg.max_epochs,
            lr_now,
            train_metrics["log_loss"],
            val_metrics["log_loss"],
            float(train_metrics["accuracy"]),
            float(val_metrics["accuracy"]),
            best_epoch,
            bad_epochs,
        )

        if scheduler is not None:
            old_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step(float(val_metrics["log_loss"]))
            new_lr = float(optimizer.param_groups[0]["lr"])
            if new_lr < old_lr:
                logger.info("学习率调整：%.6g -> %.6g", old_lr, new_lr)
                if bool(getattr(cfg, "reset_early_stop_on_lr_change", True)):
                    bad_epochs = 0

        if bad_epochs >= int(cfg.early_stop_patience):
            min_epochs = int(getattr(cfg, "early_stop_min_epochs", 0))
            if epoch < min_epochs:
                continue
            logger.info("触发早停：连续 %d 个 epoch 无提升。", bad_epochs)
            break

    if best_state is None:
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
    model: nn.Module,
    X_meta: np.ndarray,
    X_image: np.ndarray,
    len_image: np.ndarray,
    X_text: np.ndarray,
    len_text: np.ndarray,
    y: np.ndarray,
    cfg: MlpDLConfig,
) -> Dict[str, Any]:
    """评估一个数据切分，并返回指标与预测概率。"""
    device = _get_device()
    model = model.to(device)
    prob = _positive_proba(
        model,
        X_meta,
        X_image,
        len_image,
        X_text,
        len_text,
        device=device,
        batch_size=cfg.batch_size,
    )
    metrics = compute_binary_metrics(y, prob, threshold=cfg.threshold)
    return {"metrics": metrics, "prob": prob}

