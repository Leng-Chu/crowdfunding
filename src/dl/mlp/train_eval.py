# -*- coding: utf-8 -*-
"""
训练与评估（mlp）：
- meta：MLP（无序列长度）
- image/text：1D CNN（带 lengths 做 mask）
- multimodal：可选分支融合（meta/image/text 任意组合）
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MlpConfig
from utils import compute_binary_metrics


def _get_device(cfg: MlpConfig) -> torch.device:
    """根据配置选择 device：auto/cpu/cuda/cuda:N。"""
    device_str = str(getattr(cfg, "device", "auto") or "auto").strip().lower()

    if device_str in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cpu":
        return torch.device("cpu")

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境不可用 CUDA，但配置了 device=cuda。")
        return torch.device("cuda")

    if device_str.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"当前环境不可用 CUDA，但配置了 device={device_str}。")

        try:
            gpu_index = int(device_str.split(":", 1)[1])
        except Exception as e:
            raise ValueError(f"不合法的 device={device_str!r}，期望形如 cuda:0 / cuda:1。") from e

        n = int(torch.cuda.device_count())
        if gpu_index < 0 or gpu_index >= n:
            raise ValueError(f"device={device_str!r} 越界：当前可见 GPU 数量为 {n}。")
        return torch.device(f"cuda:{gpu_index}")

    raise ValueError(f"不支持的 device={device_str!r}，可选：auto/cpu/cuda/cuda:N。")


# -----------------------------
# meta（单分支）
# -----------------------------


@torch.no_grad()
def _positive_proba_tabular(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
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


def train_meta_with_early_stopping(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MlpConfig,
    logger,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    """对齐 meta：训练 + 早停。"""
    device = _get_device(cfg)
    model = model.to(device)

    x_train = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    train_loader = DataLoader(
        TensorDataset(x_train, y_train_t),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
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

    for epoch in range(1, cfg.max_epochs + 1):
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

            max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            bs = int(xb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        train_prob = _positive_proba_tabular(model, X_train, device=device, batch_size=cfg.batch_size)
        val_prob = _positive_proba_tabular(model, X_val, device=device, batch_size=cfg.batch_size)

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
            score = -float(val_metrics["log_loss"]) if score is None else float(score)

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

        if bad_epochs >= cfg.early_stop_patience:
            min_epochs = int(getattr(cfg, "early_stop_min_epochs", 0))
            if epoch < min_epochs:
                continue
            logger.info("触发早停：连续 %d 个 epoch 无提升。", bad_epochs)
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history) if history else 0
    model.load_state_dict(best_state)

    best_val = next((h for h in history if h.get("epoch") == best_epoch), None) if history else None
    best_info = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric_for_best": cfg.metric_for_best,
        "best_val_row": best_val,
    }
    return model, history, best_info


def evaluate_meta_split(model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: MlpConfig) -> Dict[str, Any]:
    device = _get_device(cfg)
    model = model.to(device)
    prob = _positive_proba_tabular(model, X, device=device, batch_size=cfg.batch_size)
    metrics = compute_binary_metrics(y, prob, threshold=cfg.threshold)
    return {"metrics": metrics, "prob": prob}


# -----------------------------
# image/text（单分支：序列）
# -----------------------------


@torch.no_grad()
def _positive_proba_sequence(
    model: nn.Module,
    X: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
    len_tensor = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
    loader = DataLoader(
        TensorDataset(x_tensor, len_tensor),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )

    probs: List[np.ndarray] = []
    for xb, lb in loader:
        xb = xb.to(device)
        lb = lb.to(device)
        logits = model(xb, lengths=lb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob.astype(np.float64, copy=False))
    if not probs:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(probs, axis=0).reshape(-1)


def train_sequence_with_early_stopping(
    model: nn.Module,
    X_train: np.ndarray,
    len_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    len_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MlpConfig,
    logger,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    device = _get_device(cfg)
    model = model.to(device)

    x_train = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    len_train_t = torch.from_numpy(np.asarray(len_train, dtype=np.int64))
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    train_loader = DataLoader(
        TensorDataset(x_train, len_train_t, y_train_t),
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

        for xb, lb, yb in train_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lengths=lb)
            loss = criterion(logits, yb)
            loss.backward()

            max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            bs = int(xb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        train_prob = _positive_proba_sequence(model, X_train, len_train, device=device, batch_size=cfg.batch_size)
        val_prob = _positive_proba_sequence(model, X_val, len_val, device=device, batch_size=cfg.batch_size)

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
            score = -float(val_metrics["log_loss"]) if score is None else float(score)

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

    best_val = next((h for h in history if h.get("epoch") == best_epoch), None) if history else None
    best_info = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric_for_best": cfg.metric_for_best,
        "best_val_row": best_val,
    }
    return model, history, best_info


def evaluate_sequence_split(
    model: nn.Module,
    X: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    cfg: MlpConfig,
) -> Dict[str, Any]:
    device = _get_device(cfg)
    model = model.to(device)
    prob = _positive_proba_sequence(model, X, lengths, device=device, batch_size=cfg.batch_size)
    metrics = compute_binary_metrics(y, prob, threshold=cfg.threshold)
    return {"metrics": metrics, "prob": prob}


# -----------------------------
# multimodal（两路/三路）
# -----------------------------


@torch.no_grad()
def _positive_proba_multimodal(
    model: nn.Module,
    use_meta: bool,
    use_image: bool,
    use_text: bool,
    X_meta: np.ndarray | None,
    X_image: np.ndarray | None,
    len_image: np.ndarray | None,
    X_text: np.ndarray | None,
    len_text: np.ndarray | None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    tensors: List[torch.Tensor] = []
    if use_meta:
        tensors.append(torch.from_numpy(np.asarray(X_meta, dtype=np.float32)))
    if use_image:
        tensors.append(torch.from_numpy(np.asarray(X_image, dtype=np.float32)))
        tensors.append(torch.from_numpy(np.asarray(len_image, dtype=np.int64)))
    if use_text:
        tensors.append(torch.from_numpy(np.asarray(X_text, dtype=np.float32)))
        tensors.append(torch.from_numpy(np.asarray(len_text, dtype=np.int64)))

    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )

    probs: List[np.ndarray] = []
    for batch in loader:
        idx = 0
        mb = None
        ib = None
        lib = None
        tb = None
        ltb = None

        if use_meta:
            mb = batch[idx].to(device)
            idx += 1
        if use_image:
            ib = batch[idx].to(device)
            lib = batch[idx + 1].to(device)
            idx += 2
        if use_text:
            tb = batch[idx].to(device)
            ltb = batch[idx + 1].to(device)
            idx += 2

        logits = model(x_meta=mb, x_image=ib, len_image=lib, x_text=tb, len_text=ltb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob.astype(np.float64, copy=False))

    if not probs:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(probs, axis=0).reshape(-1)


def train_multimodal_with_early_stopping(
    model: nn.Module,
    use_meta: bool,
    use_image: bool,
    use_text: bool,
    X_meta_train: np.ndarray | None,
    X_image_train: np.ndarray | None,
    len_image_train: np.ndarray | None,
    X_text_train: np.ndarray | None,
    len_text_train: np.ndarray | None,
    y_train: np.ndarray,
    X_meta_val: np.ndarray | None,
    X_image_val: np.ndarray | None,
    len_image_val: np.ndarray | None,
    X_text_val: np.ndarray | None,
    len_text_val: np.ndarray | None,
    y_val: np.ndarray,
    cfg: MlpConfig,
    logger,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    device = _get_device(cfg)
    model = model.to(device)

    tensors: List[torch.Tensor] = []
    if use_meta:
        tensors.append(torch.from_numpy(np.asarray(X_meta_train, dtype=np.float32)))
    if use_image:
        tensors.append(torch.from_numpy(np.asarray(X_image_train, dtype=np.float32)))
        tensors.append(torch.from_numpy(np.asarray(len_image_train, dtype=np.int64)))
    if use_text:
        tensors.append(torch.from_numpy(np.asarray(X_text_train, dtype=np.float32)))
        tensors.append(torch.from_numpy(np.asarray(len_text_train, dtype=np.int64)))
    tensors.append(torch.from_numpy(np.asarray(y_train, dtype=np.float32)))

    train_loader = DataLoader(
        TensorDataset(*tensors),
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

        for batch in train_loader:
            idx = 0
            mb = None
            ib = None
            lib = None
            tb = None
            ltb = None

            if use_meta:
                mb = batch[idx].to(device)
                idx += 1
            if use_image:
                ib = batch[idx].to(device)
                lib = batch[idx + 1].to(device)
                idx += 2
            if use_text:
                tb = batch[idx].to(device)
                ltb = batch[idx + 1].to(device)
                idx += 2

            yb = batch[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_meta=mb, x_image=ib, len_image=lib, x_text=tb, len_text=ltb)
            loss = criterion(logits, yb)
            loss.backward()

            max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            bs = int(yb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        train_prob = _positive_proba_multimodal(
            model,
            use_meta=use_meta,
            use_image=use_image,
            use_text=use_text,
            X_meta=X_meta_train,
            X_image=X_image_train,
            len_image=len_image_train,
            X_text=X_text_train,
            len_text=len_text_train,
            device=device,
            batch_size=cfg.batch_size,
        )
        val_prob = _positive_proba_multimodal(
            model,
            use_meta=use_meta,
            use_image=use_image,
            use_text=use_text,
            X_meta=X_meta_val,
            X_image=X_image_val,
            len_image=len_image_val,
            X_text=X_text_val,
            len_text=len_text_val,
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
            score = -float(val_metrics["log_loss"]) if score is None else float(score)

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

    best_val = next((h for h in history if h.get("epoch") == best_epoch), None) if history else None
    best_info = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric_for_best": cfg.metric_for_best,
        "best_val_row": best_val,
    }
    return model, history, best_info


def evaluate_multimodal_split(
    model: nn.Module,
    use_meta: bool,
    use_image: bool,
    use_text: bool,
    X_meta: np.ndarray | None,
    X_image: np.ndarray | None,
    len_image: np.ndarray | None,
    X_text: np.ndarray | None,
    len_text: np.ndarray | None,
    y: np.ndarray,
    cfg: MlpConfig,
) -> Dict[str, Any]:
    device = _get_device(cfg)
    model = model.to(device)
    prob = _positive_proba_multimodal(
        model,
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        X_meta=X_meta,
        X_image=X_image,
        len_image=len_image,
        X_text=X_text,
        len_text=len_text,
        device=device,
        batch_size=cfg.batch_size,
    )
    metrics = compute_binary_metrics(y, prob, threshold=cfg.threshold)
    return {"metrics": metrics, "prob": prob}
