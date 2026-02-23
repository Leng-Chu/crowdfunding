# -*- coding: utf-8 -*-
"""
训练与评估（mlp）：

- 多模态训练/评估逻辑：meta/image/text 任意组合
- 入口脚本：`src/dl/mlp/main.py`
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MlpConfig
from utils import compute_binary_metrics

_WARMUP_RATIO = 0.1


def _build_adamw(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    norm_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))

    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not bool(p.requires_grad):
            continue
        if name.endswith(".bias") or id(p) in norm_param_ids:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups: List[Dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups, lr=float(lr))


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    steps_per_epoch: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(max_epochs) * max(1, int(steps_per_epoch)))
    warmup_steps = max(1, int(_WARMUP_RATIO * float(total_steps)))

    base_lr = float(optimizer.param_groups[0].get("lr", 0.0))
    if base_lr <= 0.0:
        min_lr_ratio = 0.0
    else:
        min_lr_ratio = float(min_lr) / float(base_lr)
    min_lr_ratio = float(np.clip(min_lr_ratio, 0.0, 1.0))

    def lr_lambda(step: int) -> float:
        step = int(step)
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * float(np.clip(progress, 0.0, 1.0))))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    """返回正类概率（shape: [N]）。"""
    model.eval()

    tensors: List[torch.Tensor] = []
    if bool(use_meta):
        if X_meta is None:
            raise ValueError("use_meta=True 时，X_meta 不能为空。")
        tensors.append(torch.from_numpy(np.asarray(X_meta, dtype=np.float32)))

    if bool(use_image):
        if X_image is None or len_image is None:
            raise ValueError("use_image=True 时，X_image/len_image 不能为空。")
        tensors.append(torch.from_numpy(np.asarray(X_image, dtype=np.float32)))
        tensors.append(torch.from_numpy(np.asarray(len_image, dtype=np.int64)))

    if bool(use_text):
        if X_text is None or len_text is None:
            raise ValueError("use_text=True 时，X_text/len_text 不能为空。")
        tensors.append(torch.from_numpy(np.asarray(X_text, dtype=np.float32)))
        tensors.append(torch.from_numpy(np.asarray(len_text, dtype=np.int64)))

    loader = DataLoader(TensorDataset(*tensors), batch_size=max(1, int(batch_size)), shuffle=False)

    probs: List[np.ndarray] = []
    for batch in loader:
        idx = 0
        xb_meta = None
        xb_img = None
        lb_img = None
        xb_txt = None
        lb_txt = None

        if bool(use_meta):
            xb_meta = batch[idx].to(device)
            idx += 1
        if bool(use_image):
            xb_img = batch[idx].to(device)
            lb_img = batch[idx + 1].to(device)
            idx += 2
        if bool(use_text):
            xb_txt = batch[idx].to(device)
            lb_txt = batch[idx + 1].to(device)

        logits = model(
            x_meta=xb_meta,
            x_image=xb_img,
            len_image=lb_img,
            x_text=xb_txt,
            len_text=lb_txt,
        )
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
    """训练 + 早停（仅多模态）。"""
    device = _get_device(cfg)
    model = model.to(device)

    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))

    train_tensors: List[torch.Tensor] = []
    if bool(use_meta):
        if X_meta_train is None:
            raise ValueError("use_meta=True 时，X_meta_train 不能为空。")
        train_tensors.append(torch.from_numpy(np.asarray(X_meta_train, dtype=np.float32)))
    if bool(use_image):
        if X_image_train is None or len_image_train is None:
            raise ValueError("use_image=True 时，X_image_train/len_image_train 不能为空。")
        train_tensors.append(torch.from_numpy(np.asarray(X_image_train, dtype=np.float32)))
        train_tensors.append(torch.from_numpy(np.asarray(len_image_train, dtype=np.int64)))
    if bool(use_text):
        if X_text_train is None or len_text_train is None:
            raise ValueError("use_text=True 时，X_text_train/len_text_train 不能为空。")
        train_tensors.append(torch.from_numpy(np.asarray(X_text_train, dtype=np.float32)))
        train_tensors.append(torch.from_numpy(np.asarray(len_text_train, dtype=np.int64)))
    train_tensors.append(y_train_t)

    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = _build_adamw(
        model,
        lr=float(cfg.learning_rate_init),
        weight_decay=float(cfg.alpha),
    )
    scheduler = _build_warmup_cosine_scheduler(
        optimizer,
        max_epochs=int(cfg.max_epochs),
        steps_per_epoch=len(train_loader),
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
            xb_meta = None
            xb_img = None
            lb_img = None
            xb_txt = None
            lb_txt = None

            if bool(use_meta):
                xb_meta = batch[idx].to(device)
                idx += 1
            if bool(use_image):
                xb_img = batch[idx].to(device)
                lb_img = batch[idx + 1].to(device)
                idx += 2
            if bool(use_text):
                xb_txt = batch[idx].to(device)
                lb_txt = batch[idx + 1].to(device)
                idx += 2

            yb = batch[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(
                x_meta=xb_meta,
                x_image=xb_img,
                len_image=lb_img,
                x_text=xb_txt,
                len_text=lb_txt,
            )
            loss = criterion(logits, yb)
            loss.backward()

            max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

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

        row: Dict[str, Any] = {"epoch": int(epoch)}
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
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        lr_now = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %d/%d | lr=%.6g | train_loss=%.4f val_loss=%.4f | train_acc=%.4f val_acc=%.4f | best_epoch=%d bad=%d",
            int(epoch),
            int(cfg.max_epochs),
            lr_now,
            float(train_metrics["log_loss"]),
            float(val_metrics["log_loss"]),
            float(train_metrics["accuracy"]),
            float(val_metrics["accuracy"]),
            int(best_epoch),
            int(bad_epochs),
        )

        if bad_epochs >= int(cfg.early_stop_patience):
            min_epochs = int(getattr(cfg, "early_stop_min_epochs", 0))
            if int(epoch) < min_epochs:
                continue
            logger.info("触发早停：连续 %d 个 epoch 无提升。", int(bad_epochs))
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history) if history else 0
    model.load_state_dict(best_state)

    best_val = next((h for h in history if h.get("epoch") == best_epoch), None) if history else None
    best_info = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
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

