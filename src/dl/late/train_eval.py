# -*- coding: utf-8 -*-
"""
训练与评估（late）：

    - 固定使用 image/text 两个分支（可选 meta）
    - 损失：BCEWithLogitsLoss
    - 优化器：AdamW
    - best checkpoint：优先使用 val_auc；若验证集为单类导致 AUC 不可用，则回退为 val_log_loss
    - 训练阶段逐 epoch 仅记录阈值无关指标：roc_auc / log_loss
    - 阈值选择：在 best epoch 确定后由 main.py 在验证集上搜索一次（max F1）
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import LateConfig
from utils import compute_threshold_free_metrics

_LABEL_SMOOTHING_EPS = 0.05
_EMA_DECAY = 0.999
_WARMUP_RATIO = 0.1
_DEFAULT_MAX_GRAD_NORM = 1.0


def _build_grad_scaler(enabled: bool):
    """
    构造 AMP GradScaler。

    说明：本项目仅在 CUDA 设备上启用 AMP；CPU 上返回禁用的 scaler。
    """
    if bool(enabled):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler(enabled=False)


def _amp_autocast(device: torch.device, enabled: bool):
    if not bool(enabled):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=str(device.type), enabled=True)


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


class _EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if not torch.is_floating_point(v):
                continue
            self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = float(self.decay)
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            if not torch.is_floating_point(v):
                continue
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def ema_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        out = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            if k in out and torch.is_floating_point(out[k]):
                out[k] = v.detach().clone()
        return out

    @contextlib.contextmanager
    def swap_to_ema(self, model: nn.Module):
        backup: Dict[str, torch.Tensor] = {}
        state = model.state_dict()
        for k, v in self.shadow.items():
            if k not in state:
                continue
            if not torch.is_floating_point(state[k]):
                continue
            backup[k] = state[k].detach().clone()
            state[k].copy_(v)
        try:
            yield
        finally:
            state2 = model.state_dict()
            for k, v in backup.items():
                if k in state2:
                    state2[k].copy_(v)


def _get_device(cfg: LateConfig) -> torch.device:
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
def _positive_proba_late(
    model: nn.Module,
    use_meta: bool,
    X_meta: np.ndarray | None,
    X_image: np.ndarray,
    len_image: np.ndarray,
    attr_image: np.ndarray,
    X_text: np.ndarray,
    len_text: np.ndarray,
    attr_text: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """返回正类概率（shape: [N]）。"""
    model.eval()
    use_amp = device.type == "cuda"

    tensors: List[torch.Tensor] = []
    if bool(use_meta):
        if X_meta is None:
            raise ValueError("use_meta=True 时，X_meta 不能为空。")
        tensors.append(torch.from_numpy(np.asarray(X_meta, dtype=np.float32)))

    tensors.append(torch.from_numpy(np.asarray(X_image, dtype=np.float32)))
    tensors.append(torch.from_numpy(np.asarray(len_image, dtype=np.int64)))
    tensors.append(torch.from_numpy(np.asarray(attr_image, dtype=np.float32)))
    tensors.append(torch.from_numpy(np.asarray(X_text, dtype=np.float32)))
    tensors.append(torch.from_numpy(np.asarray(len_text, dtype=np.int64)))
    tensors.append(torch.from_numpy(np.asarray(attr_text, dtype=np.float32)))

    loader = DataLoader(TensorDataset(*tensors), batch_size=max(1, int(batch_size)), shuffle=False)

    probs: List[np.ndarray] = []
    for batch in loader:
        idx = 0
        xb_meta = None
        if bool(use_meta):
            xb_meta = batch[idx].to(device)
            idx += 1

        xb_img = batch[idx].to(device)
        lb_img = batch[idx + 1].to(device)
        ab_img = batch[idx + 2].to(device)
        xb_txt = batch[idx + 3].to(device)
        lb_txt = batch[idx + 4].to(device)
        ab_txt = batch[idx + 5].to(device)

        with _amp_autocast(device, enabled=bool(use_amp)):
            logits = model(
                x_meta=xb_meta,
                x_image=xb_img,
                len_image=lb_img,
                attr_image=ab_img,
                x_text=xb_txt,
                len_text=lb_txt,
                attr_text=ab_txt,
            )
        prob = torch.sigmoid(logits.float()).detach().cpu().numpy()
        probs.append(prob.astype(np.float64, copy=False))

    if not probs:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(probs, axis=0).reshape(-1)


def train_late_with_early_stopping(
    model: nn.Module,
    use_meta: bool,
    X_meta_train: np.ndarray | None,
    X_image_train: np.ndarray,
    len_image_train: np.ndarray,
    attr_image_train: np.ndarray,
    X_text_train: np.ndarray,
    len_text_train: np.ndarray,
    attr_text_train: np.ndarray,
    y_train: np.ndarray,
    X_meta_val: np.ndarray | None,
    X_image_val: np.ndarray,
    len_image_val: np.ndarray,
    attr_image_val: np.ndarray,
    X_text_val: np.ndarray,
    len_text_val: np.ndarray,
    attr_text_val: np.ndarray,
    y_val: np.ndarray,
    cfg: LateConfig,
    logger,
) -> Tuple[Dict[str, torch.Tensor], int, List[Dict[str, Any]], Dict[str, Any]]:
    """训练 + 早停（late baseline）。"""
    device = _get_device(cfg)
    model = model.to(device)
    use_amp = device.type == "cuda"

    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))

    train_tensors: List[torch.Tensor] = []
    if bool(use_meta):
        if X_meta_train is None:
            raise ValueError("use_meta=True 时，X_meta_train 不能为空。")
        train_tensors.append(torch.from_numpy(np.asarray(X_meta_train, dtype=np.float32)))

    train_tensors.append(torch.from_numpy(np.asarray(X_image_train, dtype=np.float32)))
    train_tensors.append(torch.from_numpy(np.asarray(len_image_train, dtype=np.int64)))
    train_tensors.append(torch.from_numpy(np.asarray(attr_image_train, dtype=np.float32)))
    train_tensors.append(torch.from_numpy(np.asarray(X_text_train, dtype=np.float32)))
    train_tensors.append(torch.from_numpy(np.asarray(len_text_train, dtype=np.int64)))
    train_tensors.append(torch.from_numpy(np.asarray(attr_text_train, dtype=np.float32)))
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
    scaler = _build_grad_scaler(enabled=bool(use_amp))
    ema = _EMA(model, decay=_EMA_DECAY)
    ema_logged = False

    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_auc = -float("inf")
    best_val_log_loss = float("inf")
    bad_epochs = 0

    history: List[Dict[str, Any]] = []

    y_val_1d = np.asarray(y_val).astype(int).reshape(-1)
    n_val_pos = int(np.sum(y_val_1d == 1))
    n_val_neg = int(np.sum(y_val_1d == 0))
    metric_for_best = "val_auc" if (n_val_pos > 0 and n_val_neg > 0) else "val_log_loss"
    tie_breaker = (
        "val_auc(desc) -> val_log_loss(asc) -> epoch(asc)"
        if metric_for_best == "val_auc"
        else "val_log_loss(asc) -> epoch(asc)"
    )

    for epoch in range(1, int(cfg.max_epochs) + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        grad_norm_sum = 0.0
        grad_norm_n = 0

        for batch in train_loader:
            idx = 0
            xb_meta = None
            if bool(use_meta):
                xb_meta = batch[idx].to(device)
                idx += 1

            xb_img = batch[idx].to(device)
            lb_img = batch[idx + 1].to(device)
            ab_img = batch[idx + 2].to(device)
            xb_txt = batch[idx + 3].to(device)
            lb_txt = batch[idx + 4].to(device)
            ab_txt = batch[idx + 5].to(device)
            idx += 6

            yb = batch[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            yb = yb * float(1.0 - _LABEL_SMOOTHING_EPS) + float(0.5 * _LABEL_SMOOTHING_EPS)
            with _amp_autocast(device, enabled=bool(use_amp)):
                logits = model(
                    x_meta=xb_meta,
                    x_image=xb_img,
                    len_image=lb_img,
                    attr_image=ab_img,
                    x_text=xb_txt,
                    len_text=lb_txt,
                    attr_text=ab_txt,
                )
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))
            if not (max_grad_norm > 0.0):
                max_grad_norm = float(_DEFAULT_MAX_GRAD_NORM)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm)))
            grad_norm_sum += float(grad_norm)
            grad_norm_n += 1

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            bs = int(yb.shape[0])
            running_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

        train_epoch_loss = running_loss / max(1, n_seen)

        if not ema_logged:
            logger.info("EMA：swap to ema / restore（评估使用 EMA 权重）")
            ema_logged = True

        with ema.swap_to_ema(model):
            train_prob = _positive_proba_late(
                model,
                use_meta=use_meta,
                X_meta=X_meta_train,
                X_image=X_image_train,
                len_image=len_image_train,
                attr_image=attr_image_train,
                X_text=X_text_train,
                len_text=len_text_train,
                attr_text=attr_text_train,
                device=device,
                batch_size=cfg.batch_size,
            )
            val_prob = _positive_proba_late(
                model,
                use_meta=use_meta,
                X_meta=X_meta_val,
                X_image=X_image_val,
                len_image=len_image_val,
                attr_image=attr_image_val,
                X_text=X_text_val,
                len_text=len_text_val,
                attr_text=attr_text_val,
                device=device,
                batch_size=cfg.batch_size,
            )

        train_tf = compute_threshold_free_metrics(y_train, train_prob)
        val_tf = compute_threshold_free_metrics(y_val, val_prob)

        row: Dict[str, Any] = {"epoch": int(epoch)}
        for k, v in train_tf.items():
            row[f"train_{k}"] = v
        for k, v in val_tf.items():
            row[f"val_{k}"] = v
        row["train_epoch_loss"] = float(train_epoch_loss)
        row["train_grad_norm"] = float(grad_norm_sum / max(1, grad_norm_n))
        history.append(row)

        val_log_loss = float(val_tf.get("log_loss", 0.0))
        val_auc = val_tf.get("roc_auc", None)

        improved = False
        if metric_for_best == "val_auc":
            cur_auc = -float("inf") if val_auc is None else float(val_auc)
            if cur_auc > best_val_auc + 1e-12:
                improved = True
            elif abs(cur_auc - best_val_auc) <= 1e-12 and val_log_loss < best_val_log_loss - 1e-12:
                improved = True
        else:
            if val_log_loss < best_val_log_loss - 1e-12:
                improved = True

        if improved:
            best_epoch = int(epoch)
            if metric_for_best == "val_auc" and val_auc is not None:
                best_val_auc = float(val_auc)
            best_val_log_loss = float(val_log_loss)
            best_state = ema.ema_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1

        lr_now = float(optimizer.param_groups[0]["lr"])
        val_auc_str = "None" if val_auc is None else f"{float(val_auc):.6f}"
        logger.info(
            "Epoch %d/%d | lr=%.6g | train_loss=%.4f val_loss=%.4f | val_auc=%s | grad_norm=%.4f | best_epoch=%d bad=%d",
            int(epoch),
            int(cfg.max_epochs),
            lr_now,
            float(train_tf.get("log_loss", 0.0)),
            float(val_log_loss),
            val_auc_str,
            float(row["train_grad_norm"]),
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
        best_state = ema.ema_state_dict(model)
        best_epoch = len(history) if history else 0
        if history:
            best_val_log_loss = float(history[-1].get("val_log_loss", 0.0))
            if metric_for_best == "val_auc":
                last_auc = history[-1].get("val_roc_auc", None)
                if last_auc is not None:
                    best_val_auc = float(last_auc)
    model.load_state_dict(best_state)

    best_val = next((h for h in history if h.get("epoch") == best_epoch), None) if history else None
    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_auc": None if metric_for_best != "val_auc" else (None if not np.isfinite(best_val_auc) else float(best_val_auc)),
        "best_val_log_loss": float(best_val_log_loss),
        "metric_for_best": str(metric_for_best),
        "tie_breaker": str(tie_breaker),
        "best_val_row": best_val,
    }
    best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state_cpu, int(best_epoch), history, best_info


def evaluate_late_split(
    model: nn.Module,
    use_meta: bool,
    X_meta: np.ndarray | None,
    X_image: np.ndarray,
    len_image: np.ndarray,
    attr_image: np.ndarray,
    X_text: np.ndarray,
    len_text: np.ndarray,
    attr_text: np.ndarray,
    y: np.ndarray,
    cfg: LateConfig,
) -> Dict[str, Any]:
    device = _get_device(cfg)
    model = model.to(device)
    prob = _positive_proba_late(
        model,
        use_meta=use_meta,
        X_meta=X_meta,
        X_image=X_image,
        len_image=len_image,
        attr_image=attr_image,
        X_text=X_text,
        len_text=len_text,
        attr_text=attr_text,
        device=device,
        batch_size=cfg.batch_size,
    )
    return {"prob": prob}
