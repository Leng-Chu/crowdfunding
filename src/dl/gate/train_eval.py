# -*- coding: utf-8 -*-
"""
训练与评估（gate）：

- 任务：二分类（BCEWithLogitsLoss）
- 优化器：Adam
- 早停：early_stop_patience（支持 early_stop_min_epochs）
- 指标：accuracy / precision / recall / f1 / roc_auc / log_loss

注意：
- 阈值最终由验证集选择（max F1），并用于测试集评估（见 main.py）。
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import GateConfig
from utils import compute_binary_metrics

_LABEL_SMOOTHING_EPS = 0.05
_EMA_DECAY = 0.999
_WARMUP_RATIO = 0.1
_DEFAULT_MAX_GRAD_NORM = 1.0


def _build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type="cuda", enabled=bool(enabled))
        except TypeError:
            try:
                return torch.amp.GradScaler("cuda", enabled=bool(enabled))
            except TypeError:
                return torch.amp.GradScaler(enabled=bool(enabled))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _amp_autocast(device: torch.device, enabled: bool):
    if not bool(enabled):
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=str(device.type), enabled=True)
        except TypeError:
            try:
                return torch.amp.autocast(str(device.type), enabled=True)
            except TypeError:
                pass
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()


def _compute_pos_weight(y_train: np.ndarray) -> float:
    y = np.asarray(y_train).astype(int).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(n_neg / max(1, n_pos))


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


def _get_device(cfg: GateConfig) -> torch.device:
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


class GateDataset(Dataset):
    def __init__(
        self,
        X_meta: Optional[np.ndarray],
        title_blurb: np.ndarray,
        cover: np.ndarray,
        X_img: np.ndarray,
        X_txt: np.ndarray,
        seq_type: np.ndarray,
        seq_attr: np.ndarray,
        seq_mask: np.ndarray,
        y: np.ndarray,
        project_ids: List[str],
    ) -> None:
        super().__init__()
        self.X_meta = X_meta
        self.title_blurb = title_blurb
        self.cover = cover
        self.X_img = X_img
        self.X_txt = X_txt
        self.seq_type = seq_type
        self.seq_attr = seq_attr
        self.seq_mask = seq_mask
        self.y = y
        self.project_ids = list(project_ids)

        n = int(self.X_img.shape[0])
        if (
            int(self.X_txt.shape[0]) != n
            or int(self.seq_type.shape[0]) != n
            or int(self.seq_mask.shape[0]) != n
            or int(self.title_blurb.shape[0]) != n
            or int(self.cover.shape[0]) != n
        ):
            raise ValueError("样本数不一致：first impression / seq。")
        if int(len(self.project_ids)) != n:
            raise ValueError("project_ids 长度与样本数不一致。")
        if int(self.y.shape[0]) != n:
            raise ValueError("y 长度与样本数不一致。")
        if self.X_meta is not None and int(self.X_meta.shape[0]) != n:
            raise ValueError("X_meta 样本数与 seq 不一致。")

    def __len__(self) -> int:
        return int(self.X_img.shape[0])

    def __getitem__(self, idx: int):
        i = int(idx)

        tb = torch.from_numpy(self.title_blurb[i])
        c = torch.from_numpy(self.cover[i])
        x_img = torch.from_numpy(self.X_img[i])
        x_txt = torch.from_numpy(self.X_txt[i])
        t = torch.from_numpy(self.seq_type[i])
        a = torch.from_numpy(self.seq_attr[i])
        m = torch.from_numpy(self.seq_mask[i])
        y = torch.tensor(float(self.y[i]), dtype=torch.float32)

        if self.X_meta is None:
            x_meta = torch.empty((0,), dtype=torch.float32)
        else:
            x_meta = torch.from_numpy(self.X_meta[i])
        return x_meta, tb, c, x_img, x_txt, t, a, m, y


@torch.no_grad()
def _positive_proba_gate(
    model: nn.Module,
    X_meta: np.ndarray | None,
    title_blurb: np.ndarray,
    cover: np.ndarray,
    X_img: np.ndarray,
    X_txt: np.ndarray,
    seq_type: np.ndarray,
    seq_attr: np.ndarray,
    seq_mask: np.ndarray,
    y: np.ndarray,
    project_ids: List[str],
    cfg: GateConfig,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    use_amp = device.type == "cuda"
    ds = GateDataset(
        X_meta=X_meta,
        title_blurb=title_blurb,
        cover=cover,
        X_img=X_img,
        X_txt=X_txt,
        seq_type=seq_type,
        seq_attr=seq_attr,
        seq_mask=seq_mask,
        y=y,
        project_ids=project_ids,
    )
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=False)

    probs: List[np.ndarray] = []
    for batch in loader:
        x_meta, tb, c, x_img, x_txt, t, a, m, _y = batch
        with _amp_autocast(device, enabled=bool(use_amp)):
            logits = model(
                title_blurb=tb.to(device),
                cover=c.to(device),
                x_img=x_img.to(device),
                x_txt=x_txt.to(device),
                seq_type=t.to(device),
                seq_attr=a.to(device),
                seq_mask=m.to(device),
                x_meta=x_meta.to(device),
            )
        prob = torch.sigmoid(logits.float()).detach().cpu().numpy().astype(np.float64, copy=False)
        probs.append(prob)

    if not probs:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(probs, axis=0).reshape(-1)


def train_gate_with_early_stopping(
    model: nn.Module,
    X_meta_train: np.ndarray | None,
    title_blurb_train: np.ndarray,
    cover_train: np.ndarray,
    X_img_train: np.ndarray,
    X_txt_train: np.ndarray,
    seq_type_train: np.ndarray,
    seq_attr_train: np.ndarray,
    seq_mask_train: np.ndarray,
    y_train: np.ndarray,
    train_project_ids: List[str],
    X_meta_val: np.ndarray | None,
    title_blurb_val: np.ndarray,
    cover_val: np.ndarray,
    X_img_val: np.ndarray,
    X_txt_val: np.ndarray,
    seq_type_val: np.ndarray,
    seq_attr_val: np.ndarray,
    seq_mask_val: np.ndarray,
    y_val: np.ndarray,
    val_project_ids: List[str],
    cfg: GateConfig,
    logger,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    """训练 + 早停。"""
    device = _get_device(cfg)
    model = model.to(device)
    use_amp = device.type == "cuda"

    ds_train = GateDataset(
        X_meta=X_meta_train,
        title_blurb=title_blurb_train,
        cover=cover_train,
        X_img=X_img_train,
        X_txt=X_txt_train,
        seq_type=seq_type_train,
        seq_attr=seq_attr_train,
        seq_mask=seq_mask_train,
        y=y_train,
        project_ids=train_project_ids,
    )
    train_loader = DataLoader(ds_train, batch_size=max(1, int(cfg.batch_size)), shuffle=True)

    pos_weight = _compute_pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))

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
    best_score = -float("inf")
    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(cfg.max_epochs) + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        grad_norm_sum = 0.0
        grad_norm_n = 0

        for batch in train_loader:
            x_meta, tb, c, x_img, x_txt, t, a, m, yb = batch
            optimizer.zero_grad(set_to_none=True)
            yb = yb.to(device)
            yb = yb * float(1.0 - _LABEL_SMOOTHING_EPS) + float(0.5 * _LABEL_SMOOTHING_EPS)
            with _amp_autocast(device, enabled=bool(use_amp)):
                logits = model(
                    title_blurb=tb.to(device),
                    cover=c.to(device),
                    x_img=x_img.to(device),
                    x_txt=x_txt.to(device),
                    seq_type=t.to(device),
                    seq_attr=a.to(device),
                    seq_mask=m.to(device),
                    x_meta=x_meta.to(device),
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
            train_prob = _positive_proba_gate(
                model,
                X_meta=X_meta_train,
                title_blurb=title_blurb_train,
                cover=cover_train,
                X_img=X_img_train,
                X_txt=X_txt_train,
                seq_type=seq_type_train,
                seq_attr=seq_attr_train,
                seq_mask=seq_mask_train,
                y=y_train,
                project_ids=train_project_ids,
                cfg=cfg,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            val_prob = _positive_proba_gate(
                model,
                X_meta=X_meta_val,
                title_blurb=title_blurb_val,
                cover=cover_val,
                X_img=X_img_val,
                X_txt=X_txt_val,
                seq_type=seq_type_val,
                seq_attr=seq_attr_val,
                seq_mask=seq_mask_val,
                y=y_val,
                project_ids=val_project_ids,
                cfg=cfg,
                device=device,
                batch_size=int(cfg.batch_size),
            )

        train_metrics = compute_binary_metrics(y_train, train_prob, threshold=cfg.threshold)
        val_metrics = compute_binary_metrics(y_val, val_prob, threshold=cfg.threshold)

        row: Dict[str, Any] = {"epoch": int(epoch)}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        row["train_epoch_loss"] = float(train_epoch_loss)
        row["train_grad_norm"] = float(grad_norm_sum / max(1, grad_norm_n))
        history.append(row)

        score = -float(val_metrics["log_loss"])

        improved = score > best_score
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = ema.ema_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1

        lr_now = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %d/%d | lr=%.6g | train_loss=%.4f val_loss=%.4f | train_acc=%.4f val_acc=%.4f | grad_norm=%.4f | best_epoch=%d bad=%d",
            int(epoch),
            int(cfg.max_epochs),
            lr_now,
            float(train_metrics["log_loss"]),
            float(val_metrics["log_loss"]),
            float(train_metrics["accuracy"]),
            float(val_metrics["accuracy"]),
            float(row["train_grad_norm"]),
            int(best_epoch),
            int(bad_epochs),
        )

        if False and scheduler is not None:
            old_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step(float(val_metrics["log_loss"]))
            new_lr = float(optimizer.param_groups[0]["lr"])
            if new_lr < old_lr:
                logger.info("学习率调整：%.6g -> %.6g", old_lr, new_lr)
                if bool(getattr(cfg, "reset_early_stop_on_lr_change", True)):
                    bad_epochs = 0

        if bad_epochs >= int(cfg.early_stop_patience):
            min_epochs = int(getattr(cfg, "early_stop_min_epochs", 0))
            if int(epoch) < min_epochs:
                continue
            logger.info("触发早停：连续 %d 个 epoch 无提升。", int(bad_epochs))
            break

    if best_state is None:
        best_state = ema.ema_state_dict(model)
        best_epoch = len(history) if history else 0
    model.load_state_dict(best_state)

    best_val = next((h for h in history if h.get("epoch") == best_epoch), None) if history else None
    best_info = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "metric_for_best": "val_log_loss",
        "best_val_row": best_val,
    }
    return model, history, best_info


def evaluate_gate_split(
    model: nn.Module,
    X_meta: np.ndarray | None,
    title_blurb: np.ndarray,
    cover: np.ndarray,
    X_img: np.ndarray,
    X_txt: np.ndarray,
    seq_type: np.ndarray,
    seq_attr: np.ndarray,
    seq_mask: np.ndarray,
    y: np.ndarray,
    project_ids: List[str],
    cfg: GateConfig,
) -> Dict[str, Any]:
    device = _get_device(cfg)
    model = model.to(device)
    prob = _positive_proba_gate(
        model,
        X_meta=X_meta,
        title_blurb=title_blurb,
        cover=cover,
        X_img=X_img,
        X_txt=X_txt,
        seq_type=seq_type,
        seq_attr=seq_attr,
        seq_mask=seq_mask,
        y=y,
        project_ids=project_ids,
        cfg=cfg,
        device=device,
        batch_size=int(cfg.batch_size),
    )
    metrics = compute_binary_metrics(y, prob, threshold=cfg.threshold)
    return {"metrics": metrics, "prob": prob}
