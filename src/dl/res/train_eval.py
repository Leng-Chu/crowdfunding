# -*- coding: utf-8 -*-
"""
训练与评估（res）：

    - 任务：二分类（BCEWithLogitsLoss）
    - 优化器：AdamW
    - 早停：early_stop_patience（支持 early_stop_min_epochs）
    - best checkpoint：优先使用 val_auc；若验证集为单类导致 AUC 不可用，则回退为 val_log_loss
    - 训练阶段逐 epoch 仅记录阈值无关指标：roc_auc / log_loss

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

from config import ResConfig
from utils import compute_threshold_free_metrics

_LABEL_SMOOTHING_EPS = 0.05
_EMA_DECAY = 0.999
_WARMUP_RATIO = 0.1
_DEFAULT_MAX_GRAD_NORM = 1.0
_NO_WEIGHT_DECAY_NAME_SUFFIXES = (
    "delta_scale_raw",
    "key_alpha_raw",
    "gate_bias",
    "gate_scale_raw",
)


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
        if name.endswith(".bias") or id(p) in norm_param_ids or str(name).endswith(_NO_WEIGHT_DECAY_NAME_SUFFIXES):
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


def _get_device(cfg: ResConfig) -> torch.device:
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


class ResDataset(Dataset):
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
def _positive_proba_res(
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
    cfg: ResConfig,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    use_amp = device.type == "cuda"
    ds = ResDataset(
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


@torch.no_grad()
def _val_residual_debug(
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
    cfg: ResConfig,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    baseline_mode="res" 的验证集调试输出：

    - 返回最终概率 prob_final（sigmoid(z)）
    - 额外返回 debug 字段（写入 history/log）：
      delta_abs_mean / delta_abs_p90 / delta_y1 / delta_y0 / auc_base / auc_final
      delta_scale / z_res_raw_abs_mean / z_res_raw_abs_p90
    """
    model.eval()
    use_amp = device.type == "cuda"

    if not hasattr(model, "forward_res"):
        raise RuntimeError("当前模型不支持 forward_res，无法计算 residual debug 指标。")

    ds = ResDataset(
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

    prob_final_list: List[np.ndarray] = []
    prob_base_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    z_res_raw_list: List[np.ndarray] = []

    delta_scale_value: float | None = None
    if hasattr(model, "effective_delta_scale"):
        try:
            delta_scale_value = float(model.effective_delta_scale().detach().cpu().item())
        except Exception:
            delta_scale_value = None

    for batch in loader:
        x_meta, tb, c, x_img, x_txt, t, a, m, _y = batch
        with _amp_autocast(device, enabled=bool(use_amp)):
            z_res_raw = None
            try:
                z, z_base, delta, _delta_scale, z_res_raw = model.forward_res(  # type: ignore[misc]
                    title_blurb=tb.to(device),
                    cover=c.to(device),
                    x_img=x_img.to(device),
                    x_txt=x_txt.to(device),
                    seq_type=t.to(device),
                    seq_attr=a.to(device),
                    seq_mask=m.to(device),
                    x_meta=x_meta.to(device),
                    return_debug=True,
                )
            except TypeError:
                z, z_base, delta = model.forward_res(  # type: ignore[misc]
                    title_blurb=tb.to(device),
                    cover=c.to(device),
                    x_img=x_img.to(device),
                    x_txt=x_txt.to(device),
                    seq_type=t.to(device),
                    seq_attr=a.to(device),
                    seq_mask=m.to(device),
                    x_meta=x_meta.to(device),
                )

        prob_final = torch.sigmoid(z.float()).detach().cpu().numpy().astype(np.float64, copy=False)
        prob_base = torch.sigmoid(z_base.float()).detach().cpu().numpy().astype(np.float64, copy=False)
        delta_np = delta.float().detach().cpu().numpy().astype(np.float64, copy=False)
        prob_final_list.append(prob_final)
        prob_base_list.append(prob_base)
        delta_list.append(delta_np)
        if z_res_raw is not None:
            z_res_raw_np = z_res_raw.float().detach().cpu().numpy().astype(np.float64, copy=False)
            z_res_raw_list.append(z_res_raw_np)

    if not prob_final_list:
        empty_prob = np.zeros((0,), dtype=np.float64)
        debug = {
            "val_delta_abs_mean": None,
            "val_delta_abs_p90": None,
            "val_delta_y1": None,
            "val_delta_y0": None,
            "val_auc_base": None,
            "val_auc_final": None,
            "val_delta_scale": delta_scale_value,
            "val_z_res_raw_abs_mean": None,
            "val_z_res_raw_abs_p90": None,
        }
        return empty_prob, debug

    prob_final_arr = np.concatenate(prob_final_list, axis=0).reshape(-1)
    prob_base_arr = np.concatenate(prob_base_list, axis=0).reshape(-1)
    delta_arr = np.concatenate(delta_list, axis=0).reshape(-1)
    z_res_raw_arr = np.concatenate(z_res_raw_list, axis=0).reshape(-1) if z_res_raw_list else None

    abs_delta = np.abs(delta_arr)
    delta_abs_mean = float(np.mean(abs_delta)) if int(abs_delta.size) > 0 else None
    delta_abs_p90 = float(np.quantile(abs_delta, 0.9)) if int(abs_delta.size) > 0 else None

    y_true = np.asarray(y).astype(int).reshape(-1)
    if int(y_true.shape[0]) != int(delta_arr.shape[0]):
        raise ValueError(f"val y/delta 长度不一致：{y_true.shape} vs {delta_arr.shape}")

    m1 = y_true == 1
    m0 = y_true == 0
    delta_y1 = float(np.mean(delta_arr[m1])) if bool(np.any(m1)) else None
    delta_y0 = float(np.mean(delta_arr[m0])) if bool(np.any(m0)) else None

    auc_base = compute_threshold_free_metrics(y_true, prob_base_arr).get("roc_auc", None)
    auc_final = compute_threshold_free_metrics(y_true, prob_final_arr).get("roc_auc", None)

    z_res_raw_abs_mean = None
    z_res_raw_abs_p90 = None
    if z_res_raw_arr is not None:
        abs_z_res_raw = np.abs(z_res_raw_arr)
        if int(abs_z_res_raw.size) > 0:
            z_res_raw_abs_mean = float(np.mean(abs_z_res_raw))
            z_res_raw_abs_p90 = float(np.quantile(abs_z_res_raw, 0.9))

    debug = {
        "val_delta_abs_mean": delta_abs_mean,
        "val_delta_abs_p90": delta_abs_p90,
        "val_delta_y1": delta_y1,
        "val_delta_y0": delta_y0,
        "val_auc_base": auc_base,
        "val_auc_final": auc_final,
        "val_delta_scale": delta_scale_value,
        "val_z_res_raw_abs_mean": z_res_raw_abs_mean,
        "val_z_res_raw_abs_p90": z_res_raw_abs_p90,
    }
    return prob_final_arr, debug


def train_res_with_early_stopping(
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
    cfg: ResConfig,
    logger,
) -> Tuple[Dict[str, torch.Tensor], int, List[Dict[str, Any]], Dict[str, Any]]:
    """训练 + 早停。"""
    device = _get_device(cfg)
    model = model.to(device)
    use_amp = device.type == "cuda"

    ds_train = ResDataset(
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
            train_prob = _positive_proba_res(
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

            baseline_mode = str(getattr(model, "baseline_mode", "") or "").strip().lower()
            debug_residual_stats = bool(getattr(cfg, "debug_residual_stats", False))
            if baseline_mode == "res" and debug_residual_stats:
                val_prob, val_debug = _val_residual_debug(
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
            else:
                val_prob = _positive_proba_res(
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
                val_debug = {}

        train_tf = compute_threshold_free_metrics(y_train, train_prob)
        val_tf = compute_threshold_free_metrics(y_val, val_prob)

        row: Dict[str, Any] = {"epoch": int(epoch)}
        for k, v in train_tf.items():
            row[f"train_{k}"] = v
        for k, v in val_tf.items():
            row[f"val_{k}"] = v

        if val_debug:
            for k, v in val_debug.items():
                row[k] = v
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
        if val_debug and str(getattr(model, "baseline_mode", "")).strip().lower() == "res":
            d_mean = row.get("val_delta_abs_mean", None)
            d_p90 = row.get("val_delta_abs_p90", None)
            d_y1 = row.get("val_delta_y1", None)
            d_y0 = row.get("val_delta_y0", None)
            d_scale = row.get("val_delta_scale", None)
            zrr_mean = row.get("val_z_res_raw_abs_mean", None)
            zrr_p90 = row.get("val_z_res_raw_abs_p90", None)
            auc_base = row.get("val_auc_base", None)
            auc_final = row.get("val_auc_final", None)
            d_mean_str = "None" if d_mean is None else f"{float(d_mean):.6g}"
            d_p90_str = "None" if d_p90 is None else f"{float(d_p90):.6g}"
            d_y1_str = "None" if d_y1 is None else f"{float(d_y1):.6g}"
            d_y0_str = "None" if d_y0 is None else f"{float(d_y0):.6g}"
            d_scale_str = "None" if d_scale is None else f"{float(d_scale):.6g}"
            zrr_mean_str = "None" if zrr_mean is None else f"{float(zrr_mean):.6g}"
            zrr_p90_str = "None" if zrr_p90 is None else f"{float(zrr_p90):.6g}"
            auc_base_str = "None" if auc_base is None else f"{float(auc_base):.6f}"
            auc_final_str = "None" if auc_final is None else f"{float(auc_final):.6f}"
            logger.info(
                "Epoch %d/%d | lr=%.6g | train_loss=%.4f val_loss=%.4f | val_auc=%s | "
                "delta_abs_mean=%s delta_abs_p90=%s delta_y1=%s delta_y0=%s delta_scale=%s | "
                "z_res_raw_abs_mean=%s z_res_raw_abs_p90=%s | "
                "auc_base=%s auc_final=%s | grad_norm=%.4f | best_epoch=%d bad=%d",
                int(epoch),
                int(cfg.max_epochs),
                lr_now,
                float(train_tf.get("log_loss", 0.0)),
                float(val_log_loss),
                val_auc_str,
                d_mean_str,
                d_p90_str,
                d_y1_str,
                d_y0_str,
                d_scale_str,
                zrr_mean_str,
                zrr_p90_str,
                auc_base_str,
                auc_final_str,
                float(row["train_grad_norm"]),
                int(best_epoch),
                int(bad_epochs),
            )
        else:
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


def evaluate_res_split(
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
    cfg: ResConfig,
) -> Dict[str, Any]:
    device = _get_device(cfg)
    model = model.to(device)
    prob = _positive_proba_res(
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
    return {"prob": prob}
