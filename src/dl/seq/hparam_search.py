# -*- coding: utf-8 -*-
"""
seq 超参数搜索脚本（随机搜索 + 最多 4 路并行 GPU）。

默认针对你给的命令做搜索：
python src/dl/seq/main.py --run-name clip --baseline-mode trm_pos --image-embedding-type clip --text-embedding-type clip --no-use-meta

特性：
- 并行：最多同时跑 4 个 trial，并绑定到 4 张显卡（可配置 GPU 列表）
- 记录：把每个 trial 的超参 + val 指标写到 results.csv，方便后续复盘
- 选优：默认按 val_roc_auc 最大化；可切换为 val_accuracy / val_log_loss 等

注意：
- 该脚本只负责“调度训练”；训练细节仍由 src/dl/seq/main.py 控制（早停、scheduler 等）。
- 请从仓库根目录运行，避免相对路径出错。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple


RESULT_COLUMNS: List[str] = [
    # 基本信息
    "trial_id",
    "gpu_id",
    "status",
    "run_name",
    "seconds",
    "run_dir",
    "metrics_json",
    "score",
    # 超参
    "learning_rate",
    "weight_decay",
    "batch_size",
    "fusion_dropout",
    "token_dropout",
    "transformer_dropout",
    "max_grad_norm",
    "lr_scheduler_patience",
    "lr_scheduler_factor",
    "reset_early_stop_on_lr_change",
    # 指标（val）
    "val_accuracy",
    "val_roc_auc",
    "val_log_loss",
    # 指标（test，仅作参考）
    "test_accuracy",
    "test_roc_auc",
    "test_log_loss",
    # 错误信息
    "error",
]


def _parse_gpus(text: str) -> List[int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    gpus: List[int] = []
    for p in parts:
        try:
            g = int(p)
        except Exception as e:
            raise ValueError(f"不合法的 GPU 列表：{text!r}，期望形如 0,1,2,3。") from e
        if g < 0:
            raise ValueError("GPU 编号必须为非负整数。")
        gpus.append(g)
    if not gpus:
        raise ValueError("GPU 列表不能为空。")
    return gpus


def _loguniform(rng: random.Random, lo: float, hi: float) -> float:
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError("loguniform 需要 lo>0 且 hi>lo。")
    u = rng.random()
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


@dataclass(frozen=True)
class Trial:
    trial_id: int
    params: Dict[str, Any]


def sample_trial(rng: random.Random, trial_id: int) -> Trial:
    # 这里的搜索空间偏“先稳住训练”，再争取更高指标。
    lr = _loguniform(rng, 5e-5, 5e-4)
    wd = rng.choice([0.0, 1e-5, 5e-5, 1e-4, 5e-4])
    fusion_dropout = rng.uniform(0.2, 0.6)
    token_dropout = rng.uniform(0.0, 0.2)
    transformer_dropout = rng.choice([0.1, 0.2, 0.3])
    max_grad_norm = rng.choice([0.5, 1.0, 2.0])
    batch_size = rng.choice([128, 256])
    lr_scheduler_patience = rng.choice([2, 4])
    lr_scheduler_factor = rng.choice([0.5, 0.7])

    params = {
        "learning_rate": float(lr),
        "weight_decay": float(wd),
        "fusion_dropout": float(fusion_dropout),
        "token_dropout": float(token_dropout),
        "transformer_dropout": float(transformer_dropout),
        "max_grad_norm": float(max_grad_norm),
        "batch_size": int(batch_size),
        "lr_scheduler_patience": int(lr_scheduler_patience),
        "lr_scheduler_factor": float(lr_scheduler_factor),
        # 建议默认开启：降 lr 后给模型“喘口气”
        "reset_early_stop_on_lr_change": True,
    }
    return Trial(trial_id=int(trial_id), params=params)


def _format_run_name(base: str, trial: Trial) -> str:
    p = trial.params
    lr = p["learning_rate"]
    wd = p["weight_decay"]
    fd = p["fusion_dropout"]
    gn = p["max_grad_norm"]
    # 文件夹名尽量短一点，避免过长
    return f"{base}_hs_t{trial.trial_id:03d}_lr{lr:.2e}_wd{wd:.1e}_fd{fd:.2f}_gn{gn:.1f}"


def _metric_to_score(metric_name: str, val_metrics: Dict[str, Any]) -> float:
    """
    将 val 指标映射为“越大越好”的 score。
    支持：
    - val_roc_auc / val_accuracy：直接取值
    - val_log_loss：取负数（因为越小越好）
    """
    name = str(metric_name).strip().lower()
    if name in {"val_roc_auc", "roc_auc", "auc"}:
        v = val_metrics.get("roc_auc")
        return -float("inf") if v is None else float(v)
    if name in {"val_accuracy", "accuracy", "acc"}:
        return float(val_metrics.get("accuracy", 0.0))
    if name in {"val_log_loss", "log_loss", "loss"}:
        return -float(val_metrics.get("log_loss", 1e9))
    raise ValueError(f"不支持的 metric={metric_name!r}，可选：val_roc_auc/val_accuracy/val_log_loss。")


def _read_metrics_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_row_csv(path: Path, row: Dict[str, Any], lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(RESULT_COLUMNS))
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})


def _build_cmd(
    python_exe: str,
    run_name: str,
    device: str,
    fixed_args: List[str],
    params: Dict[str, Any],
    extra_overrides: Dict[str, Any],
) -> List[str]:
    cmd: List[str] = [
        str(python_exe),
        "src/dl/seq/main.py",
        "--run-name",
        str(run_name),
        "--device",
        str(device),
    ]
    cmd.extend(list(fixed_args))

    cmd.extend(["--learning-rate", str(params["learning_rate"])])
    cmd.extend(["--weight-decay", str(params["weight_decay"])])
    cmd.extend(["--batch-size", str(params["batch_size"])])
    cmd.extend(["--fusion-dropout", str(params["fusion_dropout"])])
    cmd.extend(["--token-dropout", str(params["token_dropout"])])
    cmd.extend(["--transformer-dropout", str(params["transformer_dropout"])])
    cmd.extend(["--max-grad-norm", str(params["max_grad_norm"])])
    cmd.extend(["--lr-scheduler-patience", str(params["lr_scheduler_patience"])])
    cmd.extend(["--lr-scheduler-factor", str(params["lr_scheduler_factor"])])
    if bool(params.get("reset_early_stop_on_lr_change", True)):
        cmd.append("--reset-early-stop-on-lr-change")
    else:
        cmd.append("--no-reset-early-stop-on-lr-change")

    # 可选：为了加速搜索，覆盖训练轮数/早停
    if extra_overrides.get("max_epochs") is not None:
        cmd.extend(["--max-epochs", str(int(extra_overrides["max_epochs"]))])
    if extra_overrides.get("early_stop_patience") is not None:
        cmd.extend(["--early-stop-patience", str(int(extra_overrides["early_stop_patience"]))])
    if extra_overrides.get("early_stop_min_epochs") is not None:
        cmd.extend(["--early-stop-min-epochs", str(int(extra_overrides["early_stop_min_epochs"]))])

    return cmd


def _run_one_trial(
    project_root: Path,
    python_exe: str,
    fixed_args: List[str],
    trial: Trial,
    gpu_id: int,
    metric: str,
    results_csv: Path,
    results_lock: Lock,
    print_lock: Lock,
    env_mode: str,
    extra_overrides: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    run_name = _format_run_name("clip", trial)
    p = dict(trial.params)

    if dry_run:
        cmd = _build_cmd(
            python_exe=python_exe,
            run_name=run_name,
            device=f"cuda:{int(gpu_id)}",
            fixed_args=fixed_args,
            params=p,
            extra_overrides=extra_overrides,
        )
        with print_lock:
            print(f"[trial={trial.trial_id:03d} gpu={gpu_id}] DRY-RUN 命令：{' '.join(cmd)}")
        return {
            "trial_id": int(trial.trial_id),
            "gpu_id": int(gpu_id),
            "status": "dry_run",
            "run_name": run_name,
            "seconds": None,
            "score": None,
            "metrics_json": None,
            "run_dir": None,
            "error": None,
            **trial.params,
        }

    env = os.environ.copy()
    mode = str(env_mode).strip().lower()
    if mode not in {"device", "cuda_visible_devices"}:
        raise ValueError(f"不支持的 env_mode={env_mode!r}")

    device = f"cuda:{int(gpu_id)}"
    if mode == "cuda_visible_devices":
        # 让每个子进程只“看见”一张卡，避免 cuda:0/1 映射混乱
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        device = "cuda:0"

    cmd = _build_cmd(
        python_exe=python_exe,
        run_name=run_name,
        device=device,
        fixed_args=fixed_args,
        params=p,
        extra_overrides=extra_overrides,
    )

    prefix = f"[trial={trial.trial_id:03d} gpu={gpu_id}] "
    run_dir: Optional[str] = None
    metrics_json: Optional[str] = None

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.rstrip("\n")
        if s.startswith("RUN_DIR="):
            run_dir = s.split("=", 1)[1].strip()
        elif s.startswith("METRICS_JSON="):
            metrics_json = s.split("=", 1)[1].strip()
        with print_lock:
            print(prefix + s)
    rc = int(proc.wait())
    dt = float(time.time() - t0)

    row: Dict[str, Any] = {
        "trial_id": int(trial.trial_id),
        "gpu_id": int(gpu_id),
        "status": "ok" if rc == 0 else f"failed({rc})",
        "run_name": run_name,
        "seconds": round(dt, 3),
        "run_dir": run_dir,
        "metrics_json": metrics_json,
        "score": None,
        "val_accuracy": None,
        "val_roc_auc": None,
        "val_log_loss": None,
        "test_accuracy": None,
        "test_roc_auc": None,
        "test_log_loss": None,
        "error": None,
        **trial.params,
    }

    if rc == 0 and metrics_json:
        try:
            mj = Path(metrics_json)
            if not mj.is_absolute():
                mj = project_root / mj
            metrics = _read_metrics_json(mj)
            val_metrics = dict(metrics.get("val", {}) or {})
            test_metrics = dict(metrics.get("test", {}) or {})
            score = _metric_to_score(metric, val_metrics)

            row.update(
                {
                    "score": score,
                    "val_accuracy": val_metrics.get("accuracy"),
                    "val_roc_auc": val_metrics.get("roc_auc"),
                    "val_log_loss": val_metrics.get("log_loss"),
                    "test_accuracy": test_metrics.get("accuracy"),
                    "test_roc_auc": test_metrics.get("roc_auc"),
                    "test_log_loss": test_metrics.get("log_loss"),
                }
            )
        except Exception as e:
            row["status"] = f"metrics_error({type(e).__name__})"
            row["score"] = None
            row["error"] = str(e)
    else:
        row["score"] = None

    _write_row_csv(results_csv, row, lock=results_lock)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="seq 超参数搜索（最多 4 路并行 GPU）。")
    parser.add_argument("--n-trials", type=int, default=24, help="trial 数量（随机搜索）。")
    parser.add_argument("--max-workers", type=int, default=4, help="并行上限（最多 4）。")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="使用哪些 GPU（逗号分隔）。")
    parser.add_argument("--metric", type=str, default="val_roc_auc", help="选优指标：val_roc_auc/val_accuracy/val_log_loss。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（控制 trial 采样）。")
    parser.add_argument(
        "--env-mode",
        type=str,
        default="device",
        choices=["device", "cuda_visible_devices"],
        help="GPU 绑定方式：device=直接传 --device cuda:N；cuda_visible_devices=设置 CUDA_VISIBLE_DEVICES 并用 cuda:0。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际运行。")

    # 固定为你给的“基准命令”
    parser.add_argument("--baseline-mode", type=str, default="trm_pos")
    parser.add_argument("--image-embedding-type", type=str, default="clip")
    parser.add_argument("--text-embedding-type", type=str, default="clip")
    parser.add_argument("--use-meta", action=argparse.BooleanOptionalAction, default=False)

    # 额外覆盖：为了加速搜索（可选）
    parser.add_argument("--max-epochs", type=int, default=None, help="覆盖 max_epochs（用于加速搜索）。")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="覆盖 early_stop_patience（用于加速搜索）。")
    parser.add_argument("--early-stop-min-epochs", type=int, default=None, help="覆盖 early_stop_min_epochs（用于加速搜索）。")

    args = parser.parse_args()

    n_trials = int(args.n_trials)
    if n_trials <= 0:
        raise ValueError("--n-trials 需要 > 0。")

    gpus = _parse_gpus(args.gpus)
    max_workers = max(1, min(int(args.max_workers), 4, len(gpus)))

    project_root = Path(__file__).resolve().parents[3]
    out_dir = project_root / "experiments" / "seq" / "_hparam_search" / _now_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    meta_json = out_dir / "search_meta.json"

    fixed_args = [
        "--baseline-mode",
        str(args.baseline_mode),
        "--image-embedding-type",
        str(args.image_embedding_type),
        "--text-embedding-type",
        str(args.text_embedding_type),
    ]
    if bool(args.use_meta):
        fixed_args.append("--use-meta")
    else:
        fixed_args.append("--no-use-meta")

    extra_overrides: Dict[str, Any] = {
        "max_epochs": args.max_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_epochs": args.early_stop_min_epochs,
    }

    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": _now_tag(),
                "n_trials": n_trials,
                "max_workers": max_workers,
                "gpus": gpus,
                "metric": str(args.metric),
                "seed": int(args.seed),
                "env_mode": str(args.env_mode),
                "fixed_args": fixed_args,
                "extra_overrides": extra_overrides,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"输出目录：{out_dir.as_posix()}")
    print(f"并行度：{max_workers} | GPUs：{gpus} | 指标：{args.metric}")

    rng = random.Random(int(args.seed))
    trials = [sample_trial(rng, i + 1) for i in range(n_trials)]

    gpu_queue: Queue[int] = Queue()
    for g in gpus:
        gpu_queue.put(int(g))

    results_lock = Lock()
    print_lock = Lock()

    def _worker(trial: Trial) -> Dict[str, Any]:
        gpu_id = int(gpu_queue.get())
        try:
            return _run_one_trial(
                project_root=project_root,
                python_exe=sys.executable,
                fixed_args=fixed_args,
                trial=trial,
                gpu_id=gpu_id,
                metric=str(args.metric),
                results_csv=results_csv,
                results_lock=results_lock,
                print_lock=print_lock,
                env_mode=str(args.env_mode),
                extra_overrides=extra_overrides,
                dry_run=bool(args.dry_run),
            )
        finally:
            gpu_queue.put(gpu_id)

    # 这里用线程池只做“进程调度 + 日志采集”，真正训练仍在子进程里跑（GPU 负载不受 GIL 影响）。
    from concurrent.futures import ThreadPoolExecutor, as_completed

    best_row: Optional[Dict[str, Any]] = None
    best_score = -float("inf")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, t) for t in trials]
        for fut in as_completed(futures):
            row = fut.result()
            score = row.get("score")
            if isinstance(score, (int, float)) and float(score) > float(best_score):
                best_score = float(score)
                best_row = dict(row)
                with (out_dir / "best.json").open("w", encoding="utf-8") as f:
                    json.dump(best_row, f, ensure_ascii=False, indent=2)

    if bool(args.dry_run):
        print("DRY-RUN 完成：已打印全部命令。")
        return 0

    if best_row is None:
        print("未找到可用的 best（可能所有 trial 都失败了）。")
        return 2

    print("搜索完成。")
    print(f"最佳 score={best_score:.6f} | trial_id={best_row.get('trial_id')} | run_dir={best_row.get('run_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
