# -*- coding: utf-8 -*-
"""
Optuna 超参搜索（seq：只针对 trm_pos+meta）。

特点：
- 每个 trial 通过“调用现有 CLI -> 读取 reports/metrics.json -> 返回目标值”的黑盒方式执行；
- 默认目标：最大化 val_f1（可通过 --objective 切换为 val_auc / val_accuracy / test_*）；
- 每个 trial 固定 random_seed，尽量保证可复现；
- 自动汇总：输出 summary.csv（trial_id、params、objective、run_dir、关键 val/test 指标）。

推荐运行方式（从仓库根目录）：
  conda run -n crowdfunding python src/dl/seq/optuna_search.py --device cuda:0 --n-trials 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_optuna() -> None:
    try:
        import optuna  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "未安装 optuna。请先在环境中安装：\n"
            "  pip install optuna\n"
            "然后重试。"
        ) from e


def _read_single_row_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            raise RuntimeError(f"CSV 为空：{path}")
        return dict(row)


def _read_metrics_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    """
    读取 seq/main.py 的输出文件，返回统一扁平结构，便于做 objective 与 summary。

    返回结构（字段可能为 None）：
    {
      "selected_threshold": float|None,
      "val_accuracy": float|None,
      "val_precision": float|None,
      "val_recall": float|None,
      "val_f1": float|None,
      "val_auc": float|None,
      "test_accuracy": float|None,
      "test_precision": float|None,
      "test_recall": float|None,
      "test_f1": float|None,
      "test_auc": float|None,
    }
    """
    metrics_json = run_dir / "reports" / "metrics.json"
    if metrics_json.exists():
        obj = json.loads(metrics_json.read_text(encoding="utf-8"))
        val = dict(obj.get("val", {}) or {})
        test = dict(obj.get("test", {}) or {})
        out = {
            "selected_threshold": obj.get("selected_threshold", None),
            "val_accuracy": val.get("accuracy", None),
            "val_precision": val.get("precision", None),
            "val_recall": val.get("recall", None),
            "val_f1": val.get("f1", None),
            "val_auc": val.get("roc_auc", None),
            "test_accuracy": test.get("accuracy", None),
            "test_precision": test.get("precision", None),
            "test_recall": test.get("recall", None),
            "test_f1": test.get("f1", None),
            "test_auc": test.get("roc_auc", None),
        }
        return out

    # 回退：只读 result.csv（仅包含测试集关键指标）
    result_csv = run_dir / "result.csv"
    if result_csv.exists():
        row = _read_single_row_csv(result_csv)
        out: Dict[str, Any] = {}

        def _to_float(x: Any) -> Optional[float]:
            if x is None:
                return None
            s = str(x).strip()
            if s == "" or s.lower() == "none":
                return None
            return float(s)

        out["selected_threshold"] = _to_float(row.get("threshold"))
        out["test_accuracy"] = _to_float(row.get("test_accuracy"))
        out["test_precision"] = _to_float(row.get("test_precision"))
        out["test_recall"] = _to_float(row.get("test_recall"))
        out["test_f1"] = _to_float(row.get("test_f1"))
        out["test_auc"] = _to_float(row.get("test_auc"))
        out["val_accuracy"] = None
        out["val_precision"] = None
        out["val_recall"] = None
        out["val_f1"] = None
        out["val_auc"] = None
        return out

    raise FileNotFoundError(f"未找到 reports/metrics.json 或 result.csv：{run_dir}")


def _write_summary_row(summary_path: Path, row: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _load_json_arg(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    s = str(value).strip()
    if not s:
        return {}
    if s.startswith("{"):
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("--fixed-overrides 期望为 JSON 对象(dict)。")
        return obj
    p = Path(s)
    if not p.exists():
        raise FileNotFoundError(f"--fixed-overrides 指定的文件不存在：{p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("--fixed-overrides 文件内容期望为 JSON 对象(dict)。")
    return obj


def _suggest_params(trial) -> Dict[str, Any]:
    """
    返回的是 SeqConfig 字段名/值，最终通过环境变量覆盖到训练脚本。
    本搜索只针对 trm_pos+meta，因此不提供 baseline_mode/use_meta 的搜索分支。
    """
    lr = trial.suggest_float("learning_rate_init", 1e-5, 5e-4, log=True)
    wd = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    d_model = trial.suggest_categorical("d_model", [128, 192, 256])
    # nn.TransformerEncoderLayer 要求 d_model % nhead == 0
    valid_heads = [h for h in (2, 4, 8) if int(d_model) % int(h) == 0]
    n_heads = trial.suggest_categorical("transformer_num_heads", valid_heads)
    n_layers = trial.suggest_int("transformer_num_layers", 1, 4)
    ff_ratio = trial.suggest_categorical("ffn_ratio", [2, 4, 6])
    ff_dim = int(ff_ratio) * int(d_model)

    token_dropout = trial.suggest_float("token_dropout", 0.0, 0.35)
    transformer_dropout = trial.suggest_float("transformer_dropout", 0.0, 0.35)
    meta_hidden_dim = trial.suggest_categorical("meta_hidden_dim", [64, 128, 256, 512])
    meta_dropout = trial.suggest_float("meta_dropout", 0.0, 0.7)
    fusion_dropout = trial.suggest_float("fusion_dropout", 0.0, 0.9)

    # fusion_hidden_dim：0 表示自动（2 * fusion_in_dim）
    fusion_hidden_dim = trial.suggest_categorical(
        "fusion_hidden_dim",
        [0, int(d_model), int(2 * d_model), int(4 * d_model)],
    )

    # 早停相关（不建议调太大；否则 trial 太慢）
    early_stop_patience = trial.suggest_int("early_stop_patience", 5, 15)

    # 影响 best_model 选择的指标（最终 objective 仍按 --objective）
    metric_for_best = trial.suggest_categorical("metric_for_best", ["val_accuracy", "val_auc", "val_loss"])

    return {
        "learning_rate_init": float(lr),
        "alpha": float(wd),
        "batch_size": int(batch_size),
        "d_model": int(d_model),
        "transformer_num_heads": int(n_heads),
        "transformer_num_layers": int(n_layers),
        "transformer_dim_feedforward": int(ff_dim),
        "token_dropout": float(token_dropout),
        "transformer_dropout": float(transformer_dropout),
        "meta_hidden_dim": int(meta_hidden_dim),
        "meta_dropout": float(meta_dropout),
        "fusion_hidden_dim": int(fusion_hidden_dim),
        "fusion_dropout": float(fusion_dropout),
        "early_stop_patience": int(early_stop_patience),
        "metric_for_best": str(metric_for_best),
    }


def _build_train_cmd(
    image_embedding_type: str,
    text_embedding_type: str,
    run_name: str,
    device: Optional[str],
    gpu: Optional[int],
) -> list[str]:
    cmd = [
        sys.executable,
        "src/dl/seq/main.py",
        "--baseline-mode",
        "trm_pos",
        "--use-meta",
        "--image-embedding-type",
        str(image_embedding_type),
        "--text-embedding-type",
        str(text_embedding_type),
        "--run-name",
        str(run_name),
    ]
    if device is not None:
        cmd.extend(["--device", str(device)])
    elif gpu is not None:
        cmd.extend(["--gpu", str(int(gpu))])
    return cmd


def _objective_key_to_column(key: str) -> str:
    k = str(key or "").strip().lower()
    if k in {"val_f1", "f1"}:
        return "val_f1"
    if k in {"val_auc", "auc"}:
        return "val_auc"
    if k in {"val_accuracy", "val_acc", "acc", "accuracy"}:
        return "val_accuracy"
    if k in {"test_f1"}:
        return "test_f1"
    if k in {"test_auc"}:
        return "test_auc"
    if k in {"test_accuracy", "test_acc"}:
        return "test_accuracy"
    raise ValueError(
        f"不支持的 objective={key!r}，可选："
        "val_f1/val_auc/val_accuracy/test_f1/test_auc/test_accuracy"
    )


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="seq Optuna 超参搜索（仅 trm_pos+meta；每个 trial 调用现有 CLI 黑盒训练）。")
    p.add_argument("--image-embedding-type", default="clip", choices=["clip", "siglip", "resnet"])
    p.add_argument("--text-embedding-type", default="clip", choices=["bge", "clip", "siglip"])
    p.add_argument("--device", default=None, help="训练设备：auto/cpu/cuda/cuda:0...（优先于 --gpu）")
    p.add_argument("--gpu", type=int, default=None, help="选择第 N 块 GPU（等价于 --device cuda:N）")

    p.add_argument("--n-trials", type=int, default=20, help="trial 数量")
    p.add_argument("--timeout", type=int, default=None, help="总超时（秒），可选")
    p.add_argument("--study-name", default=None, help="study 名称（默认自动生成）")
    p.add_argument("--objective", default="val_f1", help="目标（默认最大化 val_f1）")
    p.add_argument("--random-seed", type=int, default=42, help="固定 random_seed（每个 trial 相同）")
    p.add_argument("--sampler-seed", type=int, default=42, help="Optuna sampler 的随机种子")
    p.add_argument("--fixed-overrides", default=None, help="对所有 trial 生效的 SeqConfig 覆盖项：JSON 字符串或 JSON 文件路径")
    p.add_argument("--fail-value", type=float, default=-1.0, help="trial 失败时返回的目标值（默认 -1.0）")

    p.add_argument("--save-plots", action="store_true", help="是否保存 plots（默认不保存以加速调参）")
    p.add_argument("--keep-trial-logs", action="store_true", help="是否保留每个 trial 的 stdout/stderr 日志文件")
    return p


def main() -> int:
    args = _arg_parser().parse_args()
    _ensure_optuna()
    import optuna

    if args.device is not None and args.gpu is not None:
        raise ValueError("参数冲突：--device 与 --gpu 不能同时使用。")
    if args.gpu is not None and int(args.gpu) < 0:
        raise ValueError("--gpu 需要是非负整数。")

    image_embedding_type = str(args.image_embedding_type).strip().lower()
    text_embedding_type = str(args.text_embedding_type).strip().lower()
    objective_col = _objective_key_to_column(args.objective)

    study_name = (
        str(args.study_name).strip()
        if args.study_name is not None and str(args.study_name).strip()
        else f"seq_trm_pos_meta_{image_embedding_type}_{text_embedding_type}_{objective_col}"
    )

    root = _project_root()
    study_dir = root / "experiments" / "seq" / "optuna" / study_name
    logs_dir = study_dir / "trial_logs"
    study_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = study_dir / "summary.csv"

    fixed_overrides = _load_json_arg(args.fixed_overrides)
    fixed_overrides = dict(fixed_overrides)
    # 强制只搜索 trm_pos+meta
    fixed_overrides["baseline_mode"] = "trm_pos"
    fixed_overrides["use_meta"] = True
    fixed_overrides["random_seed"] = int(args.random_seed)
    fixed_overrides["save_plots"] = bool(args.save_plots)

    db_path = study_dir / "study.db"
    storage_url = f"sqlite:///{db_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=int(args.sampler_seed))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial) -> float:
        trial_id = int(trial.number)
        run_name = f"{study_name}_t{trial_id:04d}"
        run_dir_file = study_dir / "run_dir" / f"trial_{trial_id:04d}.txt"
        run_dir_file.parent.mkdir(parents=True, exist_ok=True)
        if run_dir_file.exists():
            try:
                run_dir_file.unlink()
            except Exception:
                pass

        trial_params = _suggest_params(trial)
        overrides = dict(fixed_overrides)
        overrides.update(trial_params)

        env = dict(os.environ)
        env["SEQ_CFG_OVERRIDES"] = json.dumps(overrides, ensure_ascii=False)
        env["SEQ_RUN_DIR_FILE"] = str(run_dir_file)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONHASHSEED"] = str(int(args.random_seed))

        cmd = _build_train_cmd(
            image_embedding_type=image_embedding_type,
            text_embedding_type=text_embedding_type,
            run_name=run_name,
            device=args.device,
            gpu=args.gpu,
        )

        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        t1 = time.time()

        log_text = (
            "command: "
            + " ".join(cmd)
            + "\n"
            + f"returncode: {proc.returncode}\n"
            + f"elapsed_sec: {t1 - t0:.3f}\n"
            + "\n[stdout]\n"
            + (proc.stdout or "")
            + "\n\n[stderr]\n"
            + (proc.stderr or "")
        )
        trial_log_path = logs_dir / f"trial_{trial_id:04d}.log"
        if bool(args.keep_trial_logs) or proc.returncode != 0:
            trial_log_path.write_text(log_text, encoding="utf-8")

        status = "ok" if proc.returncode == 0 else "fail"
        run_dir_str = ""
        if run_dir_file.exists():
            try:
                run_dir_str = run_dir_file.read_text(encoding="utf-8").strip()
            except Exception:
                run_dir_str = ""

        run_dir = Path(run_dir_str) if run_dir_str else None
        metrics: Dict[str, Any] = {}
        objective_val: Optional[float] = None
        error_msg = ""
        if status == "ok" and run_dir is not None and run_dir.exists():
            try:
                metrics = _read_metrics_from_run_dir(run_dir)
                obj = metrics.get(objective_col, None)
                objective_val = None if obj is None else float(obj)
            except Exception as e:
                status = "fail"
                error_msg = f"{type(e).__name__}: {e}"
        else:
            if run_dir is None:
                error_msg = "未获取到 run_dir（SEQ_RUN_DIR_FILE 为空或不存在）。"
            elif not run_dir.exists():
                error_msg = f"run_dir 不存在：{run_dir}"

        if objective_val is None:
            objective_val = float(args.fail_value)
            if (not trial_log_path.exists()) and not bool(args.keep_trial_logs):
                trial_log_path.write_text(log_text, encoding="utf-8")

        summary_row = {
            "trial_id": trial_id,
            "status": status,
            "objective": str(objective_col),
            "objective_value": float(objective_val),
            "baseline_mode": "trm_pos",
            "use_meta": True,
            "image_embedding_type": str(image_embedding_type),
            "text_embedding_type": str(text_embedding_type),
            "run_dir": "" if run_dir is None else str(run_dir),
            "params_json": json.dumps(dict(trial.params), ensure_ascii=False, sort_keys=True),
            "val_accuracy": metrics.get("val_accuracy", None),
            "val_precision": metrics.get("val_precision", None),
            "val_recall": metrics.get("val_recall", None),
            "val_f1": metrics.get("val_f1", None),
            "val_auc": metrics.get("val_auc", None),
            "test_accuracy": metrics.get("test_accuracy", None),
            "test_precision": metrics.get("test_precision", None),
            "test_recall": metrics.get("test_recall", None),
            "test_f1": metrics.get("test_f1", None),
            "test_auc": metrics.get("test_auc", None),
            "selected_threshold": metrics.get("selected_threshold", None),
            "elapsed_sec": float(t1 - t0),
            "returncode": int(proc.returncode),
            "error": str(error_msg),
        }
        _write_summary_row(summary_path, summary_row)

        # 让 Optuna 的 trial 也记录关键字段，方便后续对比
        trial.set_user_attr("run_dir", summary_row["run_dir"])
        trial.set_user_attr("status", status)
        trial.set_user_attr("val_f1", metrics.get("val_f1", None))
        trial.set_user_attr("test_f1", metrics.get("test_f1", None))
        return float(objective_val)

    study.optimize(objective, n_trials=int(args.n_trials), timeout=args.timeout)

    best = {
        "study_name": study_name,
        "baseline_mode": "trm_pos",
        "use_meta": True,
        "objective": objective_col,
        "best_value": float(study.best_value) if study.best_trial is not None else None,
        "best_params": dict(study.best_params) if study.best_trial is not None else {},
        "best_run_dir": study.best_trial.user_attrs.get("run_dir") if study.best_trial is not None else None,
    }
    (study_dir / "best.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：study={study_name} | best_value={best.get('best_value')} | best_run_dir={best.get('best_run_dir')}")
    print(f"summary.csv：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

