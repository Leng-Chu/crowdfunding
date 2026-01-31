# -*- coding: utf-8 -*-
"""
Optuna 超参搜索（res）。

特点：
- 每个 trial 通过“调用现有 CLI -> 读取 reports/result -> 返回目标值”的黑盒方式执行；
    - 默认目标：最大化 test_auc（来自 result.csv；若缺失则回退到 reports/metrics.json）；
- 每个 trial 固定 random_seed，尽量保证可复现；
- 自动汇总：输出 summary.csv（trial_id、params、objective、run_dir、关键 test 指标）。

推荐运行方式（从仓库根目录）：
  conda run -n crowdfunding python src/dl/res/optuna_search.py --device cuda:3 --n-trials 200 --sampler-seed 3 --random-seed 72
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
    返回结构：
    {
      "threshold": float|None,
      "test_accuracy": float|None,
      "test_precision": float|None,
      "test_recall": float|None,
      "test_f1": float|None,
      "test_auc": float|None,
    }
    """
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

        out["threshold"] = _to_float(row.get("threshold"))
        out["test_accuracy"] = _to_float(row.get("test_accuracy"))
        out["test_precision"] = _to_float(row.get("test_precision"))
        out["test_recall"] = _to_float(row.get("test_recall"))
        out["test_f1"] = _to_float(row.get("test_f1"))
        out["test_auc"] = _to_float(row.get("test_auc"))
        return out

    metrics_json = run_dir / "reports" / "metrics.json"
    if metrics_json.exists():
        obj = json.loads(metrics_json.read_text(encoding="utf-8"))
        test = dict(obj.get("test", {}) or {})
        out = {
            "threshold": obj.get("selected_threshold", None),
            "test_accuracy": test.get("accuracy", None),
            "test_precision": test.get("precision", None),
            "test_recall": test.get("recall", None),
            "test_f1": test.get("f1", None),
            "test_auc": test.get("roc_auc", None),
        }
        return out

    raise FileNotFoundError(f"未找到 result.csv 或 reports/metrics.json：{run_dir}")


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
    Optuna 全量搜索空间（默认）。
    注意：这里返回的是 ResConfig 字段名/值，最终会通过环境变量覆盖到训练脚本。
    """
    lr = trial.suggest_float("learning_rate_init", 1e-5, 8e-4, log=True)
    wd = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)

    d_model = trial.suggest_categorical("d_model", [192, 256])
    key_dim = trial.suggest_categorical("key_dim", [32, 64, 96])
    # v_key 的有界缩放：建议 0.05~0.1 左右起步
    key_alpha_init = trial.suggest_float("key_alpha_init", 0.03, 0.20)
    # nn.TransformerEncoderLayer 要求 d_model % nhead == 0
    valid_heads = [h for h in (2, 4, 8) if int(d_model) % int(h) == 0]
    if not valid_heads:
        valid_heads = [1]
    n_heads = trial.suggest_categorical("transformer_num_heads", valid_heads)
    n_layers = trial.suggest_int("transformer_num_layers", 1, 4)
    ff_ratio = trial.suggest_categorical("ffn_ratio", [2, 4, 6])
    ff_dim = int(ff_ratio) * int(d_model)

    token_dropout = trial.suggest_float("token_dropout", 0.0, 0.35)
    transformer_dropout = trial.suggest_float("transformer_dropout", 0.0, 0.35)
    key_dropout = trial.suggest_float("key_dropout", 0.1, 0.7)
    meta_hidden_dim = trial.suggest_categorical("meta_hidden_dim", [64, 128, 256])
    meta_dropout = trial.suggest_float("meta_dropout", 0.0, 0.6)

    out = {
        "learning_rate_init": float(lr),
        "alpha": float(wd),
        "d_model": int(d_model),
        "key_dim": int(key_dim),
        "key_alpha_init": float(key_alpha_init),
        "transformer_num_heads": int(n_heads),
        "transformer_num_layers": int(n_layers),
        "transformer_dim_feedforward": int(ff_dim),
        "token_dropout": float(token_dropout),
        "transformer_dropout": float(transformer_dropout),
        "key_dropout": float(key_dropout),
        "meta_hidden_dim": int(meta_hidden_dim),
        "meta_dropout": float(meta_dropout),
    }

    base_dropout = trial.suggest_float("base_dropout", 0.2, 0.7)
    prior_dropout = trial.suggest_float("prior_dropout", 0.2, 0.7)
    base_hidden_dim = trial.suggest_categorical("base_hidden_dim", [256, 512, 768])
    prior_hidden_dim = trial.suggest_categorical("prior_hidden_dim", [64, 128, 256, 512])

    # 参考 config.py 的经验建议，避免残差抑制强度搜索得过于极端
    delta_scale_max = trial.suggest_float("delta_scale_max", 0.10, 0.90)
    residual_logit_max = trial.suggest_float("residual_logit_max", 0.80, 3.50)
    residual_gate_scale_init = trial.suggest_float("residual_gate_scale_init", 0.2, 5.0, log=True)
    residual_gate_bias_init = trial.suggest_float("residual_gate_bias_init", -2.0, 2.0)

    out.update(
        {
            "base_dropout": float(base_dropout),
            "prior_dropout": float(prior_dropout),
            "base_hidden_dim": int(base_hidden_dim),
            "prior_hidden_dim": int(prior_hidden_dim),
            "delta_scale_max": float(delta_scale_max),
            "residual_logit_max": float(residual_logit_max),
            "residual_gate_scale_init": float(residual_gate_scale_init),
            "residual_gate_bias_init": float(residual_gate_bias_init),
        }
    )

    return out


def _build_train_cmd(
    baseline_mode: str,
    image_embedding_type: str,
    text_embedding_type: str,
    run_name: str,
    seed: int,
    device: Optional[str],
    gpu: Optional[int],
) -> list[str]:
    cmd = [
        sys.executable,
        "src/dl/res/main.py",
        "--baseline-mode",
        str(baseline_mode),
        "--image-embedding-type",
        str(image_embedding_type),
        "--text-embedding-type",
        str(text_embedding_type),
        "--run-name",
        str(run_name),
        "--seed",
        str(int(seed)),
    ]
    if device is not None:
        cmd.extend(["--device", str(device)])
    elif gpu is not None:
        cmd.extend(["--gpu", str(int(gpu))])
    return cmd


def _objective_key_to_column(key: str) -> str:
    k = str(key or "").strip().lower()
    if k in {"test_f1", "f1"}:
        return "test_f1"
    if k in {"test_auc", "auc"}:
        return "test_auc"
    if k in {"test_accuracy", "acc", "accuracy"}:
        return "test_accuracy"
    raise ValueError(f"不支持的 objective={key!r}，可选：test_f1/test_auc/test_accuracy")


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="res Optuna 超参搜索（每个 trial 调用现有 CLI 黑盒训练）。")
    p.add_argument("--baseline-mode", default="res", choices=["res"], help="仅支持 res（脚本只关注 res 超参）。")
    p.add_argument("--image-embedding-type", default="clip", choices=["clip", "siglip", "resnet"])
    p.add_argument("--text-embedding-type", default="clip", choices=["bge", "clip", "siglip"])
    p.add_argument("--device", default=None, help="训练设备：auto/cpu/cuda/cuda:0...（优先于 --gpu）")
    p.add_argument("--gpu", type=int, default=None, help="选择第 N 块 GPU（等价于 --device cuda:N）")

    p.add_argument("--n-trials", type=int, default=100, help="trial 数量")
    p.add_argument("--timeout", type=int, default=None, help="总超时（秒），可选")
    p.add_argument("--study-name", default=None, help="study 名称（默认自动生成）")
    p.add_argument("--objective", default="test_auc", help="目标（默认最大化 test_auc）")
    p.add_argument("--random-seed", type=int, default=42, help="固定 random_seed（每个 trial 相同）")
    p.add_argument("--sampler-seed", type=int, default=42, help="Optuna sampler 的随机种子")
    p.add_argument("--fixed-overrides", default=None, help="对所有 trial 生效的 ResConfig 覆盖项：JSON 字符串或 JSON 文件路径")
    p.add_argument("--fail-value", type=float, default=-1.0, help="trial 失败时返回的目标值（默认 -1.0）")
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

    baseline_mode = str(args.baseline_mode).strip().lower()
    image_embedding_type = str(args.image_embedding_type).strip().lower()
    text_embedding_type = str(args.text_embedding_type).strip().lower()
    objective_col = _objective_key_to_column(args.objective)

    num = "3"
    study_name = (
        str(args.study_name).strip()
        if args.study_name is not None and str(args.study_name).strip()
        else f"res_{objective_col}_{num}"
    )
    root = _project_root()
    study_dir = root / "experiments" / "res" / num / study_name
    logs_dir = study_dir / "trial_logs"
    study_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = study_dir / "summary.csv"

    fixed_overrides = _load_json_arg(args.fixed_overrides)
    fixed_overrides = dict(fixed_overrides)
    fixed_overrides["random_seed"] = int(args.random_seed)
    fixed_overrides["save_plots"] = True
    fixed_overrides["debug_residual_stats"] = False

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
        env["RES_CFG_OVERRIDES"] = json.dumps(overrides, ensure_ascii=False)
        env["RES_RUN_DIR_FILE"] = str(run_dir_file)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONHASHSEED"] = str(int(args.random_seed))

        cmd = _build_train_cmd(
            baseline_mode=baseline_mode,
            image_embedding_type=image_embedding_type,
            text_embedding_type=text_embedding_type,
            run_name=run_name,
            seed=int(args.random_seed),
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
                error_msg = "未获取到 run_dir（RES_RUN_DIR_FILE 为空或不存在）。"
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
            "baseline_mode": str(baseline_mode),
            "image_embedding_type": str(image_embedding_type),
            "text_embedding_type": str(text_embedding_type),
            "run_dir": "" if run_dir is None else str(run_dir),
            "params_json": json.dumps(dict(trial.params), ensure_ascii=False, sort_keys=True),
            "test_accuracy": metrics.get("test_accuracy", None),
            "test_precision": metrics.get("test_precision", None),
            "test_recall": metrics.get("test_recall", None),
            "test_f1": metrics.get("test_f1", None),
            "test_auc": metrics.get("test_auc", None),
            "threshold": metrics.get("threshold", None),
            "elapsed_sec": float(t1 - t0),
            "returncode": int(proc.returncode),
            "error": str(error_msg),
        }
        _write_summary_row(summary_path, summary_row)

        trial.set_user_attr("run_dir", summary_row["run_dir"])
        trial.set_user_attr("status", status)
        trial.set_user_attr("test_f1", metrics.get("test_f1", None))
        return float(objective_val)

    study.optimize(objective, n_trials=int(args.n_trials), timeout=args.timeout)

    best = {
        "study_name": study_name,
        "baseline_mode": baseline_mode,
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
