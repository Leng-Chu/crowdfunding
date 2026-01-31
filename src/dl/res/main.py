# -*- coding: utf-8 -*-
"""
res 主程序入口（Residual Baselines）：

说明：
- 本目录代码可独立运行，不 import 其它 `src/dl/*` 子模块的代码
- 工程规范完全参考 `docs/dl_guidelines.md`（best checkpoint / 阈值选择 / 产物结构）

运行（在项目根目录）：
- 使用默认配置：
  `python src/dl/res/main.py`
- 覆盖常用参数（只覆盖少量配置项，其余请改 config.py）：
  `python src/dl/res/main.py --baseline-mode res --image-embedding-type clip --text-embedding-type bge --device cuda:0`
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import platform
import sys
from dataclasses import replace
from pathlib import Path

from config import ResConfig
from env_overrides import apply_config_overrides_from_env


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="res 训练入口（支持命令行覆盖少量常用配置）。")
    parser.add_argument("--run-name", default=None, help="实验名称后缀，用于产物目录命名。")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子（覆盖 config.py 的 random_seed）")
    parser.add_argument(
        "--baseline-mode",
        default=None,
        choices=["mlp", "res"],
        help="实验组：mlp / res。",
    )
    parser.add_argument(
        "--use-first-impression",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="是否使用第一印象分支（仅对 baseline_mode=mlp 生效）。",
    )
    parser.add_argument(
        "--image-embedding-type",
        default=None,
        choices=["clip", "siglip", "resnet"],
        help="图片嵌入类型（默认读取 config.py）。",
    )
    parser.add_argument(
        "--text-embedding-type",
        default=None,
        choices=["bge", "clip", "siglip"],
        help="文本嵌入类型（默认读取 config.py）。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="训练设备：auto/cpu/cuda/cuda:0/cuda:1 ...（默认读取 config.py）。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="选择第 N 张 GPU（等价于 --device cuda:N；与 --device 互斥）。",
    )
    args = parser.parse_args(argv)

    if args.device is not None and args.gpu is not None:
        raise ValueError("参数冲突：--device 与 --gpu 不能同时使用。")
    if args.gpu is not None and int(args.gpu) < 0:
        raise ValueError("--gpu 需要是非负整数。")

    if args.run_name is not None and not str(args.run_name).strip():
        args.run_name = None
    return args


def _save_predictions_csv(
    save_path: Path,
    project_ids: list[str],
    y_true,
    y_prob,
    threshold: float,
) -> None:
    import numpy as np
    import pandas as pd

    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= float(threshold)).astype(int)

    df = pd.DataFrame(
        {
            "project_id": project_ids,
            "y_true": y_true.astype(int),
            "y_prob": y_prob.astype(float),
            "y_pred": y_pred.astype(int),
        }
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8")


def _save_single_row_csv(save_path: Path, row: dict) -> None:
    """保存“单行结果”CSV（包含表头 + 1 条数据）。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = _parse_args()
    cfg = ResConfig()

    # Windows 下 torch 与 numpy/pandas 的 OpenMP 运行库可能有冲突；
    # 先 import torch，可避免部分环境出现 DLL 初始化失败（对齐 seq/gate 的做法）。
    import torch  # noqa: F401

    from data import prepare_res_data
    from model import build_res_model
    from train_eval import evaluate_res_split, train_res_with_early_stopping
    from utils import (
        compute_binary_metrics,
        find_best_f1_threshold,
        make_run_dirs,
        plot_history,
        plot_roc,
        save_json,
        save_text,
        set_global_seed,
        setup_logger,
    )

    if args.run_name is not None:
        cfg = replace(cfg, run_name=str(args.run_name))
    if args.baseline_mode is not None:
        cfg = replace(cfg, baseline_mode=str(args.baseline_mode))
    if getattr(args, "use_first_impression", None) is not None:
        cfg = replace(cfg, use_first_impression=bool(args.use_first_impression))
    if args.image_embedding_type is not None:
        cfg = replace(cfg, image_embedding_type=str(args.image_embedding_type))
    if args.text_embedding_type is not None:
        cfg = replace(cfg, text_embedding_type=str(args.text_embedding_type))

    if args.device is not None:
        cfg = replace(cfg, device=str(args.device))
    elif args.gpu is not None:
        cfg = replace(cfg, device=f"cuda:{int(args.gpu)}")

    # 允许通过环境变量覆盖少量超参（主要用于自动化脚本；不影响默认训练）。
    cfg = apply_config_overrides_from_env(cfg)

    if args.seed is not None:
        cfg = replace(cfg, random_seed=int(args.seed))

    baseline_mode = str(getattr(cfg, "baseline_mode", "res")).strip().lower()
    if baseline_mode not in {"mlp", "res"}:
        raise ValueError(f"不支持的 baseline_mode={baseline_mode!r}，可选：mlp/res")
    meta_enabled = True
    mode = baseline_mode
    if baseline_mode == "mlp":
        use_key = bool(getattr(cfg, "use_first_impression", True))
        mode += "-use-key" if use_key else "-no-key"

    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / cfg.data_csv
    projects_root = project_root / cfg.projects_root

    experiment_root = project_root / cfg.experiment_root / mode
    experiment_root.mkdir(parents=True, exist_ok=True)

    run_id, artifacts_dir, reports_dir, plots_dir = make_run_dirs(experiment_root, run_name=cfg.run_name)
    run_dir = reports_dir.parent

    # 供外部脚本可靠地拿到本次 run_dir。
    run_dir_file = str(os.getenv("RES_RUN_DIR_FILE", "") or "").strip()
    if run_dir_file:
        try:
            p = Path(run_dir_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(run_dir), encoding="utf-8")
        except Exception as e:
            print(f"写入 RES_RUN_DIR_FILE 失败：{e}", file=sys.stderr)

    logger = setup_logger(run_dir / "train.log")

    logger.info(
        "模式=%s | run_id=%s | baseline_mode=%s | meta_enabled=%s | use_first_impression=%s",
        mode,
        run_id,
        baseline_mode,
        bool(meta_enabled),
        bool(getattr(cfg, "use_first_impression", True)),
    )
    logger.info("python=%s | 平台=%s", sys.version.replace("\n", " "), platform.platform())
    logger.info("data_csv=%s", str(csv_path))
    logger.info("projects_root=%s", str(projects_root))
    logger.info("device=%s", str(getattr(cfg, "device", "auto")))
    logger.info("random_seed=%d", int(getattr(cfg, "random_seed", 0)))
    logger.info(
        "embedding：image=%s text=%s | max_seq_len=%d trunc=%s | seq=trm_pos(sin)",
        cfg.image_embedding_type,
        cfg.text_embedding_type,
        int(getattr(cfg, "max_seq_len", 0)),
        str(getattr(cfg, "truncation_strategy", "first")),
    )

    set_global_seed(cfg.random_seed)

    prepared = prepare_res_data(
        csv_path=csv_path,
        projects_root=projects_root,
        cfg=cfg,
        logger=logger,
    )
    logger.info(
        "数据集：train=%d val=%d test=%d | meta_dim=%d | img_dim=%d txt_dim=%d | max_seq_len=%d",
        int(prepared.y_train.shape[0]),
        int(prepared.y_val.shape[0]),
        int(prepared.y_test.shape[0]),
        int(prepared.meta_dim),
        int(prepared.image_embedding_dim),
        int(prepared.text_embedding_dim),
        int(prepared.max_seq_len),
    )
    if prepared.stats:
        logger.info("统计信息：%s", prepared.stats)

    if prepared.preprocessor is not None:
        with (artifacts_dir / "preprocessor.pkl").open("wb") as f:
            pickle.dump(prepared.preprocessor, f)
        if prepared.feature_names:
            save_text(prepared.feature_names, artifacts_dir / "feature_names.txt")

    try:
        import pandas as pd

        rows = []
        for pid, y in zip(prepared.train_project_ids, prepared.y_train):
            rows.append({"project_id": pid, "split": "train", "state": int(y)})
        for pid, y in zip(prepared.val_project_ids, prepared.y_val):
            rows.append({"project_id": pid, "split": "val", "state": int(y)})
        for pid, y in zip(prepared.test_project_ids, prepared.y_test):
            rows.append({"project_id": pid, "split": "test", "state": int(y)})
        pd.DataFrame(rows).to_csv(reports_dir / "splits.csv", index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("保存 splits.csv 失败：%s", e)

    model = build_res_model(
        cfg,
        meta_input_dim=int(prepared.meta_dim),
        image_embedding_dim=int(prepared.image_embedding_dim),
        text_embedding_dim=int(prepared.text_embedding_dim),
    )

    save_json(
        {
            "run_id": run_id,
            "mode": mode,
            "baseline_mode": baseline_mode,
            "meta_enabled": bool(meta_enabled),
            "meta_dim": int(prepared.meta_dim),
            "image_embedding_dim": int(prepared.image_embedding_dim),
            "text_embedding_dim": int(prepared.text_embedding_dim),
            "max_seq_len": int(prepared.max_seq_len),
            **cfg.to_dict(),
        },
        reports_dir / "config.json",
    )

    best_state, best_epoch, history, best_info = train_res_with_early_stopping(
        model,
        X_meta_train=prepared.X_meta_train,
        title_blurb_train=prepared.title_blurb_train,
        cover_train=prepared.cover_train,
        X_img_train=prepared.X_img_train,
        X_txt_train=prepared.X_txt_train,
        seq_type_train=prepared.seq_type_train,
        seq_attr_train=prepared.seq_attr_train,
        seq_mask_train=prepared.seq_mask_train,
        y_train=prepared.y_train,
        train_project_ids=prepared.train_project_ids,
        X_meta_val=prepared.X_meta_val,
        title_blurb_val=prepared.title_blurb_val,
        cover_val=prepared.cover_val,
        X_img_val=prepared.X_img_val,
        X_txt_val=prepared.X_txt_val,
        seq_type_val=prepared.seq_type_val,
        seq_attr_val=prepared.seq_attr_val,
        seq_mask_val=prepared.seq_mask_val,
        y_val=prepared.y_val,
        val_project_ids=prepared.val_project_ids,
        cfg=cfg,
        logger=logger,
    )

    model.load_state_dict(best_state)
    best_model = model

    try:
        import pandas as pd

        pd.DataFrame(history).to_csv(reports_dir / "history.csv", index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("保存 history.csv 失败：%s", e)

    train_out = evaluate_res_split(
        best_model,
        X_meta=prepared.X_meta_train,
        title_blurb=prepared.title_blurb_train,
        cover=prepared.cover_train,
        X_img=prepared.X_img_train,
        X_txt=prepared.X_txt_train,
        seq_type=prepared.seq_type_train,
        seq_attr=prepared.seq_attr_train,
        seq_mask=prepared.seq_mask_train,
        y=prepared.y_train,
        project_ids=prepared.train_project_ids,
        cfg=cfg,
    )
    val_out = evaluate_res_split(
        best_model,
        X_meta=prepared.X_meta_val,
        title_blurb=prepared.title_blurb_val,
        cover=prepared.cover_val,
        X_img=prepared.X_img_val,
        X_txt=prepared.X_txt_val,
        seq_type=prepared.seq_type_val,
        seq_attr=prepared.seq_attr_val,
        seq_mask=prepared.seq_mask_val,
        y=prepared.y_val,
        project_ids=prepared.val_project_ids,
        cfg=cfg,
    )
    test_out = evaluate_res_split(
        best_model,
        X_meta=prepared.X_meta_test,
        title_blurb=prepared.title_blurb_test,
        cover=prepared.cover_test,
        X_img=prepared.X_img_test,
        X_txt=prepared.X_txt_test,
        seq_type=prepared.seq_type_test,
        seq_attr=prepared.seq_attr_test,
        seq_mask=prepared.seq_mask_test,
        y=prepared.y_test,
        project_ids=prepared.test_project_ids,
        cfg=cfg,
    )

    best_threshold, _best_val_f1 = find_best_f1_threshold(prepared.y_val, val_out["prob"])
    train_metrics = compute_binary_metrics(prepared.y_train, train_out["prob"], threshold=float(best_threshold))
    val_metrics = compute_binary_metrics(prepared.y_val, val_out["prob"], threshold=float(best_threshold))
    test_metrics = compute_binary_metrics(prepared.y_test, test_out["prob"], threshold=float(best_threshold))
    logger.info("阈值选择：best_threshold=%.6f（val_f1=%.6f）", float(best_threshold), float(val_metrics["f1"]))

    results = {
        "run_id": run_id,
        "best_info": best_info,
        "selected_threshold": float(best_threshold),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(results, reports_dir / "metrics.json")

    # 保存可复现产物（与 docs/dl_guidelines.md 对齐）
    import torch

    delta_scale_raw_val = None
    delta_scale_eff_val = None
    gate_bias_val = None
    gate_scale_val = None
    try:
        if hasattr(best_model, "delta_scale_raw"):
            delta_scale_raw_val = float(getattr(best_model, "delta_scale_raw").detach().cpu().item())
        if hasattr(best_model, "effective_delta_scale"):
            delta_scale_eff_val = float(best_model.effective_delta_scale().detach().cpu().item())
        if hasattr(best_model, "gate_bias") and getattr(best_model, "gate_bias") is not None:
            gate_bias_val = float(getattr(best_model, "gate_bias").detach().cpu().item())
        if hasattr(best_model, "gate_scale_raw") and getattr(best_model, "gate_scale_raw") is not None:
            import torch.nn.functional as F

            gate_scale_val = float(F.softplus(getattr(best_model, "gate_scale_raw")).detach().cpu().item())
    except Exception:
        delta_scale_raw_val = None
        delta_scale_eff_val = None
        gate_bias_val = None
        gate_scale_val = None

    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": int(best_epoch),
            "best_val_auc": best_info.get("best_val_auc", None),
            "best_val_log_loss": best_info.get("best_val_log_loss", None),
            "metric_for_best": best_info.get("metric_for_best", None),
            "tie_breaker": best_info.get("tie_breaker", None),
            "best_threshold": float(best_threshold),
            "baseline_mode": baseline_mode,
            "meta_enabled": bool(meta_enabled),
            "meta_dim": int(prepared.meta_dim),
            "image_embedding_dim": int(prepared.image_embedding_dim),
            "text_embedding_dim": int(prepared.text_embedding_dim),
            "image_embedding_type": cfg.image_embedding_type,
            "text_embedding_type": cfg.text_embedding_type,
            "max_seq_len": int(prepared.max_seq_len),
            "d_model": int(getattr(cfg, "d_model", 256)),
            "transformer_num_layers": int(getattr(cfg, "transformer_num_layers", 0)),
            "transformer_num_heads": int(getattr(cfg, "transformer_num_heads", 0)),
            "transformer_dim_feedforward": int(getattr(cfg, "transformer_dim_feedforward", 0)),
            "transformer_dropout": float(getattr(cfg, "transformer_dropout", 0.0)),
            "meta_hidden_dim": int(getattr(cfg, "meta_hidden_dim", 0)),
            "meta_dropout": float(getattr(cfg, "meta_dropout", 0.0)),
            "fusion_hidden_dim": int(getattr(cfg, "fusion_hidden_dim", 0)),
            "fusion_dropout": float(getattr(cfg, "fusion_dropout", 0.0)),
            "base_hidden_dim": int(getattr(cfg, "base_hidden_dim", 0)),
            "base_dropout": float(getattr(cfg, "base_dropout", 0.0)),
            "prior_hidden_dim": int(getattr(cfg, "prior_hidden_dim", 0)),
            "prior_dropout": float(getattr(cfg, "prior_dropout", 0.0)),
            "delta_scale_init": float(getattr(cfg, "delta_scale_init", 0.0)),
            "delta_scale_max": float(getattr(cfg, "delta_scale_max", 0.0)),
            "residual_logit_max": float(getattr(cfg, "residual_logit_max", 0.0)),
            "residual_gate_mode": str(getattr(cfg, "residual_gate_mode", "")),
            "residual_gate_scale_init": float(getattr(cfg, "residual_gate_scale_init", 0.0)),
            "residual_gate_bias_init": float(getattr(cfg, "residual_gate_bias_init", 0.0)),
            "residual_detach_base_in_gate": bool(getattr(cfg, "residual_detach_base_in_gate", True)),
            "delta_scale_raw_value": delta_scale_raw_val,
            "delta_scale_value": delta_scale_eff_val,
            "gate_bias_value": gate_bias_val,
            "gate_scale_value": gate_scale_val,
            "missing_strategy": str(getattr(cfg, "missing_strategy", "")),
            "truncation_strategy": str(getattr(cfg, "truncation_strategy", "")),
        },
        artifacts_dir / "model.pt",
    )

    try:
        _save_predictions_csv(
            reports_dir / "predictions_val.csv",
            prepared.val_project_ids,
            prepared.y_val,
            val_out["prob"],
            threshold=float(best_threshold),
        )
        _save_predictions_csv(
            reports_dir / "predictions_test.csv",
            prepared.test_project_ids,
            prepared.y_test,
            test_out["prob"],
            threshold=float(best_threshold),
        )
    except Exception as e:
        logger.warning("保存预测 CSV 失败：%s", e)

    if cfg.save_plots:
        plot_history(history, plots_dir / "history.png")
        plot_roc(prepared.y_val, val_out["prob"], plots_dir / "roc_val.png")
        plot_roc(prepared.y_test, test_out["prob"], plots_dir / "roc_test.png")

    try:
        _save_single_row_csv(
            run_dir / "result.csv",
            {
                "mode": mode,
                "baseline_mode": baseline_mode,
                "image_embedding_type": cfg.image_embedding_type,
                "text_embedding_type": cfg.text_embedding_type,
                "threshold": float(best_threshold),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_precision": test_metrics.get("precision"),
                "test_recall": test_metrics.get("recall"),
                "test_f1": test_metrics.get("f1"),
                "test_auc": test_metrics.get("roc_auc"),
            },
        )
    except Exception as e:
        logger.warning("保存单行结果 CSV 失败：%s", e)

    logger.info("完成：产物已保存到 %s", str(run_dir))
    logger.info("测试集指标：%s", test_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
