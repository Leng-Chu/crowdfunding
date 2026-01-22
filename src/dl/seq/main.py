# -*- coding: utf-8 -*-
"""
seq 主程序入口（Chapter 1：图文内容块序列建模）：

说明：
- 本目录代码不依赖 `src/dl/mlp` 的训练代码，可独立运行
- 工程行为与输出结构对齐 mlp baseline，便于横向对比

运行（在项目根目录）：
- 使用默认配置：
  `conda run -n crowdfunding python src/dl/seq/main.py`
- 覆盖常用参数（只覆盖少量配置项，其余请改 config.py）：
  `conda run -n crowdfunding python src/dl/seq/main.py --baseline-mode trm_pos --use-meta --image-embedding-type clip --text-embedding-type clip --device cuda:0`
"""

from __future__ import annotations

import argparse
import csv
import pickle
import platform
import sys
from dataclasses import replace
from pathlib import Path

from config import SeqConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="seq 训练入口（支持命令行覆盖少量常用配置）。")
    parser.add_argument("--run-name", default=None, help="实验名称后缀，用于产物目录命名。")
    parser.add_argument(
        "--baseline-mode",
        default=None,
        choices=["set_mean", "set_attn", "trm_no_pos", "trm_pos", "trm_pos_shuffled"],
        help="实验组：仅修改该参数即可切换 baseline（其余配置保持一致）。",
    )
    parser.add_argument(
        "--use-meta",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="是否启用 meta 分支（出现该参数时，以命令行覆盖配置）。",
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

    # -----------------------------
    # 常用超参覆盖（用于批量实验/超参搜索）
    # -----------------------------
    parser.add_argument("--learning-rate", type=float, default=None, help="初始学习率（覆盖 learning_rate_init）。")
    parser.add_argument("--weight-decay", type=float, default=None, help="权重衰减（覆盖 alpha）。")
    parser.add_argument("--batch-size", type=int, default=None, help="batch_size（注意显存占用）。")

    parser.add_argument("--fusion-dropout", type=float, default=None, help="分类头 dropout（覆盖 fusion_dropout）。")
    parser.add_argument("--token-dropout", type=float, default=None, help="token encoder dropout（覆盖 token_dropout）。")
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=None,
        help="Transformer dropout（覆盖 transformer_dropout）。",
    )
    parser.add_argument("--max-grad-norm", type=float, default=None, help="梯度裁剪阈值（覆盖 max_grad_norm）。")

    parser.add_argument("--max-epochs", type=int, default=None, help="最大训练轮数（覆盖 max_epochs）。")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="早停耐心值（覆盖 early_stop_patience）。")
    parser.add_argument("--early-stop-min-epochs", type=int, default=None, help="早停最小 epoch（覆盖 early_stop_min_epochs）。")

    parser.add_argument("--lr-scheduler-patience", type=int, default=None, help="LR scheduler patience（覆盖 lr_scheduler_patience）。")
    parser.add_argument("--lr-scheduler-factor", type=float, default=None, help="LR scheduler factor（覆盖 lr_scheduler_factor）。")
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=None, help="LR scheduler min_lr（覆盖 lr_scheduler_min_lr）。")
    parser.add_argument(
        "--reset-early-stop-on-lr-change",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="学习率下降时是否重置早停计数（覆盖 reset_early_stop_on_lr_change）。",
    )

    parser.add_argument("--random-seed", type=int, default=None, help="随机种子（覆盖 random_seed）。")
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
    cfg = SeqConfig()

    # Windows 下 torch 与 numpy/pandas 的 OpenMP 运行库可能有冲突；
    # 先 import torch，可避免部分环境出现 DLL 初始化失败（对齐 mlp/main.py 的做法）。
    import torch  # noqa: F401

    from data import prepare_seq_data
    from model import build_seq_model
    from train_eval import evaluate_seq_split, train_seq_with_early_stopping
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
    if args.use_meta is not None:
        cfg = replace(cfg, use_meta=bool(args.use_meta))
    if args.image_embedding_type is not None:
        cfg = replace(cfg, image_embedding_type=str(args.image_embedding_type))
    if args.text_embedding_type is not None:
        cfg = replace(cfg, text_embedding_type=str(args.text_embedding_type))

    if args.device is not None:
        cfg = replace(cfg, device=str(args.device))
    elif args.gpu is not None:
        cfg = replace(cfg, device=f"cuda:{int(args.gpu)}")

    # 超参覆盖（仅在提供参数时生效）
    if args.learning_rate is not None:
        cfg = replace(cfg, learning_rate_init=float(args.learning_rate))
    if args.weight_decay is not None:
        cfg = replace(cfg, alpha=float(args.weight_decay))
    if args.batch_size is not None:
        cfg = replace(cfg, batch_size=int(args.batch_size))

    if args.fusion_dropout is not None:
        cfg = replace(cfg, fusion_dropout=float(args.fusion_dropout))
    if args.token_dropout is not None:
        cfg = replace(cfg, token_dropout=float(args.token_dropout))
    if args.transformer_dropout is not None:
        cfg = replace(cfg, transformer_dropout=float(args.transformer_dropout))
    if args.max_grad_norm is not None:
        cfg = replace(cfg, max_grad_norm=float(args.max_grad_norm))

    if args.max_epochs is not None:
        cfg = replace(cfg, max_epochs=int(args.max_epochs))
    if args.early_stop_patience is not None:
        cfg = replace(cfg, early_stop_patience=int(args.early_stop_patience))
    if args.early_stop_min_epochs is not None:
        cfg = replace(cfg, early_stop_min_epochs=int(args.early_stop_min_epochs))

    if args.lr_scheduler_patience is not None:
        cfg = replace(cfg, lr_scheduler_patience=int(args.lr_scheduler_patience))
    if args.lr_scheduler_factor is not None:
        cfg = replace(cfg, lr_scheduler_factor=float(args.lr_scheduler_factor))
    if args.lr_scheduler_min_lr is not None:
        cfg = replace(cfg, lr_scheduler_min_lr=float(args.lr_scheduler_min_lr))
    if args.reset_early_stop_on_lr_change is not None:
        cfg = replace(cfg, reset_early_stop_on_lr_change=bool(args.reset_early_stop_on_lr_change))

    if args.random_seed is not None:
        cfg = replace(cfg, random_seed=int(args.random_seed))

    baseline_mode = str(getattr(cfg, "baseline_mode", "set_mean")).strip().lower()
    mode = baseline_mode + ("+meta" if bool(getattr(cfg, "use_meta", False)) else "")

    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / cfg.data_csv
    projects_root = project_root / cfg.projects_root
    cache_dir = project_root / cfg.cache_dir

    experiment_root = project_root / cfg.experiment_root / mode
    experiment_root.mkdir(parents=True, exist_ok=True)

    run_id, artifacts_dir, reports_dir, plots_dir = make_run_dirs(experiment_root, run_name=cfg.run_name)
    run_dir = reports_dir.parent

    logger = setup_logger(run_dir / "train.log")

    logger.info("模式=%s | run_id=%s | baseline_mode=%s | use_meta=%s", mode, run_id, baseline_mode, bool(cfg.use_meta))
    logger.info("python=%s | 平台=%s", sys.version.replace("\n", " "), platform.platform())
    logger.info("data_csv=%s", str(csv_path))
    logger.info("projects_root=%s", str(projects_root))
    logger.info("device=%s", str(getattr(cfg, "device", "auto")))
    logger.info(
        "embedding：image=%s text=%s | max_seq_len=%d trunc=%s | pos=sin(fixed)",
        cfg.image_embedding_type,
        cfg.text_embedding_type,
        int(getattr(cfg, "max_seq_len", 0)),
        str(getattr(cfg, "truncation_strategy", "first")),
    )
    logger.info(
        "缓存：use_cache=%s | cache_dir=%s（默认不刷新、不压缩）",
        bool(getattr(cfg, "use_cache", False)),
        str(cache_dir),
    )

    set_global_seed(cfg.random_seed)

    prepared = prepare_seq_data(
        csv_path=csv_path,
        projects_root=projects_root,
        cfg=cfg,
        use_meta=bool(cfg.use_meta),
        cache_dir=cache_dir,
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

    if cfg.use_meta and prepared.preprocessor is not None:
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

    model = build_seq_model(
        cfg,
        meta_input_dim=int(prepared.meta_dim),
        image_embedding_dim=int(prepared.image_embedding_dim),
        text_embedding_dim=int(prepared.text_embedding_dim),
    )
    fusion_hidden_dim = int(getattr(model, "fusion_hidden_dim", 0)) or None
    if fusion_hidden_dim is not None:
        logger.info("fusion_hidden_dim（自动）：%d", int(fusion_hidden_dim))

    save_json(
        {
            "run_id": run_id,
            "mode": mode,
            "baseline_mode": baseline_mode,
            "use_meta": bool(cfg.use_meta),
            "meta_dim": int(prepared.meta_dim),
            "image_embedding_dim": int(prepared.image_embedding_dim),
            "text_embedding_dim": int(prepared.text_embedding_dim),
            "max_seq_len": int(prepared.max_seq_len),
            "fusion_hidden_dim": None if fusion_hidden_dim is None else int(fusion_hidden_dim),
            **cfg.to_dict(),
        },
        reports_dir / "config.json",
    )

    best_model, history, best_info = train_seq_with_early_stopping(
        model,
        use_meta=bool(cfg.use_meta),
        X_meta_train=prepared.X_meta_train,
        X_img_train=prepared.X_img_train,
        X_txt_train=prepared.X_txt_train,
        seq_type_train=prepared.seq_type_train,
        seq_attr_train=prepared.seq_attr_train,
        seq_mask_train=prepared.seq_mask_train,
        y_train=prepared.y_train,
        train_project_ids=prepared.train_project_ids,
        X_meta_val=prepared.X_meta_val,
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

    try:
        import pandas as pd

        pd.DataFrame(history).to_csv(reports_dir / "history.csv", index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("保存 history.csv 失败：%s", e)

    train_out = evaluate_seq_split(
        best_model,
        use_meta=bool(cfg.use_meta),
        X_meta=prepared.X_meta_train,
        X_img=prepared.X_img_train,
        X_txt=prepared.X_txt_train,
        seq_type=prepared.seq_type_train,
        seq_attr=prepared.seq_attr_train,
        seq_mask=prepared.seq_mask_train,
        y=prepared.y_train,
        project_ids=prepared.train_project_ids,
        cfg=cfg,
    )
    val_out = evaluate_seq_split(
        best_model,
        use_meta=bool(cfg.use_meta),
        X_meta=prepared.X_meta_val,
        X_img=prepared.X_img_val,
        X_txt=prepared.X_txt_val,
        seq_type=prepared.seq_type_val,
        seq_attr=prepared.seq_attr_val,
        seq_mask=prepared.seq_mask_val,
        y=prepared.y_val,
        project_ids=prepared.val_project_ids,
        cfg=cfg,
    )
    test_out = evaluate_seq_split(
        best_model,
        use_meta=bool(cfg.use_meta),
        X_meta=prepared.X_meta_test,
        X_img=prepared.X_img_test,
        X_txt=prepared.X_txt_test,
        seq_type=prepared.seq_type_test,
        seq_attr=prepared.seq_attr_test,
        seq_mask=prepared.seq_mask_test,
        y=prepared.y_test,
        project_ids=prepared.test_project_ids,
        cfg=cfg,
    )

    # 用验证集为本次模型选择阈值（最大化 F1），并用该阈值计算 train/val/test 指标。
    best_threshold = find_best_f1_threshold(prepared.y_val, val_out["prob"])
    train_out["metrics"] = compute_binary_metrics(prepared.y_train, train_out["prob"], threshold=best_threshold)
    val_out["metrics"] = compute_binary_metrics(prepared.y_val, val_out["prob"], threshold=best_threshold)
    test_out["metrics"] = compute_binary_metrics(prepared.y_test, test_out["prob"], threshold=best_threshold)
    logger.info("阈值选择：best_threshold=%.6f（val_f1=%.6f）", float(best_threshold), float(val_out["metrics"]["f1"]))

    results = {
        "run_id": run_id,
        "best_info": best_info,
        "selected_threshold": float(best_threshold),
        "train": train_out["metrics"],
        "val": val_out["metrics"],
        "test": test_out["metrics"],
    }
    save_json(results, reports_dir / "metrics.json")

    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "baseline_mode": baseline_mode,
            "use_meta": bool(cfg.use_meta),
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
            "fusion_hidden_dim": None if fusion_hidden_dim is None else int(fusion_hidden_dim),
            "fusion_dropout": float(getattr(cfg, "fusion_dropout", 0.0)),
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
        test_metrics = dict(test_out.get("metrics", {}) or {})
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
    logger.info("测试集指标：%s", test_out["metrics"])

    # 便于外部脚本（例如超参搜索）解析产物位置
    print(f"RUN_DIR={run_dir.as_posix()}")
    print(f"METRICS_JSON={(reports_dir / 'metrics.json').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
