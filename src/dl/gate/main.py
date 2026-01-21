# -*- coding: utf-8 -*-
"""
gate 主程序入口：
- 固定三分支：meta / 第一印象 / 图文序列
- 两阶段门控融合三路向量，输出项目成功概率（logits）

运行（在项目根目录）：
- 使用默认配置：
  `conda run -n crowdfunding python src/dl/gate/main.py`
- 指定 run_name / 嵌入类型 / 显卡：
  `conda run -n crowdfunding python src/dl/gate/main.py --run-name gate --image-embedding-type clip --text-embedding-type clip --device cuda:0`
- 使用第 2 张 GPU（等价写法）：
  `conda run -n crowdfunding python src/dl/gate/main.py --gpu 1`
"""

from __future__ import annotations

import argparse
import csv
import pickle
import platform
import sys
from dataclasses import replace
from pathlib import Path

from config import GateConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gate 训练入口（支持命令行覆盖部分配置）。")
    parser.add_argument("--run-name", default=None, help="实验名称后缀，用于产物目录命名。")
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = _parse_args()
    cfg = GateConfig()

    import numpy as np
    import torch

    from data import iter_gate_kfold_data, prepare_gate_data
    from model import build_gate_model
    from train_eval import evaluate_gate_split, train_gate_with_early_stopping
    from utils import (
        compute_binary_metrics,
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
    if args.image_embedding_type is not None:
        cfg = replace(cfg, image_embedding_type=str(args.image_embedding_type))
    if args.text_embedding_type is not None:
        cfg = replace(cfg, text_embedding_type=str(args.text_embedding_type))

    if args.device is not None:
        cfg = replace(cfg, device=str(args.device))
    elif args.gpu is not None:
        cfg = replace(cfg, device=f"cuda:{int(args.gpu)}")

    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / cfg.data_csv
    projects_root = project_root / cfg.projects_root
    cache_dir = project_root / cfg.cache_dir

    experiment_root = project_root / cfg.experiment_root
    experiment_root.mkdir(parents=True, exist_ok=True)

    run_id, artifacts_dir, reports_dir, plots_dir = make_run_dirs(experiment_root, run_name=cfg.run_name)
    run_dir = reports_dir.parent
    logger = setup_logger(run_dir / "train.log")

    logger.info("run_id=%s", run_id)
    logger.info("python=%s | 平台=%s", sys.version.replace("\n", " "), platform.platform())
    logger.info("data_csv=%s", str(csv_path))
    logger.info("projects_root=%s", str(projects_root))
    logger.info("device=%s", str(getattr(cfg, "device", "auto")))
    logger.info(
        "嵌入类型：image=%s text=%s | 缺失策略=%s",
        cfg.image_embedding_type,
        cfg.text_embedding_type,
        cfg.missing_strategy,
    )
    logger.info(
        "缓存：use_cache=%s | cache_dir=%s（默认不刷新、不压缩）",
        bool(getattr(cfg, "use_cache", False)),
        str(cache_dir),
    )

    set_global_seed(cfg.random_seed)

    split_mode = str(getattr(cfg, "split_mode", "ratio") or "ratio").strip().lower()
    if split_mode == "kfold":
        logger.info(
            "启用 K 折交叉验证：k=%d | stratify=%s | shuffle=%s | k_fold_index=%d",
            int(getattr(cfg, "k_folds", 5)),
            bool(getattr(cfg, "kfold_stratify", True)),
            bool(getattr(cfg, "kfold_shuffle", True)),
            int(getattr(cfg, "k_fold_index", -1)),
        )

        save_json(
            {
                "run_id": run_id,
                "split_mode": "kfold",
                "cv": {
                    "k_folds": int(getattr(cfg, "k_folds", 5)),
                    "k_fold_index": int(getattr(cfg, "k_fold_index", -1)),
                    "kfold_stratify": bool(getattr(cfg, "kfold_stratify", True)),
                    "kfold_shuffle": bool(getattr(cfg, "kfold_shuffle", True)),
                },
                **cfg.to_dict(),
            },
            reports_dir / "config.json",
        )

        fold_results: list[dict] = []
        oof_project_ids: list[str] = []
        oof_y_true: list[int] = []
        oof_y_prob: list[float] = []

        for fold_idx, prepared in iter_gate_kfold_data(
            csv_path=csv_path,
            projects_root=projects_root,
            cfg=cfg,
            cache_dir=cache_dir,
            logger=logger,
        ):
            fold_tag = f"fold_{int(fold_idx):02d}"
            fold_artifacts_dir = artifacts_dir / fold_tag
            fold_reports_dir = reports_dir / fold_tag
            fold_plots_dir = plots_dir / fold_tag
            fold_artifacts_dir.mkdir(parents=True, exist_ok=True)
            fold_reports_dir.mkdir(parents=True, exist_ok=True)
            fold_plots_dir.mkdir(parents=True, exist_ok=True)

            set_global_seed(int(cfg.random_seed) + int(fold_idx))

            logger.info(
                "fold=%d 数据：train=%d test=%d | meta_dim=%d | image_dim=%d text_dim=%d",
                int(fold_idx),
                int(prepared.y_train.shape[0]),
                int(prepared.y_test.shape[0]),
                int(prepared.meta_dim),
                int(prepared.image_embedding_dim),
                int(prepared.text_embedding_dim),
            )
            logger.info("fold=%d 统计：%s", int(fold_idx), prepared.stats)

            model = build_gate_model(
                cfg,
                meta_input_dim=int(prepared.meta_dim),
                image_embedding_dim=int(prepared.image_embedding_dim),
                text_embedding_dim=int(prepared.text_embedding_dim),
            )

            model, history, best_info = train_gate_with_early_stopping(
                model=model,
                X_meta_train=prepared.X_meta_train,
                X_cover_train=prepared.X_cover_train,
                len_cover_train=prepared.len_cover_train,
                X_title_blurb_train=prepared.X_title_blurb_train,
                len_title_blurb_train=prepared.len_title_blurb_train,
                X_image_train=prepared.X_image_train,
                len_image_train=prepared.len_image_train,
                X_text_train=prepared.X_text_train,
                len_text_train=prepared.len_text_train,
                y_train=prepared.y_train,
                X_meta_val=prepared.X_meta_val,
                X_cover_val=prepared.X_cover_val,
                len_cover_val=prepared.len_cover_val,
                X_title_blurb_val=prepared.X_title_blurb_val,
                len_title_blurb_val=prepared.len_title_blurb_val,
                X_image_val=prepared.X_image_val,
                len_image_val=prepared.len_image_val,
                X_text_val=prepared.X_text_val,
                len_text_val=prepared.len_text_val,
                y_val=prepared.y_val,
                cfg=cfg,
                logger=logger,
            )

            # 保存模型与预处理器
            torch.save(model.state_dict(), fold_artifacts_dir / "model.pt")
            with (fold_artifacts_dir / "preprocessor.pkl").open("wb") as f:
                pickle.dump(prepared.preprocessor, f)
            save_text(prepared.feature_names, fold_artifacts_dir / "feature_names.txt")

            # 评估（按 fold 的 test）
            test_out = evaluate_gate_split(
                model=model,
                X_meta=prepared.X_meta_test,
                X_cover=prepared.X_cover_test,
                X_title_blurb=prepared.X_title_blurb_test,
                len_cover=prepared.len_cover_test,
                len_title_blurb=prepared.len_title_blurb_test,
                X_image=prepared.X_image_test,
                len_image=prepared.len_image_test,
                X_text=prepared.X_text_test,
                len_text=prepared.len_text_test,
                y=prepared.y_test,
                cfg=cfg,
            )

            save_json(
                {
                    "fold": int(fold_idx),
                    "best": best_info,
                    "stats": prepared.stats,
                    "test": test_out["metrics"],
                },
                fold_reports_dir / "metrics.json",
            )

            # 保存 history.csv
            import pandas as pd

            pd.DataFrame(history).to_csv(fold_reports_dir / "history.csv", index=False, encoding="utf-8")

            _save_predictions_csv(
                fold_reports_dir / "predictions_test.csv",
                project_ids=prepared.test_project_ids,
                y_true=prepared.y_test,
                y_prob=test_out["prob"],
                threshold=cfg.threshold,
            )

            if bool(getattr(cfg, "save_plots", True)):
                plot_history(history, fold_plots_dir / "history.png")
                plot_roc(np.asarray(prepared.y_test), np.asarray(test_out["prob"]), fold_plots_dir / "roc_test.png")

            # 汇总
            fold_row = {"fold": int(fold_idx), **{f"test_{k}": v for k, v in test_out["metrics"].items()}}
            fold_results.append(fold_row)
            oof_project_ids.extend(prepared.test_project_ids)
            oof_y_true.extend(np.asarray(prepared.y_test).astype(int).tolist())
            oof_y_prob.extend(np.asarray(test_out["prob"]).astype(float).tolist())

        # 交叉验证整体汇总
        oof_y_true_arr = np.asarray(oof_y_true, dtype=np.int64)
        oof_y_prob_arr = np.asarray(oof_y_prob, dtype=np.float64)
        cv_metrics = compute_binary_metrics(oof_y_true_arr, oof_y_prob_arr, threshold=cfg.threshold)

        save_json(
            {
                "cv_metrics": cv_metrics,
                "n_oof": int(oof_y_true_arr.size),
                "n_folds": int(len(fold_results)),
            },
            reports_dir / "cv_metrics.json",
        )
        _save_predictions_csv(
            reports_dir / "cv_predictions_test.csv",
            project_ids=oof_project_ids,
            y_true=oof_y_true_arr,
            y_prob=oof_y_prob_arr,
            threshold=cfg.threshold,
        )

        if fold_results:
            import pandas as pd

            pd.DataFrame(fold_results).to_csv(reports_dir / "fold_metrics.csv", index=False, encoding="utf-8")

        logger.info("CV 完成：n_folds=%d | oof_auc=%s", int(len(fold_results)), str(cv_metrics.get("roc_auc")))
        return 0

    # -----------------------------
    # ratio（默认）
    # -----------------------------

    save_json({"run_id": run_id, "split_mode": "ratio", **cfg.to_dict()}, reports_dir / "config.json")

    prepared = prepare_gate_data(
        csv_path=csv_path,
        projects_root=projects_root,
        cfg=cfg,
        cache_dir=cache_dir,
        logger=logger,
    )
    logger.info(
        "数据：train=%d val=%d test=%d | meta_dim=%d | image_dim=%d text_dim=%d",
        int(prepared.y_train.shape[0]),
        int(prepared.y_val.shape[0]),
        int(prepared.y_test.shape[0]),
        int(prepared.meta_dim),
        int(prepared.image_embedding_dim),
        int(prepared.text_embedding_dim),
    )
    logger.info("统计：%s", prepared.stats)

    model = build_gate_model(
        cfg,
        meta_input_dim=int(prepared.meta_dim),
        image_embedding_dim=int(prepared.image_embedding_dim),
        text_embedding_dim=int(prepared.text_embedding_dim),
    )

    model, history, best_info = train_gate_with_early_stopping(
        model=model,
        X_meta_train=prepared.X_meta_train,
        X_cover_train=prepared.X_cover_train,
        len_cover_train=prepared.len_cover_train,
        X_title_blurb_train=prepared.X_title_blurb_train,
        len_title_blurb_train=prepared.len_title_blurb_train,
        X_image_train=prepared.X_image_train,
        len_image_train=prepared.len_image_train,
        X_text_train=prepared.X_text_train,
        len_text_train=prepared.len_text_train,
        y_train=prepared.y_train,
        X_meta_val=prepared.X_meta_val,
        X_cover_val=prepared.X_cover_val,
        len_cover_val=prepared.len_cover_val,
        X_title_blurb_val=prepared.X_title_blurb_val,
        len_title_blurb_val=prepared.len_title_blurb_val,
        X_image_val=prepared.X_image_val,
        len_image_val=prepared.len_image_val,
        X_text_val=prepared.X_text_val,
        len_text_val=prepared.len_text_val,
        y_val=prepared.y_val,
        cfg=cfg,
        logger=logger,
    )

    torch.save(model.state_dict(), artifacts_dir / "model.pt")
    with (artifacts_dir / "preprocessor.pkl").open("wb") as f:
        pickle.dump(prepared.preprocessor, f)
    save_text(prepared.feature_names, artifacts_dir / "feature_names.txt")

    train_out = evaluate_gate_split(
        model=model,
        X_meta=prepared.X_meta_train,
        X_cover=prepared.X_cover_train,
        X_title_blurb=prepared.X_title_blurb_train,
        len_cover=prepared.len_cover_train,
        len_title_blurb=prepared.len_title_blurb_train,
        X_image=prepared.X_image_train,
        len_image=prepared.len_image_train,
        X_text=prepared.X_text_train,
        len_text=prepared.len_text_train,
        y=prepared.y_train,
        cfg=cfg,
    )
    val_out = evaluate_gate_split(
        model=model,
        X_meta=prepared.X_meta_val,
        X_cover=prepared.X_cover_val,
        X_title_blurb=prepared.X_title_blurb_val,
        len_cover=prepared.len_cover_val,
        len_title_blurb=prepared.len_title_blurb_val,
        X_image=prepared.X_image_val,
        len_image=prepared.len_image_val,
        X_text=prepared.X_text_val,
        len_text=prepared.len_text_val,
        y=prepared.y_val,
        cfg=cfg,
    )
    test_out = evaluate_gate_split(
        model=model,
        X_meta=prepared.X_meta_test,
        X_cover=prepared.X_cover_test,
        X_title_blurb=prepared.X_title_blurb_test,
        len_cover=prepared.len_cover_test,
        len_title_blurb=prepared.len_title_blurb_test,
        X_image=prepared.X_image_test,
        len_image=prepared.len_image_test,
        X_text=prepared.X_text_test,
        len_text=prepared.len_text_test,
        y=prepared.y_test,
        cfg=cfg,
    )

    save_json(
        {
            "best": best_info,
            "stats": prepared.stats,
            "train": train_out["metrics"],
            "val": val_out["metrics"],
            "test": test_out["metrics"],
        },
        reports_dir / "metrics.json",
    )

    import pandas as pd

    pd.DataFrame(history).to_csv(reports_dir / "history.csv", index=False, encoding="utf-8")

    _save_predictions_csv(
        reports_dir / "predictions_train.csv",
        project_ids=prepared.train_project_ids,
        y_true=prepared.y_train,
        y_prob=train_out["prob"],
        threshold=cfg.threshold,
    )
    _save_predictions_csv(
        reports_dir / "predictions_val.csv",
        project_ids=prepared.val_project_ids,
        y_true=prepared.y_val,
        y_prob=val_out["prob"],
        threshold=cfg.threshold,
    )
    _save_predictions_csv(
        reports_dir / "predictions_test.csv",
        project_ids=prepared.test_project_ids,
        y_true=prepared.y_test,
        y_prob=test_out["prob"],
        threshold=cfg.threshold,
    )

    if bool(getattr(cfg, "save_plots", True)):
        plot_history(history, plots_dir / "history.png")
        plot_roc(np.asarray(prepared.y_test), np.asarray(test_out["prob"]), plots_dir / "roc_test.png")

    # 便于快速查看的单行汇总
    row = {"run_id": run_id, **{f"test_{k}": v for k, v in test_out["metrics"].items()}}
    _save_single_row_csv(reports_dir / "result.csv", row)

    logger.info("完成：test_auc=%s test_acc=%.4f", str(test_out["metrics"].get("roc_auc")), float(test_out["metrics"]["accuracy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
