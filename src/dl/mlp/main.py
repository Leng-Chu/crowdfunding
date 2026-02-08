# -*- coding: utf-8 -*-
"""
mlp 主程序入口：
- 三路输入：metadata / image / text
- 通过 use_meta / use_image / use_text 三个开关自由组合分支

说明：
- 本目录代码不依赖 `src/dl/mlp` 之外的其他训练代码，可独立运行
- fusion_hidden_dim 会在构建模型时根据实际启用的分支自动计算

运行（在项目根目录）：
- 使用默认配置：
  `conda run -n crowdfunding python src/dl/mlp/main.py`
- 指定 run_name / 嵌入类型 / 显卡：
  `conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --image-embedding-type clip --text-embedding-type clip --device cuda:0`
- 使用第 2 张 GPU（等价写法）：
  `conda run -n crowdfunding python src/dl/mlp/main.py --gpu 1`
"""

from __future__ import annotations

import argparse
import csv
import pickle
import platform
import sys
from dataclasses import replace
from pathlib import Path

from config import MlpConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mlp 训练入口（支持命令行覆盖部分配置）。")
    parser.add_argument("--run-name", default=None, help="实验名称后缀，用于产物目录命名。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖 config.py 的 random_seed）")
    parser.add_argument(
        "--use-meta",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="是否启用 meta 分支（出现任一 use_* 参数时，以命令行覆盖配置；未指定的 use_* 视为 False）。",
    )
    parser.add_argument(
        "--use-image",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="是否启用 image 分支（出现任一 use_* 参数时，以命令行覆盖配置；未指定的 use_* 视为 False）。",
    )
    parser.add_argument(
        "--use-text",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="是否启用 text 分支（出现任一 use_* 参数时，以命令行覆盖配置；未指定的 use_* 视为 False）。",
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
    """
    保存“单行结果”CSV（包含表头 + 1 条数据）。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _mode_name(use_meta: bool, use_image: bool, use_text: bool) -> str:
    parts: list[str] = []
    if use_meta:
        parts.append("meta")
    if use_image:
        parts.append("image")
    if use_text:
        parts.append("text")
    return "+".join(parts) if parts else "none"


def main() -> int:
    args = _parse_args()
    cfg = MlpConfig()

    import torch

    from data import prepare_multimodal_data
    from model import build_multimodal_model
    from train_eval import evaluate_multimodal_split, train_multimodal_with_early_stopping
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
    if args.seed is not None:
        cfg = replace(cfg, random_seed=int(args.seed))
    if args.use_meta is not None or args.use_image is not None or args.use_text is not None:
        cfg = replace(
            cfg,
            use_meta=bool(args.use_meta),
            use_image=bool(args.use_image),
            use_text=bool(args.use_text),
        )
    if args.image_embedding_type is not None:
        cfg = replace(cfg, image_embedding_type=str(args.image_embedding_type))
    if args.text_embedding_type is not None:
        cfg = replace(cfg, text_embedding_type=str(args.text_embedding_type))

    if args.device is not None:
        cfg = replace(cfg, device=str(args.device))
    elif args.gpu is not None:
        cfg = replace(cfg, device=f"cuda:{int(args.gpu)}")

    use_meta = bool(cfg.use_meta)
    use_image = bool(cfg.use_image)
    use_text = bool(cfg.use_text)

    if not use_image and cfg.image_embedding_type is not None:
        cfg = replace(cfg, image_embedding_type=None)
    if not use_text and cfg.text_embedding_type is not None:
        cfg = replace(cfg, text_embedding_type=None)

    n_enabled = int(use_meta) + int(use_image) + int(use_text)
    if n_enabled <= 0:
        raise ValueError("至少需要开启一个分支：use_meta/use_image/use_text。")

    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / cfg.data_csv
    projects_root = project_root / cfg.projects_root

    mode = _mode_name(use_meta, use_image, use_text)
    experiment_root = project_root / cfg.experiment_root / mode
    experiment_root.mkdir(parents=True, exist_ok=True)

    run_id, artifacts_dir, reports_dir, plots_dir = make_run_dirs(experiment_root, run_name=cfg.run_name)
    run_dir = reports_dir.parent
    logger = setup_logger(run_dir / "train.log")

    logger.info("模式=%s | run_id=%s | use_meta=%s use_image=%s use_text=%s", mode, run_id, use_meta, use_image, use_text)
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

    set_global_seed(cfg.random_seed)

    prepared = prepare_multimodal_data(
        csv_path=csv_path,
        projects_root=projects_root,
        cfg=cfg,
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        logger=logger,
    )
    logger.info(
        "数据集：train=%d val=%d test=%d | meta_dim=%d | image_dim=%d text_dim=%d | max_img_len=%d max_txt_len=%d",
        int(prepared.y_train.shape[0]),
        int(prepared.y_val.shape[0]),
        int(prepared.y_test.shape[0]),
        int(prepared.meta_dim),
        int(prepared.image_embedding_dim),
        int(prepared.text_embedding_dim),
        int(prepared.max_image_seq_len),
        int(prepared.max_text_seq_len),
    )
    if prepared.stats:
        logger.info("统计信息：%s", prepared.stats)

    if use_meta and prepared.preprocessor is not None:
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

    model = build_multimodal_model(
        cfg,
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        meta_input_dim=int(prepared.meta_dim),
        image_embedding_dim=int(prepared.image_embedding_dim),
        text_embedding_dim=int(prepared.text_embedding_dim),
    )
    fusion_hidden_dim = int(getattr(getattr(model, "fusion_fc", None), "out_features", 0)) or None
    if fusion_hidden_dim is not None:
        logger.info("fusion_hidden_dim（自动）：%d", int(fusion_hidden_dim))

    save_json(
        {
            "run_id": run_id,
            "mode": mode,
            "use_meta": use_meta,
            "use_image": use_image,
            "use_text": use_text,
            "meta_dim": int(prepared.meta_dim),
            "image_embedding_dim": int(prepared.image_embedding_dim),
            "text_embedding_dim": int(prepared.text_embedding_dim),
            "fusion_hidden_dim": None if fusion_hidden_dim is None else int(fusion_hidden_dim),
            **cfg.to_dict(),
        },
        reports_dir / "config.json",
    )

    best_model, history, best_info = train_multimodal_with_early_stopping(
        model,
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        X_meta_train=prepared.X_meta_train,
        X_image_train=prepared.X_image_train,
        len_image_train=prepared.len_image_train,
        X_text_train=prepared.X_text_train,
        len_text_train=prepared.len_text_train,
        y_train=prepared.y_train,
        X_meta_val=prepared.X_meta_val,
        X_image_val=prepared.X_image_val,
        len_image_val=prepared.len_image_val,
        X_text_val=prepared.X_text_val,
        len_text_val=prepared.len_text_val,
        y_val=prepared.y_val,
        cfg=cfg,
        logger=logger,
    )

    try:
        import pandas as pd

        pd.DataFrame(history).to_csv(reports_dir / "history.csv", index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("保存 history.csv 失败：%s", e)

    train_out = evaluate_multimodal_split(
        best_model,
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        X_meta=prepared.X_meta_train,
        X_image=prepared.X_image_train,
        len_image=prepared.len_image_train,
        X_text=prepared.X_text_train,
        len_text=prepared.len_text_train,
        y=prepared.y_train,
        cfg=cfg,
    )
    val_out = evaluate_multimodal_split(
        best_model,
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        X_meta=prepared.X_meta_val,
        X_image=prepared.X_image_val,
        len_image=prepared.len_image_val,
        X_text=prepared.X_text_val,
        len_text=prepared.len_text_val,
        y=prepared.y_val,
        cfg=cfg,
    )
    test_out = evaluate_multimodal_split(
        best_model,
        use_meta=use_meta,
        use_image=use_image,
        use_text=use_text,
        X_meta=prepared.X_meta_test,
        X_image=prepared.X_image_test,
        len_image=prepared.len_image_test,
        X_text=prepared.X_text_test,
        len_text=prepared.len_text_test,
        y=prepared.y_test,
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
            "use_meta": bool(use_meta),
            "use_image": bool(use_image),
            "use_text": bool(use_text),
            "meta_dim": int(prepared.meta_dim),
            "image_embedding_dim": int(prepared.image_embedding_dim),
            "text_embedding_dim": int(prepared.text_embedding_dim),
            "image_embedding_type": cfg.image_embedding_type,
            "text_embedding_type": cfg.text_embedding_type,
            "missing_strategy": str(cfg.missing_strategy),
            "meta_hidden_dim": int(cfg.meta_hidden_dim),
            "meta_dropout": float(cfg.meta_dropout),
            "image_conv_channels": int(cfg.image_conv_channels),
            "image_conv_kernel_size": int(cfg.image_conv_kernel_size),
            "image_input_dropout": float(cfg.image_input_dropout),
            "image_dropout": float(cfg.image_dropout),
            "image_use_batch_norm": bool(cfg.image_use_batch_norm),
            "text_conv_kernel_size": int(cfg.text_conv_kernel_size),
            "text_input_dropout": float(cfg.text_input_dropout),
            "text_dropout": float(cfg.text_dropout),
            "text_use_batch_norm": bool(cfg.text_use_batch_norm),
            "fusion_hidden_dim": None if fusion_hidden_dim is None else int(fusion_hidden_dim),
            "fusion_dropout": float(cfg.fusion_dropout),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

