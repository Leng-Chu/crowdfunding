# -*- coding: utf-8 -*-
"""
主程序入口：
- 读取 CSV（只用 project_id + state）
- 从 data/projects/.../<project_id>/ 读取文本嵌入（title_blurb + text）
- 训练 PyTorch 1D CNN（二分类）
- 保存产物到 experiments/text_dl/<run_id>/{artifacts,reports,plots}

运行示例（在项目根目录）：
conda run -n crowdfunding python src/dl/text/main.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch

from config import TextDLConfig
from data import prepare_data
from model import build_cnn
from train_eval import evaluate_split, train_with_early_stopping
from utils import make_run_dirs, plot_history, plot_roc, save_json, set_global_seed, setup_logger


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


def main() -> int:
    cfg = TextDLConfig()

    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / cfg.data_csv
    projects_root = project_root / cfg.projects_root
    experiment_root = project_root / cfg.experiment_root
    experiment_root.mkdir(parents=True, exist_ok=True)
    cache_dir = project_root / cfg.cache_dir

    run_id, artifacts_dir, reports_dir, plots_dir = make_run_dirs(experiment_root, run_name=cfg.run_name)
    run_dir = reports_dir.parent
    logger = setup_logger(reports_dir / "train.log")

    logger.info("run_id=%s", run_id)
    logger.info("python=%s | platform=%s", sys.version.replace("\n", " "), platform.platform())
    logger.info("data_csv=%s", str(csv_path))
    logger.info("projects_root=%s", str(projects_root))
    logger.info("embedding_type=%s | missing_strategy=%s", cfg.embedding_type, cfg.missing_strategy)
    logger.info(
        "缓存：use_cache=%s refresh_cache=%s compress=%s | cache_dir=%s",
        bool(getattr(cfg, "use_cache", False)),
        bool(getattr(cfg, "refresh_cache", False)),
        bool(getattr(cfg, "cache_compress", False)),
        str(cache_dir),
    )

    save_json({"run_id": run_id, **cfg.to_dict()}, reports_dir / "config.json")
    set_global_seed(cfg.random_seed)

    # 1) 数据准备
    prepared = prepare_data(csv_path=csv_path, projects_root=projects_root, cfg=cfg, cache_dir=cache_dir, logger=logger)
    logger.info(
        "数据集：train=%s val=%s test=%s | embedding_dim=%d | max_seq_len=%d",
        prepared.X_train.shape,
        prepared.X_val.shape,
        prepared.X_test.shape,
        prepared.embedding_dim,
        prepared.max_seq_len,
    )
    if prepared.stats:
        logger.info("统计信息：%s", prepared.stats)

    # 保存 split 明细（便于复现与排查）
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

    # 2) 模型训练
    model = build_cnn(cfg, embedding_dim=int(prepared.embedding_dim))
    best_model, history, best_info = train_with_early_stopping(
        model,
        prepared.X_train,
        prepared.len_train,
        prepared.y_train,
        prepared.X_val,
        prepared.len_val,
        prepared.y_val,
        cfg,
        logger,
    )

    # 3) 评估与保存
    try:
        import pandas as pd

        pd.DataFrame(history).to_csv(reports_dir / "history.csv", index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("保存 history.csv 失败：%s", e)

    train_out = evaluate_split(best_model, prepared.X_train, prepared.len_train, prepared.y_train, cfg)
    val_out = evaluate_split(best_model, prepared.X_val, prepared.len_val, prepared.y_val, cfg)
    test_out = evaluate_split(best_model, prepared.X_test, prepared.len_test, prepared.y_test, cfg)

    results = {
        "run_id": run_id,
        "best_info": best_info,
        "train": train_out["metrics"],
        "val": val_out["metrics"],
        "test": test_out["metrics"],
    }
    save_json(results, reports_dir / "metrics.json")

    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "embedding_dim": int(prepared.embedding_dim),
            "embedding_type": str(cfg.embedding_type),
            "missing_strategy": str(cfg.missing_strategy),
            "conv_kernel_size": int(getattr(cfg, "conv_kernel_size", 3)),
            "fc_hidden_dim": int(getattr(cfg, "fc_hidden_dim", 0)),
            "input_dropout": float(getattr(cfg, "input_dropout", 0.0)),
            "dropout": float(getattr(cfg, "dropout", 0.3)),
            "use_batch_norm": bool(getattr(cfg, "use_batch_norm", False)),
        },
        artifacts_dir / "model.pt",
    )

    # 保存预测结果（便于快速检查）
    try:
        _save_predictions_csv(
            reports_dir / "predictions_val.csv",
            prepared.val_project_ids,
            prepared.y_val,
            val_out["prob"],
            threshold=cfg.threshold,
        )
        _save_predictions_csv(
            reports_dir / "predictions_test.csv",
            prepared.test_project_ids,
            prepared.y_test,
            test_out["prob"],
            threshold=cfg.threshold,
        )
    except Exception as e:
        logger.warning("保存预测 CSV 失败：%s", e)

    if cfg.save_plots:
        plot_history(history, plots_dir / "history.png")
        plot_roc(prepared.y_val, val_out["prob"], plots_dir / "roc_val.png")
        plot_roc(prepared.y_test, test_out["prob"], plots_dir / "roc_test.png")

    logger.info("完成：产物已保存到 %s", str(run_dir))
    logger.info("测试集指标：%s", test_out["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

