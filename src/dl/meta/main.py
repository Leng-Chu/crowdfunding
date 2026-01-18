# -*- coding: utf-8 -*-
"""
主程序入口：
- 读取数据
- 预处理（one-hot + 标准化）
- 训练 PyTorch MLP（二分类）
- 保存实验产物到 experiments/meta_dl/<run_id>/{artifacts,reports,plots}

运行示例（在项目根目录）：
conda run -n crowdfunding python src/dl/meta/main.py
"""

from __future__ import annotations

import pickle
import platform
import sys
from pathlib import Path

import torch

from config import MetaDLConfig
from data import prepare_data
from model import build_mlp
from train_eval import evaluate_split, train_with_early_stopping
from utils import make_run_dirs, plot_history, plot_roc, save_json, save_text, set_global_seed, setup_logger


def main() -> int:
    # 你不需要在命令行里传任何参数：统一在 config.py 里改
    cfg = MetaDLConfig()

    # 项目根目录：.../src/dl/meta/main.py -> 上 3 层
    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / cfg.data_csv
    experiment_root = project_root / cfg.experiment_root
    experiment_root.mkdir(parents=True, exist_ok=True)

    run_id, artifacts_dir, reports_dir, plots_dir = make_run_dirs(experiment_root, run_name=cfg.run_name)
    run_dir = reports_dir.parent
    logger = setup_logger(reports_dir / "train.log")

    logger.info("run_id=%s", run_id)
    logger.info("python=%s | platform=%s", sys.version.replace("\n", " "), platform.platform())
    logger.info("data_csv=%s", str(csv_path))

    save_json({"run_id": run_id, **cfg.to_dict()}, reports_dir / "config.json")

    set_global_seed(cfg.random_seed)

    # 1) 数据准备
    prepared = prepare_data(csv_path, cfg)
    logger.info(
        "数据集：train=%s val=%s test=%s | 特征维度=%d",
        prepared.X_train.shape,
        prepared.X_val.shape,
        prepared.X_test.shape,
        prepared.X_train.shape[1],
    )

    # 保存预处理器与特征名（方便后续复用/排查）
    with (artifacts_dir / "preprocessor.pkl").open("wb") as f:
        pickle.dump(prepared.preprocessor, f)
    if prepared.feature_names:
        save_text(prepared.feature_names, artifacts_dir / "feature_names.txt")

    # 2) 模型训练
    model = build_mlp(cfg, input_dim=int(prepared.X_train.shape[1]))
    best_model, history, best_info = train_with_early_stopping(
        model,
        prepared.X_train,
        prepared.y_train,
        prepared.X_val,
        prepared.y_val,
        cfg,
        logger,
    )

    # 3) 评估与保存
    # 保存训练历史（便于后续画图/复盘）
    try:
        import pandas as pd

        pd.DataFrame(history).to_csv(reports_dir / "history.csv", index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("保存 history.csv 失败：%s", e)

    train_out = evaluate_split(best_model, prepared.X_train, prepared.y_train, cfg)
    val_out = evaluate_split(best_model, prepared.X_val, prepared.y_val, cfg)
    test_out = evaluate_split(best_model, prepared.X_test, prepared.y_test, cfg)

    results = {
        "run_id": run_id,
        "best_info": best_info,
        "train": train_out["metrics"],
        "val": val_out["metrics"],
        "test": test_out["metrics"],
    }
    save_json(results, reports_dir / "metrics.json")

    # 保存 PyTorch 权重（包含必要的结构信息，便于复现/加载）
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "input_dim": int(prepared.X_train.shape[1]),
            "hidden_layer_sizes": list(cfg.hidden_layer_sizes),
            "activation": cfg.activation,
            "dropout": cfg.dropout,
            "use_batch_norm": cfg.use_batch_norm,
        },
        artifacts_dir / "model.pt",
    )

    # 保存训练曲线与 ROC
    if cfg.save_plots:
        plot_history(history, plots_dir / "history.png")
        plot_roc(prepared.y_val, val_out["prob"], plots_dir / "roc_val.png")
        plot_roc(prepared.y_test, test_out["prob"], plots_dir / "roc_test.png")

    logger.info("完成：产物已保存到 %s", str(run_dir))
    logger.info("测试集指标：%s", test_out["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
