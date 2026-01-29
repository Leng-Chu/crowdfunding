# Residual Baselines（res）

本模块位于 `src/dl/res`，用于在“三路分支”输入一致的前提下，提供两种对照 baseline：

- `baseline_mode="mlp"`：三路拼接 + 2-layer MLP head
- `baseline_mode="res"`：`z = z_base + delta_scale * z_res` 的残差式基线（用于验证 first-impression 信息是否只是“先验修正”）

工程规范完全遵循 `docs/dl_guidelines.md`（best checkpoint 选择、阈值选择、产物结构与字段口径）。

---

## 1. 输入与三路分支

三路分支与 `src/dl/gate` 保持一致（但代码完全独立，不 import gate/seq/late）：

- `v_meta`：metadata 表格特征（来自 `data/metadata/now_processed.csv`；categorical one-hot + numeric 标准化；仅在训练集上拟合预处理器）
- `h_seq`：正文内容序列表示（输入为预先计算的块 embedding + token features；不包含 title_blurb/cover_image token）
- `v_key`：第一印象表示（`title_blurb` + `cover_image` 的融合；若缺失则用零向量替代，并计数）

> 注意：图里不要有中文；本模块的训练曲线与 ROC 图均使用英文标题/坐标轴。

---

## 2. 两种 baseline_mode

### 2.1 `baseline_mode="mlp"`

- 融合：
  - `logit = Head( concat( LN(h_seq), LN(meta_proj(v_meta)), LN(key_proj(v_key)) ) )`
- `meta_proj/key_proj`：投影到与 `h_seq` 同维度 `d`（若已为 `d` 则 identity）
- `Head`：2-layer MLP：`Linear -> (ReLU/GELU) -> Dropout -> Linear(1)`

### 2.2 `baseline_mode="res"`

- 基线：
  - `z_base = MLP_base( concat( LN(h_seq), LN(meta_proj(v_meta)) ) )`
- 残差：
  - `z_res = MLP_prior( concat( LN(key_proj(v_key)), LN(meta_proj(v_meta)), LN(key_proj(v_key) ⊙ meta_proj(v_meta)) ) )`
- 最终：
  - `z = z_base + delta_scale * z_res`
  - `delta_scale` 为可学习标量，初始化默认为 `0.0`（可通过 `ResConfig.delta_scale_init` 覆盖）

关键要求：

- `MLP_base` 的结构与 `src/dl/seq/model.py` 的融合头完全一致（用于严格对齐对照）。

---

## 3. 运行方式

从仓库根目录运行：

```bash
python src/dl/res/main.py --baseline-mode mlp
python src/dl/res/main.py --baseline-mode res
```

常用参数：

- `--run-name`：实验名后缀
- `--image-embedding-type`：`clip/siglip/resnet`
- `--text-embedding-type`：`bge/clip/siglip`
- `--device`：`auto/cpu/cuda/cuda:0...`
- `--gpu`：等价于 `--device cuda:N`

通过环境变量覆盖配置（JSON）：

- `RES_CFG_OVERRIDES='{"batch_size": 256, "max_epochs": 10, "random_seed": 42}'`

---

## 4. 输出目录结构（验收口径）

每次运行固定生成：

- `experiments/res/<baseline_mode>/<run_id>/`
  - `artifacts/`
    - `model.pt`
    - `preprocessor.pkl`
    - `feature_names.txt`
  - `reports/`
    - `train.log`
    - `config.json`
    - `history.csv`
    - `metrics.json`
    - `splits.csv`
    - `predictions_val.csv`
    - `predictions_test.csv`
  - `plots/`（若 `save_plots=True`）
    - `history.png`
    - `roc_val.png`
    - `roc_test.png`

当 `baseline_mode="res"` 时，`reports/history.csv` 会额外包含验证集残差调试字段：

- `val_delta_abs_mean / val_delta_abs_p90 / val_delta_y1 / val_delta_y0`
- `val_auc_base / val_auc_final`

上述字段受配置开关控制：`ResConfig.debug_residual_stats=True` 才会输出（默认为 True）。

---

## 5. Optuna 超参搜索

脚本：`src/dl/res/optuna_search.py`（黑盒调用 `src/dl/res/main.py`）。

示例：

```bash
conda run -n crowdfunding python src/dl/res/optuna_search.py --baseline-mode res --device cuda:0 --n-trials 50
```

输出：

- `experiments/res/optuna/<study_name>/summary.csv`
- `experiments/res/optuna/<study_name>/best.json`
- `experiments/res/optuna/<study_name>/study.db`
