# 深度学习模块通用工程规范

本文总结 `src/dl/seq`、`src/dl/late`、`src/dl/gate`、`src/dl/res` 四套训练代码的共同工程规范，用于保证实验可复现、指标口径一致、产物结构稳定，便于横向对比与自动化汇总。

该规范也可作为其它 `src/dl/*` 子模块的参考（尤其是二分类阈值与产物结构部分）。

---

## 1. 运行与路径规范

- 从仓库根目录运行脚本（代码内部会基于 `__file__` 推断仓库根路径，避免工作目录差异导致路径错误）。
- 所有路径使用相对仓库根目录的相对路径（例如 `data/...`、`experiments/...`），避免硬编码绝对路径。
- 输出目录统一写入 `experiments/<module>/<mode>/<run_id>/`，其中：
  - `<module> ∈ {seq, late, gate, res}`
  - `<mode>` 为实验组名称（由 `baseline_mode` 及开关组合得到）
  - `<run_id>` 为时间戳（可附带 `run_name` 后缀）

---

## 2. 数据加载与切分规范

### 2.1 输入数据位置与必备文件

- 元数据 CSV：`data/metadata/now_processed.csv`
  - 关键字段：`project_id`（样本 ID）、`state`（二分类标签）
- 项目目录：`data/projects/now/<project_id>/`
  - 必备：`content.json`（包含 `content_sequence` 及预处理统计信息）
  - 预计算 embedding（至少包含以下两类）：
    - `image_<image_embedding_type>.npy`
    - `text_<text_embedding_type>.npy`
  - 部分模型还会读取：
    - `cover_image_<image_embedding_type>.npy`
    - `title_blurb_<text_embedding_type>.npy`

### 2.2 缺失处理（missing_strategy）

- `missing_strategy="error"`：遇到缺失文件/异常样本直接报错，保证实验严格一致性。
- `missing_strategy="skip"`：跳过缺失样本，并在返回的 `stats` 中记录跳过数量；若某个 split 最终无可用样本则报错。

### 2.3 数据切分（split）

- 切分比例默认：`train/val/test = 0.6/0.2/0.2`
- 默认 `shuffle_before_split=False`，保证与数据顺序严格对齐（便于复现实验与对齐外部汇总）。
- `splits.csv` 必须写入 `reports/`，包含每个 `project_id` 的 `split` 与标签，便于复核。

### 2.4 meta 表格特征预处理（可选）

当启用 meta 分支（`use_meta=True` 或对应的 meta 开关为真）时：

- 表格特征来自 CSV 的 categorical/numeric 列：
  - categorical：one-hot
  - numeric：按训练集统计做标准化（mean/std）
- 预处理器必须只在训练集上拟合，然后应用到 val/test（禁止信息泄漏）。
- 产物必须保存到 `artifacts/`：
  - `preprocessor.pkl`
  - `feature_names.txt`

### 2.5 统一截断（truncation_strategy）

- `max_seq_len` 表示块序列长度上限，用于对齐实验与控制计算量。
- `truncation_strategy="first"`：始终取最前窗口。
- `truncation_strategy="random"`：随机窗口必须可复现，且与样本绑定：seed 由 `hashlib.sha256(project_id)` 与 `random_seed` 混合得到，避免顺序/进程差异引入不稳定。

---

## 3. 训练过程通用规范

### 3.1 可复现性

- 必须在训练开始前调用全局种子设置（`random/numpy/torch`），使用 `random_seed` 控制。
- 数据端的“随机截断/随机打乱”等操作必须可复现，且与样本绑定：
  - 统一使用 `hashlib.sha256(project_id)` 生成稳定哈希，并与 `random_seed` 混合得到样本 seed
  - 禁止依赖 Python 原生 `hash()`（不同进程/不同运行可能不稳定）

### 3.2 损失与类别不平衡

- 任务：二分类，loss 使用 `BCEWithLogitsLoss`
- 统一使用轻量 label smoothing（将标签从 {0,1} 平滑到接近 0/1 的浮点值）

### 3.3 优化器、学习率与梯度

- 优化器：AdamW
  - 学习率：`learning_rate_init`
  - 权重衰减：`alpha`（作为 `weight_decay`）
- 学习率调度：warmup + cosine（每 step 更新）
  - 最小学习率：`lr_scheduler_min_lr`
- 梯度裁剪：按 `max_grad_norm` 做全局 norm clip，并记录 `grad_norm`

### 3.4 AMP 与 EMA

- 设备为 CUDA 时启用 AMP（autocast + GradScaler），并保持实现对不同 PyTorch 版本兼容。
- 为保证口径一致：early stopping、阈值选择、最终评估所用的 `prob` 必须来自同一权重模式（推荐使用 EMA 权重进行验证/推理），减少指标抖动并提升稳定性。

---

## 4. 指标与阈值规范

二分类指标统一包括：

- `accuracy, precision, recall, f1, roc_auc, log_loss`
- `tn, fp, fn, tp`
- `threshold`（本次计算指标所使用的阈值）

其中 `roc_auc` 在某个 split 只包含单一类别时可能为 `None`，并记录 `roc_auc_error` 说明原因。

### 4.1 训练阶段：best checkpoint 选择（阈值无关）

对每个 epoch：

1. 得到验证集预测概率 `val_prob`
2. 仅计算阈值无关指标：
   - `val_auc`（即 `roc_auc`）
   - `val_log_loss`

early stopping 与 best checkpoint 选择口径：

- 默认使用 `val_auc`（越大越好）
- 若验证集为单类导致 `val_auc` 不可用，则回退为 `val_log_loss`（越小越好）

要求：

- 训练阶段不进行阈值搜索；history/log 中不记录任何逐 epoch 阈值字段

### 4.2 best_info 与并列判定

当出现新的 best epoch 时，必须同步保存：

- `best_epoch`
- `best_val_auc`（可为空）
- `best_val_log_loss`
- `metric_for_best`（`val_auc` 或 `val_log_loss`）
- `tie_breaker`（并列判定固定规则）
- `best_state`（best checkpoint 权重）

并列判定固定规则建议：

- 若 `metric_for_best=val_auc`：按 `val_auc` 降序；并列时按 `val_log_loss` 升序；仍并列时按 `epoch` 升序
- 若 `metric_for_best=val_log_loss`：按 `val_log_loss` 升序；并列时按 `epoch` 升序

### 4.3 阈值搜索（find_best_f1_threshold）

- 输入：`y_true` 与预测概率 `prob`
- 输出：`best_threshold` 与 `best_f1`
- 搜索要求：
  - 不使用随机数
  - 候选阈值来自 `prob` 的稳定集合（例如 distinct 值）
  - 并列最优 F1 时采用固定规则保证可复现（当前实现选择**较小阈值**）

### 4.4 最终评估口径

- 在 `best_epoch` 确定后，使用该 checkpoint 对验证集推理得到的 `val_prob` 搜索一次 `best_threshold`（最大化 F1）。
- 最终 train/val/test 的阈值相关指标（`precision/recall/f1/acc` 等）必须全部使用该 `best_threshold` 计算。
- 最终评估基于 best checkpoint 推理得到的 `prob` 计算。

---

## 5. 产物与文件保存规范

### 5.1 目录结构

每次运行固定生成：

- `artifacts/`：可复现产物
- `reports/`：日志与指标
- `plots/`：训练曲线与 ROC（若开启）

### 5.2 reports/（日志与指标）

必须包含（文件名固定，编码 UTF-8）：

- `train.log`：训练日志
- `config.json`：本次运行完整配置与关键信息（run_id、mode、embedding 类型、维度等）
- `history.csv`：逐 epoch 训练/验证指标历史
- `metrics.json`：最终 train/val/test 指标（统一口径）
- `splits.csv`：project_id 与 split 映射（含标签）
- `predictions_val.csv` / `predictions_test.csv`：预测明细
  - 列：`project_id, y_true, y_prob, y_pred`
- `result.csv`：单行汇总（便于批量汇总多个实验）

其中 `metrics.json` 的结构固定为：

```json
{
  "run_id": "...",
  "best_info": {
    "best_epoch": 0,
    "best_val_auc": 0.0,
    "best_val_log_loss": 0.0,
    "metric_for_best": "val_auc",
    "tie_breaker": "val_auc(desc) -> val_log_loss(asc) -> epoch(asc)",
    "best_val_row": {}
  },
  "selected_threshold": 0.0,
  "train": { "accuracy": 0.0, "...": 0 },
  "val": { "accuracy": 0.0, "...": 0 },
  "test": { "accuracy": 0.0, "...": 0 }
}
```

（字段可能随指标扩展而增加，但上述顶层结构与关键 key 不应改变。）

### 5.3 artifacts/（可复现产物）

必须保存 `model.pt`，并包含：

- `state_dict`：best model 权重
- `best_epoch / best_val_auc / best_val_log_loss / best_threshold`
- 与模型结构/数据对齐相关的关键配置（例如 embedding 类型与维度、序列长度、head/transformer 结构参数等）

当启用 meta 分支时，还必须包含：

- `preprocessor.pkl`
- `feature_names.txt`

### 5.4 plots/（图像产物）

- 图中不允许出现中文（标题/坐标轴/legend 使用英文）
- 常见产物：`history.png`、`roc_val.png`、`roc_test.png`
