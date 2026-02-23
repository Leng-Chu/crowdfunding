# 图文 Cross-Attention baseline（`src/dl/dcan`）说明

## 1. 任务定义
- 任务：Kickstarter 项目二分类（`state`，0/1）。
- 输入：`image` + `text` 必选，`meta` 可选（`use_meta` 控制）。
- token 属性（文本长度/图片面积）可选（`use_token_attr` 控制，默认开启）。
- 输出：二分类 logits（训练使用 `BCEWithLogitsLoss`）。

## 2. 数据语义（与 late 对齐）
`dcan` 采用统一序列截断语义，不再分别截断图文：

1. 先构建统一序列：`title_blurb` 前缀 + `cover_image` 前缀 + `content_sequence` 正文。
2. 在统一序列上整体截断（`max_seq_len` + `truncation_strategy`）。
3. 将截断后的 token 映射回 image/text 两个模态序列，供 DCAN 使用。

配置项：
- `max_seq_len`：统一序列长度上限。
- `truncation_strategy`：`first` 或 `random`。
- `random` 策略按 `random_seed + sha256(project_id)` 生成稳定随机窗口，保证可复现。
- token 属性：`attr_image=log(max(1, area))`、`attr_text=log(max(1, length))`。
- 其中 `area` 来自 `cover_image` 与 `content_sequence` 中 image 的 `width*height`；
  `length` 来自 `title/blurb` 与 `content_sequence` 中 text 的 `content_length`（缺失时回退文本长度）。
- 模型在 projection 阶段显式融合 `attr_image/attr_text`（`use_token_attr=True`）。
- 当 `use_token_attr=False` 时，仍保留统一截断与 mask 语义，但不向投影特征注入属性分支。

## 3. 训练与 best checkpoint 规则
训练在 `src/dl/dcan/train_eval.py`：

- 优化器：`AdamW`（`alpha` 作为 `weight_decay`）。
- 学习率：warmup + cosine，按 step 更新（`lr_scheduler_min_lr` 控制最小学习率）。
- 训练技巧：`label smoothing` + `EMA` + `AMP(CUDA)`。
- 梯度裁剪：`max_grad_norm`。

best checkpoint 规则固定为：
- 优先 `val_auc`（越大越好）。
- 若验证集单类导致 AUC 不可用，回退 `val_log_loss`（越小越好）。
- 并列规则：
  - `val_auc` 模式：`val_auc` 降序 -> `val_log_loss` 升序 -> `epoch` 升序。
  - `val_log_loss` 模式：`val_log_loss` 升序 -> `epoch` 升序。

## 4. 阈值选择规则
best epoch 确定后：

1. 使用该 checkpoint 的 `val_prob` 搜索阈值（最大化 F1）。
2. 若多个阈值 F1 相同，固定选择更小阈值。
3. 用该阈值统一计算 train/val/test 的阈值相关指标。

## 5. 关键默认值（与 late 对齐）
- `batch_size = 256`
- `max_epochs = 80`
- `alpha = 4e-4`
- `fusion_hidden_dim = 512`（可在 `src/dl/dcan/config.py` 手动配置；若设为 `<=0` 则自动回退为 `2 * fusion_in_dim`）

## 6. 运行方式
在项目根目录执行：

- 默认运行：
  - `conda run -n crowdfunding python src/dl/dcan/main.py`
- 指定随机种子与设备：
  - `conda run -n crowdfunding python src/dl/dcan/main.py --seed 42 --device cuda:0`
- 关闭 meta：
  - `conda run -n crowdfunding python src/dl/dcan/main.py --no-use-meta`
- 关闭 token 属性（不使用文本长度/图片面积）：
  - `conda run -n crowdfunding python src/dl/dcan/main.py --no-use-token-attr`
- 同时关闭 meta 与 token 属性：
  - `conda run -n crowdfunding python src/dl/dcan/main.py --no-use-meta --no-use-token-attr`

## 7. 输出产物
默认输出到 `experiments/dcan/<mode>/<run_id>/`：

- `artifacts/model.pt`：包含 `state_dict`、`best_epoch`、`best_val_auc`、`best_val_log_loss`、`best_threshold`、`use_token_attr` 等。
- `reports/config.json` / `reports/history.csv` / `reports/metrics.json` / `reports/splits.csv`
- `reports/predictions_val.csv` / `reports/predictions_test.csv`
- `result.csv`（单行汇总）
- `plots/*.png`（若 `save_plots=True`）

## 8. 批量脚本
已提供脚本：`src/scripts/run/dcan_run_all.py`

- 单个 seed 跑全部组合（`use_meta` + `no-use-meta`）：
  - `python src/scripts/run/dcan_run_all.py all --seed 42`
- 指定是否使用 token 属性：
  - `python src/scripts/run/dcan_run_all.py all --seed 42 --no-use-token-attr`
- 在 seed 区间内批量运行：
  - `python src/scripts/run/dcan_run_all.py single --experiment-mode all --start-seed 42 --end-seed 46`
