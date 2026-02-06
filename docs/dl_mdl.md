# MDL 多模态二分类模型（代码目录：`src/dl/mdl`）说明

> 说明：本 baseline 的工程命名（日志/产物/mode）统一使用 `mdl`；代码中真正的 MLP 组件（例如 `MetaMLPEncoder`）保留原名。

## 1. 任务定义与输入输出

- **任务**：Kickstarter 项目二分类（成功/失败）。
- **标签**：来自表格数据 `state` 字段（**数值 0/1**；正类为 1）。
- **输入模态**（可按开关自由组合）：
  - `meta`：表格元数据特征（类别 + 数值）。
  - `image`：项目封面与正文图片的向量序列（embedding sequence；不做顺序建模，仅作为序列输入 CNN）。
  - `text`：项目标题/简介与正文文本的向量序列（embedding sequence；不做顺序建模，仅作为序列输入 CNN）。
- **输出**：二分类 **logits**（训练使用 `BCEWithLogitsLoss`），推理时对 logits 施加 `sigmoid` 得到正类概率。

代码层面通过 `use_meta / use_image / use_text` 三个布尔开关控制启用哪些分支，并自动根据启用分支计算融合层输入维度与隐藏层大小。

## 2. 数据组织与文件约定（关键）

### 2.1 表格数据（metadata CSV）

默认配置见 `src/dl/mdl/config.py:MdlConfig`：

- `data_csv`：默认 `data/metadata/now_processed.csv`
- 关键列名：
  - `id_col`：`project_id`
  - `target_col`：`state`（必须为 0/1 数值）
  - `categorical_cols`：`category, country, currency`
  - `numeric_cols`：`duration_days, log_usd_goal`
  - `drop_cols`：默认丢弃 `project_id, time`

表格预处理由 `src/dl/mdl/data.py:TabularPreprocessor` 完成（one-hot + 数值标准化），并在产物中保存，便于复现。

### 2.2 项目目录（projects_root）

`projects_root` 默认 `data/projects/now`，每个项目一个文件夹：

```
data/projects/<dataset>/<project_id>/
  content.json
  cover_image_{emb_type}.npy
  title_blurb_{emb_type}.npy
  image_{emb_type}.npy          (可选：当 content_sequence 中 image 数量为 0 时可缺失)
  text_{emb_type}.npy           (可选：当 content_sequence 中 text 数量为 0 时可缺失)
```

其中 `content.json` 必须包含字段 `content_sequence`，它是按页面呈现顺序排列的列表；每个元素包含 `type ∈ {"text", "image"}`。

### 2.3 向量文件命名（与 `src/preprocess/embedding` 对齐）

`src/dl/mdl/data.py` 约定如下文件名（`{emb_type}` 来自配置项 `image_embedding_type/text_embedding_type`）：

- **图片侧**（`image_embedding_type ∈ {clip, siglip, resnet}`）
  - 必须：`cover_image_{emb_type}.npy`（封面图，形状通常为 `[1, D]`）
  - 可选：`image_{emb_type}.npy`（正文图片序列，形状通常为 `[K, D]`）
- **文本侧**（`text_embedding_type ∈ {bge, clip, siglip}`）
  - 必须：`title_blurb_{emb_type}.npy`（标题+简介，形状可能为 `[1, D]` 或 `[2, D]`）
  - 可选：`text_{emb_type}.npy`（正文文本分段序列，形状通常为 `[T, D]`）

向量生成细节请参见 `docs/preprocess_embedding.md`。

### 2.4 统一序列截断（与工程规范对齐）

为保证与 `src/dl/late` / `src/dl/seq` 的样本口径一致，本 baseline 使用 `content.json` 的 `content_sequence` 做**统一序列截断**：

- `max_seq_len`：内容块窗口长度上限（控制计算量）
- `truncation_strategy`：
  - `first`：取最前窗口
  - `random`：在所有长度为 `max_seq_len` 的窗口中随机取一个；随机性与样本绑定（`sha256(project_id)` 与 `random_seed` 混合），保证可复现

截断窗口确定后：

- 对 `content_sequence` 中落在窗口内的 `image/text` 内容块，分别映射回 `image_*.npy / text_*.npy` 的子集；
- 再把 `cover_image_*.npy / title_blurb_*.npy` 分别拼到 image/text 序列最前面，得到最终输入序列。

### 2.5 缺失策略

当项目目录、`content.json` 或“必须的 embedding 文件”缺失时，由 `missing_strategy` 控制：

- `error`：直接抛错（默认，更利于保证论文实验的数据一致性）。
- `skip`：跳过该样本（会减少可用数据量；统计信息会记录跳过数量；若某个 split 最终无可用样本则报错）。

## 3. 数据划分

按比例切分 train/val/test（可选是否在切分前打乱），对应 `src/dl/mdl/config.py:MdlConfig`：

- `train_ratio / val_ratio / test_ratio`：默认 `0.6/0.2/0.2`
- `shuffle_before_split`：默认 `True`

每次运行会在 `reports/splits.csv` 写出每个 `project_id` 的 split 归属与标签，便于复核。

## 4. 模型结构（实现描述）

模型实现位于 `src/dl/mdl/model.py`，包含多分支融合网络；单/双分支通过关闭部分分支做消融，结构为：分支编码器 + 特征拼接 + 融合头。

### 4.1 meta 分支（表格特征）

- 输入：预处理后的表格特征向量 `x_meta ∈ R^{B×F}`。
- 结构：全连接 MLP（`MetaMLPEncoder`）
  - 典型默认：`Linear(F→256) + ReLU + Dropout(0.3)`

### 4.2 image 分支（图片向量序列）

- 输入：图片向量序列 `x_image ∈ R^{B×L×D}`。
- 结构：两层 1D CNN + 池化 + padding mask + 全局最大池化（`ImageCNNEncoder`）。

### 4.3 text 分支（文本向量序列）

- 输入：文本向量序列 `x_text ∈ R^{B×L×D}`。
- 结构：TextCNN（`TextCNNEncoder`），与 image 分支类似，但层数/配置以 `model.py` 实现为准。

### 4.4 多模态融合

- 对启用的分支分别得到定长向量表示后拼接；
- 融合头：`Linear → ReLU → Dropout → Linear(→1)`，其中 `fusion_hidden_dim` 在构建模型时按启用分支自动计算（默认 `2 * fusion_in_dim`）。

## 5. 训练与评估（工程规范对齐）

训练入口：`src/dl/mdl/main.py`，训练与评估逻辑：`src/dl/mdl/train_eval.py`。

- 损失：`BCEWithLogitsLoss` + 轻量 label smoothing
- 优化器：AdamW（`learning_rate_init`，`alpha` 作为 `weight_decay`）
- 学习率调度：warmup + cosine（每 step 更新；最小学习率由 `lr_scheduler_min_lr` 控制）
- 梯度裁剪：按 `max_grad_norm` 做全局 norm clip，并记录 `grad_norm`
- AMP：CUDA 时启用 autocast + GradScaler
- EMA：评估/early stopping 使用 EMA 权重，减少指标抖动并提升稳定性
- best checkpoint 口径：逐 epoch 仅计算阈值无关指标 `val_auc` / `val_log_loss`；优先使用 `val_auc`（验证集为单类导致 AUC 不可用则回退为 `val_log_loss`）；history/log 中不记录逐 epoch 阈值相关字段
- 阈值选择：best epoch 确定后，用该 checkpoint 的 `val_prob` 搜索一次 `best_threshold`（最大化 F1，并列取较小阈值），并用该阈值计算最终 train/val/test 指标；详见 `docs/dl_guidelines.md`

## 6. 配置文件与命令行参数

配置为单 dataclass：`src/dl/mdl/config.py:MdlConfig`。

命令行参数仅覆盖少量常用项：

- `--run-name`
- `--seed`（覆盖 `random_seed`）
- `--use-meta / --no-use-meta`
- `--use-image / --no-use-image`
- `--use-text / --no-use-text`
- `--image-embedding-type`
- `--text-embedding-type`
- `--device`
- `--gpu`（等价于 `--device cuda:N`，与 `--device` 互斥）

说明：只要命令行出现任一 `--use-meta/--use-image/--use-text`（或 `--no-use-*`），就以命令行为准，未显式指定的分支默认关闭。

## 7. 运行方法与产物位置

### 7.1 运行命令（从仓库根目录）

- 默认配置运行：
  - `conda run -n crowdfunding python src/dl/mdl/main.py`
- 指定嵌入类型与设备：
  - `conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed 42 --image-embedding-type clip --text-embedding-type clip --device cuda:0`
- 只使用 meta（关闭 image/text）：
  - `conda run -n crowdfunding python src/dl/mdl/main.py --run-name meta --seed 42 --use-meta --device cuda:0`
- 仅指定 GPU 序号（等价于 `--device cuda:N`）：
  - `conda run -n crowdfunding python src/dl/mdl/main.py --gpu 1`

### 7.2 输出目录结构

默认写入 `experiments/newtest/<mode>/<run_id>/`：

- `mode`：`mdl_meta+image+text` / `mdl_image+text` / `mdl_meta+text` ...（由分支开关组合得到）
- `run_id`：时间戳（可附带 `run_name` 后缀）

目录结构：

- `artifacts/`：`model.pt`（best 权重 + best_info + best_threshold 等）、`preprocessor.pkl` / `feature_names.txt`（仅 `use_meta=True`）
- `reports/`：`train.log`、`config.json`、`history.csv`、`metrics.json`、`splits.csv`、`predictions_val.csv` / `predictions_test.csv`、`result.csv`
- `plots/`：`history.png`、`roc_val.png`、`roc_test.png`（若 `save_plots=True`；图内无中文）
