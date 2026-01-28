# seq 图文内容块序列建模（`src/dl/seq`）说明

## 1. 任务定义与输入输出

- **任务**：Kickstarter 项目二分类（成功/失败）。
- **标签**：来自表格数据 `state` 字段（`successful` 视为正类，其余为负类；同时兼容 0/1 数值标签）。
- **输入**：预先计算好的 embedding（不涉及原始模态特征的编码器）。
  - **内容序列侧**：由 `content.json` 的 `title/blurb/cover_image` 与 `content_sequence` 构造的“图文交替统一序列”，并读取对应的 embedding 文件：
    - 前缀（固定拼在最前面）：`title → blurb → cover_image`
    - 正文：按 `content_sequence` 的原始顺序追加
    - 对应向量：`cover_image_*.npy / title_blurb_*.npy / image_*.npy / text_*.npy`
  - **可选 meta**：表格元数据特征（类别 + 数值），仅在分类前与 pooled 表示拼接（`use_meta=True` 时启用）。
- **输出**：二分类 **logits**（训练使用 `BCEWithLogitsLoss`），推理时对 logits 施加 `sigmoid` 得到正类概率。

> 约束（用于论文/实验对齐）：
> - 同一配置下，仅修改 `baseline_mode` 即可复现实验组。
> - 所有 baseline 的 token 输入 `X` 必须完全一致（该阶段不包含任何位置信息）。
> - 顺序信息仅通过 position encoding 注入（仅对 `trm_pos / trm_pos_shuffled` 生效）。

---

## 2. 数据组织与文件约定（关键）

### 2.1 表格数据（metadata CSV）

默认配置见 `src/dl/seq/config.py:SeqConfig`：

- `data_csv`：默认 `data/metadata/now_processed.csv`
- 关键列名（与 `docs/dl_mlp.md` 对齐）：
  - `id_col`：`project_id`
  - `target_col`：`state`
  - `categorical_cols`：`category, country, currency`
  - `numeric_cols`：`duration_days, log_usd_goal`
  - `drop_cols`：默认丢弃 `project_id, time`

其中 `duration_days` 与 `log_usd_goal` 为派生特征（生成方式见 `docs/preprocess_table.md`）。

### 2.2 项目目录（projects_root）

`projects_root` 默认 `data/projects/now`，每个项目一个文件夹：

```
data/projects/<dataset>/<project_id>/
  content.json
  cover_image_{emb_type}.npy
  title_blurb_{emb_type}.npy
  image_{emb_type}.npy
  text_{emb_type}.npy
```

说明：
- 本实验同时使用 **标题/简介/封面图** 与 **正文内容块序列**：
  - 封面图：`cover_image_{emb_type}.npy`（必须存在，形状通常为 `[1, D_img]`）
  - 标题/简介：`title_blurb_{emb_type}.npy`（必须存在，形状为 `[1, D_txt]` 或 `[2, D_txt]`）
  - 正文：`image_{emb_type}.npy` 与 `text_{emb_type}.npy`（当正文该模态数量为 0 时允许缺失）
- 图片尺寸与文本长度属性已预处理写入 `content.json`（见下节字段要求），代码不读取本地图片文件，以避免数据加载速度过慢。

### 2.3 统一序列与 embedding 对齐规则

`content.json` 中必须包含：

- `title`（字符串，可为空）
- `blurb`（字符串，可为空）
- `cover_image`（必须包含 `width/height` 的对象）
- `content_sequence`：按页面呈现顺序排列的列表；每个元素至少包含：

- `type ∈ {"text", "image"}`
- 若 `type="text"`：必须包含 `content_length`（预处理好的文本长度）；`content` 可选
- 若 `type="image"`：必须包含 `width/height`（预处理好的图片尺寸）；`filename` 可选

在 data loader 中，根据 `content_sequence` 构造统一的内容块序列：

- 统一序列的输入顺序为：
  1. `title`（若非空）
  2. `blurb`（若非空）
  3. `cover_image`（固定存在）
  4. `content_sequence`（按原始顺序）
- 维护两个指针：
  - `img_idx`：遇到 `"image"` 时取 `image_emb[img_idx]`
  - `txt_idx`：遇到 `"text"` 时取 `text_emb[txt_idx]`
- 按 `content_sequence` 顺序生成：
  - `seq_type = [0/1]`（0=text，1=image）
  - `seq_attr = [a_1, ..., a_L]`（属性，见下节）
  - 以及对应的 `image/text` embedding 取值

一致性要求（默认严格）：

- `cover_image_{emb_type}.npy` 必须存在，且行数为 1。
- `title_blurb_{emb_type}.npy` 必须存在，其行数必须等于 `title/blurb` 中非空字段的个数；且顺序固定为 `[title, blurb]`（缺失则不占位）。
- `content_sequence` 中 `"image"` 数量必须与 `image_{emb_type}.npy` 的行数一致。
- `content_sequence` 中 `"text"` 数量必须与 `text_{emb_type}.npy` 的行数一致。
- 不一致默认直接报错并记录项目 id（`missing_strategy=error`）。

### 2.4 内容块属性（attr）

对每个 token 计算 1 个标量属性，并在模型端做线性映射：

- **title/blurb**：属性为 `log(max(1, len(text)))`
- **封面图**：直接读取 `cover_image.width/cover_image.height`，计算 `area = width * height`，属性为 `log(max(1, area))`
- **正文文本块**：读取 `content_length`，属性为 `log(max(1, content_length))`
- **正文图片块**：读取 `width/height`，属性为 `log(max(1, area))`

字段要求：

- `type="text"`：必须包含 `content_length`（int）
- `type="image"`：必须包含 `width` 与 `height`（int）

### 2.5 截断与 mask

只采用一种截断方式：对交替后的统一序列按 `max_seq_len` 截断，支持两种策略：

- `truncation_strategy="first"`：取前 `max_seq_len`
- `truncation_strategy="random"`：在 `[0, L-max_seq_len]` 中随机选取一个窗口（为可复现，随机种子为 `random_seed + hash(project_id)`）

注意：截断发生在 **拼接完 title/blurb/cover_image 前缀之后**，因此当 `max_seq_len` 较小，会优先保留前缀并截掉部分正文 token。

输出：

- `seq_mask ∈ [B, max_seq_len]`：`True` 表示有效位置

### 2.6 缺失策略

当项目目录或必需文件缺失（或解析失败）时，由 `missing_strategy` 控制：

- `error`：直接抛错（默认，更利于保证论文实验的数据一致性）
- `skip`：跳过该样本（会减少可用数据量；统计信息与日志会记录缺失/跳过数量）

常见缺失/错误包括：
- 项目目录不存在
- `content.json` 缺失
- `content_sequence` 不合法
- 必需 embedding 文件缺失（当 `content_sequence` 中对应模态数量 > 0 时）
- 图片文件缺失或无法解析尺寸（影响 image 块属性）

---

## 3. 内容块的统一 token 表示（不含位置信息）

对每个内容块构造统一 token 表示（模型端实现见 `src/dl/seq/model.py:TokenEncoder`）：

- 内容 embedding：来自 image 或 text embedding
- 类型标识：image / text
- 块属性：`log(length)` 或 `log(area)`

映射方式：

- `img_proj: Linear(D_img → d_model) + ReLU`
- `txt_proj: Linear(D_txt → d_model) + ReLU`
- `type_embedding: Embedding(2, d_model)`
- `attr_proj: Linear(1 → d_model)`（输入为 `seq_attr`）

最终 token 表示：

```
x_i = proj(e_i) + type_embedding(t_i) + attr_proj(a_i)
X ∈ R^{B×L×d_model}
```

说明：

- 该阶段 **不包含任何位置信息**。
- 所有 baseline 必须使用完全相同的 `X`（顺序差异只允许体现在后续是否加入 position encoding / 是否打乱顺序）。

---

## 4. Baseline 模型（`baseline_mode`）

通过 `baseline_mode` 控制实验组（见 `src/dl/seq/config.py:SeqConfig.baseline_mode` 与 `src/dl/seq/model.py:SeqBinaryClassifier`）：

### 4.1 `set_mean`：Set-Mean Pooling

- 输入：`X, seq_mask`
- 操作：masked mean pooling
- 输出：`h ∈ [B, d_model]`

### 4.2 `set_attn`：Set-Attention Pooling

- 输入：`X, seq_mask`
- 使用单个全局 query 对序列做单头 attention pooling
- 不使用位置编码
- 输出 pooled 向量 `h`

### 4.3 `trm_no_pos`：Transformer without Position Encoding

- Transformer Encoder（不加入任何 position encoding）
- 使用 `src_key_padding_mask=~seq_mask`
- encoder 输出后做 masked mean pooling 得到 `h`

### 4.4 `trm_pos`：Transformer with Position Encoding（正确顺序）

- 在 `X` 上加入 position encoding（**固定使用 sinusoidal**，不提供可选项）
- 输入顺序为 `content_sequence` 的原始顺序
- Transformer Encoder + masked mean pooling 得到 `h`

### 4.5 `trm_pos_shuffled`：Transformer with Position Encoding + Shuffled Order

- 在 data 阶段对每个样本的有效 token 随机打乱（`X_img/X_txt/seq_type/seq_attr/seq_mask` 同步打乱；随机种子同样为 `random_seed + hash(project_id)`，保证可复现）
- position encoding 按打乱后的顺序重新编号（即新的位置 0..L-1）
- 其余结构与 `trm_pos` 完全一致

> 重要：该 baseline 不得泄露原始顺序或位置；顺序信息仅通过 position encoding 注入。

---

## 5. 融合与分类头（与 mlp baseline 对齐）

### 5.1 pooled 表示

得到 pooled 表示 `h ∈ [B, d_model]` 后：

- 若 `use_meta=True`：
  - meta encoder 与 `mlp baseline` 的结构与处理方式一致（one-hot + 标准化；encoder 为 `Linear → ReLU → Dropout`）
  - meta 仅在分类前与 `h` concat：`concat([h, h_meta])`

### 5.2 分类头

分类头结构与 `mlp baseline` 一致：

- `Linear → ReLU → Dropout → Linear(→1)`
- `fusion_hidden_dim<=0` 时自动取 `2 * fusion_in_dim`

---

## 6. 训练与评估

训练入口：`src/dl/seq/main.py`

训练与评估逻辑：`src/dl/seq/train_eval.py`

- 损失：`BCEWithLogitsLoss`
- 优化器：AdamW（`learning_rate_init`，`alpha` 作为 `weight_decay`）
- 学习率调度：warmup + cosine（每 step 更新；最小学习率由 `lr_scheduler_min_lr` 控制）
- 早停：`early_stop_patience`，并支持最小训练轮数 `early_stop_min_epochs`
- 评估指标：
  - `accuracy, precision, recall, f1, roc_auc, log_loss`
  - 若某个 split 只包含单一类别，`roc_auc` 可能为 `None`，并在指标里记录 `roc_auc_error`
- 阈值与 best 口径（工程规范）：训练阶段不做阈值搜索，每个 epoch 仅计算阈值无关指标 `val_auc`（若验证集为单类导致 AUC 不可用则回退为 `val_log_loss`），early stopping 与 best checkpoint 选择均基于该指标；在 best epoch 确定后，用该模型的 `val_prob` 搜索一次 `best_threshold`（最大化 F1），并用该阈值计算最终 train/val/test 的阈值相关指标；详见 `docs/dl_guidelines.md`
- 评估函数：`evaluate_seq_split` 仅返回 `prob`；二分类指标由调用方显式传入 `best_threshold` 计算（避免隐式阈值）

---

## 7. 产物目录结构

### 7.1 输出目录结构

默认写入 `experiments/seq/<mode>/<run_id>/`，其中：

- `mode = baseline_mode` 或 `baseline_mode+meta`（当 `use_meta=True` 时）

目录结构：

- `artifacts/`：
  - `model.pt`：best model 权重 + `best_epoch/best_val_auc/best_val_log_loss/best_threshold` + 关键超参
  - `preprocessor.pkl`：表格预处理器（仅 `use_meta=True`）
  - `feature_names.txt`：特征名（仅 `use_meta=True`）
- `reports/`：
  - `config.json`：完整配置（含 run_id、切分信息等）
  - `metrics.json`：train/val/test 指标
  - `history.csv`：逐 epoch 指标历史
  - `splits.csv`：每个 project_id 的 split 归属
  - `predictions_val.csv / predictions_test.csv`：预测概率与分类结果
  - `result.csv`：单行汇总（便于汇总多个实验）
  - `train.log`：训练日志
- `plots/`（若 `save_plots=True`）：
  - `history.png`：LogLoss/ROC-AUC 曲线（图内无中文）
  - `roc_val.png / roc_test.png`：ROC 曲线（图内无中文）

---

## 8. 运行方法（从仓库根目录）

- 默认配置运行：
  - `conda run -n crowdfunding python src/dl/seq/main.py`
- 指定 run_name（用于产物目录命名）：
  - `conda run -n crowdfunding python src/dl/seq/main.py --run-name exp1`
- 切换 baseline（建议只改这一项做横向对比）：
  - `conda run -n crowdfunding python src/dl/seq/main.py --baseline-mode set_mean`
  - `conda run -n crowdfunding python src/dl/seq/main.py --baseline-mode set_attn`
  - `conda run -n crowdfunding python src/dl/seq/main.py --baseline-mode trm_no_pos`
  - `conda run -n crowdfunding python src/dl/seq/main.py --baseline-mode trm_pos`
  - `conda run -n crowdfunding python src/dl/seq/main.py --baseline-mode trm_pos_shuffled`
- 指定嵌入类型与设备：
  - `conda run -n crowdfunding python src/dl/seq/main.py --baseline-mode trm_pos --image-embedding-type clip --text-embedding-type clip --device cuda:0`
- 仅指定 GPU 序号（等价于 `--device cuda:N`）：
  - `conda run -n crowdfunding python src/dl/seq/main.py --gpu 1`

说明：
- 命令行参数仅覆盖少数常用项；其余超参建议直接编辑 `src/dl/seq/config.py` 的默认值。

### 8.1 Optuna 自动化调参（仅 `trm_pos+meta`）

本仓库提供 `src/dl/seq/optuna_search.py` 用于自动化调参（黑盒调用现有训练入口 `src/dl/seq/main.py`）。

- 安装依赖：
  - `pip install optuna`
- 运行示例（默认最大化 `test_f1`）：
  - `conda run -n crowdfunding python src/dl/seq/optuna_search.py --device cuda:0 --n-trials 30`
- 切换优化目标（可选）：
  - `--objective val_auc` 或 `--objective val_accuracy`（也支持 `test_*`，但不建议用测试集做调参目标）
- 为所有 trial 固定覆盖项（可选，便于加速调参）：
  - `--fixed-overrides "{\"max_epochs\": 20, \"early_stop_min_epochs\": 3}"`

输出：
- Optuna 产物：`experiments/seq/optuna/<study_name>/summary.csv`、`best.json`、`study.db`、`trial_logs/`
- 每个 trial 的训练产物仍会写入：`experiments/ch1/trm_pos+meta/<run_id>/...`
