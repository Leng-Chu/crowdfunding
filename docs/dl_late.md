# 图文分别建模 + 晚期融合 baseline（`src/dl/late`）说明

## 1. 任务定义与输入输出

- **任务**：Kickstarter 项目二分类（成功/失败）。
- **标签**：来自表格数据 `state` 字段（`successful` 视为正类，其余为负类）。
- **输入**：
  - `image`：项目图片向量集合（embedding set，不建模顺序），其中 **封面图向量** 会拼接在集合最前面。
  - `text`：项目文本向量集合（embedding set，不建模顺序），其中 **title/blurb 向量** 会拼接在集合最前面。
  - `meta`（可选）：表格元数据特征（类别 + 数值），开关为 `use_meta`。
- **输出**：二分类 **logits**（训练使用 `BCEWithLogitsLoss`），推理时对 logits 施加 `sigmoid` 得到正类概率。

该 baseline 的核心约束为：
- 使用 `content_sequence` 做**统一序列截断**（控制每个样本使用的内容块数量），但不进行图文交替顺序建模。
- 不使用任何 position encoding，不泄露位置信息。
- 不使用跨模态 attention，不进行图文 token 级交互。

## 2. 数据组织与文件约定（关键）

### 2.1 表格数据（metadata CSV）

默认配置见 `src/dl/late/config.py:LateConfig`：

- `data_csv`：默认 `data/metadata/now_processed.csv`
- 关键列名：
  - `id_col`：`project_id`
  - `target_col`：`state`
  - `categorical_cols`：`category, country, currency`
  - `numeric_cols`：`duration_days, log_usd_goal`
  - `drop_cols`：默认丢弃 `project_id, time`（由预处理器内部决定是否使用）

表格预处理由 `src/dl/late/data.py:TabularPreprocessor` 完成（one-hot + 数值标准化），并在产物中保存，便于复现。

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

其中 `content.json` 必须包含字段 `content_sequence`，它是按页面呈现顺序排列的列表；每个元素包含 `type ∈ {"text", "image"}`。

### 2.3 向量文件命名（与 `src/preprocess/embedding` 对齐）

`src/dl/late/data.py` 约定如下文件名（`{emb_type}` 来自配置项 `image_embedding_type/text_embedding_type`）：

- **图片侧**（`image_embedding_type ∈ {clip, siglip, resnet}`）
  - `cover_image_{emb_type}.npy`，形状通常为 `[1, D_img]`（必须存在）
  - `image_{emb_type}.npy`，形状 `[N_img, D_img]`
- **文本侧**（`text_embedding_type ∈ {bge, clip, siglip}`）
  - `title_blurb_{emb_type}.npy`，形状为 `[1, D_txt]` 或 `[2, D_txt]`（必须存在）
  - `text_{emb_type}.npy`，形状 `[N_txt, D_txt]`

一致性要求（默认严格）：
- `cover_image_{emb_type}.npy` 与 `title_blurb_{emb_type}.npy` 必须存在，且其维度必须分别与正文 `image/text` 向量维度一致。
- `content_sequence` 中 `image/text` 的总数量必须与 `image_*.npy / text_*.npy` 的行数一致；不一致默认直接报错并包含项目 id。
- 若某项目 `content_sequence` 中某一模态的数量为 0，则允许该模态的 `.npy` 文件缺失（该样本该模态集合长度视为 0）。

### 2.4 缺失策略

当项目目录、`content.json` 或“必须的 embedding 文件”缺失时，由 `missing_strategy` 控制：

- `error`：直接抛错（默认，更利于保证论文实验的数据一致性）。
- `skip`：跳过该样本（会减少可用数据量；统计信息会记录缺失数量）。

## 3. 数据划分

按比例切分 train/val/test（可选是否在切分前打乱），对应 `src/dl/late/config.py:LateConfig`：

- `train_ratio / val_ratio / test_ratio`：三段比例（默认 `0.6/0.2/0.2`）
- `shuffle_before_split`：是否在切分前打乱（默认 `False`）

为保证与其他 baseline 横向对比，建议固定 `random_seed`，并保持数据划分配置一致。

## 4. 统一序列截断与映射回模态集合（关键）

### 4.1 统一序列截断

先读取 `content.json` 的 `content_sequence`，按 `max_seq_len` 对其截断得到前 `L` 个内容块：

- `truncation_strategy=first`：取前 `L` 个内容块
- `truncation_strategy=random`：从所有长度为 `L` 的窗口中随机选一个（对每个项目基于 `random_seed + hash(project_id)` 固定，保证可复现）

注意：该截断仅用于控制每个样本使用的内容块数量，不用于顺序建模。

### 4.2 从截断后的序列映射回图像集合与文本集合

对截断后的 `L` 个内容块，维护两个计数器 `img_idx=0, txt_idx=0`，逐块扫描：

- 若当前块为 `image`：将当前 `img_idx` 追加到 `keep_img_indices`，然后 `img_idx += 1`
- 若当前块为 `text`：将当前 `txt_idx` 追加到 `keep_txt_indices`，然后 `txt_idx += 1`

随后根据索引从 embedding 中取子集：

- `img_keep = image_emb[keep_img_indices]`
- `txt_keep = text_emb[keep_txt_indices]`

最后将前缀向量拼接到集合最前面：

- `img_keep = concat([cover_image, img_keep])`
- `txt_keep = concat([title_blurb, txt_keep])`

注意：`cover_image/title_blurb` **不参与** `content_sequence` 的统一序列截断，它们总是被保留。

对 batch 做 padding：

- `img_emb ∈ [B, N_img_keep_max, D_img]`，并保存 `len_image`（或等价 mask）
- `txt_emb ∈ [B, N_txt_keep_max, D_txt]`，并保存 `len_text`（或等价 mask）

## 5. 模型结构：分别集合编码 + 晚期融合

模型实现位于 `src/dl/late/model.py`，结构为：模态投影 + 模态内集合编码 + concat 融合 + 分类头。

### 5.1 Modality Projection

- `img_proj: Linear(D_img → d_model) + ReLU`
- `txt_proj: Linear(D_txt → d_model) + ReLU`

得到：
- `Img ∈ [B, N_img, d_model]`
- `Txt ∈ [B, N_txt, d_model]`

### 5.2 模态内集合编码（不使用顺序）

配置项 `baseline_mode ∈ {attn_pool, trm_no_pos}`：

1) `baseline_mode=attn_pool`

- 每个模态一个可学习全局 query：`q_img, q_txt ∈ R^{d_model}`
- 单头 scaled dot-product attention（手写实现）
  - 使用 mask；padding 位置在 softmax 前置为 `-inf`
  - `K = Linear(d_model → d_model)`，`V = Linear(d_model → d_model)`
- 输出：
  - `h_img ∈ [B, d_model]`
  - `h_txt ∈ [B, d_model]`

2) `baseline_mode=trm_no_pos`（容量对照）

- 分别对 `Img` 和 `Txt` 使用 `TransformerEncoder`（不使用任何 position encoding）
- 使用 `src_key_padding_mask`
- Encoder 输出后对每个模态做 masked mean pooling
- 输出：
  - `h_img ∈ [B, d_model]`
  - `h_txt ∈ [B, d_model]`

通用禁止项（两种 `baseline_mode` 均适用）：
- 不使用 position encoding
- 不使用跨模态 attention
- 不进行图文 token 级交互

### 5.3 晚期融合与分类头（与 mlp baseline 对齐）

- 融合：`h = concat(h_img, h_txt)`
- 若 `use_meta=True`：
  - `meta_h = MetaEncoder(meta_features)`（结构与 mlp baseline 一致）
  - `h = concat(h, meta_h)`
- 分类头（与 mlp baseline 完全一致）：`Linear → ReLU → Dropout → Linear(→1)`
- 输出：`logits ∈ [B]`

## 6. 训练与评估

训练入口见 `src/dl/late/main.py`，训练与评估逻辑在 `src/dl/late/train_eval.py`。

- 损失：`BCEWithLogitsLoss`
- 优化器：Adam
  - 学习率：`learning_rate_init`
  - 权重衰减（L2）：`alpha`
- 早停：`early_stop_patience`，并支持最小训练轮数 `early_stop_min_epochs`
- 评估指标：`accuracy, precision, recall, f1, roc_auc, log_loss`
- 阈值（工程规范）：最终报告指标时，阈值由验证集选取（最大化 F1），并将该阈值用于测试集评估；详见 `docs/dl_threshold.md`

注意：作图时图里不要有中文，因此图标题/坐标轴保持英文（实现见 `src/dl/late/utils.py`）。

## 7. 配置文件与命令行参数

配置文件为单 dataclass：`src/dl/late/config.py:LateConfig`。

命令行参数仅包括（出现则覆盖配置文件对应项）：

- `--run-name`
- `--use-meta / --no-use-meta`
- `--image-embedding-type`
- `--text-embedding-type`
- `--baseline-mode`
- `--device`
- `--gpu`（等价于 `--device cuda:N`，与 `--device` 互斥）

其余超参建议直接编辑 `LateConfig` 默认值以保持实验可控与可复现。

## 8. 运行方法与产物位置

### 8.1 运行命令（从仓库根目录）

- 默认配置运行：
  - `conda run -n crowdfunding python src/dl/late/main.py`
- 指定 run_name 后缀（用于产物目录命名）：
  - `conda run -n crowdfunding python src/dl/late/main.py --run-name debug`
- 关闭 meta（只做图文晚期融合）：
  - `conda run -n crowdfunding python src/dl/late/main.py --no-use-meta`
- 指定嵌入类型 / baseline 模式 / 显卡：
  - `conda run -n crowdfunding python src/dl/late/main.py --image-embedding-type clip --text-embedding-type bge --baseline-mode attn_pool --device cuda:0`

### 8.2 输出目录结构（与 mlp baseline 对齐）

默认写入 `experiments/late/<mode>/<run_id>/`：

- `artifacts/`：模型权重（`model.pt`）、表格预处理器（`preprocessor.pkl`）等可复现产物
- `reports/`：`config.json`、`metrics.json`、`history.csv`、`splits.csv`、预测结果 CSV 等
- `plots/`：训练曲线与 ROC 图（若 `save_plots=True`）

其中：
- `mode` 取 `late_attn_pool` / `late_trm_no_pos`，并按需追加 `+meta`
- `baseline_mode`（配置项）取 `attn_pool` 或 `trm_no_pos`

## 9. 缓存与工程性细节（简述）

为避免每次训练都逐个 `np.load` 读取 embedding，`src/dl/late/data.py` 支持将“已构建好的 numpy 特征张量”缓存为 `.npz`：

- 开关：`use_cache`
- 目录：`cache_dir`（默认 `experiments/late/_cache`）
- cache key 包含：embedding type（image/text）、`max_seq_len`、`truncation_strategy`（以及数据切分/列配置等复现信息）
