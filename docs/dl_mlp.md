# MLP 多模态二分类模型（`src/dl/mlp`）说明

## 1. 任务定义与输入输出

- **任务**：Kickstarter 项目二分类（成功/失败）。
- **标签**：来自表格数据 `state` 字段（`successful` 视为正类，其余为负类）。
- **输入模态**（可按开关自由组合）：
  - `meta`：表格元数据特征（类别 + 数值）。
  - `image`：项目封面与正文图片的向量序列（embedding sequence）。
  - `text`：项目标题/简介与正文文本的向量序列（embedding sequence）。
- **输出**：二分类 **logits**（训练使用 `BCEWithLogitsLoss`），推理时对 logits 施加 `sigmoid` 得到正类概率。

代码层面通过 `use_meta / use_image / use_text` 三个布尔开关控制启用哪些分支，并自动根据启用分支计算融合层输入维度与隐藏层大小。

## 2. 数据组织与文件约定（关键）

### 2.1 表格数据（metadata CSV）

默认配置见 `src/dl/mlp/config.py:MlpConfig`：

- `data_csv`：默认 `data/metadata/now_processed.csv`
- 关键列名：
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
  cover/
    cover_image.jpg|jpeg|png|webp
  content.json
  (向量文件若干 .npy，见下节)
```

### 2.3 向量文件命名（与 `src/preprocess/embedding` 对齐）

`src/dl/mlp/data.py` 约定如下文件名（`{emb_type}` 来自配置项 `image_embedding_type/text_embedding_type`）：

- **图片侧**（`image_embedding_type ∈ {clip, siglip, resnet}`）
  - 必须：`cover_image_{emb_type}.npy`（封面图，形状通常为 `[1, D]`）
  - 可选：`image_{emb_type}.npy`（正文图片序列，形状通常为 `[K, D]`）
- **文本侧**（`text_embedding_type ∈ {bge, clip, siglip}`）
  - 必须：`title_blurb_{emb_type}.npy`（标题+简介，形状可能为 `[1, D]` 或 `[2, D]`）
  - 可选：`text_{emb_type}.npy`（正文文本分段序列，形状通常为 `[T, D]`）

向量生成细节请参见 `docs/preprocess_embedding.md`。

### 2.4 缺失策略

当项目目录或必须向量文件缺失时，由 `missing_strategy` 控制：

- `error`：直接抛错（默认，更利于保证论文实验的数据一致性）。
- `skip`：跳过该样本（会减少可用数据量；统计信息会记录缺失数量）。

## 3. 数据划分（ratio / kfold）

由 `split_mode` 控制：

- `ratio`：按比例切分 train/val/test（默认 `0.7/0.15/0.15`，可选是否在切分前打乱）。
- `kfold`：K 折交叉验证（实现为轻量级 KFold/StratifiedKFold，不依赖 sklearn）：
  - 先按 `test_ratio` 固定一个**独立测试集**（可通过 `shuffle_before_split` 控制是否在切分前打乱；若为 `False`，通常等价于取 CSV 最后 `test_ratio` 的样本作为测试集）
  - 测试集不参与任何折、不用于早停/调参
  - 在剩余 `trainval` 上做 K 折：每折用 `k-1` 折做 train，用剩余 `1` 折做 val，轮换 `k` 次
  - `k_folds`：折数
  - `kfold_shuffle`：是否打乱
  - `kfold_stratify`：是否按标签分层（默认开启）
  - `k_fold_index`：`-1` 表示跑全折；指定 `0..k-1` 可只跑某一折（便于调试）

## 4. 模型结构（可直接写入论文的实现描述）

模型实现位于 `src/dl/mlp/model.py`，包含单分支与多分支融合两类网络，结构为：分支编码器 + 特征拼接 + 融合头。

### 4.1 meta 分支（表格特征）

- 输入：预处理后的表格特征向量 `x_meta ∈ R^{B×F}`。
- 结构：全连接 MLP（`MLPBinaryClassifier` / `MetaMLPEncoder`）
  - 典型默认：`Linear(F→256) + ReLU + Dropout(0.3)`（作为 encoder 输出定长向量）
  - 单分支分类器会在末尾追加 `Linear(·→1)` 输出 logits。
- 初始化：线性层使用 Xavier 初始化。

表格预处理在 `src/dl/mlp/data.py:prepare_meta_data` 中完成（one-hot 编码 + 数值标准化等），并将预处理器与特征名保存到实验产物中，便于复现。

### 4.2 image 分支（图片向量序列）

- 输入：图片向量序列 `x_image ∈ R^{B×L×D}`，其中 `L` 为序列长度，`D` 为图片向量维度。
- 结构：两层 1D CNN + 池化 + 全局最大池化（`ImageCNNEncoder` / `ImageCNNBinaryClassifier`）
  - 将输入转置为 `R^{B×D×L}` 后做 `Conv1d`（same padding，kernel size 为奇数）
  - 每层卷积后：`BatchNorm(可选) + ReLU + MaxPool1d(stride=2, ceil_mode=True) + Dropout`
  - 通过 `lengths` 对 padding 位置 mask 后做 **全局最大池化** 得到定长向量
  - 经过 `Linear + ReLU + Dropout` 后进入分类头（或作为 encoder 输出）
- 初始化：卷积层 Kaiming 初始化、线性层 Xavier 初始化。

### 4.3 text 分支（文本向量序列）

- 输入：文本向量序列 `x_text ∈ R^{B×L×D}`。
- 结构：与 image 分支类似的 TextCNN（`TextCNNEncoder` / `TextCNNBinaryClassifier`），差异在于文本侧使用三层卷积/池化（以适配更长的序列；具体实现以 `model.py` 为准）。

### 4.4 多模态融合（两路/三路）

- 对启用的分支分别得到定长向量表示：
  - `h_meta ∈ R^{B×d_m}`
  - `h_image ∈ R^{B×d_i}`
  - `h_text ∈ R^{B×d_t}`
- 拼接：`h = concat([h_*], dim=1) ∈ R^{B×(d_m+d_i+d_t)}`
- 融合头：`Linear(fusion_in_dim → fusion_hidden_dim) + ReLU + Dropout + Linear(→1)`
  - 其中 `fusion_hidden_dim` 若未显式指定，则取 `2 * fusion_in_dim`（代码自动计算，确保与启用分支一致）。

## 5. 训练与评估

训练入口见 `src/dl/mlp/main.py`，训练与评估逻辑在 `src/dl/mlp/train_eval.py`。

- 损失：`BCEWithLogitsLoss`
- 优化器：Adam
  - 学习率：`learning_rate_init`
  - 权重衰减（L2）：`alpha`
- 学习率调度（可选）：`ReduceLROnPlateau`（监控 `val_log_loss`）
- 早停：`early_stop_patience`，并支持最小训练轮数 `early_stop_min_epochs`
- 评估指标：`accuracy, precision, recall, f1, roc_auc, log_loss`（实现于 `src/dl/mlp/utils.py`）
- 阈值：`threshold`（默认 0.5）

## 6. 配置文件说明（`src/dl/mlp/config.py`）

配置采用单文件 dataclass：`MlpConfig`。实验记录中保存有 `reports/config.json`（代码已自动输出）：

1. **数据与路径**：`data_csv, projects_root, experiment_root`
2. **分支开关**：`use_meta, use_image, use_text`
3. **嵌入类型**：`image_embedding_type, text_embedding_type`
4. **序列截断策略**：
   - `max_image_vectors, image_select_strategy ∈ {first, random}`
   - `max_text_vectors, text_select_strategy ∈ {first, random}`
5. **模型超参**：
   - `meta_hidden_dim, meta_dropout`
   - `image_conv_channels, image_conv_kernel_size, image_input_dropout, image_dropout, image_use_batch_norm`
   - `text_conv_kernel_size, text_input_dropout, text_dropout, text_use_batch_norm`
   - `fusion_dropout`（融合层 dropout）
6. **训练超参**：
   - `learning_rate_init, alpha, batch_size`
   - `max_epochs, early_stop_*`
   - `use_lr_scheduler, lr_scheduler_*`
   - `random_seed`

## 7. 运行方法与产物位置

### 7.1 运行命令（从仓库根目录）

- 默认配置运行：
  - `conda run -n crowdfunding python src/dl/mlp/main.py`
- 指定嵌入类型与设备：
  - `conda run -n crowdfunding python src/dl/mlp/main.py --run-name clip --image-embedding-type clip --text-embedding-type clip --device cuda:0`
- 仅指定 GPU 序号（等价于 `--device cuda:N`）：
  - `conda run -n crowdfunding python src/dl/mlp/main.py --gpu 1`

命令行参数仅覆盖少数常用项；其余超参建议直接编辑 `src/dl/mlp/config.py` 中的默认值。

### 7.2 输出目录结构

默认写入 `experiments/mlp/<run_id>/`：

- `artifacts/`：模型权重（`model.pt`）、表格预处理器（`preprocessor.pkl`）等可复现产物
- `reports/`：`config.json`、`metrics.json`、`history.csv`、`splits.csv`、预测结果 CSV 等
- `plots/`：训练曲线与 ROC 图（若 `save_plots=True`）

当 `split_mode=kfold` 时，会在 `reports/fold_XX/` 等子目录下写入每折结果，并额外生成：
- `reports/cv_metrics.json`：汇总每折指标 + OOF 验证集指标 + 测试集集成指标
- `reports/cv_predictions_val.csv`：OOF 验证集预测（每个样本只在其所属 val 折被预测一次）
- `reports/test_predictions_ensemble.csv`：固定测试集的集成预测（对每个测试样本汇聚各折模型的概率并取平均）

## 8. 缓存与工程性细节（简述）

为避免每次训练都逐个 `np.load` 读取向量，`src/dl/mlp/data.py` 支持将“已构建好的 numpy 特征张量”缓存为 `.npz`：

- 开关：`use_cache`
- 目录：`cache_dir`（默认 `experiments/mlp/_cache`）
