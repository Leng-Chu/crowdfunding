# MDL 多模态二分类说明（`src/dl/mdl`）

## 1. 任务与输入输出

- 任务：Kickstarter 项目二分类（`state`，0/1）。
- 输入：
  - `meta`（可选）：表格特征（类别 one-hot + 数值标准化）。
  - `image`：图像 embedding 序列。
  - `text`：文本 embedding 序列。
- 输出：单个 logit（训练用 `BCEWithLogitsLoss`，推理用 `sigmoid` 转概率）。

说明：当前 `main.py` 固定启用 `image+text`，`meta` 通过 `use_meta` 开关控制。

## 2. 数据口径与统一截断（已与 late 对齐）

数据文件约定（默认）：

- 元数据：`data/metadata/now_processed.csv`
- 项目目录：`data/projects/now/<project_id>/`
- 必要文件：
  - `content.json`
  - `cover_image_{image_embedding_type}.npy`
  - `title_blurb_{text_embedding_type}.npy`
- 条件必要文件：
  - `image_{image_embedding_type}.npy`（当正文有 image 块时）
  - `text_{text_embedding_type}.npy`（当正文有 text 块时）

统一截断语义：

1. 先构建统一序列（按 token 顺序）：
   - `title_blurb` 的每一行（text prefix）
   - `cover_image`（image prefix）
   - `content_sequence` 中正文块（保持原顺序）
2. 再在统一序列上做整体窗口截断（`max_seq_len` + `truncation_strategy`）。
3. 最后将截断后的 token 映射回 image/text 两个序列。

同时生成 token 级属性（与 `late` 对齐）：

- `attr_image`：每个 image token 对应 `log(max(1, area))`。
  - `cover_image` 的面积来自 `content.json.cover_image.width/height`。
  - 正文 image 的面积来自 `content_sequence` 中该块的 `width/height`。
- `attr_text`：每个 text token 对应 `log(max(1, length))`。
  - `title/blurb` 长度优先取 `content_length`，缺失时回退 `content` 字符串长度。
  - 正文 text 长度同样优先取 `content_length`。

属性开关（`use_seq_attr`）：

- `use_seq_attr=True`（默认）：按上面的规则构造并使用属性。
- `use_seq_attr=False`：仍保留 `attr_image/attr_text` 张量，但全部置 0，模型前向不注入属性。

`truncation_strategy`：

- `first`：取前 `max_seq_len`。
- `random`：从所有长度为 `max_seq_len` 的窗口随机取一个，seed 为 `random_seed + sha256(project_id)`，保证可复现。

注意：prefix 会参与统一截断，不再“永远保留”。因此某个样本在截断后可能出现单模态空序列（`len=0`），模型侧已做兼容。

## 3. 模型结构

代码位置：`src/dl/mdl/model.py`

- `MetaMLPEncoder`：处理表格特征。
- `ImageCNNEncoder`：处理图像 embedding 序列。
- `TextCNNEncoder`：处理文本 embedding 序列。
- `MultiModalBinaryClassifier`：拼接分支特征后，经过融合 head 输出二分类 logit。

实现细节：

- image/text 分支会显式接收 `attr_image/attr_text`，并通过线性投影后加到 token embedding 上，再进入 CNN。
- image/text 分支均支持 `length=0` 样本（返回零向量，避免数值异常）。
- 融合层 `fusion_hidden_dim` 支持手动配置（默认 `512`，与 `seq` 一致）；当配置 `<=0` 时自动取 `2 * fusion_in_dim`。

## 4. 训练与 best checkpoint 规则（已与 late 对齐）

代码位置：`src/dl/mdl/train_eval.py`

训练技巧：

- Label Smoothing：启用（`_LABEL_SMOOTHING_EPS = 0.05`）。
- EMA：启用（`_EMA_DECAY = 0.999`）。
- AMP：CUDA 下启用（训练和概率推理均使用 autocast/GradScaler）。
- 学习率调度：warmup + cosine，按 step 更新（与 `late` 一致）。

best checkpoint 规则：

- 固定优先 `val_auc`（越大越好）。
- 若验证集单类导致 AUC 不可用，回退到 `val_log_loss`（越小越好）。
- 并列规则：
  - 当 metric 为 `val_auc`：`val_auc` 降序 -> `val_log_loss` 升序 -> `epoch` 升序。
  - 当 metric 为 `val_log_loss`：`val_log_loss` 升序 -> `epoch` 升序。

## 5. 阈值规则（并列取更小阈值）

代码位置：`src/dl/mdl/utils.py`

- 在 best epoch 对应 checkpoint 的 `val_prob` 上搜索 `best_threshold`（最大化 F1）。
- 若多个阈值 F1 并列，取更小阈值。
- 最终 train/val/test 的阈值相关指标统一使用该阈值计算。

## 6. 默认超参（与 late 对齐）

配置位置：`src/dl/mdl/config.py`

- `batch_size = 256`
- `max_epochs = 80`
- `alpha = 4e-4`（AdamW 的 `weight_decay`）

其余超参以 `MdlConfig` 为准。

## 7. 运行方式

从仓库根目录运行：

- 默认：
  - `conda run -n crowdfunding python src/dl/mdl/main.py`
- 指定 run_name、seed、设备：
  - `conda run -n crowdfunding python src/dl/mdl/main.py --run-name clip --seed 42 --device cuda:0`
- 指定 embedding 类型：
  - `conda run -n crowdfunding python src/dl/mdl/main.py --image-embedding-type clip --text-embedding-type clip --device cuda:0`
- 关闭 meta：
  - `conda run -n crowdfunding python src/dl/mdl/main.py --no-use-meta --device cuda:0`
- 关闭属性注入：
  - `conda run -n crowdfunding python src/dl/mdl/main.py --no-use-seq-attr --device cuda:0`

## 8. 批量脚本

仓库已提供批量运行脚本：

- `src/scripts/run/mdl_run_all.py`

示例：

- 单个 seed 运行预设组合：
  - `python src/scripts/run/mdl_run_all.py all --seed 42`
- seed 区间运行：
  - `python src/scripts/run/mdl_run_all.py single --start-seed 42 --end-seed 46`
- 批量运行时关闭属性注入：
  - `python src/scripts/run/mdl_run_all.py all --seed 42 --no-use-seq-attr`

## 9. 产物目录

输出目录：`experiments/newtest/<mode>/<run_id>/`

- `artifacts/`：`model.pt`、`preprocessor.pkl`（启用 meta 时）等。
- `reports/`：`config.json`、`history.csv`、`metrics.json`、`predictions_*.csv`、`splits.csv`、`result.csv`。
- `plots/`：`history.png`、`roc_val.png`、`roc_test.png`（图中不使用中文）。
