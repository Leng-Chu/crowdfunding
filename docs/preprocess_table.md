# 表格数据预处理方案（`src/preprocess/table`）说明

本文件描述 `src/preprocess/table` 中用于构建“表格元数据特征（meta）”的数据清洗与特征工程流程，便于论文写作时说明数据处理步骤与可复现产物。实现脚本以一次性处理脚本为主，工程细节仅简述。

## 1. 输入与输出

### 1.1 输入

- 原始 metadata CSV：通常为 `data/metadata/all.csv` 或按时间段/年份切分的 CSV。
- 对应的项目目录：`data/projects/<dataset>/<project_id>/...`（用于后续 embedding 与训练阶段对齐）。

### 1.2 输出（核心）

- “可用于训练的表格数据”CSV：`data/metadata/now_processed.csv`
  - 包含：`project_id`、`time`（由 `launched_at` 转换）、少量结构化特征、以及二分类标签 `state`（0/1）。
- “被过滤/被移动样本”的记录 CSV：例如 `data/metadata/move1.csv`、`data/metadata/move2.csv`（用于审计被删除样本原因）。
- （可选）被移动的项目文件夹：例如 `data/projects/move1/`、`data/projects/move2/`（用于与 metadata 同步裁剪数据集）。

## 2. 清洗与裁剪策略

该目录下脚本可概括为 3 个阶段：

1. **样本过滤（质量控制 + 低频裁剪）**：删除不可靠/过于稀疏的样本与类别。
2. **按年份类别平衡（可选）**：在每个年份内部对成功/失败样本进行下采样以减小类别偏斜。
3. **特征工程与格式化**：生成用于模型训练的结构化特征列，并输出最终 CSV。

下面按脚本说明具体规则。

## 3. 具体脚本与规则

### 3.1 `filter.py`：过滤低质量样本 + 移动对应项目目录

目标：将“缺失关键字段/低频类别/异常持续时间”的项目剔除，并同步移动对应项目文件夹，确保 metadata 与 projects 目录一致。

主要规则（以代码为准）：

- **缺失 `category_parent`** 的样本剔除。
- **持续时间过短**剔除：
  - 计算 `duration_days = (deadline - launched_at) / 86400`
  - 若 `duration_days < 5` 则剔除。
- **低频裁剪**剔除（统计频次来自原始 CSV）：
  - `currency` 出现次数 `< 100`
  - `country` 出现次数 `< 100`
  - `category` 出现次数 `< 50`
- 对被剔除样本：将 `data/projects/<dataset>/<project_id>/` 移动到 `data/projects/move1/<project_id>/`，并写出被移动样本清单 CSV（便于审计）。

该步骤的核心目的是减少长尾类别带来的极端稀疏特征，同时确保后续 embedding 与训练阶段不会在 projects_root 中读到“metadata 已删除但目录仍存在”的样本。

### 3.2 `balance_year.py`：按年份进行类别平衡（可选）

目标：在每个年份内部，将成功/失败样本数量平衡（通过对多数类下采样），缓解标签不均衡对训练与评估的影响。

处理逻辑：

- 从 `launched_at` 提取年份 `year`。
- 对每个 `year`：
  - 分别统计 `successful` 与 `failed` 数量。
  - 对多数类随机抽样下采样，使两类样本数相等（`random_state=42`）。
- 同步移动被删除样本的项目目录到 `data/projects/move2/`，并记录 `move2.csv`。

### 3.3 `clean.py`：生成最终训练用表格数据（`now_processed.csv`）

目标：从清洗后的 metadata 构建最终可训练的结构化特征，并生成 `data/metadata/now_processed.csv`。

关键处理：

1. **保留对齐用字段**
   - `project_id`
   - `time`：将 `launched_at` 转换为日期字符串（`YYYY-MM-DD`），并重命名为 `time`（便于按时间排序/做时序划分）。
2. **删除不用于 meta 特征的列**
   - 例如：抓取状态字段、URL 字段、文本字段（`title/blurb`）等。
3. **标签构建**
   - `state = 1` 若原始 `state == "successful"`，否则为 `0`。
4. **数值特征构建**
   - `duration_days`：`(deadline - launched_at)` 的天数差
   - `log_usd_goal`：对 `usd_goal` 做 `log1p` 变换，并删除原始 `usd_goal`
5. **按时间排序**
   - 使用 `time` 排序，保证整行数据随时间移动（为后续“按时间顺序切分”做准备）。

输出列（默认训练用）与 `src/dl/mlp/config.py` 对齐：`category/country/currency/duration_days/log_usd_goal/state`，以及对齐字段 `project_id/time`。

### 3.4 `check_status.py`：数据概览与缺失检查（辅助）

目标：快速检查清洗后的表格数据的列类型、缺失率、类别分布等，用于人工 sanity check。

输出内容包括：

- 特征矩阵 `X` 的 shape、成功率（`y.mean()`）
- 数值列/类别列列表
- 缺失率 top-k
- 数值列统计（`describe()`）
- 类别列 value counts

### 3.5 `machine_learning.py`：传统机器学习基线（辅助）

目标：在 `now_processed.csv` 上给出传统 ML 基线结果，作为深度模型的参照。

实现要点：

- 特征处理：
  - 类别特征 one-hot
  - 数值特征标准化（StandardScaler）
- 模型：Logistic Regression / Decision Tree / Random Forest
- 切分方式：
  - random split（80/20）
  - sequential split（按 CSV 顺序前 80% 训练，后 20% 测试）
- 结果保存：`experiments/meta_ml/ml_results_report.txt`
