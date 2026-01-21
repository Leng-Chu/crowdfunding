# Gate 两阶段门控多模态模型（`src/dl/gate`）说明

## 1. 模型结构

该模型固定包含三条分支，不再提供选择分支的开关：

1. **meta 分支**：表格元数据（类别 + 数值）→ MLP 编码得到向量。
2. **第一印象分支**：`title_blurb` 序列嵌入与 `cover_image` 序列嵌入分别编码（1D CNN + global max pooling），再通过两层 MLP 融合得到向量。
3. **图文序列分支**：正文 `text` 序列嵌入与正文 `image` 序列嵌入分别编码（1D CNN + global max pooling），再通过两层 MLP 融合得到向量。

最终融合不再使用 MLP，而是使用**两阶段门控网络**融合三路向量：

- 第 1 阶段：融合 `meta` 与 `第一印象` → 生成“项目初步认知特征”。
- 第 2 阶段：融合 `初步认知特征` 与 `图文序列特征` → 得到最终融合特征向量。

分类头为线性层，输出二分类 logits（训练使用 `BCEWithLogitsLoss`，推理时对 logits 取 `sigmoid` 得到成功概率）。

## 2. 数据与文件约定

### 2.1 表格数据

默认读取 `data/metadata/now_processed.csv`，关键列名与 `src/dl/mlp` 保持一致（可在 `src/dl/gate/config.py` 修改）：

- `id_col`: `project_id`
- `target_col`: `state`（`successful` 视为正类，其余为负类）

### 2.2 项目目录与嵌入文件

默认项目目录为 `data/projects/now/<project_id>/`（可在 `src/dl/gate/config.py` 修改）。

第一印象分支需要的嵌入文件（**必需**）：

- `cover_image_{image_embedding_type}.npy`
- `title_blurb_{text_embedding_type}.npy`

图文序列分支需要的嵌入文件（**允许缺失**，缺失时按空序列处理）：

- `image_{image_embedding_type}.npy`
- `text_{text_embedding_type}.npy`

其中 `image_embedding_type ∈ {clip, siglip, resnet}`，`text_embedding_type ∈ {bge, clip, siglip}`。

## 3. 运行方式

从仓库根目录运行：

- `conda run -n crowdfunding python src/dl/gate/main.py`

常用参数（覆盖部分配置项）：

- `--run-name gate`
- `--image-embedding-type clip|siglip|resnet`
- `--text-embedding-type bge|clip|siglip`
- `--device auto|cpu|cuda|cuda:0 ...` 或 `--gpu N`

默认产物目录：`experiments/gate/<run_id>/`（包含 `artifacts/`、`reports/`、`plots/`）。
