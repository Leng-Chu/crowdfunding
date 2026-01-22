# Embedding 方案（`src/preprocess/embedding`）说明（忽略 qwen）

本文件描述 `src/preprocess/embedding` 中“文本/图像向量化”的实现与数据落盘约定，用于说明多模态特征构建方式。

## 1. 目标与总体流程

目标：为每个项目生成可被直接读取的 `.npy` 向量文件，包括：

- **文本侧**：标题/简介（`title + blurb`）与正文文本分段（来自 `content.json`）
- **图像侧**：封面图（`cover_image`）与正文图片序列（来自 `content.json`）

总体流程建议按如下顺序执行：

1. 先用 `vectorize_csv_data.py` 生成 **title/blurb** 与 **cover_image** 的向量（两者在训练阶段均为“必需文件”）。
2. 再用 `vectorize_content.py` 生成正文 **text** 与正文 **image** 的向量（训练阶段允许缺失，但缺失会降低信息量）。

## 2. 数据组织与落盘规范

### 2.1 项目目录

默认每个项目一个文件夹：

```
data/projects/<dataset>/<project_id>/
  cover/
    cover_image.jpg|jpeg|png|webp
  content.json
```

### 2.2 生成的向量文件名

向量文件以 `.npy` 存放在 `project_id` 目录下，命名规则为：

- **封面图向量**：`cover_image_{model}.npy`
- **正文图片向量序列**：`image_{model}.npy`
- **标题/简介向量**：`title_blurb_{model}.npy`
- **正文文本向量序列**：`text_{model}.npy`

其中 `{model}` 为命令行参数 `--text-model/--image-model` 的取值（常用：`bge/clip/siglip/resnet`）。

### 2.3 形状约定（训练阶段的关键假设）

尽管不同后端的向量维度 `D` 不同，但代码要求“同一模态内部维度一致”：

- `cover_image_{model}.npy`：通常为 `[1, D]`
- `image_{model}.npy`：通常为 `[K, D]`（K 为正文图片数量；可为空但文件可不生成）
- `title_blurb_{model}.npy`：为 `[1, D]` 或 `[2, D]`（取决于 title/blurb 是否同时存在）
- `text_{model}.npy`：通常为 `[T, D]`（T 为正文文本段数量；可为空但文件可不生成）

训练侧（`src/dl/mlp` / `src/dl/seq` / `src/dl/late`）的读取规则（需要在论文中明确）：

- 文本侧：`title_blurb_{text_embedding_type}.npy` **必须存在**；`text_{text_embedding_type}.npy` 可缺失。
- 图像侧：`cover_image_{image_embedding_type}.npy` **必须存在**；`image_{image_embedding_type}.npy` 可缺失。

## 3. 向量化入口脚本

### 3.1 `vectorize_csv_data.py`：从 metadata CSV 生成“标题/简介 + 封面图”向量

输入：

- `data/metadata/<dataset>.csv`（需要包含 `project_id/title/blurb` 三列）
- `data/projects/<dataset>/<project_id>/cover/cover_image.*`

输出（写入项目目录）：

- `title_blurb_{text_model}.npy`
- `cover_image_{image_model}.npy`

行为要点：

- 若输出文件已存在则跳过（避免重复计算）。
- `title_blurb` 由 `[title, blurb]` 组成；若其中一个缺失则只对存在的文本做向量化。

### 3.2 `vectorize_content.py`：从 `content.json` 生成“正文 text + 正文 image”向量

输入：

- `data/projects/<dataset>/<project_id>/content.json`
  - 需要包含 `content_sequence`，其中每个 item 形如：
    - 文本：`{"type": "text", "content": "..."}`
    - 图片：`{"type": "image", "filename": "photo/xxx.jpg"}`

输出（写入项目目录）：

- `text_{text_model}.npy`
- `image_{image_model}.npy`

行为要点：

- 对每个项目先做校验：若 `content_sequence` 为空、文本为空、图片文件不存在等，会停止处理该项目。
- 若输出文件已存在则跳过该模态的向量化。
- 脚本内提供 `enable_text_vector/enable_image_vector` 两个全局开关，用于只跑某一侧向量化。

## 4. 本地 embedding 后端

后端选择由 `src/preprocess/embedding/backends.py` 完成，并带有简单的进程内缓存（同一进程多次调用不重复加载模型）。

### 4.1 文本后端：BGE（`embedding_bge.py`）

- 默认模型：`BAAI/bge-m3`
- 向量抽取：取 `last_hidden_state` 做 attention-mask mean pooling
- 归一化：L2 normalize

### 4.2 图文后端：CLIP（`embedding_clip.py`）

- 默认模型：`openai/clip-vit-base-patch32`
- 文本：
  - 对短文本批处理
  - 对长文本做滑窗切块（stride 固定），块向量平均后再归一化
- 图像：`get_image_features`，并做 L2 normalize

### 4.3 图文后端：SigLIP（`embedding_siglip.py`）

- 默认模型：`google/siglip-base-patch16-224`
- 文本：超过最大长度时做滑窗切块，块向量平均后归一化
- 图像：`get_image_features`（或 fallback 到 `image_embeds`），并做 L2 normalize

### 4.4 图像后端：ResNet（`embedding_resnet.py`）

- 默认模型：`microsoft/resnet-50`
- 向量抽取：
  - 优先 `pooler_output`（常见为 `[N, 2048]`）
  - 否则对 `last_hidden_state` 在空间维做均值
- 归一化：L2 normalize

## 5. 运行方式

- 生成封面图与 title/blurb：
  - `python src/preprocess/embedding/vectorize_csv_data.py --dataset test --text-model bge --image-model clip`
- 生成正文 text 与 image：
  - `python src/preprocess/embedding/vectorize_content.py --dataset test --text-model bge --image-model clip`
