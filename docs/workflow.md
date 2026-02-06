# 数据构建与实验复现流程（顶层说明）

本文面向“论文写作 + 代码复现”，以更高层级总结本仓库的数据构建与实验流程。目标是说明 **数据从哪里来、如何变成可训练的数据契约（CSV + 项目目录 + embedding）、以及如何得到可比较的实验结果**。实现细节以现有模块文档与代码为准。

---

## 0. 总览：从原始数据到可比较结果

建议按如下顺序组织复现（括号内为主要输入/输出）：

1. **元数据构建**（`data/metadata/all.json` → `data/metadata/all.csv` + `data/metadata/years/*.csv`）
2. **项目抓取与解析**（metadata CSV → `data/projects/<dataset>/<project_id>/{page.html, content.json}` + `content_status`）
3. **资源下载 + 写回属性**（`content.json` → `cover/cover_image.[jpg|jpeg|png|webp]` + `photo/*` + 可选视频 + `download_status`；并写回图片尺寸/修正扩展名）
4. **表格数据清洗与筛选**（`all.csv`/年度 CSV → `data/metadata/now_processed.csv`，并同步裁剪 `data/projects/now/`）
5. **向量化（embedding）**（项目目录 → `.npy` 文件，作为训练输入）
6. **训练与评估（可比口径）**（统一 best checkpoint/阈值/产物结构，输出 `result.csv`）
7. **结果汇总**（批量合并与分组平均，产出论文表格的候选结果）

其中第 4/5/6 步的详细口径分别见：

- 表格清洗：`docs/preprocess_table.md`
- embedding：`docs/preprocess_embedding.md`
- 深度学习评估口径：`docs/dl_guidelines.md`（并参考 `docs/dl_seq.md` / `docs/dl_late.md` / `docs/dl_mdl.md`）

---

## 1. 元数据构建（metadata）

### 1.1 数据来源

元数据来自 Kickstarter 的公开数据集（通常为逐行 JSON）。本仓库约定将其放在：

- `data/metadata/all.json`（一行一个 JSON 对象）

### 1.2 转表脚本与产物

使用 `src/crawlers/json_to_table.py` 将 `all.json` 转成 CSV：

- 输入：`data/metadata/all.json`
- 输出：
  - `data/metadata/all.csv`
  - `data/metadata/years/<year>_<count>.csv`

脚本会进行基础过滤/整理（以代码为准）：

- 仅保留 `state ∈ {successful, failed}`
- 仅保留 `year > 2014`
- `project_id` 去重
- 计算 `usd_goal = goal * static_usd_rate`

字段含义参见 `docs/csv_fields.md`。

---

## 2. 爬虫：抓 HTML 并解析为 `content.json`

### 2.1 输入与输出

输入通常为一个 metadata CSV（例如 `data/metadata/<dataset>.csv`），至少包含：

- `project_id`
- `project_url`
- `cover_url`（可为空；若为空将导致封面下载与后续流程不完整）
- （可选）`content_status`：用于标记抓取/解析是否成功

输出为项目目录（一个项目一个文件夹）：

```
data/projects/<dataset>/<project_id>/
  page.html
  content.json
  cover/
  photo/
```

`content.json` 由解析脚本生成，最小结构形如：

```json
{
  "project_url": "https://...",
  "title": {"content": "...", "content_length": 12},
  "blurb": {"content": "...", "content_length": 34},
  "cover_image": {"url": "https://...", "filename": "cover/cover_image.jpg", "width": 680, "height": 453},
  "video": "https://...base.mp4",
  "content_sequence": [
    {"type": "text", "content": "...", "content_length": 123},
    {"type": "image", "url": "https://...", "filename": "photo/story_image_1.jpg", "width": 680, "height": 453}
  ]
}
```

说明：

- `title/blurb/cover_image/content_sequence[*].content_length/width/height` 均为训练侧需要用到的属性字段。
- 解析阶段（`crawl_batch*.py`）会写入 `title/blurb` 与正文 text 的 `content_length`；下载阶段（`download_batch.py`）会自动写回图片的 `width/height` 并修正图片扩展名（见下一节）。

### 2.2 入口脚本

- 顺序版（复用同一个浏览器实例，稳定优先）：`src/crawlers/crawl_batch.py`
- 多线程版（速度优先，稳定性可能更差）：`src/crawlers/crawl_batch_muti.py`

两者都会：

1. 抓取 `page.html`（`src/crawlers/fetch_html.py`）
2. 解析 `story-content` 得到 `content_sequence`（`src/crawlers/parse_content.py`）
3. 将 `content_status` 写回到 CSV（`success` 或 `failed: <原因>`）

说明：

- 这两个脚本在 `main()` 中有硬编码的 `csv_path/output_root/start_row/end_row/...`，复现时通常需要先按自己的数据集修改配置区。
- 抓 HTML 使用 `DrissionPage`（Chromium），可能受本地浏览器环境/网络影响；建议先小规模跑通再批量扩展。

---

## 3. 资源下载：封面/正文图片/可选视频

在 `content_status=success` 的前提下，使用：

- `src/crawlers/download_batch.py`：批量下载并回写 `download_status`
- `src/crawlers/download_assets.py`：针对单个 `content.json` 下载资源（被 `download_batch.py` 调用）

默认落盘位置：

- 封面：`data/projects/<dataset>/<project_id>/cover/cover_image.[jpg|jpeg|png|webp]`（以真实格式为准）
- 正文图片：`data/projects/<dataset>/<project_id>/photo/*`
- （可选）视频：`data/projects/<dataset>/<project_id>/cover/project_video.mp4`

`download_status` 会写回到 CSV（`success` 或 `failed: <原因>`），便于后续筛选与审计。

下载阶段会同时保证 `content.json` 的关键字段完整（无需额外 “update 脚本”）：

- `cover_image` 统一为 dict，并写回 `filename/width/height`
- 对正文图片：写回每个 image item 的 `width/height`
- 对正文图片与封面：若下载得到的真实格式与扩展名不一致，会自动 **修正扩展名并同步更新 `filename`**
- 兼容旧数据：若正文 text 缺少 `content_length`，会自动补齐

---

## 4. 训练侧需要的 `content.json` 字段（与爬虫对齐）

为保证 `src/dl/seq` / `src/dl/late` / `src/dl/mdl` 的读取口径一致，训练阶段默认依赖 `content.json` 中的以下字段：

- `title/blurb`：dict（包含 `content/content_length`），用于构造 `title_blurb_{emb}.npy` 的 token attr
- `cover_image`：dict（包含 `url/filename/width/height`），用于构造封面 token 的 attr（面积）
- `content_sequence`：
  - text item：包含 `content/content_length`
  - image item：包含 `url/filename/width/height`

这些字段由爬虫流水线保证：

- `crawl_batch*.py`：负责写入 `title/blurb` 与正文 text 的 `content_length`
- `download_batch.py`：负责下载资源、修正扩展名，并写入封面/正文图片的 `filename/width/height`

---

## 5. 表格数据清洗与数据集筛选（生成最终训练集）

表格特征（meta）与最终训练集 CSV 的构建在 `src/preprocess/table`，详细规则与产物见：

- `docs/preprocess_table.md`

该阶段的目标是输出一份可训练的表格数据，并同步裁剪项目目录，得到训练默认使用的数据契约：

- 表格：`data/metadata/now_processed.csv`
- 项目目录：`data/projects/now/<project_id>/...`

论文写作时通常需要说明的要点是：

- **质量控制过滤**（缺失字段、异常持续时间、低频类别等）
- **（可选）按年份的类别平衡**（下采样多数类）
- **特征工程**（如 `duration_days`、`log_usd_goal`）
- **保证 metadata 与 projects 目录一致**（过滤样本时同步移动/删除项目文件夹）

---

## 6. 向量化（embedding）：生成训练输入的 `.npy`

embedding 生成在 `src/preprocess/embedding`，细节、文件命名与形状约定见：

- `docs/preprocess_embedding.md`

顶层推荐顺序：

1. 先生成 **title/blurb + cover_image**（训练阶段通常作为“前缀 token”必需）
2. 再生成正文 **text/image**（可选，但信息量更完整；缺失时可能需要 `missing_strategy="skip"` 或对应逻辑）

产物落盘到每个项目目录下（示例）：

```
data/projects/now/<project_id>/
  cover_image_clip.npy
  title_blurb_bge.npy
  image_clip.npy        (可选)
  text_bge.npy          (可选)
```

---

## 7. 训练与评估：统一口径得到可比 `result.csv`

深度学习训练代码位于 `src/dl/*`：

- `seq`：图文内容块序列建模（见 `docs/dl_seq.md`）
- `late`：图文晚期融合 baseline（见 `docs/dl_late.md`）
- `mdl`：多分支（meta/image/text）baseline（见 `docs/dl_mdl.md`）

统一工程规范（best checkpoint / 阈值选择 / 产物结构）见：

- `docs/dl_guidelines.md`

最终每次运行都会在 `experiments/newtest/<mode>/<run_id>/reports/result.csv` 产出单行汇总，便于批量合并对比。

---

## 8. 结果汇总：合并与分组平均（论文表格友好）

常用脚本位于 `src/scripts/`：

- `src/scripts/merge_result_csv.py`
  - 遍历 `experiments/newtest/*/*/result.csv` 合并为 `experiments/newtest/merge.csv`
- `src/scripts/group_and_average_results.py`
  - 读取 `merge.csv`，按关键配置分组求均值，输出 `experiments/newtest/averaged_results.csv`

建议将这些结果作为论文表格的候选基线对比入口，再结合实验设计筛选最终汇报指标（例如以 `test_auc/test_f1` 排序）。

---

## 9. 常见维护/审计脚本（非核心流程）

这类脚本通常用于数据一致性检查、清理无效样本、排查 embedding 落盘问题；不建议作为“稳定 API”，但在做大规模数据时很实用：

- `src/preprocess/clean/update_status.py`：基于磁盘实际文件回写 `content_status/download_status`
- `src/preprocess/clean/check_embedding.py`：检查每个项目 `.npy` 文件集合是否完整，并可移动不合格项目到 trash，同时更新 CSV
- `src/scripts/check_npy_files.py`：随机抽取项目检查 `.npy` 的 shape/dtype/范数等（embedding sanity check）

实践建议：

- 尽量对 CSV 做备份（生成 `_updated` 或带行数后缀的新文件），避免不可逆修改。
- 在移动/删除项目文件夹前先抽样检查，确认规则符合当前实验口径。
