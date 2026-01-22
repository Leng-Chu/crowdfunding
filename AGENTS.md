# 仓库指南 (Repository Guidelines)

## 项目结构与模块组织

- `src/`: Python 源代码 (以脚本为主的仓库)
  - `crawlers/`: 获取 Kickstarter 页面/资源并写入 `data/projects/...`
  - `preprocess/`: 清洗 + 特征/嵌入生成 (`clean/`, `embedding/`, `table/`)
  - `dl/`: 模型训练代码 (例如 `dl/mlp/`)
- `data/`: 本地数据集 + 爬虫输出 (`metadata/`, `projects/`)
- `experiments/`: 运行输出 (日志/指标/图表)
- `docs/`: 文档
  - `docs/csv_fields.md`: 数据集列说明
  - `docs/dl_mlp.md`: `src/dl/mlp` 模型与复现说明
  - `docs/dl_late.md`: `src/dl/late` 图文晚期融合 baseline 说明
  - `docs/dl_seq.md`: `src/dl/seq` 图文内容块序列建模说明
  - `docs/preprocess_table.md`: `src/preprocess/table` 表格预处理说明
  - `docs/preprocess_embedding.md`: `src/preprocess/embedding` 向量化方案说明（忽略 qwen）

## 构建、测试和开发命令

从仓库根目录运行脚本；代码已使用以仓库根目录为基准的相对路径。

- 抓取项目页面: `python src/crawlers/crawl_batch.py` (读取 `data/metadata/*.csv`, 写入 `data/projects/...`)
- 转换 JSON → 表格: `python src/crawlers/json_to_table.py`
- 生成嵌入: `python src/preprocess/embedding/vectorize_content.py` 或 `python src/preprocess/embedding/vectorize_csv_data.py`
- 训练 MLP 多模态模型: `python src/dl/mlp/main.py` (在 `src/dl/mlp/config.py` 中编辑参数)

注意: 某些嵌入后端会下载大型预训练模型; 请优先使用本地缓存并记录任何需要的权重。

## 编码风格与命名约定

- Python: 4个空格缩进, PEP 8, 函数/变量使用 `snake_case`, 类使用 `PascalCase`。
- 保持文件为 UTF-8 编码 (许多注释/文档是中文)。
- 优先使用 `pathlib.Path`; 避免绝对路径。将派生输出写在 `data/` 或 `experiments/` 下。

## 提交与拉取请求指南

- Git 历史使用简短的单行消息 (中英文混合; "debug/update/tmp" 出现)。保持提交小而具有描述性。
- 默认情况下不要提交大的数据/媒体/模型二进制文件; 链接到外部存储 (或使用 Git LFS) 并在需要时更新 `.gitignore`。

## 重要提示

- 回答时均需要使用中文，代码中的注释和输出均需要使用中文。
- 作图时图里不要有中文。
- 请注意使用UTF-8编码以支持中文字符的正确显示。
- 工程规范：`src/dl/*` 二分类任务最终报告指标时，阈值必须在验证集上选择（最大化 F1），并用该阈值计算测试集指标（详见 `docs/dl_threshold.md`）。
- 每次修改代码后需要检查 `AGENTS.md` 是否需要更新。
- 只参考当前要求参考的内容，其余没有提到的项目中的内容默认与当前任务无关。
