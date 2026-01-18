# 仓库指南 (Repository Guidelines)

## 项目结构与模块组织

- `src/`: Python 源代码 (以脚本为主的仓库)
  - `crawlers/`: 获取 Kickstarter 页面/资源并写入 `data/projects/...`
  - `preprocess/`: 清洗 + 特征/嵌入生成 (`clean/`, `embedding/`, `table/`)
  - `dl/`: 模型训练代码 (例如 `dl/meta/`)
- `data/`: 本地数据集 + 爬虫输出 (`metadata/`, `projects/`)
- `experiments/`: 运行输出 (日志/指标/图表)
- `docs/`: 文档 (参见 `docs/csv_fields.md` 了解数据集列)

## 构建、测试和开发命令

从仓库根目录运行脚本，以便像 `data/...` 这样的相对路径能正确解析。

- 抓取项目页面: `python src/crawlers/crawl_batch.py` (读取 `data/metadata/*.csv`, 写入 `data/projects/...`)
- 转换 JSON → 表格: `python src/crawlers/json_to_table.py`
- 生成嵌入: `python src/preprocess/embedding/vectorize_content.py` 或 `python src/preprocess/embedding/vectorize_csv_data.py`
- 训练元深度学习模型: `conda run -n crowdfunding python src/dl/meta/main.py` (在 `src/dl/meta/config.py` 中编辑参数)

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
