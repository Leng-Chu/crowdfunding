# paper.md 人工转换为 LaTeX 的操作指南（不直接转换）

本文档用于指导你手工将 `paper/paper.md` 转为 `paper/` 模板可编译的 LaTeX 论文。

## 0. 先确认模板结构

`paper/` 目录中与模板相关的关键文件：

- `template.tex`：模板示例文件（建议先阅读结构）。
- `main.tex`：当前主文件（建议作为最终编译入口）。
- `ructhesis.cls`、`bibspacing.sty`：模板依赖样式文件。
- `figures/`：图片目录（`paper.md` 已改为本地相对路径）。
- `REF/`、`textfiles/`：可用于参考模板已有拆分方式。

先看两处“硬约束”再动手迁移：

- `main.tex` 已固定图表编号形式为“节号-序号”（如 `2-1`），不要手工改编号。
- `main.tex` 已加载 `booktabs`、`threeparttable`、`enumerate`、`float`，说明表格和列表格式应尽量按模板示例来写，而不是随意发挥。

## 1. 建议的人工迁移顺序

建议按“先框架、后内容、再公式图表、最后参考文献”的顺序，减少返工。

1. 在 `main.tex` 中搭好章节骨架。
2. 按章节把 `paper.md` 正文迁入对应的 LaTeX 章节命令。
3. 逐段处理：标题、列表、表格、公式、图片、交叉引用。
4. 最后统一处理参考文献与编号。

## 2. Markdown 元素到 LaTeX 的映射规则

### 2.1 标题

- `# 一级标题` → `\chapter{}`（若模板是学位论文类，通常一级对应章）
- `## 二级标题` → `\section{}`
- `### 三级标题` → `\subsection{}`
- `#### 四级标题` → `\subsubsection{}`

> 注意：以 `template.tex` 的层级定义为准，避免章/节级别错位。

### 2.2 段落与强调

- 普通段落直接粘贴并处理转义字符。
- `**加粗**` → `\textbf{}`。
- `*斜体*` → `\textit{}`。

需要重点转义的字符：`_ % & # $ { } ~ ^ \\`。

### 2.3 列表

- 无序列表 `-` → `itemize`。
- 有序列表 `1.` → `enumerate`，并按模板显式写编号样式。

模板中常见两种写法，建议优先使用第 1 种：

1. 全列表统一编号样式：

```latex
\begin{enumerate}[(1)]
  \item 条目A
  \item 条目B
\end{enumerate}
```

2. 对单条目手工指定标签（在“研究结论/创新点”类段落很常见）：

```latex
\begin{itemize}
  \item[(1)] \textbf{创新点1.} 说明文字。
  \item[(2)] \textbf{创新点2.} 说明文字。
\end{itemize}
```

> 若你从 Markdown 的 `1. 2. 3.` 迁移，默认不要写成裸 `\item`，应改为 `\begin{enumerate}[(1)]` 或 `\item[(1)]` 风格，保证与模板一致。

### 2.4 公式

- 行内公式 `$...$` 通常可直接保留。
- 块公式建议改为：
  - 无编号：`\[ ... \]`
  - 有编号：`\begin{equation} ... \end{equation}`

同时检查数学命令是否依赖宏包（如 `amsmath`）。

### 2.5 图片

`paper.md` 已使用本地路径（例如 `figures/MSTC_overview.jpg`）。

人工迁移时建议统一改为：

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{figures/MSTC_overview.jpg}
  \caption{MSTC 模型架构概览}
  \label{fig:mstc_overview}
\end{figure}
```

注意事项：

- 保持路径相对 `main.tex`。
- 每张图都给 `\label{}`，正文中用 `\ref{}` 引用。
- 图片文件名建议不含空格。

### 2.6 表格

Markdown 表格需手工改为 `table + tabular`，并尽量贴近 `template.tex` 的“标准样式”：

```latex
\begin{table}[H]
\centering
\caption{表题}
\label{tab:example}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{列1} & \textbf{列2} & \textbf{列3} \\ \midrule
内容A         & 内容B         & 内容C         \\ \bottomrule
\end{tabular}
\end{table}
```

请特别注意：

1. 浮动体位置优先用 `[H]`（模板已加载 `float`）。
2. 线型优先用 `\toprule / \midrule / \bottomrule`（`booktabs` 风格），避免满屏竖线。
3. 列格式优先使用 `@{}...@{}` 去除两侧多余留白。
4. 复杂表可用 `\multirow`；带注释的表优先套 `threeparttable + tablenotes`。
5. 表题和标签顺序按模板：`\caption{}` 在前，`\label{}` 在后。

## 3. 参考文献人工迁移建议

你当前 `paper.md` 的参考文献已经按正文出现顺序排好编号。人工转 LaTeX 时建议二选一：

### 方案 A（推荐）：BibTeX/BibLaTeX

1. 新建 `refs.bib`，把每条文献写成 Bib 条目。
2. 在正文中把 `[n]` 改为 `\cite{key}`。
3. 在 `main.tex` 按模板要求加入参考文献命令（例如 `\bibliography{refs}`）。

优点：后续改格式最省事。

### 方案 B：手工 thebibliography

1. 直接在文末写 `thebibliography` 环境。
2. 每条文献使用 `\bibitem{}`，并按当前顺序排列。
3. 正文用 `\cite{}` 引用对应 `\bibitem`。

优点：一次性快速定稿；缺点：后续维护成本高。

## 4. 与模板对齐的关键检查点

在首次可编译后，请逐项检查：

1. 封面、摘要、中英文关键词是否符合模板要求。
2. 目录、图目录、表目录是否自动生成且页码正确。
3. 图表标题样式、编号样式是否与学校模板一致。
4. 公式编号是否按章连续或全文连续（按模板规范）。
5. 参考文献样式是否符合提交规范。
6. 有序列表是否统一为 `[(1)]` 风格（`\begin{enumerate}[(1)]` 或 `\item[(1)]`）。
7. 表格是否统一采用 `booktabs` 风格，复杂表是否使用 `threeparttable`。

## 5. 推荐的人工工作流（最稳妥）

1. 先只迁移“摘要 + 第1章”，编译通过。
2. 再逐章迁移，每次迁移后都编译。
3. 所有正文完成后统一处理图表编号与交叉引用。
4. 最后处理参考文献、附录、致谢等收尾部分。

## 6. 常见问题与处理建议

- **问题：中文乱码**  
  处理：确保文件为 UTF-8；按模板推荐的编译链（如 XeLaTeX）。

- **问题：图片不显示**  
  处理：确认路径相对 `main.tex`，并检查扩展名大小写一致。

- **问题：引用显示为问号**  
  处理：多编译几次（LaTeX 交叉引用需要多轮）。

- **问题：表格超页或超宽**  
  处理：拆表、缩放、或改用支持自动换行的表格环境。

## 7. 你可以直接执行的人工清单

- [ ] 先阅读 `template.tex`，确认章/节层级和前置部分写法。
- [ ] 在 `main.tex` 建立完整章节骨架。
- [ ] 逐章粘贴 `paper.md` 内容并完成 LaTeX 语法替换。
- [ ] 全部图片替换为 `figure` 环境并加 `caption+label`。
- [ ] 全部表格替换为 `table/tabular` 并加 `caption+label`。
- [ ] 检查所有有序列表：统一成 `\begin{enumerate}[(1)]` 或 `\item[(1)]`。
- [ ] 检查所有表格：优先 `[H] + booktabs + @{}...@{}`，必要时 `threeparttable`。
- [ ] 将正文数字引用替换为 `\cite{}`。
- [ ] 构建并校对参考文献（BibTeX 或 `thebibliography`）。
- [ ] 全文编译检查格式、页码、引用、图表目录。

---

如果你希望，我下一步可以只做“转换前检查”类工作（例如：帮你列出 `paper.md` 里所有表格、公式、图、引用的位置清单），方便你人工迁移时逐项勾选。
