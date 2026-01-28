# gate 三分支 + Two-stage fusion（Stage2=FiLM/AdaLN）（`src/dl/gate`）说明

## 1. 任务与输入输出

- **任务**：Kickstarter 项目二分类（成功/失败）。
- **标签**：来自表格数据 `state` 字段（`successful` 视为正类，其余为负类；同时兼容 0/1 数值标签）。
- **输入**：预先计算好的 embedding（不涉及原始模态特征的编码器）。
  - **First Impression 分支**（固定使用 *title/blurb/cover*，不参与正文截断）：
    - `title_blurb_{txt_emb_type}.npy`：形状 `[2, D_txt]`，顺序固定 `[title, blurb]`
    - `cover_image_{img_emb_type}.npy`：形状 `[1, D_img]`
  - **Content Seq 分支**（仅正文 `content_sequence`，不含 title/blurb/cover）：
    - `content.json` 的 `content_sequence`（按页面顺序）
    - `image_{img_emb_type}.npy`：形状 `[N_img, D_img]`
    - `text_{txt_emb_type}.npy`：形状 `[N_txt, D_txt]`
  - **meta**：表格元数据特征（类别 one-hot + 数值标准化），除 `seq_only` / `key_only` 外均启用。
- **输出**：二分类 **logits**（训练使用 `BCEWithLogitsLoss`），推理时对 logits 施加 `sigmoid` 得到正类概率。

> 与 `docs/dl_seq.md` 的主要区别：本章内容序列分支不再包含 title/blurb/cover，它们只属于 First Impression 分支。

---

## 2. 数据组织与文件约定

### 2.1 表格数据（metadata CSV）

默认配置见 `src/dl/gate/config.py:GateConfig`：

- `data_csv`：默认 `data/metadata/now_processed.csv`
- 关键列名（与 `docs/dl_seq.md` 对齐）：
  - `id_col`：`project_id`
  - `target_col`：`state`
  - `categorical_cols`：`category, country, currency`
  - `numeric_cols`：`duration_days, log_usd_goal`

### 2.2 项目目录（projects_root）

`projects_root` 默认 `data/projects/now`，每个项目一个文件夹：

```
data/projects/<dataset>/<project_id>/
  content.json
  cover_image_{img_emb_type}.npy
  title_blurb_{txt_emb_type}.npy
  image_{img_emb_type}.npy
  text_{txt_emb_type}.npy
```

一致性要求（默认严格，`missing_strategy=error`）：

- `cover_image_{img_emb_type}.npy` 必须存在，且行数为 1。
- `title_blurb_{txt_emb_type}.npy` 必须存在，且行数必须为 2（固定 `[title, blurb]`）。
- `content_sequence` 中 `"image"` 数量必须与 `image_{img_emb_type}.npy` 行数一致（为 0 时允许缺失）。
- `content_sequence` 中 `"text"` 数量必须与 `text_{txt_emb_type}.npy` 行数一致（为 0 时允许缺失）。

### 2.3 正文 token 属性（attr）

对每个正文 token 计算 1 个标量属性，并在模型端做线性映射：

- 正文文本块：`log(max(1, content_length))`
- 正文图片块：`log(max(1, width*height))`

字段要求：

- `type="text"`：必须包含 `content_length`
- `type="image"`：必须包含 `width` 与 `height`

### 2.4 截断与 mask

仅对正文 `content_sequence` 做截断（First Impression 永远保留）：

- `truncation_strategy="first"`：取前 `max_seq_len`
- `truncation_strategy="random"`：在 `[0, L-max_seq_len]` 中随机选窗口（随机种子为 `random_seed + hash(project_id)`，实现中使用稳定 hash）

输出：

- `seq_mask ∈ [B, max_seq_len]`：`True` 表示有效位置

---

## 3. 模型结构（固定）

三分支：

1) **MetaEncoder**：`Linear(d_meta_in→d)→ReLU→Dropout→Linear(d→d)→ReLU`，输出 `v_meta`
2) **FirstImpressionEncoder**：title/blurb 与 cover 的 gated 交互 + concat+proj，输出 `v_key`
3) **ContentSeqEncoder（trm_pos）**：TokenEncoder + sinusoidal PE + TransformerEncoder + masked mean pooling，输出 `h_seq`

融合（Two-stage fusion）：

- Stage1：`v_key` 与 `v_meta` 得先验 `p`
- Stage2：用 `p` 生成 FiLM 参数，对 `h_seq` 做 AdaLN 条件调制得到 `h_final`

Head：

- `Linear(d→d_head)→ReLU→Dropout→Linear(d_head→1)`

---

## 4. 实验组（baseline_mode）

`src/dl/gate/config.py:GateConfig.baseline_mode`：

- `late_concat`：concat(`h_seq, v_key, v_meta`) 后投影到 `d` 再接 head
- `stage1_only`：只计算 `p`，不做 Stage2；使用 `LN(Linear(concat(h_seq,p)))` 再接 head
- `stage2_only`：不做 Stage1；`p0=LN(Linear(concat(v_key,v_meta)))`，再做 Stage2 得 `h_final` 接 head
- `two_stage`：完整 Two-stage（主模型）
- `seq_only`：只使用 Content Seq 分支（`h_seq`）接 head
- `key_only`：只使用 First Impression 分支（`v_key`）接 head
- `meta_only`：只使用 Meta 分支（`v_meta`）接 head

要求：除融合逻辑外，其余三分支完全一致。

---

## 5. 训练与评估（对齐 seq）

- 损失：`BCEWithLogitsLoss`
- 优化器：AdamW（`learning_rate_init`，`alpha` 作为 `weight_decay`）
- 学习率调度：warmup + cosine（每 step 更新；最小学习率由 `lr_scheduler_min_lr` 控制）
- 早停：`early_stop_patience`（支持 `early_stop_min_epochs`）
- 指标：`accuracy, precision, recall, f1, roc_auc, log_loss`
- 阈值与 best 口径：训练阶段不做阈值搜索，每个 epoch 仅计算阈值无关指标 `val_auc`（若验证集为单类导致 AUC 不可用则回退为 `val_log_loss`），early stopping 与 best checkpoint 选择均基于该指标；在 best epoch 确定后，用该模型的 `val_prob` 搜索一次 `best_threshold`（最大化 F1），并用该阈值计算最终 train/val/test 的阈值相关指标；详见 `docs/dl_guidelines.md`
- 评估函数：`evaluate_gate_split` 仅返回 `prob`；二分类指标由调用方显式传入 `best_threshold` 计算（避免隐式阈值）

---

## 6. 产物目录结构（对齐 seq）

默认写入 `experiments/gate/<mode>/<run_id>/`，其中：

- `mode = baseline_mode`（除 `seq_only` / `key_only` 外默认都使用 meta，因此不再额外区分）

目录结构：

- `artifacts/`
  - `model.pt`（含 `best_epoch/best_val_auc/best_val_log_loss/best_threshold`）
  - `preprocessor.pkl` / `feature_names.txt`（除 `seq_only/key_only` 外均会生成）
- `reports/`
  - `config.json`
  - `metrics.json`
  - `history.csv`
  - `splits.csv`
  - `predictions_val.csv` / `predictions_test.csv`
  - `result.csv`
  - `train.log`
- `plots/`（若 `save_plots=True`）
  - `history.png`
  - `roc_val.png` / `roc_test.png`

---

## 7. 运行方法（从仓库根目录）

- 默认配置运行：
  - `conda run -n crowdfunding python src/dl/gate/main.py`
- 只切换 baseline（推荐的横向对比方式）：
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode two_stage`
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode stage1_only`
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode stage2_only`
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode late_concat`
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode seq_only`
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode key_only`
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode meta_only`
- 指定嵌入类型与设备：
  - `conda run -n crowdfunding python src/dl/gate/main.py --baseline-mode two_stage --image-embedding-type clip --text-embedding-type bge --device cuda:0`

---

## 8. Optuna 自动调参（黑盒）

本仓库提供 `src/dl/gate/optuna_search.py` 作为调参入口：每个 trial 会直接调用现有 `src/dl/gate/main.py` CLI 进行一次完整训练，然后读取 `result.csv`（或回退读取 `reports/metrics.json`）得到目标值。

- 典型用法（调 `two_stage`，目标最大化 `test_f1`）：
  - `conda run -n crowdfunding python src/dl/gate/optuna_search.py --baseline-mode two_stage --device cuda:0 --n-trials 30`
- 输出目录：
  - `experiments/gate/optuna/<study_name>/summary.csv`：每个 trial 的汇总（trial_id、params、objective、run_dir、关键 test 指标）
  - `experiments/gate/optuna/<study_name>/best.json`：最优 trial 的参数与 run_dir
