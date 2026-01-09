"""
多线程版本
批量解析HTML生成JSON
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from parse_content import parse_story_content

def simple_log(message: str) -> None:
    """简化日志输出，直接打印时间戳和消息。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _format_prefix(csv_row_index: int, project_id: str) -> str:
    return f"[CSV第{csv_row_index}行][{project_id}]"


def process_project(args) -> tuple:
    (
        project_url, project_id, output_root, csv_row_index,
        overwrite_content, cover_url,
    ) = args

    prefix = _format_prefix(csv_row_index, project_id)

    def log_with_prefix(message: str) -> None:
        simple_log(f"{prefix} {message}")

    log_with_prefix("开始处理项目")

    project_dir = output_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    html_path = project_dir / "page.html"

    try:
        result = parse_story_content(
            str(html_path),
            str(project_dir),
            project_url=project_url,
            cover_url=cover_url,
            overwrite_content=overwrite_content,
            logger=log_with_prefix,
        )
    except Exception as exc:
        log_with_prefix(f"解析内容失败: {exc}")
        return project_id, "failed: parse_error"

    issues = []
    if not result or not result.get("cover_image"):
        issues.append("missing_cover_image")
    if not result or not result.get("content_sequence"):
        issues.append("empty_content_sequence")

    if issues:
        issue_str = ",".join(issues)
        log_with_prefix(f"项目失败，问题: {issue_str}")
        return project_id, f"failed: {issue_str}"
    else:
        log_with_prefix("项目流水线执行成功")
        return project_id, "success"


def update_csv_results(csv_path: Path, results: list) -> None:
    """在所有处理完成后，一次性更新CSV文件"""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    
    if "content_status" not in df.columns:
        df["content_status"] = ""
    if "project_id" not in df.columns:
        df["project_id"] = ""
    
    # 根据处理结果更新DataFrame
    for project_id, status in results:
        mask = df["project_id"].astype(str) == str(project_id)
        if mask.any():
            df.loc[mask, "content_status"] = status

    df = df.fillna("")
    df.to_csv(csv_path, index=False, encoding="utf-8")


def main() -> None:
    csv_path = Path("data/metadata/years/2024_20760.csv")
    output_root = Path("data/projects/2024")
    overwrite_content = True
    start_row = 1  # 从第几行开始处理（从1开始计数）
    end_row = None  # 处理到第几行（None表示处理到文件末尾）
    max_workers = 16 # 并发线程数

    output_root.mkdir(parents=True, exist_ok=True)

    args_list = []
    simple_log(f"开始处理CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    for row_idx, row in df.iterrows():
        if row_idx < start_row - 1:
            continue
        if end_row is not None and row_idx >= end_row:
            break

        content_status = row.get("content_status", "")
        if content_status == "success":
            continue

        project_id = row.get("project_id")
        project_url = row.get("project_url")
        cover_url = row.get("cover_url")

        # 如果output_root中没有名为project_id的文件夹，跳过
        if not (output_root / project_id).exists():
            continue

        args_list.append((
            project_url, project_id, output_root,
            row_idx + 1,
            overwrite_content,
            cover_url,
        ))

    simple_log(
        f"开始并发处理 {len(args_list)} 个项目，"
        f"范围：第{start_row}行到第{end_row or '末尾'}行，"
        f"线程数：{max_workers}"
    )

    # 存储处理结果
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_project, args) for args in args_list]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                simple_log(f"线程任务异常: {exc}")

    # 在所有处理完成后，一次性更新CSV文件
    update_csv_results(csv_path, results)

    simple_log("所有项目处理完成")


if __name__ == "__main__":
    main()
