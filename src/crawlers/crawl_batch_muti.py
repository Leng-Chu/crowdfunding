"""
多线程版本
1. 爬取HTML
2. 解析HTML生成JSON
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_html import fetch_html
from parse_content import parse_story_content

def simple_log(message: str) -> None:
    """简化日志输出，直接打印时间戳和消息。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _format_prefix(csv_row_index: int, project_id: str) -> str:
    return f"[CSV第{csv_row_index}行][{project_id}]"


def update_csv_with_status(csv_path: Path, project_id: str, status: str, lock: threading.Lock) -> None:
    """加锁更新CSV，避免并发写入冲突。"""
    with lock:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

        if "content_status" not in df.columns:
            df["content_status"] = ""
        if "project_id" not in df.columns:
            df["project_id"] = ""

        mask = df["project_id"].astype(str) == str(project_id)
        if mask.any():
            df.loc[mask, "content_status"] = status

        df = df.fillna("")
        df.to_csv(csv_path, index=False, encoding="utf-8")

def process_project(args, csv_path: Path, csv_lock: threading.Lock) -> None:
    (
        project_url, project_id, output_root, csv_row_index,
        overwrite_html, overwrite_content,
        cover_url,
    ) = args

    prefix = _format_prefix(csv_row_index, project_id)

    def log_with_prefix(message: str) -> None:
        simple_log(f"{prefix} {message}")

    log_with_prefix("开始处理项目")

    project_dir = output_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    html_path = project_dir / "page.html"

    try:
        fetch_html(
            project_url,
            str(html_path),
            overwrite_html=overwrite_html,
            logger=log_with_prefix,
        )
    except Exception as exc:
        update_csv_with_status(csv_path, project_id, "failed: fetch_html_error", csv_lock)
        log_with_prefix(f"抓取HTML失败: {exc}")
        return

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
        update_csv_with_status(csv_path, project_id, "failed: parse_error", csv_lock)
        log_with_prefix(f"解析内容失败: {exc}")
        return

    issues = []
    if not result or not result.get("cover_image"):
        issues.append("missing_cover_image")
    if not result or not result.get("content_sequence"):
        issues.append("empty_content_sequence")


    if issues:
        issue_str = ",".join(issues)
        update_csv_with_status(csv_path, project_id, f"failed: {issue_str}", csv_lock)
        log_with_prefix(f"项目失败，问题: {issue_str}")
    else:
        update_csv_with_status(csv_path, project_id, "success", csv_lock)
        log_with_prefix("项目流水线执行成功")


def main() -> None:
    csv_path = Path("data/metadata/years/2024_20760.csv")
    output_root = Path("data/projects/2024")
    overwrite_html = True
    overwrite_content = True
    start_row = 1  # 从第几行开始处理（从1开始计数）
    end_row = 5000  # 处理到第几行（None表示处理到文件末尾）
    max_workers = 6 # 并发线程数

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

        args_list.append((
            project_url, project_id, output_root,
            row_idx + 1,
            overwrite_html, overwrite_content,
            cover_url,
        ))

    simple_log(
        f"开始并发处理 {len(args_list)} 个项目，"
        f"范围：第{start_row}行到第{end_row or '末尾'}行，"
        f"线程数：{max_workers}"
    )

    csv_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_project, args, csv_path, csv_lock) for args in args_list]
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                simple_log(f"线程任务异常: {exc}")

    simple_log("所有项目处理完成")


if __name__ == "__main__":
    main()
