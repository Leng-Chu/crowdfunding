"""
1. 抓取HTML
2. 解析HTML生成JSON
3. 下载资源文件
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue

import pandas as pd
from DrissionPage import ChromiumOptions, ChromiumPage

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_html import fetch_html
from parse_content import parse_story_content
from download_assets import download_assets_from_json


def _build_options() -> ChromiumOptions:
    options = ChromiumOptions()
    options.auto_port()
    options.set_argument("--disable-notifications")
    options.set_argument("--start-minimized")
    options.set_argument("--window-position=-32000,-32000")  # 设置为屏幕外位置
    options.set_argument("--blink-settings=imagesEnabled=false")  # 禁止加载图片
    return options


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

        if "success_status" not in df.columns:
            df["success_status"] = ""
        if "project_id" not in df.columns:
            df["project_id"] = ""

        mask = df["project_id"].astype(str) == str(project_id)
        if mask.any():
            df.loc[mask, "success_status"] = status

        df = df.fillna("")
        df.to_csv(csv_path, index=False, encoding="utf-8")


def _init_browser_pool(pool_size: int) -> Queue:
    browser_pool = Queue(maxsize=pool_size)
    for _ in range(pool_size):
        options = _build_options()
        browser_pool.put(ChromiumPage(options))
    return browser_pool


def process_project(args, browser_pool: Queue, csv_path: Path, csv_lock: threading.Lock) -> None:
    (
        project_url, project_id, output_root, csv_row_index,
        overwrite_html, overwrite_content, overwrite_assets,
        download_assets, download_workers, cover_url,
    ) = args

    prefix = _format_prefix(csv_row_index, project_id)

    def log_with_prefix(message: str) -> None:
        simple_log(f"{prefix} {message}")

    log_with_prefix("开始处理项目")

    project_dir = output_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    html_path = project_dir / "page.html"

    browser = None
    try:
        browser = browser_pool.get()
        fetch_html(
            project_url,
            str(html_path),
            overwrite_html=overwrite_html,
            logger=log_with_prefix,
            browser_page=browser,
        )
    except Exception as exc:
        update_csv_with_status(csv_path, project_id, "failed: fetch_html_error", csv_lock)
        log_with_prefix(f"抓取HTML失败: {exc}")
        return
    finally:
        if browser is not None:
            browser_pool.put(browser)

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

    if download_assets and not issues:
        content_json_path = project_dir / "content.json"
        if content_json_path.exists():
            download_failures = download_assets_from_json(
                str(content_json_path),
                str(project_dir),
                max_workers=download_workers,
                overwrite_files=overwrite_assets,
                logger=log_with_prefix,
            )
            if download_failures:
                issues.append("asset_download_failed")
        else:
            issues.append("missing_content_json")

    if issues:
        issue_str = ",".join(issues)
        update_csv_with_status(csv_path, project_id, f"failed: {issue_str}", csv_lock)
        log_with_prefix(f"项目失败，问题: {issue_str}")
    else:
        update_csv_with_status(csv_path, project_id, "success", csv_lock)
        log_with_prefix("项目流水线执行成功")


def main() -> None:
    csv_path = Path("data/metadata/years/2025_45005.csv")
    output_root = Path("data/projects/2025")
    overwrite_html = True
    overwrite_content = True
    overwrite_assets = False
    download_assets = False
    start_row = 1  # 从第几行开始处理（从1开始计数）
    end_row = None  # 处理到第几行（None表示处理到文件末尾）
    max_workers = 8 # 并发线程数

    output_root.mkdir(parents=True, exist_ok=True)

    args_list = []
    simple_log(f"开始处理CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for row_idx, row in df.iterrows():
        if row_idx < start_row - 1:
            continue
        if end_row is not None and row_idx >= end_row:
            break

        success_status = row.get("success_status", "")
        if success_status == "success":
            continue

        project_id = row.get("project_id")
        project_url = row.get("project_url")
        cover_url = row.get("cover_url")

        args_list.append((
            project_url, project_id, output_root,
            row_idx + 1,
            overwrite_html, overwrite_content, overwrite_assets,
            download_assets, 10,
            cover_url,
        ))

    simple_log(
        f"开始并发处理 {len(args_list)} 个项目，"
        f"范围：第{start_row}行到第{end_row or '末尾'}行，"
        f"线程数：{max_workers}"
    )

    csv_lock = threading.Lock()
    browser_pool = _init_browser_pool(max_workers)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_project, args, browser_pool, csv_path, csv_lock) for args in args_list]
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    simple_log(f"线程任务异常: {exc}")
    finally:
        while not browser_pool.empty():
            try:
                browser = browser_pool.get_nowait()
                browser.quit()
            except Exception:
                pass

    simple_log("所有项目处理完成")


if __name__ == "__main__":
    main()
