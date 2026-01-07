"""
三步走流水线执行器：
1. 爬取HTML
2. 解析HTML生成JSON
3. 下载资源文件
"""

import csv
from datetime import datetime
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_html import fetch_html
from parse_content import parse_story_content
from download_assets import download_assets_from_json


# 全局线程池用于下载，所有项目共享
DOWNLOAD_THREAD_POOL = None
DOWNLOAD_THREAD_POOL_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def _format_log_prefix(project_id=None, row_index=None, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    project_part = f"[{project_id}]" if project_id else "[main]"
    row_part = f"[{row_index}]" if row_index is not None else ""
    return f"[{timestamp}]{row_part}{project_part} "


def make_logger(project_id=None, row_index=None):
    def _log(message, level="INFO"):
        with PRINT_LOCK:
            print(_format_log_prefix(project_id, row_index, level) + message, flush=True)
    return _log


def get_first_value(row, keys):
    """从CSV行中获取第一个非空值"""
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def iter_rows(csv_path):
    """迭代CSV文件的行"""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def run_pipeline_for_project(project_url, project_id, output_root,
                             row_index=None,
                             overwrite_html=False, overwrite_content=False, overwrite_assets=False,
                             download_assets=True, download_workers=10):
    log = make_logger(project_id, row_index=row_index)
    project_dir = output_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    html_path = project_dir / "page.html"

    fetch_html(project_url, str(html_path), overwrite_html=overwrite_html, logger=log)

    result = parse_story_content(
        str(html_path),
        str(project_dir),
        project_url=project_url,
        overwrite_content=overwrite_content,
        logger=log,
    )

    issues = []
    if not result or not result.get("cover_image"):
        issues.append("missing_cover_image")
    if not result or not result.get("content_sequence"):
        issues.append("empty_content_sequence")

    download_failures = []
    if download_assets and not issues:
        content_json_path = project_dir / "content.json"
        if content_json_path.exists():
            download_failures = download_assets_from_json(
                str(content_json_path),
                str(project_dir),
                max_workers=download_workers,
                overwrite_files=overwrite_assets,
                logger=log,
            )
            if download_failures:
                issues.append("asset_download_failed")
        else:
            issues.append("missing_content_json")

    log("项目流水线执行完成")

    return {
        "project_id": project_id,
        "project_url": project_url,
        "issues": issues,
        "download_failures": download_failures,
    }

def run_pipeline_with_shared_download_pool(args):
    (project_url, project_id, output_root, row_index, overwrite_html,
     overwrite_content, overwrite_assets, download_assets, download_workers, row_data) = args

    global DOWNLOAD_THREAD_POOL

    with DOWNLOAD_THREAD_POOL_LOCK:
        if DOWNLOAD_THREAD_POOL is None:
            DOWNLOAD_THREAD_POOL = ThreadPoolExecutor(max_workers=download_workers)

    result = run_pipeline_for_project(
        project_url, project_id, output_root,
        row_index,
        overwrite_html, overwrite_content, overwrite_assets,
        download_assets, download_workers,
    )

    if row_data is not None:
        result["row"] = row_data

    return result

def main():
    csv_path = Path("data/test/test_problems_problems.csv")
    output_root = Path("data/projects")
    id_keys = ["project_id", "id"]
    url_keys = ["project_url", "url"]
    overwrite_html = False
    overwrite_content = True
    overwrite_assets = True
    download_assets = True
    max_rows = None
    project_workers = 5
    download_workers = 10

    output_root.mkdir(parents=True, exist_ok=True)

    args_list = []
    row_count = 0
    problems = []
    log = make_logger()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if max_rows is not None and row_count >= max_rows:
                log(f"已达到最大行数限制: {max_rows}，停止处理", level="WARN")
                break

            project_id = get_first_value(row, id_keys)
            project_url = get_first_value(row, url_keys)

            if not project_id or not project_url:
                log(f"跳过缺少项目ID或URL的行: {row}", level="WARN")
                continue

            args_list.append((
                project_url, project_id, output_root,
                row_count + 1,
                overwrite_html, overwrite_content, overwrite_assets,
                download_assets, download_workers, row,
            ))
            row_count += 1

    log(f"开始使用 {project_workers} 个线程处理 {len(args_list)} 个项目")
    with ThreadPoolExecutor(max_workers=project_workers) as executor:
        futures = [executor.submit(run_pipeline_with_shared_download_pool, args) for args in args_list]

        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result.get("issues"):
                    problems.append(result)
            except Exception as e:
                log(f"处理项目时发生错误: {e}", level="ERROR")

    if problems:
        problems_csv = csv_path.with_name(f"{csv_path.stem}_problems.csv")
        with problems_csv.open("w", encoding="utf-8", newline="") as f:
            extra_fields = ["issues", "download_error_count", "download_error_samples"]
            writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
            writer.writeheader()
            for item in problems:
                row = dict(item.get("row") or {})
                issues = item.get("issues") or []
                download_failures = item.get("download_failures") or []
                row["issues"] = ";".join(issues)
                row["download_error_count"] = str(len(download_failures))
                if download_failures:
                    samples = download_failures[:3]
                    row["download_error_samples"] = "; ".join(
                        f"{s.get('url', '')} -> {s.get('path', '')}: {s.get('error', '')}" for s in samples
                    )
                else:
                    row["download_error_samples"] = ""
                writer.writerow(row)
        log(f"已保存问题行CSV: {problems_csv}")
    else:
        log("没有发现问题行")

    log("所有项目处理完成")

def run_single_project_pipeline(project_url, project_id, output_root = Path("data/projects"), 
                              overwrite_html=False, overwrite_content=False, overwrite_assets=False, 
                              download_assets=True, download_workers=10):
    """
    为单个项目执行完整流水线的便捷函数
    """
    run_pipeline_for_project(
        project_url, project_id, output_root,
        overwrite_html, overwrite_content, overwrite_assets, 
        download_assets, download_workers
    )


if __name__ == "__main__":
    main()
    #run_single_project_pipeline("https://www.kickstarter.com/projects/halo-air/halo-air-the-environmental-sensor-for-your-phone","1923273133")
