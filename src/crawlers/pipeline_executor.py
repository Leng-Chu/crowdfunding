"""
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
    project_part = f"[{project_id}]" if project_id is not None else "[main]"
    row_part = f"[{row_index}]" if row_index is not None else ""
    return f"[{timestamp}]{row_part}{project_part} "


def make_logger(project_id=None, row_index=None):
    def _log(message, level="INFO"):
        with PRINT_LOCK:
            print(_format_log_prefix(project_id, row_index, level) + message, flush=True)
    return _log



def iter_rows(csv_path):
    """从CSV文件中逐行读取"""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def run_pipeline_for_project(project_url, project_id, output_root,
                             row_index=None,
                             overwrite_html=False, overwrite_content=False, overwrite_assets=False,
                             download_assets=True, download_workers=10,
                             start_minimized=False,
                             window_position=None, cover_url=None):  # 新增参数：cover_url
    log = make_logger(project_id, row_index=row_index)
    project_dir = output_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    html_path = project_dir / "page.html"

    # 尝试获取HTML，如果解析失败则重试一次
    result = None
    issues = []
    max_retries = 1
    retry_count = 0
    
    while retry_count <= max_retries:
        # 只有在首次或需要重试时才获取HTML
        log(f"第 {retry_count+1} 次: 获取HTML")
        if retry_count>0:
            tmp_overwrite_html = True
        else:
            tmp_overwrite_html = overwrite_html
        fetch_html(
            project_url,
            str(html_path),
            overwrite_html=tmp_overwrite_html,  # 重试时强制覆盖
            start_minimized=start_minimized,
            window_position=window_position,
            logger=log,
        )

        result = parse_story_content(
            str(html_path),
            str(project_dir),
            project_url=project_url,
            cover_url=cover_url,  # 传递封面图片URL参数
            overwrite_content=overwrite_content,
            logger=log,
        )

        # 检查是否有问题，如果有则重试
        issues = []
        if not result or not result.get("cover_image"):
            issues.append("missing_cover_image")
        if not result or not result.get("content_sequence"):
            issues.append("empty_content_sequence")
            
        if issues and retry_count < max_retries:
            log(f"解析结果有问题: {issues}，准备重试...")
            retry_count += 1
        else:
            # 没有问题或已达到最大重试次数，退出循环
            break

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
     overwrite_content, overwrite_assets, download_assets, download_workers,
     start_minimized, window_position, cover_url, row_data) = args

    global DOWNLOAD_THREAD_POOL

    with DOWNLOAD_THREAD_POOL_LOCK:
        if DOWNLOAD_THREAD_POOL is None:
            DOWNLOAD_THREAD_POOL = ThreadPoolExecutor(max_workers=download_workers)
    
    result = run_pipeline_for_project(
        project_url, project_id, output_root,
        row_index,
        overwrite_html, overwrite_content, overwrite_assets,
        download_assets, download_workers,
        start_minimized, window_position,
        cover_url,  # 传递封面图片URL
    )

    if row_data is not None:
        result["row"] = row_data

    return result


def main():
    csv_path = Path("data/metadata/years/2024_31932.csv")
    output_root = Path("data/projects/2024")
    overwrite_html = False
    overwrite_content = True
    overwrite_assets = False
    download_assets = False
    max_rows = None
    project_workers = 8
    download_workers = 10
    start_minimized = True
    window_position = (-32000, -32000)

    output_root.mkdir(parents=True, exist_ok=True)

    args_list = []
    row_count = 0
    log = make_logger()
    problems_csv = csv_path.with_name(f"{csv_path.stem}_problems.csv")
    problems_file = None
    problems_writer = None
    problems_written = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if max_rows is not None and row_count >= max_rows:
                log(f"已达到最大行数限制: {max_rows}", level="WARN")
                break

            project_id = row.get("project_id")
            project_url = row.get("project_url")
            cover_url = row.get("cover_url")

            if not project_id or not project_url:
                log(f"跳过缺少项目ID或URL的行: {row}", level="WARN")
                continue

            args_list.append((
                project_url, project_id, output_root,
                row_count + 1,
                overwrite_html, overwrite_content, overwrite_assets,
                download_assets, download_workers,
                start_minimized, window_position, cover_url, row
            ))
            row_count += 1

    log(f"开始使用 {project_workers} 个工作线程处理 {len(args_list)} 个项目")
    try:
        with ThreadPoolExecutor(max_workers=project_workers) as executor:
            futures = [executor.submit(run_pipeline_with_shared_download_pool, args) for args in args_list]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result and result.get("issues"):
                        if problems_writer is None:
                            problems_file = problems_csv.open("w", encoding="utf-8", newline="")
                            extra_fields = ["issues", "download_error_count", "download_error_samples"]
                            problems_writer = csv.DictWriter(
                                problems_file,
                                fieldnames=fieldnames + extra_fields
                            )
                            problems_writer.writeheader()
                        row = dict(result.get("row") or {})
                        issues = result.get("issues") or []
                        download_failures = result.get("download_failures") or []
                        row["issues"] = ";".join(issues)
                        row["download_error_count"] = str(len(download_failures))
                        if download_failures:
                            samples = download_failures[:3]
                            row["download_error_samples"] = "; ".join(
                                f"{s.get('url', '')} -> {s.get('path', '')}: {s.get('error', '')}" for s in samples
                            )
                        else:
                            row["download_error_samples"] = ""
                        problems_writer.writerow(row)
                        problems_file.flush()
                        problems_written += 1
                except Exception as e:
                    log(f"处理项目时发生错误: {e}", level="ERROR")
    finally:
        if problems_file is not None:
            problems_file.close()

    if problems_written:
        log(f"已保存问题行CSV: {problems_csv}")
    else:
        log("没有发现问题行")
    log("所有项目处理完成")


if __name__ == "__main__":
    main()
    # run_pipeline_for_project(
    #     project_url="https://www.kickstarter.com/projects/bigmellon/a-bed-system-and-legit-blackout-blinds-for-the-tesla-model-y",
    #     project_id="1682927218", output_root=Path("data/projects"),
    #     overwrite_html=False, overwrite_content=True, overwrite_assets=False,
    #     download_assets=True, download_workers=10,
    #     start_minimized=True,
    #     cover_url="https://example.com/cover_image.jpg"  # 添加cover_url参数示例
    # )