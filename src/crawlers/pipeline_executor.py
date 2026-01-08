"""
1. 爬取HTML
2. 解析HTML生成JSON
3. 下载资源文件
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_html import fetch_html
from parse_content import parse_story_content
from download_assets import download_assets_from_json
from DrissionPage import ChromiumOptions, ChromiumPage
from DrissionPage.errors import BrowserConnectError


def _build_options() -> ChromiumPage:
    options = ChromiumOptions()
    options.auto_port()
    options.set_argument("--disable-notifications")
    options.set_argument("--start-minimized")
    options.set_argument("--window-position=-32000,-32000")  # 设置为屏幕外位置
    # 禁止加载图片
    options.set_argument("--blink-settings=imagesEnabled=false")
    return options


def simple_log(message):
    """简化日志输出，直接打印时间戳和消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def update_csv_with_status(csv_path, project_id, status):
    # 读取当前CSV文件
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    
    if "success_status" not in df.columns:
        df["success_status"] = ""
    if "project_id" not in df.columns:
        df["project_id"] = ""

    mask = df["project_id"].astype(str) == str(project_id)
    if mask.any():
        df.loc[mask, "success_status"] = status

    df = df.fillna("")
    
    # 直接写入原文件（在锁的保护下）
    df.to_csv(csv_path, index=False, encoding="utf-8")


def main():
    csv_path = Path("data/metadata/years/2024_31932.csv")
    output_root = Path("data/projects/2024")
    overwrite_html = True
    overwrite_content = True
    overwrite_assets = False
    download_assets = False
    start_row = 1  # 从第几行开始处理（从1开始计数）
    end_row = None  # 到第几行结束处理，会处理end_row这一行（None表示处理到文件末尾）
    wait_seconds = 1 # 等待时间，指数退避

    output_root.mkdir(parents=True, exist_ok=True)

    args_list = []
    row_count = 0
    simple_log(f"开始处理 {csv_path}")
    
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for row_idx, row in df.iterrows():
        # Filter by requested row range
        if row_idx < start_row - 1:
            continue
        if end_row is not None and row_idx >= end_row:
            break

        # Only process rows not marked as success
        success_status = row.get('success_status', '')
        if success_status == 'success':
            continue  # Skip rows already marked as success

        project_id = row.get("project_id")
        project_url = row.get("project_url")
        cover_url = row.get("cover_url")

        args_list.append((
            project_url, project_id, output_root,
            row_idx + 1,  # Use 1-based CSV row number
            overwrite_html, overwrite_content, overwrite_assets,
            download_assets, 10,  # download_workers kept for interface compatibility
            cover_url, row.to_dict()
        ))
        row_count += 1

    simple_log(f"开始顺序处理 {len(args_list)} 个项目，从第{start_row}到第{end_row or '末尾'}行")
    
    # 创建浏览器实例
    options = _build_options()
    browser = ChromiumPage(options)
    
    try:
        # 顺序处理所有项目
        for args in args_list:
            (project_url, project_id, output_root, csv_row_index, overwrite_html,
             overwrite_content, overwrite_assets, download_assets, download_workers,
             cover_url, row) = args

            simple_log(f"处理项目 {project_id}，行号 {csv_row_index}")
            # 直接执行项目处理逻辑
            project_dir = output_root / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            html_path = project_dir / "page.html"

            # 获取HTML（使用复用的浏览器实例）
            # 如果连接失败，重新创建浏览器实例
            try:
                fetch_html(
                    project_url,
                    str(html_path),
                    overwrite_html=overwrite_html,
                    wait_seconds=wait_seconds,
                    logger=simple_log,
                    browser_page=browser,  # 传入复用的浏览器实例
                )
            except BrowserConnectError as e:
                simple_log(f"浏览器连接失败，重新创建浏览器实例: {e}")
                # 关闭旧的浏览器实例
                try:
                    browser.quit()
                except:
                    pass  # 忽略关闭时的错误
                
                # 创建新的浏览器实例
                options = _build_options()
                browser = ChromiumPage(options)
                
                # 重新尝试获取HTML
                fetch_html(
                    project_url,
                    str(html_path),
                    overwrite_html=overwrite_html,
                    wait_seconds=wait_seconds,
                    logger=simple_log,
                    browser_page=browser,
                )

            result = parse_story_content(
                str(html_path),
                str(project_dir),
                project_url=project_url,
                cover_url=cover_url,  # 传递封面图片URL参数
                overwrite_content=overwrite_content,
                logger=simple_log,
            )

            # 检查是否有问题
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
                        logger=simple_log,
                    )
                    if download_failures:
                        issues.append("asset_download_failed")
                else:
                    issues.append("missing_content_json")

            if issues:
                issue_str = ','.join(issues)
                # 更新CSV文件，标记处理失败及原因
                update_csv_with_status(csv_path, project_id, f"failed: {issue_str}")
                wait_seconds *= 2  # 失败后等待时间翻倍
                simple_log(f"[CSV第{csv_row_index}行][{project_id}] 项目有 {len(issues)} 个问题: {issue_str}")
            else:
                update_csv_with_status(csv_path, project_id, "success")
                wait_seconds = max(1, wait_seconds // 2)  # 成功后等待时间减半，但不低于1秒
                simple_log(f"[CSV第{csv_row_index}行][{project_id}] 项目流水线执行成功")
            simple_log(f"等待时间: {wait_seconds}")

    finally:
        # 在处理完所有项目后，关闭浏览器实例
        try:
            browser.quit()
        except:
            pass  # 忽略关闭时的错误

    simple_log("所有项目处理完成")


if __name__ == "__main__":
    main()