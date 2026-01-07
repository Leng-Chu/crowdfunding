"""
1. 爬取HTML
2. 解析HTML生成JSON
3. 下载资源文件
"""

import csv
from datetime import datetime
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_html import fetch_html
from parse_content import parse_story_content
from download_assets import download_assets_from_json


def simple_log(message):
    """简化日志输出，直接打印时间戳和消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_success_status_column(csv_path):
    """确保CSV文件包含success_status列"""
    rows = []
    fieldnames = []
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # 检查是否已有success_status列
    if 'success_status' not in fieldnames:
        fieldnames = list(fieldnames) + ['success_status']
        # 为已有数据添加默认值
        for row in rows:
            row['success_status'] = ''
    
    # 写回CSV文件
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_csv_with_status(csv_path, project_id, status):
    """更新CSV文件中的特定项目，添加处理状态"""
    rows = []
    fieldnames = []
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # 更新指定项目的状态
    for row in rows:
        if row['project_id'] == project_id:
            row['success_status'] = status
            break
    
    # 写回CSV文件
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    csv_path = Path("data/metadata/years/2024_31932.csv")
    output_root = Path("data/projects/2024")
    overwrite_html = True
    overwrite_content = True
    overwrite_assets = False
    download_assets = False
    start_row = 1  # 从第几行开始处理（从1开始计数）
    end_row = None  # 到第几行结束处理，会处理end_row这一行（None表示处理到文件末尾）
    wait_seconds = 1

    output_root.mkdir(parents=True, exist_ok=True)

    # 确保CSV文件包含success_status列
    ensure_success_status_column(csv_path)

    args_list = []
    row_count = 0
    simple_log(f"开始处理 {csv_path}")
    
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            # 检查是否在处理范围内
            if row_idx < start_row - 1:
                continue
            if end_row is not None and row_idx >= end_row:
                break

            # 只处理未成功的行
            success_status = row.get('success_status', '')
            if success_status == 'success':
                continue  # 跳过已成功的行

            project_id = row.get("project_id")
            project_url = row.get("project_url")
            cover_url = row.get("cover_url")
            
            args_list.append((
                project_url, project_id, output_root,
                row_idx + 1,  # 使用CSV文件中的实际行号
                overwrite_html, overwrite_content, overwrite_assets,
                download_assets, 10,  # download_workers 参数保留以保持接口兼容性
                True, (-32000, -32000), cover_url, row, wait_seconds
            ))
            row_count += 1

    simple_log(f"开始顺序处理 {len(args_list)} 个项目，从第{start_row}行到第{end_row or '末尾'}行")
    
    # 顺序处理所有项目
    for args in args_list:
        (project_url, project_id, output_root, csv_row_index, overwrite_html,
         overwrite_content, overwrite_assets, download_assets, download_workers,
         start_minimized, window_position, cover_url, row, wait_seconds) = args

        simple_log(f"处理项目 {project_id}，行号 {csv_row_index}")
        # 直接执行项目处理逻辑
        project_dir = output_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        html_path = project_dir / "page.html"

        # 获取HTML
        fetch_html(
            project_url,
            str(html_path),
            overwrite_html=overwrite_html,
            wait_seconds=wait_seconds,
            start_minimized=start_minimized,
            window_position=window_position,
            logger=simple_log,
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
            simple_log(f"[CSV第{csv_row_index}行][{project_id}] 项目有 {len(issues)} 个问题: {issue_str}")
            # 更新CSV文件，标记处理失败及原因
            update_csv_with_status(csv_path, project_id, f"failed: {issue_str}")
            wait_seconds*=2
        else:
            simple_log(f"[CSV第{csv_row_index}行][{project_id}] 项目流水线执行成功")
            # 更新CSV文件，标记处理成功
            update_csv_with_status(csv_path, project_id, "success")
            if wait_seconds>1:
                wait_seconds/=2
        simple_log(f"等待时间: {wait_seconds}")

    simple_log("所有项目处理完成")


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