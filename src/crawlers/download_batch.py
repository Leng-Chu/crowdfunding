"""
处理CSV中content_status为"success"的行，从content.json下载资源文件
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import threading
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from download_assets import download_assets_from_json


def _format_prefix(csv_row_index: int, project_id: str) -> str:
    return f"[CSV第{csv_row_index}行][{project_id}]"


def simple_log(message, csv_row_index=None, project_id=None):
    """简化日志输出，直接打印时间戳和消息，支持CSV行数和项目ID前缀"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if csv_row_index is not None and project_id is not None:
        prefix = _format_prefix(csv_row_index, project_id)
        print(f"[{timestamp}] {prefix} {message}", flush=True)
    else:
        print(f"[{timestamp}] {message}", flush=True)


def update_csv_with_download_status(csv_path, project_id, status):
    """更新CSV中的download_status列"""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    
    if "download_status" not in df.columns:
        df["download_status"] = ""
    if "project_id" not in df.columns:
        df["project_id"] = ""

    mask = df["project_id"].astype(str) == str(project_id)
    if mask.any():
        df.loc[mask, "download_status"] = status

    df = df.fillna("")
    df.to_csv(csv_path, index=False, encoding="utf-8")


def download_for_success_rows(csv_path, output_root, overwrite_assets=False, download_workers=10, start_row=None, end_row=None, logger=None):
    """
    仅处理CSV中content_status为"success"的行，从content.json下载资源文件
    :param csv_path: CSV文件路径
    :param output_root: 输出根目录
    :param overwrite_assets: 是否覆盖已存在的资源
    :param download_workers: 下载工作线程数
    :param start_row: 开始处理的行号（从1开始计数），None表示从第一行开始
    :param end_row: 结束处理的行号（从1开始计数），None表示处理至文件末尾
    :param logger: 日志记录器
    """
    log = logger or simple_log
    
    log(f"开始处理成功状态的项目 {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    
    # 确保download_status列存在
    if "download_status" not in df.columns:
        df["download_status"] = ""
        df.to_csv(csv_path, index=False, encoding="utf-8")
    
    # 确保content_status列存在
    if "content_status" not in df.columns:
        log("CSV文件中没有content_status列，无法筛选成功项目")
        return

    # 筛选出content_status为"success"的行
    success_mask = df["content_status"].astype(str) == "success"
    target_rows = df[success_mask]

    # 应用start_row和end_row参数
    if start_row is not None or end_row is not None:
        # 将从1开始的索引转换为从0开始的索引
        start_idx = start_row - 1 if start_row is not None else 0
        end_idx = end_row - 1 if end_row is not None else len(target_rows) - 1
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(target_rows) - 1, end_idx)
        
        # 获取实际的行索引
        actual_indices = target_rows.index[start_idx:end_idx+1]
        target_rows = target_rows.loc[actual_indices]

    log(f"总共找到 {len(df[success_mask])} 个成功状态的项目，将处理其中的 {start_row}-{end_row} 行，实际处理 {len(target_rows)} 个项目")

    for row_idx, row in target_rows.iterrows():
        project_id = row.get("project_id")
        csv_row_index = row_idx + 1  

        project_dir = output_root / project_id
        content_json_path = project_dir / "content.json"

        if not content_json_path.exists():
            log(f"content.json不存在，跳过下载: {content_json_path}", csv_row_index=csv_row_index, project_id=project_id)
            update_csv_with_download_status(csv_path, project_id, "failed: missing content.json")
            continue
        log(f"开始下载资源", csv_row_index=csv_row_index, project_id=project_id)

        try:
            download_failures = download_assets_from_json(
                str(content_json_path),
                str(project_dir),
                max_workers=download_workers,
                overwrite_files=overwrite_assets,
                logger=lambda msg: log(msg, csv_row_index=csv_row_index, project_id=project_id),
            )

            if download_failures:
                error_msg = f"failed: {len(download_failures)}"
                update_csv_with_download_status(csv_path, project_id, error_msg)
            else:
                update_csv_with_download_status(csv_path, project_id, "success")


        except Exception as e:
            error_msg = f"failed: {str(e)}"
            update_csv_with_download_status(csv_path, project_id, error_msg)
            log(f"下载过程中发生错误: {str(e)}")


def main():
    csv_path = Path("data/metadata/years/2024 copy.csv")  # 修改为实际的CSV文件路径
    output_root = Path("data/projects/2024_1_5000")  # 修改为实际的输出目录路径
    overwrite_assets = True  # 是否覆盖已存在的资源
    download_workers = 10  # 并行下载线程数
    start_row = 1  # 开始处理的行号（从1开始计数）
    end_row = 10  # 结束处理的行号（从1开始计数）

    download_for_success_rows(
        csv_path=csv_path,
        output_root=output_root,
        overwrite_assets=overwrite_assets,
        download_workers=download_workers,
        start_row=start_row,
        end_row=end_row
    )


if __name__ == "__main__":
    main()