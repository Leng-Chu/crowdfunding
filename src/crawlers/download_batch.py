"""
处理 CSV 中 content_status 为 success 的行，从 content.json 下载资源文件。
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from download_assets import download_assets_from_json


def _get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


REPO_ROOT = _get_repo_root()

def _format_prefix(csv_row_index: int, project_id: str) -> str:
    return f"[CSV第{csv_row_index}行][{project_id}]"


def simple_log(message, csv_row_index=None, project_id=None):
    """简化日志输出，支持 CSV 行号和项目 ID 前缀。"""
    timestamp = datetime.now().strftime("%H:%M:%S")

    if csv_row_index is not None and project_id is not None:
        prefix = _format_prefix(csv_row_index, project_id)
        print(f"[{timestamp}] {prefix} {message}", flush=True)
    else:
        print(f"[{timestamp}] {message}", flush=True)


def update_csv_with_download_status(csv_path, project_id, status, lock=None):
    """更新 CSV 中的 download_status 列，支持并发加锁。"""
    def _write():
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

    if lock is None:
        _write()
    else:
        with lock:
            _write()


def download_for_success_rows(
    csv_path,
    output_root,
    overwrite_assets=False,
    download_workers=10,
    download_video=False,
    start_row=None,
    end_row=None,
    logger=None,
    project_workers=None,
    skip_success=False,
):
    """
    仅处理 CSV 中 content_status 为 success 的行，从 content.json 下载资源文件。
    :param csv_path: CSV 文件路径
    :param output_root: 输出根目录
    :param overwrite_assets: 是否覆盖已存在的资源
    :param download_workers: 单项目下载并发数
    :param start_row: 开始处理的行号（从 1 开始计数）
    :param end_row: 结束处理的行号（从 1 开始计数）
    :param logger: 日志记录器
    :param project_workers: 项目级并发线程数，None 表示自动设置
    :param skip_success: 是否跳过 download_status 为 success 的行
    """
    log = logger or simple_log

    log(f"开始处理成功状态的项目 {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    if "download_status" not in df.columns:
        df["download_status"] = ""
        df.to_csv(csv_path, index=False, encoding="utf-8")

    if "content_status" not in df.columns:
        log("CSV 文件中没有 content_status 列，无法筛选成功项目")
        return

    if start_row is not None or end_row is not None:
        start_idx = start_row - 1 if start_row is not None else 0
        end_idx = end_row - 1 if end_row is not None else len(df) - 1

        start_idx = max(0, start_idx)
        end_idx = min(len(df) - 1, end_idx)

        df = df.iloc[start_idx:end_idx + 1]

    success_mask = df["content_status"].astype(str) == "success"
    target_rows = df[success_mask]

    if skip_success:
        skip_mask = target_rows["download_status"].astype(str) != "success"
        target_rows = target_rows[skip_mask]

    log(
        f"总共找到 {len(df[df['content_status'].astype(str) == 'success'])} 个成功状态的项目，"
        f"{'排除 download_status 为 success 的项目后，' if skip_success else ''}"
        f"将处理 {len(target_rows)} 个项目"
    )

    if target_rows.empty:
        log("没有需要处理的项目")
        return

    if project_workers is None:
        project_workers = 1
    project_workers = max(1, project_workers)
    effective_project_workers = min(project_workers, len(target_rows))
    csv_lock = threading.Lock()

    log(
        f"开始并发下载项目资源，项目线程数 {effective_project_workers}，"
        f"单项目下载线程数 {download_workers}"
    )

    def _download_project(row_idx, row):
        project_id = row.get("project_id")
        csv_row_index = row_idx + 1

        project_dir = output_root / project_id
        content_json_path = project_dir / "content.json"

        if not content_json_path.exists():
            log(
                f"content.json 不存在，跳过下载: {content_json_path}",
                csv_row_index=csv_row_index,
                project_id=project_id,
            )
            update_csv_with_download_status(
                csv_path, project_id, "failed: missing content.json", lock=csv_lock
            )
            return

        log("开始下载资源", csv_row_index=csv_row_index, project_id=project_id)

        try:
            download_failures = download_assets_from_json(
                str(content_json_path),
                str(project_dir),
                max_workers=download_workers,
                overwrite_files=overwrite_assets,
                download_video=download_video,
                logger=lambda msg: log(
                    msg, csv_row_index=csv_row_index, project_id=project_id
                ),
            )

            if download_failures:
                error_msg = f"failed: {len(download_failures)}"
                update_csv_with_download_status(
                    csv_path, project_id, error_msg, lock=csv_lock
                )
            else:
                update_csv_with_download_status(
                    csv_path, project_id, "success", lock=csv_lock
                )

        except Exception as e:
            error_msg = f"failed: {str(e)}"
            update_csv_with_download_status(
                csv_path, project_id, error_msg, lock=csv_lock
            )
            log(
                f"下载过程中发生错误: {str(e)}",
                csv_row_index=csv_row_index,
                project_id=project_id,
            )

    with ThreadPoolExecutor(max_workers=effective_project_workers) as executor:
        futures = [
            executor.submit(_download_project, row_idx, row)
            for row_idx, row in target_rows.iterrows()
        ]
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                log(f"线程任务异常: {exc}")


def main():
    csv_path = REPO_ROOT / "data" / "metadata" / "add_3071.csv"
    output_root = REPO_ROOT / "data" / "projects" / "tmp"
    overwrite_assets = False
    skip_success = True
    download_workers = 3
    download_video = False
    project_workers = 10
    start_row = 1
    end_row = 99999

    download_for_success_rows(
        csv_path=csv_path,
        output_root=output_root,
        overwrite_assets=overwrite_assets,
        download_workers=download_workers,
        download_video=download_video,
        project_workers=project_workers,
        start_row=start_row,
        end_row=end_row,
        skip_success=skip_success,
    )


if __name__ == "__main__":
    main()
