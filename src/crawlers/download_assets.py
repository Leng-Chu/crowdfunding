import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


BASE_URL = "https://www.kickstarter.com/"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
}
ALT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
}
DEFAULT_TIMEOUT = (10, 120)
DEFAULT_CHUNK_SIZE = 1024 * 512
DEFAULT_MAX_RETRIES = 3


def _ensure_output_dirs(output_dir):
    """确保输出目录存在，创建 cover 和 photo 子目录。"""
    output_dir = Path(output_dir)
    cover_dir = output_dir / "cover"
    photo_dir = output_dir / "photo"
    cover_dir.mkdir(parents=True, exist_ok=True)
    photo_dir.mkdir(parents=True, exist_ok=True)
    return cover_dir, photo_dir


def _download_file(
    url,
    path,
    logger=None,
    max_retries=DEFAULT_MAX_RETRIES,
    timeout=DEFAULT_TIMEOUT,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    log = logger or print
    if not url:
        return False, "empty_url"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parsed_url = urlparse(url)
    if parsed_url.scheme and parsed_url.netloc:
        full_url = url
    else:
        full_url = urljoin(BASE_URL, url)

    base_headers = DEFAULT_HEADERS.copy()
    base_headers["Accept"] = "*/*"
    base_headers["Accept-Encoding"] = "identity"
    header_variants = (base_headers, ALT_HEADERS)
    part_path = Path(str(path) + ".part")
    last_error = None

    for attempt in range(1, max_retries + 1):
        headers = header_variants[(attempt - 1) % len(header_variants)].copy()
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            with requests.get(full_url, headers=headers, stream=True, timeout=timeout) as response:
                if response.status_code == 416:
                    part_path.unlink(missing_ok=True)
                    continue

                if response.status_code == 200 and existing_size:
                    part_path.unlink(missing_ok=True)
                    existing_size = 0

                if response.status_code not in (200, 206):
                    last_error = f"status_code:{response.status_code}"
                    log(f"下载失败 {url}: {response.status_code}")
                    time.sleep(min(2 * attempt, 6))
                    continue

                open_mode = "ab" if existing_size else "wb"
                with open(part_path, open_mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

                if response.status_code == 200:
                    total_length = response.headers.get("Content-Length")
                    if total_length is not None:
                        expected = int(total_length)
                        actual = part_path.stat().st_size
                        if actual != expected:
                            raise requests.exceptions.ChunkedEncodingError(
                                f"IncompleteRead({actual} bytes read, {expected - actual} more expected)"
                            )

                part_path.replace(path)
                return True, None

        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            last_error = f"connection_error:{e}"
            log(f"下载失败 {url}: 连接中断 - {e}")
        except requests.exceptions.ContentDecodingError as e:
            last_error = f"content_decoding_error:{e}"
            log(f"下载失败 {url}: 内容解码错误 - {e}")
        except requests.exceptions.ReadTimeout as e:
            last_error = f"read_timeout:{e}"
            log(f"下载失败 {url}: 读取超时 - {e}")
        except Exception as e:
            last_error = f"error:{e}"
            log(f"下载失败 {url}: {e}")

        time.sleep(min(2 * attempt, 6))

    return False, last_error


def download_assets_from_json(
    content_json_path,
    output_dir=None,
    max_workers=6,
    overwrite_files=False,
    download_video=False,
    logger=None,
    max_retries=DEFAULT_MAX_RETRIES,
    timeout=DEFAULT_TIMEOUT,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    log = logger or print
    with open(content_json_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if output_dir is None:
        output_dir = Path(content_json_path).parent
    cover_dir, _ = _ensure_output_dirs(output_dir)

    tasks = []
    cover_image_url = content.get("cover_image")
    video_url = content.get("video")

    if cover_image_url:
        img_path = cover_dir / "cover_image.jpg"
        if overwrite_files or not img_path.exists():
            tasks.append((cover_image_url, str(img_path)))
        # else:
        #     log(f"封面图片已存在，跳过下载: {img_path}")

    if download_video and video_url:
        video_path = cover_dir / "project_video.mp4"
        if overwrite_files or not video_path.exists():
            tasks.append((video_url, str(video_path)))
        # else:
        #     log(f"视频文件已存在，跳过下载: {video_path}")

    for item in content.get("content_sequence", []):
        if item.get("type") != "image":
            continue
        filename = item.get("filename")
        img_url = item.get("url")
        if not filename or not img_url:
            continue
        img_path = os.path.join(output_dir, filename)
        path = Path(img_path)
        if overwrite_files or not path.exists():
            tasks.append((img_url, img_path))
        # else:
        #     log(f"图片文件已存在，跳过下载: {img_path}")

    if not tasks:
        log("没有需要下载的资源或所有资源已存在")
        return []

    effective_workers = max(1, min(max_workers, len(tasks)))
    # log(f"开始下载 {len(tasks)} 个资源文件，使用 {effective_workers} 个线程...")

    def _download_task(args):
        url, path = args
        ok, error = _download_file(
            url,
            path,
            logger=log,
            max_retries=max_retries,
            timeout=timeout,
            chunk_size=chunk_size,
        )
        if not ok:
            return {"url": url, "path": path, "error": error or "unknown_error"}
        return None

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        results = list(executor.map(_download_task, tasks))
    failures = [item for item in results if item]
    log(f"资源下载完成，成功 {len(tasks) - len(failures)} 个，失败 {len(failures)} 个")
    return failures


if __name__ == "__main__":
    content_json_path = "data/projects/sample/content.json"
    output_dir = "data/projects/sample"
    max_workers = 10
    overwrite_files = True
    download_video=False
    download_assets_from_json(
        content_json_path,
        output_dir,
        max_workers=max_workers,
        overwrite_files=overwrite_files,
        download_video=download_video
    )
