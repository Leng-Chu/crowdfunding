import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Any, Dict, Optional, Tuple

import requests

# 可选：用于提取图片真实格式与尺寸（用于写回 content.json 的 width/height，并修正扩展名）
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


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


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _normalize_relpath(path: Path) -> str:
    """把相对路径规范成 content.json 使用的形式（统一用 '/'）。"""
    return path.as_posix()


def _detect_image_format_and_size(path: Path) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    """
    返回 (format, width, height, error)。
    - format 为 PIL 识别出的格式，如 'JPEG'/'PNG'/'WEBP'/...
    - 若 Pillow 不可用或读取失败，返回 error。
    """
    if Image is None:
        return None, None, None, "missing_pillow"
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").strip().upper()
            w, h = img.size
        if not fmt:
            fmt = None
        return fmt, int(w), int(h), None
    except Exception as e:
        return None, None, None, f"image_open_error:{type(e).__name__}:{e}"


def _format_to_extension(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    fmt = str(fmt).strip().upper()
    if fmt == "JPEG":
        return ".jpg"
    if fmt == "PNG":
        return ".png"
    if fmt == "WEBP":
        return ".webp"
    if fmt == "GIF":
        return ".gif"
    if fmt == "AVIF":
        return ".avif"
    return "." + fmt.lower()


def _standardize_image_file(
    abs_path: Path,
    project_dir: Path,
    allow_rename: bool = True,
) -> Tuple[Path, Optional[int], Optional[int], Optional[str], Optional[str]]:
    """
    对单个图片文件：
    - 用 Pillow 读取真实格式与宽高
    - 如扩展名不匹配则重命名（仅改后缀）

    返回 (abs_path_after, width, height, detected_format, error)。
    """
    fmt, w, h, err = _detect_image_format_and_size(abs_path)
    if err is not None:
        return abs_path, None, None, None, err

    new_ext = _format_to_extension(fmt)
    if not new_ext:
        return abs_path, w, h, fmt, None

    if not allow_rename:
        return abs_path, w, h, fmt, None

    if abs_path.suffix.lower() == str(new_ext).lower():
        return abs_path, w, h, fmt, None

    new_abs_path = abs_path.with_suffix(new_ext)
    if new_abs_path.exists() and new_abs_path != abs_path:
        # 避免覆盖已有文件，保留原文件名
        return abs_path, w, h, fmt, f"rename_conflict:{new_abs_path.name}"

    try:
        abs_path.rename(new_abs_path)
    except Exception as e:
        return abs_path, w, h, fmt, f"rename_error:{type(e).__name__}:{e}"

    return new_abs_path, w, h, fmt, None


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
    title: Optional[str] = None,
    blurb: Optional[str] = None,
    cover_url: Optional[str] = None,
    max_retries=DEFAULT_MAX_RETRIES,
    timeout=DEFAULT_TIMEOUT,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    log = logger or print
    with open(content_json_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if output_dir is None:
        output_dir = Path(content_json_path).parent
    output_dir = Path(output_dir)
    cover_dir, _ = _ensure_output_dirs(output_dir)

    changed = False
    failures = []

    # 1) 先补全 title/blurb（用于训练侧 attr；不依赖下载）
    def _as_text_field(v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if isinstance(v, dict):
            text = str(v.get("content", "") or "").strip()
            if not text:
                return None
            length = _safe_int(v.get("content_length"), default=len(text))
            return {"content": text, "content_length": int(length)}
        text = str(v).strip()
        if not text:
            return None
        return {"content": text, "content_length": int(len(text))}

    if title is not None:
        cur = content.get("title")
        if _as_text_field(cur) is None:
            new_v = _as_text_field(title)
            if new_v is not None:
                content["title"] = new_v
                changed = True
    if blurb is not None:
        cur = content.get("blurb")
        if _as_text_field(cur) is None:
            new_v = _as_text_field(blurb)
            if new_v is not None:
                content["blurb"] = new_v
                changed = True

    # 如果已存在 title/blurb 但缺少 content_length，也补上
    for key in ("title", "blurb"):
        cur = content.get(key)
        fixed = _as_text_field(cur)
        if fixed is not None and fixed != cur:
            content[key] = fixed
            changed = True

    tasks = []

    # 2) 统一 cover_image 字段为 dict（兼容旧数据：str -> dict）
    cover_image_value = content.get("cover_image")
    cover_image: Optional[Dict[str, Any]] = None
    if isinstance(cover_image_value, dict):
        cover_image = cover_image_value
    elif isinstance(cover_image_value, str) and cover_image_value.strip():
        cover_image = {"url": cover_image_value.strip(), "filename": "", "width": 0, "height": 0}
        content["cover_image"] = cover_image
        changed = True
    elif cover_image_value is None:
        cover_image = None
    else:
        # 不支持的类型，直接置空，避免 downstream 误用
        cover_image = None
        if "cover_image" in content:
            content.pop("cover_image", None)
            changed = True

    cover_image_url = ""
    if isinstance(cover_image, dict):
        cover_image_url = str(cover_image.get("url", "") or "").strip()

    # 兼容：若 content.json 没有 cover_image 或 url 为空，但 CSV 提供了 cover_url，则写回
    if cover_url is not None and str(cover_url).strip():
        if cover_image is None:
            cover_image = {"url": str(cover_url).strip(), "filename": "", "width": 0, "height": 0}
            content["cover_image"] = cover_image
            cover_image_url = str(cover_url).strip()
            changed = True
        elif isinstance(cover_image, dict) and not cover_image_url:
            cover_image["url"] = str(cover_url).strip()
            cover_image_url = str(cover_url).strip()
            changed = True
    video_url = content.get("video")

    # 3) 处理封面下载（文件名：cover/cover_image.<ext>）
    cover_abs_path: Optional[Path] = None
    if isinstance(cover_image, dict):
        # 3.1 若 content.json 已记录 filename 且文件存在，优先使用
        fname = str(cover_image.get("filename", "") or "").strip()
        if fname:
            p = output_dir / Path(fname)
            if p.exists():
                cover_abs_path = p

        # 3.2 否则在 cover/ 下查找已存在的 cover_image.*
        if cover_abs_path is None:
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"):
                p = cover_dir / f"cover_image{ext}"
                if p.exists():
                    cover_abs_path = p
                    rel = _normalize_relpath(Path("cover") / p.name)
                    if str(cover_image.get("filename", "") or "") != rel:
                        cover_image["filename"] = rel
                        changed = True
                    break

        # 3.3 若仍不存在且有 url，则下载到默认路径（后续会按真实格式重命名）
        if cover_abs_path is None and cover_image_url:
            default_cover = cover_dir / "cover_image.jpg"
            cover_abs_path = default_cover
            if overwrite_files or not default_cover.exists():
                tasks.append(("cover_image", cover_image_url, default_cover))

    if download_video and video_url:
        video_path = cover_dir / "project_video.mp4"
        if overwrite_files or not video_path.exists():
            tasks.append(("video", video_url, video_path))
        # else:
        #     log(f"视频文件已存在，跳过下载: {video_path}")

    # 4) 构造正文图片下载任务；同时在这里补全 text 的 content_length（旧数据兼容）
    seq = content.get("content_sequence", [])
    if isinstance(seq, list):
        for idx, item in enumerate(seq):
            if not isinstance(item, dict):
                continue
            t = str(item.get("type", "") or "").strip().lower()
            if t == "text":
                if "content_length" not in item:
                    text = str(item.get("content", "") or "")
                    item["content_length"] = int(len(text))
                    changed = True
                continue
            if t != "image":
                continue

            filename = str(item.get("filename", "") or "").strip()
            img_url = str(item.get("url", "") or "").strip()
            if not filename:
                # 尽量给一个确定的默认名，避免后续处理失败
                filename = f"photo/story_image_{idx + 1}.jpeg"
                item["filename"] = filename
                changed = True
            abs_path = output_dir / Path(filename)
            if not abs_path.exists() and not img_url:
                failures.append({"url": "", "path": str(abs_path), "error": "missing_url"})
                continue
            if overwrite_files or not abs_path.exists():
                tasks.append((f"story_image:{idx}", img_url, abs_path))
    else:
        failures.append({"url": "", "path": str(Path(content_json_path)), "error": "bad_content_sequence"})

    # 5) 并发下载所有资源
    if tasks:
        effective_workers = max(1, min(int(max_workers), len(tasks)))

        def _download_task(args):
            kind, url, path = args
            ok, error = _download_file(
                url,
                path,
                logger=log,
                max_retries=max_retries,
                timeout=timeout,
                chunk_size=chunk_size,
            )
            if not ok:
                return {"url": url, "path": str(path), "error": error or "unknown_error", "kind": kind}
            return None

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            results = list(executor.map(_download_task, tasks))
        download_failures = [item for item in results if item]
        failures.extend(download_failures)

    # 6) 下载后：标准化扩展名 + 写入宽高（cover 与 story images）
    # 6.1 cover
    if isinstance(cover_image, dict) and cover_abs_path is not None and cover_abs_path.exists():
        std_path, w, h, fmt, err = _standardize_image_file(cover_abs_path, output_dir, allow_rename=True)
        if err is not None and not str(err).startswith("rename_conflict"):
            failures.append(
                {
                    "url": cover_image_url,
                    "path": str(cover_abs_path),
                    "error": err,
                    "kind": "cover_image_postprocess",
                }
            )
        if std_path != cover_abs_path:
            cover_abs_path = std_path
        rel = _normalize_relpath(Path("cover") / cover_abs_path.name)
        if str(cover_image.get("filename", "") or "") != rel:
            cover_image["filename"] = rel
            changed = True
        if w is not None and h is not None and (int(w) > 0 and int(h) > 0):
            if _safe_int(cover_image.get("width"), 0) != int(w) or _safe_int(cover_image.get("height"), 0) != int(h):
                cover_image["width"] = int(w)
                cover_image["height"] = int(h)
                changed = True
        else:
            failures.append(
                {
                    "url": cover_image_url,
                    "path": str(cover_abs_path),
                    "error": "invalid_cover_size",
                    "kind": "cover_image_postprocess",
                }
            )

    # 6.2 story images
    seq = content.get("content_sequence", [])
    if isinstance(seq, list):
        for idx, item in enumerate(seq):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "") or "").strip().lower() != "image":
                continue
            filename = str(item.get("filename", "") or "").strip()
            if not filename:
                continue
            abs_path = output_dir / Path(filename)
            if not abs_path.exists():
                failures.append(
                    {
                        "url": str(item.get("url", "") or ""),
                        "path": str(abs_path),
                        "error": "missing_file_after_download",
                        "kind": f"story_image:{idx}",
                    }
                )
                continue

            std_path, w, h, fmt, err = _standardize_image_file(abs_path, output_dir, allow_rename=True)
            if err is not None and not str(err).startswith("rename_conflict"):
                failures.append(
                    {
                        "url": str(item.get("url", "") or ""),
                        "path": str(abs_path),
                        "error": err,
                        "kind": f"story_image_postprocess:{idx}",
                    }
                )

            if std_path != abs_path:
                rel = _normalize_relpath(Path(filename).with_suffix(std_path.suffix))
                item["filename"] = rel
                abs_path = std_path
                changed = True

            if w is not None and h is not None and int(w) > 0 and int(h) > 0:
                if _safe_int(item.get("width"), 0) != int(w) or _safe_int(item.get("height"), 0) != int(h):
                    item["width"] = int(w)
                    item["height"] = int(h)
                    changed = True
            else:
                failures.append(
                    {
                        "url": str(item.get("url", "") or ""),
                        "path": str(abs_path),
                        "error": "invalid_image_size",
                        "kind": f"story_image_postprocess:{idx}",
                    }
                )

    # 7) 写回 content.json（只在发生变化时写）
    if changed:
        with open(content_json_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    if tasks:
        log(f"资源下载完成，成功 {len(tasks) - len([x for x in failures if x.get('kind') in {t[0] for t in tasks}])} 个，失败 {len([x for x in failures if x.get('kind') in {t[0] for t in tasks}])} 个")
    return failures


if __name__ == "__main__":
    content_json_path = "data/projects/test/sample/content.json"
    output_dir = "data/projects/test/sample"
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
