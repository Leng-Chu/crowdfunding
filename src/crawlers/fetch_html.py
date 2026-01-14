from pathlib import Path
import shutil
import tempfile
import time
import threading
from typing import Dict
from DrissionPage import ChromiumOptions, ChromiumPage
from DrissionPage.errors import BrowserConnectError


def _build_options() -> ChromiumOptions:
    options = ChromiumOptions()
    options.auto_port()
    options.set_argument("--disable-notifications")
    options.set_argument("--start-minimized")
    options.set_argument("--window-position=-32000,-32000")  # 设置为屏幕外位置
    # 禁止加载图片
    options.set_argument("--blink-settings=imagesEnabled=false")
    return options

_BROWSER_START_LOCK = threading.RLock()
_AUTOPORTDATA_DIR = Path(tempfile.gettempdir()) / "DrissionPage" / "autoPortData"


def _is_in_use_error(exc: Exception) -> bool:
    winerror = getattr(exc, "winerror", None)
    return isinstance(exc, PermissionError) or winerror in (5, 32, 33)


def _snapshot_auto_port_dirs() -> Dict[str, float]:
    if not _AUTOPORTDATA_DIR.exists():
        return {}

    result: Dict[str, float] = {}
    for child in _AUTOPORTDATA_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            result[child.name] = child.stat().st_mtime
        except Exception:
            result[child.name] = 0.0
    return result


def _delete_path_with_retry(target: Path, logger=None, timeout_seconds: float = 30.0) -> None:
    log = logger or (lambda *_: None)
    deadline = time.time() + max(0.0, timeout_seconds)
    delay_seconds = 0.2

    while True:
        try:
            if not target.exists():
                return

            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return
        except FileNotFoundError:
            return
        except Exception as exc:
            if _is_in_use_error(exc) and time.time() < deadline:
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 2.0)
                continue
            log(f"autoPortData 清理失败: {target} ({exc})")
            return


def clear_drissionpage_auto_port_data(logger=None) -> None:
    """Best-effort cleanup for DrissionPage temp cache files under autoPortData.

    Note: On Windows, the browser process may hold handles briefly after `quit()`.
    This function retries for a short period to reduce transient "file in use" errors.
    """
    log = logger or (lambda *_: None)
    with _BROWSER_START_LOCK:
        try:
            if not _AUTOPORTDATA_DIR.exists():
                return

            for item in _AUTOPORTDATA_DIR.iterdir():
                _delete_path_with_retry(item, logger=log, timeout_seconds=5.0)
        except Exception as exc:
            log(f"autoPortData 清理失败: {exc}")


def fetch_html(url: str, output_path: str,
               overwrite_html: bool = False,
               wait_seconds: float = 0,
               logger=None,
               browser_page=None) -> None:
    """
    抓取网页HTML并保存到文件
    
    Args:
        url (str): 要抓取的网页URL
        output_path (str): 输出文件路径
        overwrite_html (bool): 是否覆盖已存在的HTML文件，默认为False
        wait_seconds (float): 等待页面加载的时间（秒），默认为0
        browser_page: 可选的浏览器实例，如果提供则复用该实例，否则创建新的实例
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否存在且不需要覆盖
    log = logger or print
    if output_path.exists() and not overwrite_html:
        log(f"HTML文件已存在，跳过抓取: {output_path}")
        return
    
    # 如果没有提供浏览器实例，则创建一个新的
    should_quit = False
    auto_port_profile_dir = None
    if browser_page is None:
        with _BROWSER_START_LOCK:
            before = _snapshot_auto_port_dirs()
            options = _build_options()
            page = ChromiumPage(options)
            after = _snapshot_auto_port_dirs()
            created = set(after) - set(before)
            if created:
                dir_name = max(created, key=lambda name: after.get(name, 0.0))
                auto_port_profile_dir = _AUTOPORTDATA_DIR / dir_name
        should_quit = True
    else:
        page = browser_page
    time.sleep(wait_seconds)
    try:
        page.get(url)
        # 等待页面基本结构加载
        page.wait.doc_loaded()
        # 等待 story-content 区域出现
        page.wait.eles_loaded('.story-content', timeout=10)
        # 滚动到页面底部，触发懒加载内容
        page.scroll.to_bottom()
        
        # 滚动到顶部
        page.scroll.to_top()
        # 获取完整的HTML内容
        html = page.html
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"HTML已保存到: {output_path}")
        
    except BrowserConnectError as e:
        log(f"浏览器连接失败: {e}")
        raise e
    finally:
        # 只有在函数内部创建了浏览器实例时才退出
        if should_quit and page:
            try:
                page.quit()
            finally:
                if auto_port_profile_dir is not None:
                    _delete_path_with_retry(auto_port_profile_dir, logger=log, timeout_seconds=30.0)


if __name__ == "__main__":
    # 使用示例
    URL = "https://www.kickstarter.com/projects/capseal/capseal-craft-your-own-flavor-cafe-capsule-at-home"
    OUTPUT_HTML = "data/projects/sample/page.html"
    overwrite_html = True
    fetch_html(URL, OUTPUT_HTML, overwrite_html)
