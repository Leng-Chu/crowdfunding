from pathlib import Path
import time
from DrissionPage import ChromiumOptions, ChromiumPage

def _build_options(start_minimized=False,
                   disable_notifications=True,
                   window_position=None,
                   window_size=None) -> ChromiumOptions:
    options = ChromiumOptions()
    options.auto_port()
    if disable_notifications:
        options.set_argument("--disable-notifications")
    if start_minimized:
        options.set_argument("--start-minimized")
    if window_position:
        options.set_argument(f"--window-position={window_position[0]},{window_position[1]}")
    if window_size:
        options.set_argument(f"--window-size={window_size[0]},{window_size[1]}")
    # 仅抓取HTML时禁用图片加载以提速
    options.set_argument("--blink-settings=imagesEnabled=false")
    return options


def fetch_html(url: str, output_path: str,
               overwrite_html: bool = False,
               start_minimized: bool = False,
               window_position=None,
               logger=None) -> None:
    """
    抓取网页HTML并保存到文件
    
    Args:
        url (str): 要抓取的网页URL
        output_path (str): 输出文件路径
        overwrite_html (bool): 是否覆盖已存在的HTML文件，默认为False
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否存在且不需要覆盖
    log = logger or print
    if output_path.exists() and not overwrite_html:
        log(f"HTML文件已存在，跳过抓取: {output_path}")
        return
    
    options = _build_options(
        start_minimized=start_minimized,
        window_position=window_position,
    )
    page = ChromiumPage(options)
    page.get(url)
    
    # 等待 class="story-content" 的 div 元素及其子内容加载完成
    page.wait.eles_loaded('.story-content')
    
    html = page.html
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    page.close()
    
    log(f"HTML已保存到: {output_path}")


if __name__ == "__main__":
    # 使用示例
    URL = "https://www.kickstarter.com/projects/capseal/capseal-craft-your-own-flavor-cafe-capsule-at-home"
    OUTPUT_HTML = "data/projects/sample/page.html"
    overwrite_html = True
    fetch_html(URL, OUTPUT_HTML, overwrite_html, start_minimized=True)