from pathlib import Path
from DrissionPage import ChromiumOptions, ChromiumPage

def _build_options() -> ChromiumOptions:
    options = ChromiumOptions()
    options.auto_port()
    # 仅抓取HTML时禁用图片加载以提速
    options.set_argument("--blink-settings=imagesEnabled=false")
    return options


def fetch_html(url: str, output_path: str, overwrite_html: bool = False, logger=None) -> None:
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
    
    options = _build_options()
    page = ChromiumPage(options)
    page.get(url)
    html = page.html
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    page.close()
    
    log(f"HTML已保存到: {output_path}")


if __name__ == "__main__":
    # 使用示例
    URL = "https://www.kickstarter.com/projects/capseal/capseal-craft-your-own-flavor-cafe-capsule-at-home"
    OUTPUT_HTML = "data/projects/sample/page.html"
    overwrite_html = False
    fetch_html(URL, OUTPUT_HTML, overwrite_html)
