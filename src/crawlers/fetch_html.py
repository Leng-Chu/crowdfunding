from pathlib import Path
import time
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
    if browser_page is None:
        options = _build_options()
        page = ChromiumPage(options)
        should_quit = True
    else:
        page = browser_page
    
    time.sleep(wait_seconds)
    try:
        page.get(url)
        # 首先等待页面基本结构加载
        page.wait.doc_loaded()
        
        # 等待关键元素出现
        try:
            # 等待 story-content 区域出现
            page.wait.eles_loaded('.story-content', timeout=10)
            
            # 检查story-content是否有实际内容，如果没有则继续等待
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                story_content_ele = page.ele('.story-content', timeout=2)
                if story_content_ele and story_content_ele.html and len(story_content_ele.html.strip()) > 50:
                    # 如果story-content中有足够的内容，则认为已经加载完成
                    break
                else:
                    # 否则再等待1秒
                    time.sleep(1)
                    attempt += 1
                    log(f"等待story-content内容加载... 尝试 {attempt}/{max_attempts}")
        
        except Exception as e:
            log(f"等待story-content元素加载时出错: {e}")
        
        # 滚动到页面底部，触发懒加载内容
        page.scroll.to_bottom()
        time.sleep(1)
        
        # 滚动到顶部
        page.scroll.to_top()
        
        # 获取完整的HTML内容
        html = page.html
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        log(f"HTML已保存到: {output_path}")
        
    except BrowserConnectError as e:
        log(f"浏览器连接失败: {e}")
        # 重新抛出异常，让调用方处理
        raise e
    finally:
        # 只有在函数内部创建了浏览器实例时才退出
        if should_quit and page:
            page.quit()


if __name__ == "__main__":
    # 使用示例
    URL = "https://www.kickstarter.com/projects/capseal/capseal-craft-your-own-flavor-cafe-capsule-at-home"
    OUTPUT_HTML = "data/projects/sample/page.html"
    overwrite_html = True
    fetch_html(URL, OUTPUT_HTML, overwrite_html)