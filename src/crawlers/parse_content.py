import re
from bs4 import BeautifulSoup
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_output_dirs(output_dir):
    """确保输出目录存在，创建cover和photo子目录"""
    output_dir = Path(output_dir)
    cover_dir = output_dir / "cover"
    photo_dir = output_dir / "photo"
    cover_dir.mkdir(parents=True, exist_ok=True)
    photo_dir.mkdir(parents=True, exist_ok=True)
    return cover_dir, photo_dir


def _extract_video_url(soup, logger=None):
    """从HTML中提取视频URL，只在story-content之前查找以base.mp4结尾的URL"""
    log = logger or print
    
    # 查找 story-content 元素
    story_content_element = soup.find("div", class_="story-content")
    
    if story_content_element:
        # 获取整个HTML字符串
        html_content = str(soup)
        
        # 找到story-content元素在HTML中的位置
        story_content_str = str(story_content_element)
        story_pos = html_content.find(story_content_str)
        
        if story_pos != -1:
            # 获取story-content之前的部分
            html_before_story = html_content[:story_pos]
            # 在 story-content 之前的内容中查找视频URL
            base_mp4_urls = re.findall(r'https?://[^\s"\'<>]*base\.mp4', html_before_story)
        else:
            # 如果没找到确切位置，使用之前的逻辑
            base_mp4_urls = re.findall(r'https?://[^\s"\'<>]*base\.mp4', str(soup))
    else:
        # 如果没有找到 story-content，搜索整个HTML
        html_content = str(soup)
        base_mp4_urls = re.findall(r'https?://[^\s"\'<>]*base\.mp4', html_content)
    
    if base_mp4_urls:
        return base_mp4_urls[0]
    else:
        return None


def _is_leaf_text_div(element, block_tags):
    """Return True when a div is acting like a text paragraph."""
    if element.find(block_tags + ["div"], recursive=False):
        return False
    return True


def _as_text_field(value: Any) -> Optional[Dict[str, Any]]:
    """
    将文本转成 content.json 约定的 dict 格式：
    - {"content": "...", "content_length": 123}
    空/全空白则返回 None。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return {"content": text, "content_length": int(len(text))}


def _extract_story_content(soup, selectors, include_div=False):
    """提取故事内容中的图像和文本序列"""
    content_sequence = []
    image_counter = 1
    block_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]
    text_tags = list(block_tags)
    if include_div:
        text_tags.append("div")

    for selector in selectors:
        story_elements = soup.select(selector)
        if not story_elements:
            continue
        for story_content in story_elements:
            last_item_was_text = False
            for element in story_content.descendants:
                if element.name == "img":
                    img_url = (
                        element.get("src")
                        or element.get("data-src")
                        or element.get("data-gif-src")
                        or element.get("data-original")
                    )
                    if img_url:
                        relative_name = f"photo/story_image_{image_counter}.jpeg"
                        content_sequence.append(
                            {
                                "type": "image",
                                "url": img_url,
                                "filename": relative_name,
                                "width": 0,
                                "height": 0,
                            }
                        )
                        image_counter += 1
                        last_item_was_text = False  # 重置文本标记
                elif element.name in text_tags:
                    if element.name == "div" and not _is_leaf_text_div(element, block_tags):
                        continue
                    text = element.get_text(strip=True)
                    if text:
                        # 如果上一项也是文本，则合并到上一项
                        if last_item_was_text and content_sequence and content_sequence[-1]["type"] == "text":
                            content_sequence[-1]["content"] += "\n" + text
                            content_sequence[-1]["content_length"] = int(len(content_sequence[-1]["content"]))
                        else:
                            content_sequence.append(
                                {
                                    "type": "text",
                                    "content": text,
                                    "content_length": int(len(text)),
                                }
                            )
                            last_item_was_text = True
        break

    return content_sequence


def parse_story_content(
    html_file_path,
    output_dir,
    project_url=None,
    cover_url=None,
    title=None,
    blurb=None,
    overwrite_content=False,
    logger=None,
):
    """
    解析HTML文件中的背景故事（story-content）的图像和文本序列，
    以及封面图像和视频信息
    """
    _ensure_output_dirs(output_dir)
    result_file = os.path.join(output_dir, "content.json")

    if os.path.exists(result_file) and not overwrite_content:
        with open(result_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        #print(f"content.json已存在，跳过重新生成: {result_file}")
        return result

    log = logger or print
    
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    
    title_field = _as_text_field(title)
    blurb_field = _as_text_field(blurb)

    # 封面图片URL默认从参数传入；若为空则置为 None（由上游决定是否判失败）
    cover_image_url = str(cover_url).strip() if cover_url is not None else ""
    cover_image = None
    if cover_image_url:
        cover_image = {
            "url": cover_image_url,
            "filename": "",
            "width": 0,
            "height": 0,
        }
    
    # 从HTML中提取视频URL
    video_url = _extract_video_url(soup, logger=log)
    selectors = [
        "div.story-content",
        'div[data-element="rich_text_content"]',
        "div.rte__content"
    ]
    content_sequence = _extract_story_content(soup, selectors, include_div=False)
    if not content_sequence:
        content_sequence = _extract_story_content(soup, selectors, include_div=True)

    result = {
        "project_url": project_url,  # 添加项目URL
        "title": title_field,
        "blurb": blurb_field,
        "cover_image": cover_image,
        "video": video_url,
        "content_sequence": content_sequence,
    }

    # 移除空字段，避免 downstream 误判
    result = {k: v for k, v in result.items() if v is not None}

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log(f"解析完成，结果已保存到: {result_file} ，共找到 {len(content_sequence)} 个内容元素")
    return result


if __name__ == "__main__":
    # 使用示例（配置区域）
    html_file_path = "data/projects/sample/page.html"  # HTML文件路径
    output_dir = "data/projects/sample"  # 输出目录
    project_url = "https://www.kickstarter.com/projects/capseal/capseal-craft-your-own-flavor-cafe-capsule-at-home"  # 项目URL
    overwrite_content = True  # 是否覆盖已存在的content.json
    cover_url = "https://i.kickstarter.com/assets/041/738/680/ab09ef0d9617617b6baa03e36632a46f_original.jpeg?anim=false&fit=cover&gravity=auto&height=873&origin=ugc&q=92&v=1690365942&width=1552&sig=Finaoze44%2B%2Bk%2FEMMT28MubZnmHqXu%2F9qeHj%2B2lWXPoc%3D"
    if os.path.exists(html_file_path):
        result = parse_story_content(
            html_file_path,
            output_dir,
            project_url=project_url,
            cover_url=cover_url,
            overwrite_content=overwrite_content,
        )
        print("解析完成")
    else:
        print(f"找不到HTML文件: {html_file_path}")
