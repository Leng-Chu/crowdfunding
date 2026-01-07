import re
from bs4 import BeautifulSoup
import os
import json
from pathlib import Path


def _ensure_output_dirs(output_dir):
    """确保输出目录存在，创建cover和photo子目录"""
    output_dir = Path(output_dir)
    cover_dir = output_dir / "cover"
    photo_dir = output_dir / "photo"
    cover_dir.mkdir(parents=True, exist_ok=True)
    photo_dir.mkdir(parents=True, exist_ok=True)
    return cover_dir, photo_dir


def _extract_basic_info(soup, logger=None):
    """提取封面图片和视频信息"""
    log = logger or print
    cover_image_url = None
    cover_img = soup.select_one(".project-profile__feature_image img")
    if not cover_img:
        cover_img = soup.select_one("img.aspect-ratio--object")
    if not cover_img:
        # 如果没有找到直接的封面图片，尝试从视频元素中获取封面图片
        video_element = soup.find("div", id="video_pitch")
        if video_element:
            cover_image_url = video_element.get("data-image")
    else:
        cover_image_url = cover_img.get("src") or cover_img.get("data-src")
    if not cover_image_url:
        log("未找到封面图片")

    video_url = None
    video_element = soup.find("div", id="video_pitch")
    if video_element:
        video_data = video_element.get("data-video")
        if video_data:
            try:
                video_data = json.loads(video_data)
                video_url = video_data.get("base")
            except json.JSONDecodeError:
                pass

    return cover_image_url, video_url


def _extract_story_content(soup, selectors, include_div=False):
    """提取故事内容中的图像和文本序列"""
    content_sequence = []
    image_counter = 1
    text_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]
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
                        relative_name = f"photo/story_image_{image_counter}.jpg"
                        content_sequence.append(
                            {
                                "type": "image",
                                "url": img_url,
                                "filename": relative_name,
                            }
                        )
                        image_counter += 1
                        last_item_was_text = False  # 重置文本标记
                elif element.name in text_tags:
                    text = element.get_text(strip=True)
                    if text:
                        # 如果上一项也是文本，则合并到上一项
                        if last_item_was_text and content_sequence and content_sequence[-1]["type"] == "text":
                            content_sequence[-1]["content"] += "\n" + text
                        else:
                            content_sequence.append({"type": "text", "content": text})
                            last_item_was_text = True
                # 对于其他元素
                #else:
                    #if element.name:
                        #print(f"未知元素: {element.name}")
        break

    return content_sequence


def parse_story_content(
    html_file_path,
    output_dir,
    project_url=None,
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
    log(f"正在解析HTML: {html_file_path}")
    
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    cover_image_url, video_url = _extract_basic_info(soup, logger=log)
    selectors = [
        "div.story-content",
        'div[data-element="rich_text_content"]',
    ]
    content_sequence = _extract_story_content(soup, selectors, include_div=False)

    result = {
        "project_url": project_url,  # 添加项目URL
        "cover_image": cover_image_url,
        "video": video_url,
        "content_sequence": content_sequence,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log(f"解析完成，结果已保存到: {result_file}，共找到 {len(content_sequence)} 个内容元素")
    return result


if __name__ == "__main__":
    # 使用示例（配置区域）
    html_file_path = "data/projects/sample/page.html"  # HTML文件路径
    output_dir = "data/projects/sample"  # 输出目录
    project_url = "https://www.kickstarter.com/projects/sample"  # 项目URL
    overwrite_content = False  # 是否覆盖已存在的content.json

    if os.path.exists(html_file_path):
        result = parse_story_content(
            html_file_path,
            output_dir,
            project_url=project_url,
            overwrite_content=overwrite_content,
        )
        print("解析完成")
    else:
        print(f"找不到HTML文件: {html_file_path}")
