import os
import json
from pathlib import Path

from PIL import Image

# 可选：让 Pillow 支持 AVIF（装了 pillow-avif-plugin 才会生效）
try:
    import pillow_avif  # noqa: F401
    _AVIF_OK = True
except Exception:
    _AVIF_OK = False


def get_image_size(image_path: str):
    """
    获取图片宽高；遇到不支持的格式（如未启用 AVIF）返回 (None, None)
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None


def update_content_json(content_json_path: str, base_project_path: str):
    """
    更新单个 content.json 文件
    """
    with open(content_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data.get("content_sequence", []):
        t = item.get("type")

        if t == "text":
            content = item.get("content", "")
            if "content_length" not in item:
                item["content_length"] = len(content)

        elif t == "image":
            filename = item.get("filename")
            if not filename:
                continue

            # 已有尺寸就不重复算
            if "width" in item and "height" in item:
                continue

            image_path = os.path.join(base_project_path, filename)
            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                continue

            w, h = get_image_size(image_path)
            if w is not None and h is not None:
                item["width"] = w
                item["height"] = h

    with open(content_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    base_path = Path("data/projects/move2")
    if not base_path.exists():
        print(f"Directory {base_path} does not exist!")
        return

    for project_dir in base_path.iterdir():
        if not project_dir.is_dir():
            continue

        content_json_path = project_dir / "content.json"
        if content_json_path.exists():
            # print(f"Processing {content_json_path}")
            update_content_json(str(content_json_path), str(project_dir))
        else:
            print(f"content.json not found in {project_dir}")


if __name__ == "__main__":
    main()
