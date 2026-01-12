import base64
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List

import dashscope
import numpy as np


def _encode_image_to_base64(image_path: str) -> str:
    image_format = "jpg"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{image_format};base64,{base64_image}"


def vectorize_sequence(
    project_folder: Path,
    content_sequence: List[Dict[str, Any]],
) -> List[np.ndarray]:
    vector_list: List[np.ndarray] = []

    for item in content_sequence:
        input_data: List[Dict[str, str]] = []
        item_type = item.get("type", "")

        if item_type == "text":
            text_content = item.get("content", "")
            if not text_content:
                print("文本内容为空，停止处理整个项目。")
                return []
            input_data.append({"text": text_content})
        elif item_type == "image":
            filename = item.get("filename", "")
            if not filename:
                print("图片文件名为空，停止处理整个项目。")
                return []
            image_path = project_folder / Path(filename)
            if not image_path.exists():
                print(f"图片文件不存在: {image_path}，停止处理整个项目。")
                return []
            image_data = _encode_image_to_base64(str(image_path))
            input_data.append({"image": image_data})
        else:
            print(f"未知的项目类型 '{item_type}'，停止处理整个项目。")
            return []

        try:
            response = dashscope.MultiModalEmbedding.call(
                model="qwen2.5-vl-embedding",
                input=input_data,
                parameters={"dimension": 1024},
            )
            if response.status_code == HTTPStatus.OK:
                embeddings = response.output["embeddings"]
                for emb in embeddings:
                    vector_list.append(np.array(emb["embedding"], dtype=np.float32))
            else:
                print(
                    "向量化失败，状态码: "
                    f"{response.status_code}，错误信息: {getattr(response, 'message', '')}"
                )
                return []
        except Exception as exc:
            print(f"向量化过程发生错误: {exc}")
            return []

    return vector_list
