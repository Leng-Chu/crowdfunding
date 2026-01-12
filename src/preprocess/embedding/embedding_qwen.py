import base64
from http import HTTPStatus
from pathlib import Path
from typing import List

import dashscope
import numpy as np


def _encode_image_to_base64(image_path: str) -> str:
    import os
    # 从文件路径中提取文件扩展名作为格式
    _, ext = os.path.splitext(image_path)
    image_format = ext.lower()[1:]  # 去掉点号，例如 ".jpg" -> "jpg"
    
    # 验证是否为支持的图片格式
    supported_formats = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
    if image_format not in supported_formats:
        raise ValueError(f"Unsupported image format: {image_format}. Supported formats: {supported_formats}")
        
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{image_format};base64,{base64_image}"


def vectorize_sequence(
    content_sequence: List[str],
    model_name: str = "qwen2.5-vl-embedding",
    vector_type: str = "text"
) -> List[np.ndarray]:
    vector_list: List[np.ndarray] = []

    if vector_type == "text":
        input_data = [{"text": text_content} for text_content in content_sequence]
        
    elif vector_type == "image":
        input_data = []
        for img_path in content_sequence:
            image_data = _encode_image_to_base64(str(img_path))
            input_data.append({"image": image_data})
    
    else:
        print(f"未知的向量类型 '{vector_type}'，停止处理整个项目。")
        return []

    try:
        response = dashscope.MultiModalEmbedding.call(
            model=model_name,
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