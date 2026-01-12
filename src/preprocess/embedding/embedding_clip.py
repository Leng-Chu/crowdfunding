import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}
_CACHE_LOCK = threading.Lock()


def _get_clip_bundle(model_name: str, device: str):
    cache_key = (model_name, device)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError("CLIP 后端需要 transformers、torch 和 Pillow。") from exc

        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        _MODEL_CACHE[cache_key] = (model, processor)
        return _MODEL_CACHE[cache_key]


def _move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    for key, value in batch.items():
        if hasattr(value, "to"):
            batch[key] = value.to(device)
    return batch


def _encode_text(text: str, model, processor, device: str) -> np.ndarray:
    import torch

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = _move_to_device(inputs, device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].detach().cpu().numpy().astype(np.float32)


def _encode_image(image_path: Path, model, processor, device: str) -> np.ndarray:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = _move_to_device(inputs, device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].detach().cpu().numpy().astype(np.float32)


def vectorize_sequence(
    project_folder: Path,
    content_sequence: List[Dict[str, Any]],
) -> List[np.ndarray]:
    base_model_name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    text_model_name = os.getenv("CLIP_TEXT_MODEL", base_model_name)
    image_model_name = os.getenv("CLIP_IMAGE_MODEL", text_model_name)
    device = os.getenv("CLIP_DEVICE", "cpu")

    try:
        text_model, text_processor = _get_clip_bundle(text_model_name, device)
        image_model, image_processor = _get_clip_bundle(image_model_name, device)
    except Exception as exc:
        print(f"CLIP 初始化失败: {exc}")
        return []

    vector_list: List[np.ndarray] = []
    for item in content_sequence:
        item_type = item.get("type", "")
        if item_type == "text":
            text_content = item.get("content", "")
            if not text_content:
                print("文本内容为空，停止处理整个项目。")
                return []
            try:
                vector_list.append(
                    _encode_text(text_content, text_model, text_processor, device)
                )
            except Exception as exc:
                print(f"CLIP 文本编码失败: {exc}")
                return []
        elif item_type == "image":
            filename = item.get("filename", "")
            if not filename:
                print("图片文件名为空，停止处理整个项目。")
                return []
            image_path = project_folder / Path(filename)
            if not image_path.exists():
                print(f"图片文件不存在: {image_path}，停止处理整个项目。")
                return []
            try:
                vector_list.append(
                    _encode_image(image_path, image_model, image_processor, device)
                )
            except Exception as exc:
                print(f"CLIP 图片编码失败: {exc}")
                return []
        else:
            print(f"未知的项目类型 '{item_type}'，停止处理整个项目。")
            return []

    return vector_list
