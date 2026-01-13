import embedding_clip, embedding_qwen3, embedding_bge, embedding_siglip, embedding_resnet


def get_text_backend(text_model: str):
    """根据模型名称返回对应的文本向量化后端"""
    if "qwen" in text_model:
        return embedding_qwen3.vectorize_sequence
    elif "clip" in text_model:
        return embedding_clip.vectorize_sequence
    elif "bge" in text_model:
        return embedding_bge.vectorize_sequence
    elif "siglip" in text_model:
        return embedding_siglip.vectorize_sequence
    else:
        print(f"未知的文本向量化后端: {text_model}")
        return None
    

def get_image_backend(image_model: str):
    """根据模型名称返回对应的图像向量化后端"""
    if "qwen" in image_model:
        return embedding_qwen3.vectorize_sequence
    elif "clip" in image_model:
        return embedding_clip.vectorize_sequence
    elif "siglip" in image_model:
        return embedding_siglip.vectorize_sequence
    elif "resnet" in image_model:
        return embedding_resnet.vectorize_sequence
    else:
        print(f"未知的图像向量化后端: {image_model}")
        return None