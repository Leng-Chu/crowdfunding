import embedding_clip, embedding_qwen3, embedding_bge, embedding_siglip, embedding_resnet

# 模型缓存
_model_cache = {}

def get_text_backend(text_model: str):
    """根据模型名称返回对应的文本向量化后端"""
    # 如果指定了模型路径，使用该路径作为缓存键
    cache_key = f"text_{text_model}"
    
    if cache_key not in _model_cache:
        if "qwen" in text_model:
            _model_cache[cache_key] = embedding_qwen3.EmbeddingModel()
        elif "clip" in text_model:
            _model_cache[cache_key] = embedding_clip.EmbeddingModel()
        elif "bge" in text_model:
            _model_cache[cache_key] = embedding_bge.EmbeddingModel()
        elif "siglip" in text_model:
            _model_cache[cache_key] = embedding_siglip.EmbeddingModel()
        else:
            print(f"未知的文本向量化后端: {text_model}")
            return None
    
    return _model_cache[cache_key]


def get_image_backend(image_model: str):
    """根据模型名称返回对应的图像向量化后端"""
    # 如果指定了模型路径，使用该路径作为缓存键
    cache_key = f"image_{image_model}"
    
    if cache_key not in _model_cache:
        if "qwen" in image_model:
            _model_cache[cache_key] = embedding_qwen3.EmbeddingModel()
        elif "clip" in image_model:
            _model_cache[cache_key] = embedding_clip.EmbeddingModel()
        elif "siglip" in image_model:
            _model_cache[cache_key] = embedding_siglip.EmbeddingModel()
        elif "resnet" in image_model:
            _model_cache[cache_key] = embedding_resnet.EmbeddingModel()
        else:
            print(f"未知的图像向量化后端: {image_model}")
            return None
    
    return _model_cache[cache_key]


def cleanup_models():
    """清理模型缓存，释放内存"""
    global _model_cache
    _model_cache.clear()