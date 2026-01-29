# -*- coding: utf-8 -*-
"""
环境变量覆盖 ResConfig（供自动化脚本/调参使用）。

设计目标：
- 不影响默认训练：只有设置环境变量才生效；
- 尽量通用：用 JSON 字典一次性传入覆盖项；
- 尽量安全：忽略未知字段，按原字段类型做基本类型转换。

用法示例（bash）：
  export RES_CFG_OVERRIDES='{"max_epochs": 1, "batch_size": 64, "random_seed": 42}'
  python src/dl/res/main.py --baseline-mode res
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Dict

from config import ResConfig

ENV_CFG_OVERRIDES = "RES_CFG_OVERRIDES"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(s)


def _cast_like(value: Any, ref: Any) -> Any:
    # bool 必须放在 int 前面（bool 是 int 的子类）
    if isinstance(ref, bool):
        return _parse_bool(value)
    if isinstance(ref, int):
        return int(value)
    if isinstance(ref, float):
        return float(value)
    if isinstance(ref, str):
        return str(value)
    if isinstance(ref, tuple):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return ref
    return value


def apply_config_overrides_from_env(cfg: ResConfig, logger=None) -> ResConfig:
    """
    从环境变量读取配置覆盖项并返回新 cfg。

    环境变量：
    - RES_CFG_OVERRIDES：JSON 字典，例如 {"learning_rate_init": 1e-4, "batch_size": 512}
    """
    raw = str(os.getenv(ENV_CFG_OVERRIDES, "") or "").strip()
    if not raw:
        return cfg

    try:
        overrides = json.loads(raw)
    except Exception as e:
        if logger is not None:
            logger.warning("解析环境变量 %s 失败：%s", ENV_CFG_OVERRIDES, e)
        return cfg

    if not isinstance(overrides, dict):
        if logger is not None:
            logger.warning("环境变量 %s 期望为 JSON 对象(dict)，但得到：%r", ENV_CFG_OVERRIDES, type(overrides).__name__)
        return cfg

    valid_keys = set(cfg.to_dict().keys())
    cleaned: Dict[str, Any] = {}
    for k, v in overrides.items():
        key = str(k)
        if key not in valid_keys:
            if logger is not None:
                logger.warning("忽略未知配置字段：%s", key)
            continue

        ref = getattr(cfg, key, None)
        try:
            cleaned[key] = _cast_like(v, ref)
        except Exception as e:
            if logger is not None:
                logger.warning("配置字段 %s 类型转换失败（value=%r）：%s", key, v, e)

    if not cleaned:
        return cfg

    try:
        return replace(cfg, **cleaned)
    except Exception as e:
        if logger is not None:
            logger.warning("应用配置覆盖失败：%s", e)
        return cfg
