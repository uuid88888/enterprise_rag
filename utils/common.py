"""通用工具：日志、文本清洗、哈希去重等。"""
from __future__ import annotations

import hashlib
import logging
import re
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """返回带统一格式的 logger，重复调用不会重复挂 handler。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def clean_text(text: str) -> str:
    """文本清洗：合并多余空白、去掉不可见控制字符。"""
    if not text:
        return ""
    # 去除控制字符（保留换行）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 多个连续空白/换行压缩
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_hash(text: str) -> str:
    """对文本内容做归一化哈希，用于重复片段去重。"""
    normalized = re.sub(r"\s+", "", text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()
