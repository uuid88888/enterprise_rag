"""OCR provider registry."""
from __future__ import annotations

from rag.ocr.base import OCRProvider
from rag.ocr.qwen_vl import QwenVLOCRProvider
from utils.config import settings


def get_ocr_provider() -> OCRProvider:
    """Return configured OCR provider."""
    provider = settings.ocr_provider.lower().strip()
    if provider in {"qwen-vl", "qwen", "openai-compatible"}:
        return QwenVLOCRProvider()
    raise ValueError(f"不支持的 OCR_PROVIDER：{settings.ocr_provider}")

