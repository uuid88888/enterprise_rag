"""OpenAI-compatible vision OCR provider.

Works with Qwen-VL compatible services such as DashScope's compatible-mode API.
"""
from __future__ import annotations

import base64

from openai import OpenAI

from middleware.retry_middleware import retry_with_backoff
from rag.ocr.base import OCRProvider
from utils.common import get_logger
from utils.config import settings

logger = get_logger("rag.ocr.qwen_vl")


class QwenVLOCRProvider(OCRProvider):
    """OCR provider backed by an OpenAI-compatible vision model."""

    def __init__(self) -> None:
        api_key = settings.ocr_api_key or settings.openai_api_key
        base_url = settings.ocr_base_url or settings.openai_base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self._model = settings.ocr_model

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def extract_image_bytes(self, image: bytes, mime_type: str = "image/png") -> str:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请对图片进行 OCR，提取所有可见文字。"
                                "保留自然阅读顺序；表格可用 Markdown 表格或逐行文本表示。"
                                "只输出识别到的文字，不要解释。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        text = resp.choices[0].message.content or ""
        logger.info("OCR 调用完成，返回 %d 字符", len(text))
        return text.strip()

