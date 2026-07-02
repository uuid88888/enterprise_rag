"""全局配置管理。

集中从环境变量 / .env 读取配置，避免散落在各处的硬编码。
注意：在导入任何 sentence-transformers / huggingface 相关库之前，
必须先设置 HF_ENDPOINT 镜像环境变量，否则镜像不生效。
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 先加载 .env，保证下方 os.environ 设置镜像时已读到用户配置
load_dotenv()

# 关键：HF 镜像必须在 transformers/sentence-transformers 被 import 前设置
os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
# 关闭 tokenizers 并行告警
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Settings(BaseSettings):
    """应用配置，字段名与 .env 中的大写键自动映射（大小写不敏感）。"""

    # 大模型
    openai_api_key: str = "sk-your-key-here"
    deepseek_api_key: str = ""
    dashscope_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1

    # HF 镜像
    hf_endpoint: str = "https://hf-mirror.com"

    # 模型选型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    # 向量库
    chroma_dir: str = "./data/chroma"
    collection_name: str = "enterprise_kb"

    # 检索默认参数
    top_k: int = 5
    score_threshold: float = 0.3
    enable_hyde: bool = False
    enable_pii: bool = True
    chunk_strategy: str = "char"
    chunk_size: int = 800
    chunk_overlap: int = 150
    summary_max_tokens: int = 2000
    summary_keep_recent: int = 4

    # OCR（默认关闭；开启后图片与扫描 PDF 会调用视觉 OCR Provider）
    enable_ocr: bool = False
    ocr_provider: str = "qwen-vl"
    ocr_model: str = "qwen-vl-ocr-latest"
    ocr_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ocr_api_key: str = ""
    ocr_max_pages: int = 20
    ocr_dpi: int = 180
    ocr_pdf_min_text_chars: int = 30

    # 服务
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_port: int = 7860
    auto_open_browser: bool = True

    @staticmethod
    def _is_placeholder(value: str | None) -> bool:
        """判断配置值是否为空或占位示例，避免把占位 key 当真实 key 使用。"""
        if not value:
            return True
        normalized = value.strip().lower()
        return normalized in {
            "",
            "sk-your-key-here",
            "your-key",
            "your-api-key",
            "你的apikey",
            "你的 api key",
            "你的视觉模型 api key",
        } or "xxxx" in normalized or normalized.startswith("your-")

    @classmethod
    def _first_real(cls, values: Iterable[str | None]) -> str:
        for value in values:
            if not cls._is_placeholder(value):
                return value.strip()
        return ""

    @property
    def effective_llm_api_key(self) -> str:
        """按 LLM base_url 自动选择可用 key。"""
        base_url = (self.openai_base_url or "").lower()
        provider_first = []
        if "deepseek" in base_url:
            provider_first.append(self.deepseek_api_key)
        if "dashscope" in base_url or "aliyuncs" in base_url:
            provider_first.append(self.dashscope_api_key)
        if provider_first:
            return self._first_real(
                [*provider_first, self.openai_api_key, self.deepseek_api_key, self.dashscope_api_key]
            )
        return self._first_real([self.openai_api_key, self.deepseek_api_key, self.dashscope_api_key])

    @property
    def effective_ocr_api_key(self) -> str:
        """OCR 优先使用专用 key，其次复用百炼/通用 OpenAI 兼容 key。"""
        return self._first_real(
            [self.ocr_api_key, self.dashscope_api_key, self.openai_api_key]
        )

    @property
    def normalized_chunk_strategy(self) -> str:
        strategy = (self.chunk_strategy or "char").strip().lower()
        if strategy in {"recursive", "character", "chars"}:
            return "char"
        if strategy in {"token", "tokens", "token_approx"}:
            return "token"
        return "char"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """单例配置，避免重复解析 .env。"""
    return Settings()


settings = get_settings()
