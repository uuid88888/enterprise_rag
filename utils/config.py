"""全局配置管理。

集中从环境变量 / .env 读取配置，避免散落在各处的硬编码。
注意：在导入任何 sentence-transformers / huggingface 相关库之前，
必须先设置 HF_ENDPOINT 镜像环境变量，否则镜像不生效。
"""
from __future__ import annotations

import os
from functools import lru_cache

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
