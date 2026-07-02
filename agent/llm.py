"""LLM 客户端工厂。

统一构造 OpenAI 兼容的 ChatModel；通过 .env 的 base_url 即可切换到
本地 Qwen / Llama 的 OpenAI 兼容服务（vLLM、Ollama、LM Studio 等）。
所有对外调用包裹工具重试中间件，保证接口抖动时自动退避重试。
"""
from __future__ import annotations

import threading
from typing import List, Optional

from langchain_openai import ChatOpenAI

from middleware.retry_middleware import retry_with_backoff
from utils.common import get_logger
from utils.config import settings

logger = get_logger("agent.llm")

_llm: Optional[ChatOpenAI] = None
_lock = threading.Lock()


def get_llm() -> ChatOpenAI:
    """返回进程级单例 ChatModel。"""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                _llm = ChatOpenAI(
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    timeout=60,
                )
                logger.info("LLM 初始化：%s @ %s", settings.llm_model, settings.openai_base_url)
    return _llm


@retry_with_backoff(max_retries=3, base_delay=1.0)
def chat(messages: List[dict]) -> str:
    """同步对话调用。messages 为 {"role","content"} 字典列表，返回纯文本。"""
    llm = get_llm()
    # LangChain 接受 (role, content) 元组列表
    lc_messages = [(m["role"], m["content"]) for m in messages]
    resp = llm.invoke(lc_messages)
    return resp.content if hasattr(resp, "content") else str(resp)
