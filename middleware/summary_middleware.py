"""对话摘要中间件。

当上下文消息累计 token 超过阈值时，自动将较早的历史消息压缩为一段摘要，
保留最近 N 轮关键对话，避免超出模型上下文窗口。
"""
from __future__ import annotations

from typing import Callable, List, Optional

from utils.common import get_logger

logger = get_logger("middleware.summary")


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1 字 1 token，英文约 4 字符 1 token。

    用于无需依赖具体 tokenizer 的快速判断。
    """
    if not text:
        return 0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    others = len(text) - chinese
    return chinese + others // 4 + 1


class SummaryMiddleware:
    """对话历史压缩器。

    消息格式约定为 LangChain 风格的字典：{"role": "user"|"assistant"|"system", "content": str}
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        keep_recent: int = 4,
        summarizer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        # summarizer 注入 LLM 调用；不传则使用简单截断式摘要，保证可独立运行
        self._summarizer = summarizer or self._naive_summarize

    @staticmethod
    def _naive_summarize(text: str) -> str:
        snippet = text[:300]
        return f"【历史对话摘要】{snippet}..." if len(text) > 300 else f"【历史对话摘要】{text}"

    def _total_tokens(self, messages: List[dict]) -> int:
        return sum(_estimate_tokens(m.get("content", "")) for m in messages)

    def process(self, messages: List[dict]) -> List[dict]:
        """若超阈值则压缩早期历史，返回新的消息列表。"""
        if self._total_tokens(messages) <= self.max_tokens:
            return messages
        if len(messages) <= self.keep_recent:
            return messages

        # 系统消息始终保留在最前
        system_msgs = [m for m in messages if m.get("role") == "system"]
        dialog_msgs = [m for m in messages if m.get("role") != "system"]

        to_compress = dialog_msgs[: -self.keep_recent]
        recent = dialog_msgs[-self.keep_recent :]
        if not to_compress:
            return messages

        history_text = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in to_compress
        )
        summary = self._summarizer(history_text)
        logger.info(
            "对话历史压缩：%d 条 -> 1 条摘要 + 保留最近 %d 条",
            len(to_compress),
            len(recent),
        )
        summary_msg = {"role": "system", "content": summary}
        return system_msgs + [summary_msg] + recent
