"""PII 隐私脱敏中间件。

自动识别并掩码邮箱、手机号、身份证号、银行卡号等敏感信息，
支持输入（用户问题入模型前）与输出（模型回答返回前）双向脱敏。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Pattern, Tuple

from utils.common import get_logger

logger = get_logger("middleware.pii")


@dataclass
class PIIRule:
    name: str
    pattern: Pattern
    mask: str


def _build_rules() -> List[PIIRule]:
    return [
        PIIRule("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
        # 中国大陆手机号
        PIIRule("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
        # 18 位身份证号（含末位 X）
        PIIRule(
            "id_card",
            re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
            "[ID_CARD]",
        ),
        # 银行卡号 16~19 位
        PIIRule("bank_card", re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[BANK_CARD]"),
    ]


class PIIMiddleware:
    """可开关的 PII 脱敏处理器。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._rules = _build_rules()

    def mask(self, text: str) -> Tuple[str, int]:
        """对文本脱敏，返回（脱敏后文本, 命中数量）。"""
        if not self.enabled or not text:
            return text, 0
        total = 0
        for rule in self._rules:
            text, n = rule.pattern.subn(rule.mask, text)
            total += n
        if total:
            logger.info("PII 脱敏命中 %d 处", total)
        return text, total

    def mask_input(self, text: str) -> str:
        """输入方向脱敏（用户问题）。"""
        masked, _ = self.mask(text)
        return masked

    def mask_output(self, text: str) -> str:
        """输出方向脱敏（模型回答）。"""
        masked, _ = self.mask(text)
        return masked
