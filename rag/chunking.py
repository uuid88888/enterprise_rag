"""文本分块策略。

默认保留 LangChain 的递归字符分块；可切换到轻量 token 估算分块。
token 估算不绑定具体模型 tokenizer，适合作为 Python 轻量版本的过渡方案。
"""
from __future__ import annotations

import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "char",
    separators: List[str] | None = None,
) -> List[str]:
    """按指定策略切分文本。"""
    if strategy == "token":
        return _split_by_approx_tokens(text, chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )
    return splitter.split_text(text)


def _split_by_approx_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """轻量 token 估算切分。

    中文按单字计，英文/数字连续串按约 4 字符一组计，标点单独计。
    该策略重点是让分块长度更接近 token 预算；需要模型级精确计数时，
    后续可替换为 tiktoken / jtokkit 对应实现。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    units = _token_units(text)
    if not units:
        return []

    chunks: List[str] = []
    start = 0
    total = len(units)
    while start < total:
        end = min(start + chunk_size, total)
        piece = "".join(units[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _token_units(text: str) -> List[str]:
    units: List[str] = []
    for part in re.findall(r"\s+|[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text):
        if re.fullmatch(r"[A-Za-z0-9_]+", part):
            units.extend(part[i : i + 4] for i in range(0, len(part), 4))
        else:
            units.append(part)
    return units
