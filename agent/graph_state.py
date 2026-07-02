"""LangGraph 状态定义。

定义 Agent 在整个 ReAct 决策循环中流转的共享状态。
"""
from __future__ import annotations

from typing import List, TypedDict

from rag.retriever import RetrievedDoc


class AgentState(TypedDict, total=False):
    """Agent 状态机的共享上下文。"""

    question: str  # 用户原始问题
    history: List[dict]  # 多轮对话历史 [{"role","content"}]

    # 运行时配置
    top_k: int
    score_threshold: float
    enable_hyde: bool
    enable_pii: bool

    # 决策与中间产物
    need_retrieve: bool  # 是否需要检索
    sub_questions: List[str]  # 拆解后的子问题
    iterations: int  # 已执行的检索迭代次数
    max_iterations: int  # 最大迭代次数
    enough_info: bool  # 信息是否充足

    # 检索结果累积
    retrieved: List[RetrievedDoc]  # 去重后的所有命中片段

    # 最终输出
    answer: str
