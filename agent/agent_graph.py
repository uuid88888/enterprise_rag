"""LangGraph 智能 Agent 主逻辑。

包含：
- HyDE 查询改写
- 普通 RAG 链路（simple_rag）
- Agent ReAct 决策图（plan -> retrieve -> reflect -> generate）
  支持：判断是否检索、复杂问题拆解、多轮迭代补充检索、信息充足后生成
- 答案防幻觉：强制基于检索片段作答，附来源溯源
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from langgraph.graph import END, StateGraph

from agent.graph_state import AgentState
from agent.llm import chat
from middleware.pii_middleware import PIIMiddleware
from middleware.summary_middleware import SummaryMiddleware
from rag.retriever import RetrievedDoc, get_retriever
from utils.common import get_logger
from utils.config import settings

logger = get_logger("agent.graph")


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
_ANSWER_SYSTEM = (
    "你是企业知识库智能助手。请严格依据【参考资料】回答用户问题。\n"
    "要求：\n"
    "1. 只能使用参考资料中的信息，禁止编造或使用资料外的知识；\n"
    "2. 若参考资料不足以回答，请明确说明「根据现有资料无法回答该问题」；\n"
    "3. 回答末尾不要重复罗列来源，来源由系统单独展示。"
)

_DECIDE_PROMPT = (
    "判断回答下面这个问题是否需要检索企业知识库。\n"
    "若是闲聊、问候、与知识库无关的常识，则不需要检索。\n"
    '只输出 JSON：{{"need_retrieve": true/false}}\n'
    "问题：{question}"
)

_DECOMPOSE_PROMPT = (
    "将下面复杂问题拆解为 1-3 个相互独立、便于检索的子问题。\n"
    "若问题本身简单，则只返回原问题。\n"
    '只输出 JSON：{{"sub_questions": ["...", "..."]}}\n'
    "问题：{question}"
)

_REFLECT_PROMPT = (
    "已检索到以下资料片段（可能不完整）：\n{context}\n\n"
    "针对问题：{question}\n"
    "这些资料是否足以给出准确、完整的回答？\n"
    '只输出 JSON：{{"enough": true/false, "follow_up": "若不足则给出一个补充检索的查询，否则留空"}}'
)

_HYDE_PROMPT = (
    "针对下面的问题，写一段 100 字以内、像是从专业文档中摘录的、可能包含答案的假想段落。\n"
    "只输出该段落本身，不要解释。\n问题：{question}"
)

_SUMMARY_PROMPT = (
    "请将以下较早的多轮对话压缩为一段简洁摘要，保留用户目标、关键事实、约束条件和已确认结论。\n"
    "只输出摘要本身。\n\n{history}"
)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> Optional[dict]:
    """从 LLM 输出中尽量解析出 JSON，失败返回 None。"""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def hyde_rewrite(question: str) -> str:
    """HyDE：生成假想文档用于增强检索。失败时回退为原问题。"""
    try:
        doc = chat([{"role": "user", "content": _HYDE_PROMPT.format(question=question)}])
        hypo = doc.strip()
        logger.info("HyDE 改写成功")
        # 拼接原问题，兼顾关键词与语义
        return f"{question}\n{hypo}"
    except Exception as exc:
        logger.warning("HyDE 改写失败，回退原问题：%s", exc)
        return question


def _format_context(docs: List[RetrievedDoc], mask_pii: bool = False) -> str:
    if not docs:
        return "（无相关资料）"
    pii = PIIMiddleware(enabled=mask_pii)
    return "\n\n".join(
        f"[片段{i + 1}｜来源:{d.source}]\n{pii.mask_input(d.text)}"
        for i, d in enumerate(docs)
    )


def _dedup_docs(docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
    seen, out = set(), []
    for d in docs:
        if d.text in seen:
            continue
        seen.add(d.text)
        out.append(d)
    return out


def _summarize_history(history_text: str) -> str:
    """用 LLM 压缩较早历史；失败时 SummaryMiddleware 会由重试层抛出。"""
    return chat([{"role": "user", "content": _SUMMARY_PROMPT.format(history=history_text)}])


def _compress_messages(messages: List[dict]) -> List[dict]:
    """对长对话做摘要压缩，替代简单丢弃历史。"""
    middleware = SummaryMiddleware(
        max_tokens=settings.summary_max_tokens,
        keep_recent=settings.summary_keep_recent,
        summarizer=_summarize_history,
    )
    try:
        return middleware.process(messages)
    except Exception as exc:
        logger.warning("对话摘要失败，回退为保留最近历史：%s", exc)
        fallback = SummaryMiddleware(
            max_tokens=settings.summary_max_tokens,
            keep_recent=settings.summary_keep_recent,
        )
        return fallback.process(messages)


def _retrieve(query: str, state: AgentState) -> List[RetrievedDoc]:
    search_query = hyde_rewrite(query) if state.get("enable_hyde") else query
    return get_retriever().retrieve(
        search_query,
        top_k=state.get("top_k", settings.top_k),
        score_threshold=state.get("score_threshold", settings.score_threshold),
    )


# --------------------------------------------------------------------------- #
# 普通 RAG 链路
# --------------------------------------------------------------------------- #
def simple_rag(
    question: str,
    history: Optional[List[dict]] = None,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    enable_hyde: bool = False,
    enable_pii: bool = False,
) -> dict:
    """单轮检索 + 生成，返回 {answer, sources}。"""
    history = history or []
    query = hyde_rewrite(question) if enable_hyde else question
    docs = get_retriever().retrieve(query, top_k=top_k, score_threshold=score_threshold)

    messages = [{"role": "system", "content": _ANSWER_SYSTEM}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"参考资料：\n{_format_context(docs, mask_pii=enable_pii)}\n\n问题：{question}",
        }
    )
    messages = _compress_messages(messages)
    answer = chat(messages)
    return {"answer": answer, "sources": [d.as_dict() for d in docs]}


# --------------------------------------------------------------------------- #
# Agent 节点
# --------------------------------------------------------------------------- #
def node_decide(state: AgentState) -> AgentState:
    """判断是否需要检索。"""
    out = chat([{"role": "user", "content": _DECIDE_PROMPT.format(question=state["question"])}])
    parsed = _parse_json(out)
    need = parsed.get("need_retrieve", True) if parsed else True
    logger.info("决策：需要检索=%s", need)
    return {"need_retrieve": need, "iterations": 0, "retrieved": []}


def node_decompose(state: AgentState) -> AgentState:
    """复杂问题拆解为子问题。"""
    out = chat([{"role": "user", "content": _DECOMPOSE_PROMPT.format(question=state["question"])}])
    parsed = _parse_json(out)
    subs = parsed.get("sub_questions") if parsed else None
    if not subs or not isinstance(subs, list):
        subs = [state["question"]]
    subs = [s for s in subs if isinstance(s, str) and s.strip()][:3]
    logger.info("问题拆解为 %d 个子问题", len(subs))
    return {"sub_questions": subs}


def node_retrieve(state: AgentState) -> AgentState:
    """对每个子问题检索并累积去重。"""
    docs = list(state.get("retrieved", []))
    queries = state.get("sub_questions") or [state["question"]]
    for q in queries:
        docs.extend(_retrieve(q, state))
    docs = _dedup_docs(docs)
    iterations = state.get("iterations", 0) + 1
    logger.info("第 %d 轮检索后累计片段 %d", iterations, len(docs))
    return {"retrieved": docs, "iterations": iterations}


def node_reflect(state: AgentState) -> AgentState:
    """反思信息是否充足；不足则生成补充检索查询作为新的子问题。"""
    docs = state.get("retrieved", [])
    if not docs:
        return {"enough_info": True}  # 无资料则直接进入生成（会回答无法作答）

    out = chat(
        [
            {
                "role": "user",
                "content": _REFLECT_PROMPT.format(
                    context=_format_context(
                        docs, mask_pii=state.get("enable_pii", False)
                    )[:3000],
                    question=state["question"],
                ),
            }
        ]
    )
    parsed = _parse_json(out) or {}
    enough = parsed.get("enough", True)
    follow_up = (parsed.get("follow_up") or "").strip()

    if not enough and follow_up:
        logger.info("信息不足，补充检索：%s", follow_up)
        return {"enough_info": False, "sub_questions": [follow_up]}
    return {"enough_info": True}


def node_generate(state: AgentState) -> AgentState:
    """基于累积资料生成最终答案。"""
    docs = state.get("retrieved", [])
    history = state.get("history", [])
    messages = [{"role": "system", "content": _ANSWER_SYSTEM}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                "参考资料：\n"
                f"{_format_context(docs, mask_pii=state.get('enable_pii', False))}"
                f"\n\n问题：{state['question']}"
            ),
        }
    )
    messages = _compress_messages(messages)
    answer = chat(messages)
    return {"answer": answer}


def node_direct_answer(state: AgentState) -> AgentState:
    """无需检索时直接回答（闲聊等）。"""
    history = state.get("history", [])
    messages = [{"role": "system", "content": "你是友好的企业知识库助手。"}]
    messages.extend(history)
    messages.append({"role": "user", "content": state["question"]})
    messages = _compress_messages(messages)
    return {"answer": chat(messages), "retrieved": []}


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
def _route_after_decide(state: AgentState) -> str:
    return "decompose" if state.get("need_retrieve") else "direct"


def _route_after_reflect(state: AgentState) -> str:
    """信息不足且未达最大迭代则再次检索，否则生成。"""
    if state.get("enough_info"):
        return "generate"
    if state.get("iterations", 0) >= state.get("max_iterations", 2):
        return "generate"
    return "retrieve"


def build_agent_graph():
    """构建并编译 LangGraph 状态图。"""
    graph = StateGraph(AgentState)
    graph.add_node("decide", node_decide)
    graph.add_node("decompose", node_decompose)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("reflect", node_reflect)
    graph.add_node("generate", node_generate)
    graph.add_node("direct", node_direct_answer)

    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide", _route_after_decide, {"decompose": "decompose", "direct": "direct"}
    )
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "reflect")
    graph.add_conditional_edges(
        "reflect", _route_after_reflect, {"retrieve": "retrieve", "generate": "generate"}
    )
    graph.add_edge("generate", END)
    graph.add_edge("direct", END)
    return graph.compile()


# 编译一次复用
_compiled_graph = None


def get_agent_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


def run_agent(
    question: str,
    history: Optional[List[dict]] = None,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    enable_hyde: bool = False,
    enable_pii: bool = False,
    max_iterations: int = 2,
) -> dict:
    """运行 Agent，返回 {answer, sources}。"""
    init: AgentState = {
        "question": question,
        "history": history or [],
        "top_k": top_k or settings.top_k,
        "score_threshold": score_threshold
        if score_threshold is not None
        else settings.score_threshold,
        "enable_hyde": enable_hyde,
        "enable_pii": enable_pii,
        "max_iterations": max_iterations,
    }
    final = get_agent_graph().invoke(init)
    docs = final.get("retrieved", [])
    return {
        "answer": final.get("answer", ""),
        "sources": [d.as_dict() for d in docs],
    }
