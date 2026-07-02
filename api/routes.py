"""FastAPI 路由与可复用服务函数。

本模块同时提供：
1. 可被 Gradio 前端直接 import 调用的服务函数（ingest_paths / answer）
2. FastAPI 路由（文档入库、问答、知识库管理）

PII 脱敏在服务层统一接入：问答模式下对输入问题与输出答案双向脱敏。
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from agent.agent_graph import run_agent, simple_rag
from middleware.pii_middleware import PIIMiddleware
from rag.document_loader import load_files, supported_types
from rag.vector_store import get_vector_store
from utils.common import get_logger
from utils.config import settings

logger = get_logger("api.routes")
router = APIRouter()


# --------------------------------------------------------------------------- #
# 请求/响应模型
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    mode: str = Field("rag", description="rag=普通RAG，agent=智能Agent")
    history: List[dict] = Field(default_factory=list, description="多轮历史 [{role,content}]")
    top_k: int = Field(settings.top_k, ge=1, le=20)
    score_threshold: float = Field(settings.score_threshold, ge=0.0, le=1.0)
    enable_hyde: bool = settings.enable_hyde
    enable_pii: bool = settings.enable_pii


class SourceItem(BaseModel):
    text: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    mode: str


class IngestResponse(BaseModel):
    added: int
    skipped_duplicates: int
    total_chunks: int


class DocItem(BaseModel):
    source: str
    chunks: int


# --------------------------------------------------------------------------- #
# 可复用服务函数（UI 与 API 共用）
# --------------------------------------------------------------------------- #
def ingest_paths(
    paths: List[str], names: Optional[List[str]] = None, progress_callback=None
) -> IngestResponse:
    """将本地文件路径列表入库。

    names 可选，用于以原始文件名覆盖来源（Gradio 临时文件名无意义时使用）。
    progress_callback(done, total) 用于上报向量化进度。
    """
    if not paths:
        raise ValueError("未提供任何文件")

    # 仅解析一次：有 names 则逐文件加载以保留原始文件名，否则批量加载
    if names:
        chunks, skipped = _load_with_names(paths, names)
    else:
        result = load_files(paths)
        chunks, skipped = result.chunks, result.skipped_duplicates

    store = get_vector_store()
    added = store.add_chunks(chunks, progress_callback=progress_callback)
    return IngestResponse(
        added=added,
        skipped_duplicates=skipped,
        total_chunks=store.count(),
    )


def _load_with_names(paths: List[str], names: List[str]):
    """逐文件加载并用原始文件名覆盖来源，同时跨文件内容去重。

    返回 (chunks, skipped_duplicates)。
    """
    from rag.document_loader import load_file

    chunks = []
    seen: set[str] = set()
    skipped = 0
    for path, name in zip(paths, names):
        for chunk in load_file(path):
            chunk.source = name
            if chunk.content_hash in seen:
                skipped += 1
                continue
            seen.add(chunk.content_hash)
            chunks.append(chunk)
    return chunks, skipped


def answer(
    question: str,
    mode: str = "rag",
    history: Optional[List[dict]] = None,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    enable_hyde: bool = False,
    enable_pii: bool = True,
) -> AskResponse:
    """统一问答入口，含 PII 双向脱敏。"""
    pii = PIIMiddleware(enabled=enable_pii)
    safe_question = pii.mask_input(question)
    safe_history = _mask_history(history or [], pii)

    if mode == "agent":
        result = run_agent(
            safe_question,
            history=safe_history,
            top_k=top_k,
            score_threshold=score_threshold,
            enable_hyde=enable_hyde,
            enable_pii=enable_pii,
        )
    else:
        result = simple_rag(
            safe_question,
            history=safe_history,
            top_k=top_k,
            score_threshold=score_threshold,
            enable_hyde=enable_hyde,
            enable_pii=enable_pii,
        )

    safe_answer = pii.mask_output(result["answer"])
    sources = [SourceItem(**_mask_source(s, pii)) for s in result.get("sources", [])]
    return AskResponse(answer=safe_answer, sources=sources, mode=mode)


def _mask_history(history: List[dict], pii: PIIMiddleware) -> List[dict]:
    """对多轮历史脱敏，避免旧消息中的敏感信息进入模型或摘要链路。"""
    safe_history = []
    for item in history:
        safe_item = dict(item)
        content = safe_item.get("content")
        if isinstance(content, str):
            safe_item["content"] = pii.mask_input(content)
        safe_history.append(safe_item)
    return safe_history


def _mask_source(source: dict, pii: PIIMiddleware) -> dict:
    """对返回给前端的溯源片段脱敏。"""
    safe_source = dict(source)
    text = safe_source.get("text")
    if isinstance(text, str):
        safe_source["text"] = pii.mask_output(text)
    return safe_source


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@router.post("/ingest", response_model=IngestResponse, summary="批量上传文档入库")
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    tmp_paths: List[str] = []
    try:
        names = [f.filename or "未知" for f in files]
        for f in files:
            suffix = os.path.splitext(f.filename or "")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await f.read())
                tmp_paths.append(tmp.name)
        return ingest_paths(tmp_paths, names=names)
    except Exception as exc:
        logger.error("入库失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


@router.post("/ask", response_model=AskResponse, summary="智能问答")
async def ask(req: AskRequest) -> AskResponse:
    try:
        return answer(
            question=req.question,
            mode=req.mode,
            history=req.history,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            enable_hyde=req.enable_hyde,
            enable_pii=req.enable_pii,
        )
    except Exception as exc:
        logger.error("问答失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/documents/supported-types", summary="查看支持的文档格式")
@router.get("/doc/supported-types", summary="查看支持的文档格式（兼容路径）")
async def get_supported_types() -> dict:
    data = supported_types()
    data["chunking"] = {
        "strategy": "recursive_character",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "note": "当前按字符长度递归分块，不是严格 token 分块。",
    }
    data["vector_store"] = {
        "provider": "Chroma",
        "embedding_model": settings.embedding_model,
    }
    return data


@router.get("/documents", response_model=List[DocItem], summary="查看已入库文档列表")
async def list_documents() -> List[DocItem]:
    return [DocItem(**d) for d in get_vector_store().list_sources()]


@router.delete("/documents/source", summary="按来源文件删除文档")
async def delete_document_source(source: str = Query(..., min_length=1)) -> dict:
    store = get_vector_store()
    before = store.count()
    store.delete_source(source)
    after = store.count()
    return {
        "message": f"已删除来源文件：{source}",
        "deleted_chunks": max(before - after, 0),
        "total_chunks": after,
    }


@router.delete("/documents", summary="清空向量库")
@router.delete("/doc/clear", summary="清空向量库（兼容路径）")
async def clear_documents() -> dict:
    get_vector_store().clear()
    return {"message": "向量库已清空", "total_chunks": 0}


@router.get("/health", summary="健康检查")
async def health() -> dict:
    return {"status": "ok", "total_chunks": get_vector_store().count()}
