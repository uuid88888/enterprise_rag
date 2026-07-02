"""混合检索 + Rerank。

链路：
1. 稠密向量检索（Chroma）
2. BM25 稀疏关键词检索（rank-bm25 + jieba 中文分词）
3. RRF（Reciprocal Rank Fusion）融合两路结果
4. CrossEncoder（BGE Reranker）二次打分重排
5. 相似度阈值过滤
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

import jieba
from rank_bm25 import BM25Okapi

from rag.vector_store import get_vector_store
from utils.common import get_logger
from utils.config import settings

logger = get_logger("rag.retriever")

jieba.setLogLevel(20)  # 关闭 jieba 启动日志

_reranker = None
_rerank_lock = threading.Lock()


def _get_reranker():
    """延迟加载 BGE Reranker（CrossEncoder）。

    优先本地缓存加载，避免联网 HEAD 检查卡顿；无缓存时再联网下载。
    """
    global _reranker
    if _reranker is None:
        with _rerank_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                logger.info("加载 Rerank 模型：%s", settings.rerank_model)
                try:
                    _reranker = CrossEncoder(settings.rerank_model, local_files_only=True)
                except Exception:
                    logger.info("本地无缓存，联网下载 Rerank 模型...")
                    _reranker = CrossEncoder(settings.rerank_model)
    return _reranker


def _tokenize(text: str) -> List[str]:
    """中文分词，过滤空白 token。"""
    return [t for t in jieba.lcut(text) if t.strip()]


@dataclass
class RetrievedDoc:
    text: str
    source: str
    score: float  # rerank 后的最终分数

    def as_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "score": round(self.score, 4)}


class HybridRetriever:
    """混合检索器。BM25 索引按需基于向量库全量文档构建并缓存。"""

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[dict] = []
        self._bm25_signature = -1  # 用向量库 revision 判断是否需要重建索引
        self._lock = threading.Lock()

    def _ensure_bm25(self) -> None:
        store = get_vector_store()
        signature = store.revision
        if self._bm25 is not None and signature == self._bm25_signature:
            return
        with self._lock:
            if self._bm25 is not None and signature == self._bm25_signature:
                return
            self._bm25_docs = store.all_documents()
            if not self._bm25_docs:
                self._bm25 = None
                self._bm25_signature = signature
                return
            corpus = [_tokenize(d["text"]) for d in self._bm25_docs]
            self._bm25 = BM25Okapi(corpus)
            self._bm25_signature = signature
            logger.info("BM25 索引已构建，文档数：%d", len(self._bm25_docs))

    def _dense_search(self, query: str, top_k: int) -> List[dict]:
        return get_vector_store().query(query, top_k=top_k)

    def _sparse_search(self, query: str, top_k: int) -> List[dict]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "text": self._bm25_docs[i]["text"],
                "metadata": self._bm25_docs[i]["metadata"],
                "score": float(scores[i]),
            }
            for i in ranked
            if scores[i] > 0
        ]

    @staticmethod
    def _rrf_fuse(dense: List[dict], sparse: List[dict], k: int = 60) -> List[dict]:
        """RRF 融合：score = sum(1 / (k + rank))，按文本去重。"""
        fused: dict[str, dict] = {}

        def accumulate(results: List[dict]) -> None:
            for rank, item in enumerate(results):
                key = item["text"]
                rrf = 1.0 / (k + rank + 1)
                if key not in fused:
                    fused[key] = {
                        "text": item["text"],
                        "metadata": item.get("metadata", {}),
                        "rrf": 0.0,
                    }
                fused[key]["rrf"] += rrf

        accumulate(dense)
        accumulate(sparse)
        return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedDoc]:
        """执行完整混合检索 + 重排。"""
        top_k = top_k or settings.top_k
        score_threshold = (
            score_threshold if score_threshold is not None else settings.score_threshold
        )

        # 两路各召回更多候选，给融合与重排留空间
        recall_k = max(top_k * 3, 10)
        dense = self._dense_search(query, recall_k)
        sparse = self._sparse_search(query, recall_k)

        if not dense and not sparse:
            logger.info("两路检索均无结果")
            return []

        fused = self._rrf_fuse(dense, sparse)
        candidates = fused[: max(top_k * 3, 10)]

        # CrossEncoder 重排
        reranker = _get_reranker()
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = reranker.predict(pairs)

        results = []
        for cand, rscore in zip(candidates, rerank_scores):
            results.append(
                RetrievedDoc(
                    text=cand["text"],
                    source=cand.get("metadata", {}).get("source", "未知"),
                    score=float(rscore),
                )
            )
        results.sort(key=lambda d: d.score, reverse=True)

        # 阈值过滤（BGE reranker 输出经 sigmoid 在 0~1 之间）
        filtered = [d for d in results if d.score >= score_threshold]
        final = filtered[:top_k] if filtered else results[:top_k]
        logger.info(
            "检索完成：dense=%d sparse=%d 融合=%d 重排后返回=%d",
            len(dense),
            len(sparse),
            len(fused),
            len(final),
        )
        return final


_retriever: Optional[HybridRetriever] = None
_retriever_lock = threading.Lock()


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = HybridRetriever()
    return _retriever
