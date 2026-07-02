"""向量库管理（Chroma + BGE Embedding）。

封装：入库、删除、清空、查询、文档列表。
Embedding 使用 sentence-transformers 加载 BAAI/bge-small-zh-v1.5。
延迟加载模型，避免 import 阶段触发大文件下载 / DLL 初始化问题。
"""
from __future__ import annotations

import threading
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.document_loader import Chunk
from utils.common import get_logger
from utils.config import settings

logger = get_logger("rag.vector_store")

# 模型与客户端均为进程级单例，加锁保证线程安全（Gradio 多线程）
_embed_lock = threading.Lock()
_embedder = None


def _get_embedder():
    """延迟加载 BGE Embedding 模型。

    优先从本地缓存加载（local_files_only=True），避免每次加载都联网向
    huggingface.co 发 HEAD 检查导致卡顿；本地无缓存时再联网下载。
    """
    global _embedder
    if _embedder is None:
        with _embed_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer

                logger.info("加载 Embedding 模型：%s", settings.embedding_model)
                try:
                    _embedder = SentenceTransformer(
                        settings.embedding_model, local_files_only=True
                    )
                except Exception:
                    logger.info("本地无缓存，联网下载 Embedding 模型...")
                    try:
                        _embedder = SentenceTransformer(settings.embedding_model)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Embedding 模型加载失败：{settings.embedding_model}。"
                            "请确认网络可访问 HuggingFace/HF 镜像，或提前下载模型到本地缓存。"
                        ) from exc
    return _embedder


def warmup() -> None:
    """预热：提前加载 Embedding 模型，避免首次上传/查询时才触发加载。"""
    _get_embedder()


def embed_texts(
    texts: List[str],
    is_query: bool = False,
    batch_size: int = 64,
    progress_callback=None,
) -> List[List[float]]:
    """文本向量化。

    bge 系列建议对查询添加指令前缀以提升检索效果。
    分批编码，可通过 progress_callback(done, total) 上报进度。
    """
    model = _get_embedder()
    if is_query:
        texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]

    total = len(texts)
    all_embeddings: List[List[float]] = []
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        emb = model.encode(
            batch,
            normalize_embeddings=True,  # 归一化后内积即余弦相似度
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(emb.tolist())
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)
    return all_embeddings


class VectorStore:
    """Chroma 向量库封装。"""

    def __init__(self) -> None:
        self._revision = 0
        self._client = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("向量库就绪，当前片段数：%d", self.count())

    def count(self) -> int:
        return self._collection.count()

    @property
    def revision(self) -> int:
        """知识库变更版本号，用于下游缓存判断是否需要刷新。"""
        return self._revision

    def add_chunks(self, chunks: List[Chunk], progress_callback=None) -> int:
        """增量入库。按 content_hash 作为 id，天然实现重复片段幂等去重。

        progress_callback(done, total) 用于上报向量化进度。
        """
        if not chunks:
            return 0

        ids = [c.content_hash for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata() for c in chunks]
        embeddings = embed_texts(
            documents, is_query=False, progress_callback=progress_callback
        )

        # upsert：同 id 覆盖，实现增量更新且不报重复错误
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._revision += 1
        logger.info("入库完成，新增/更新 %d 个片段", len(ids))
        return len(ids)

    def query(self, text: str, top_k: int) -> List[dict]:
        """稠密向量检索，返回带相似度分数的片段列表。"""
        if self.count() == 0:
            return []
        query_emb = embed_texts([text], is_query=True)[0]
        res = self._collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # cosine distance -> similarity
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1.0 - float(dist),
                }
            )
        return hits

    def all_documents(self) -> List[dict]:
        """返回所有片段（供 BM25 构建全量索引）。"""
        if self.count() == 0:
            return []
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(res.get("documents", []), res.get("metadatas", []))
        ]

    def list_sources(self) -> List[dict]:
        """按来源文件聚合，返回已入库文档列表及片段数。"""
        res = self._collection.get(include=["metadatas"])
        counter: dict[str, int] = {}
        for meta in res.get("metadatas", []):
            src = meta.get("source", "未知")
            counter[src] = counter.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(counter.items())]

    def delete_source(self, source: str) -> None:
        """删除指定来源文件的全部片段。"""
        self._collection.delete(where={"source": source})
        self._revision += 1
        logger.info("已删除来源文件：%s", source)

    def clear(self) -> None:
        """清空整个向量库 collection。"""
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._revision += 1
        logger.info("向量库已清空")


# 进程级单例
_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store
