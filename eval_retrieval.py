"""检索质量评测脚本。

读取 eval/questions.jsonl，验证问题的 Top-K 召回来源与关键词命中。
该脚本不调用大模型，只覆盖入库、向量检索、BM25、RRF、Rerank。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.document_loader import load_file  # noqa: E402
from rag.retriever import get_retriever  # noqa: E402
from rag.vector_store import get_vector_store  # noqa: E402


@dataclass
class EvalCase:
    question: str
    expected_sources: List[str]
    expected_keywords: List[str]


def _load_cases(path: str) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(
                    question=data["question"],
                    expected_sources=data.get("expected_sources", []),
                    expected_keywords=data.get("expected_keywords", []),
                )
            )
    return cases


def _ingest_samples(sample_dir: str) -> None:
    store = get_vector_store()
    store.clear()
    chunks = []
    for name in os.listdir(sample_dir):
        path = os.path.join(sample_dir, name)
        if os.path.isfile(path):
            chunks.extend(load_file(path))
    store.add_chunks(chunks)


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="评测检索召回质量。")
    parser.add_argument("--cases", default=os.path.join(root, "eval", "questions.jsonl"))
    parser.add_argument("--sample-dir", default=os.path.join(root, "samples"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ingest-samples", action="store_true", help="评测前清空库并入库 samples")
    args = parser.parse_args()

    if args.ingest_samples:
        _ingest_samples(args.sample_dir)

    cases = _load_cases(args.cases)
    retriever = get_retriever()
    source_hits = 0
    keyword_hits = 0

    for case in cases:
        docs = retriever.retrieve(case.question, top_k=args.top_k)
        sources = [doc.source for doc in docs]
        joined = "\n".join(doc.text for doc in docs)
        source_ok = not case.expected_sources or any(src in sources for src in case.expected_sources)
        keyword_ok = not case.expected_keywords or any(kw in joined for kw in case.expected_keywords)
        source_hits += int(source_ok)
        keyword_hits += int(keyword_ok)

        print(f"\n问题：{case.question}")
        print(f"  来源命中：{'OK' if source_ok else 'MISS'}  期望={case.expected_sources} 实际={sources}")
        print(f"  关键词命中：{'OK' if keyword_ok else 'MISS'}  期望={case.expected_keywords}")
        for i, doc in enumerate(docs, 1):
            preview = doc.text.replace("\n", " ")[:80]
            print(f"    {i}. [{doc.source}|{doc.score:.4f}] {preview}...")

    total = len(cases)
    print("\n评测结果")
    print(f"  来源命中率：{source_hits}/{total}")
    print(f"  关键词命中率：{keyword_hits}/{total}")
    return 0 if source_hits == total and keyword_hits == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
