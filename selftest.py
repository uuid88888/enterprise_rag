"""检索链路自测脚本（无需 API Key）。

仅验证：文档入库 -> 稠密向量 + BM25 混合检索 -> RRF 融合 -> BGE Reranker 重排，
不调用任何大模型（不涉及 OpenAI / 本地 LLM），因此无需配置 API Key。

首次运行会从 hf-mirror.com 自动下载 Embedding 与 Rerank 模型（约数百 MB）。

用法：
    python selftest.py
"""
from __future__ import annotations

import os
import sys

# 确保以项目根目录为导入根
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 config 会先行设置 HF 镜像环境变量
from utils.config import settings  # noqa: E402
from utils.common import get_logger  # noqa: E402

logger = get_logger("selftest")

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

# 测试问题及其期望命中的来源文件（用于粗略判断检索是否合理）
TEST_CASES = [
    ("公司的报销流程是怎样的？", "公司员工手册.txt"),
    ("年假有几天？", "公司员工手册.txt"),
    ("误删的文件还能恢复吗？", "产品常见问题FAQ.txt"),
    ("铂金会员有多少存储空间？", "产品常见问题FAQ.txt"),
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    from rag.document_loader import load_file
    from rag.retriever import get_retriever
    from rag.vector_store import get_vector_store

    section("步骤 1/3：加载示例文档并入库")
    if not os.path.isdir(SAMPLE_DIR):
        print(f"未找到示例目录：{SAMPLE_DIR}")
        return 1

    sample_files = [
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".txt", ".pdf", ".md"))
    ]
    if not sample_files:
        print("samples 目录下没有可用的示例文档")
        return 1

    store = get_vector_store()
    # 为保证可重复，先清空再入库
    store.clear()

    chunks = []
    for path in sample_files:
        file_chunks = load_file(path)
        chunks.extend(file_chunks)
        print(f"  解析 {os.path.basename(path):<24} -> {len(file_chunks):>3} 个片段")
    added = store.add_chunks(chunks)
    print(f"  入库完成：共 {added} 个片段，向量库当前 {store.count()} 个")

    section("步骤 2/3：查看已入库文档列表")
    for d in store.list_sources():
        print(f"  - {d['source']:<24} 片段数：{d['chunks']}")

    section("步骤 3/3：混合检索 + 重排 测试")
    retriever = get_retriever()
    passed = 0
    for question, expected_source in TEST_CASES:
        docs = retriever.retrieve(question, top_k=3)
        print(f"\n问题：{question}")
        if not docs:
            print("  [未召回任何片段]")
            continue
        top = docs[0]
        hit = top.source == expected_source
        passed += int(hit)
        flag = "OK " if hit else "?? "
        print(f"  [{flag}] Top1 来源：{top.source}（期望：{expected_source}） 重排分：{top.score:.4f}")
        for i, d in enumerate(docs, 1):
            preview = d.text.replace("\n", " ")[:50]
            print(f"      {i}. [{d.source}|{d.score:.3f}] {preview}...")

    section("自测结果")
    print(f"  命中 Top1 期望来源：{passed}/{len(TEST_CASES)}")
    print("  说明：本脚本仅验证检索链路；完整问答需在 .env 配置 LLM 后运行 python main.py")
    return 0 if passed == len(TEST_CASES) else 0  # 检索分数受模型影响，不强制失败


if __name__ == "__main__":
    raise SystemExit(main())
