"""Gradio 可视化交互界面。

页面分两大模块：
- 模块 A 知识库管理：批量上传入库、查看文档列表、清空向量库、实时进度
- 模块 B 智能问答：多轮对话、RAG/Agent 模式切换、参考片段溯源折叠展示、参数面板
"""
from __future__ import annotations

from typing import List

import gradio as gr

from api.routes import answer, ingest_paths
from rag.document_loader import SUPPORTED_TYPES
from rag.vector_store import get_vector_store
from utils.common import get_logger
from utils.config import settings

logger = get_logger("ui.web_app")


# --------------------------------------------------------------------------- #
# 知识库管理回调
# --------------------------------------------------------------------------- #
def handle_upload(files, progress=gr.Progress()):
    """处理批量上传入库，返回状态文本与最新文档表格。

    分阶段反馈：解析分块 -> 向量化（按片段实时进度）-> 写入向量库。
    """
    if not files:
        return "请先选择文件", _doc_table_html(), gr.update(
            choices=_source_choices(), value=None
        )

    import os

    paths = [f.name for f in files]
    names = [os.path.basename(f.name) for f in files]
    try:
        progress(0.05, desc="解析与分块中...")

        def on_embed(done: int, total: int) -> None:
            # 向量化占进度条 0.1 ~ 0.95 区间
            frac = 0.1 + 0.85 * (done / total if total else 1)
            progress(frac, desc=f"向量化中… {done}/{total} 片段")

        result = ingest_paths(paths, names=names, progress_callback=on_embed)
        progress(1.0, desc="入库完成")
        status = (
            f"入库成功：新增/更新 {result.added} 个片段，"
            f"去重跳过 {result.skipped_duplicates} 个，"
            f"知识库当前共 {result.total_chunks} 个片段。"
        )
    except Exception as exc:
        logger.error("上传入库失败：%s", exc)
        status = f"入库失败：{exc}"
    return status, _doc_table_html(), gr.update(choices=_source_choices(), value=None)


def _esc(text: str) -> str:
    """HTML 转义，避免文件名中的特殊字符破坏表格。"""
    import html

    return html.escape(str(text))


def _doc_table_html() -> str:
    """渲染可滚动的已入库文档表格（HTML）。"""
    rows = get_vector_store().list_sources()
    total_chunks = sum(r["chunks"] for r in rows)

    if not rows:
        body = (
            '<tr><td colspan="3" style="padding:16px;text-align:center;color:#888;">'
            "暂无入库文档</td></tr>"
        )
    else:
        body = "".join(
            f"<tr>"
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;">{i + 1}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;'
            f'word-break:break-all;">{_esc(r["source"])}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;'
            f'text-align:right;white-space:nowrap;">{r["chunks"]}</td>'
            f"</tr>"
            for i, r in enumerate(rows)
        )

    return f"""
<div style="margin-bottom:6px;font-size:13px;color:#666;">
  共 {len(rows)} 个文件 / {total_chunks} 个片段
</div>
<div style="max-height:320px;overflow-y:auto;border:1px solid #e0e0e0;border-radius:8px;">
  <table style="width:100%;border-collapse:collapse;font-size:14px;">
    <thead>
      <tr style="position:sticky;top:0;background:#f5f5f5;z-index:1;">
        <th style="padding:10px 12px;text-align:left;width:48px;">#</th>
        <th style="padding:10px 12px;text-align:left;">来源文件</th>
        <th style="padding:10px 12px;text-align:right;width:80px;">片段数</th>
      </tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def handle_refresh():
    return _doc_table_html(), gr.update(choices=_source_choices(), value=None)


def handle_clear():
    get_vector_store().clear()
    return "向量库已清空", _doc_table_html(), gr.update(choices=[], value=None)


def _source_choices() -> List[str]:
    return [r["source"] for r in get_vector_store().list_sources()]


def handle_delete_source(source: str):
    if not source:
        return "请先选择要删除的来源文件", _doc_table_html(), gr.update(
            choices=_source_choices(), value=None
        )
    get_vector_store().delete_source(source)
    return (
        f"已删除来源文件：{source}",
        _doc_table_html(),
        gr.update(choices=_source_choices(), value=None),
    )


# --------------------------------------------------------------------------- #
# 问答回调
# --------------------------------------------------------------------------- #
def _format_sources(sources: List[dict]) -> str:
    if not sources:
        return "（本次回答未引用知识库片段）"
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(
            f"**片段 {i}｜来源：{s['source']}｜相关度：{s['score']}**\n\n{s['text']}\n\n---"
        )
    return "\n".join(lines)


def handle_chat(
    message: str,
    chat_history: List[dict],
    mode_label: str,
    top_k: int,
    score_threshold: float,
    enable_hyde: bool,
    enable_pii: bool,
):
    """处理一轮问答。chat_history 为 messages 格式。"""
    if not message or not message.strip():
        return chat_history, "", "（请输入问题）"

    mode = "agent" if "Agent" in mode_label else "rag"
    # 将 Gradio messages 历史转为 {role,content}
    history = [{"role": m["role"], "content": m["content"]} for m in chat_history]

    try:
        resp = answer(
            question=message,
            mode=mode,
            history=history,
            top_k=int(top_k),
            score_threshold=float(score_threshold),
            enable_hyde=bool(enable_hyde),
            enable_pii=bool(enable_pii),
        )
        answer_text = resp.answer
        sources_md = _format_sources([s.model_dump() for s in resp.sources])
    except Exception as exc:
        logger.error("问答失败：%s", exc)
        answer_text = f"出错了：{exc}"
        sources_md = "（无）"

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer_text},
    ]
    return chat_history, "", sources_md


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面。"""
    file_types = [ext for exts in SUPPORTED_TYPES.values() for ext in exts]
    with gr.Blocks(title="企业私有知识库问答平台") as demo:
        gr.Markdown(
            "# 🔍 企业私有知识库问答平台\n"
            "基于 LangChain + LangGraph 的 Agentic 混合检索 RAG 系统"
        )

        with gr.Tab("📚 知识库管理"):
            gr.Markdown("### 上传文档（支持 PDF / Office / 文本 / EPUB / OpenDocument 等）")
            with gr.Row():
                file_input = gr.File(
                    label="选择文件",
                    file_count="multiple",
                    file_types=file_types,
                )
            with gr.Row():
                upload_btn = gr.Button("上传并入库", variant="primary")
                refresh_btn = gr.Button("刷新文档列表")
                clear_btn = gr.Button("清空向量库", variant="stop")
            with gr.Row():
                delete_source_dropdown = gr.Dropdown(
                    choices=_source_choices(),
                    label="选择要删除的来源文件",
                    scale=4,
                )
                delete_source_btn = gr.Button("删除所选文件", variant="stop", scale=1)
            upload_status = gr.Textbox(label="状态", interactive=False)
            gr.Markdown("#### 已入库文档")
            doc_table = gr.HTML(value=_doc_table_html())

            upload_btn.click(
                handle_upload,
                [file_input],
                [upload_status, doc_table, delete_source_dropdown],
            )
            refresh_btn.click(handle_refresh, None, [doc_table, delete_source_dropdown])
            clear_btn.click(
                handle_clear,
                None,
                [upload_status, doc_table, delete_source_dropdown],
            )
            delete_source_btn.click(
                handle_delete_source,
                [delete_source_dropdown],
                [upload_status, doc_table, delete_source_dropdown],
            )

        with gr.Tab("💬 智能问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="对话", height=460, type="messages")
                    with gr.Row():
                        msg_box = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入你的问题，回车或点击发送...",
                            scale=5,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_chat_btn = gr.Button("清空对话")
                    with gr.Accordion("📎 检索参考原文片段（溯源）", open=False):
                        sources_box = gr.Markdown("（暂无）")

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 参数配置")
                    mode_radio = gr.Radio(
                        ["普通 RAG 模式", "Agent 智能检索模式"],
                        value="普通 RAG 模式",
                        label="问答模式",
                    )
                    top_k_slider = gr.Slider(1, 20, value=settings.top_k, step=1, label="检索条数 Top-K")
                    threshold_slider = gr.Slider(
                        0.0, 1.0, value=settings.score_threshold, step=0.05, label="相似度阈值"
                    )
                    hyde_check = gr.Checkbox(value=settings.enable_hyde, label="开启 HyDE 查询改写")
                    pii_check = gr.Checkbox(value=settings.enable_pii, label="开启隐私脱敏")

            inputs = [
                msg_box,
                chatbot,
                mode_radio,
                top_k_slider,
                threshold_slider,
                hyde_check,
                pii_check,
            ]
            outputs = [chatbot, msg_box, sources_box]
            send_btn.click(handle_chat, inputs, outputs)
            msg_box.submit(handle_chat, inputs, outputs)
            clear_chat_btn.click(lambda: ([], "（暂无）"), None, [chatbot, sources_box])

    return demo
