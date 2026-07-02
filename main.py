"""项目总入口：一键启动 FastAPI + Gradio。

启动后：
- FastAPI 接口挂载在 /api 下，文档地址 /docs
- Gradio 页面挂载在根路径 /
- 自动打开浏览器访问 Gradio 页面
"""
from __future__ import annotations

import threading
import webbrowser

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 注意：config 的 import 会先行设置 HF 镜像环境变量，必须在其它重模型库之前
from utils.config import settings
from utils.common import get_logger
from api.routes import router
from ui.web_app import build_ui

logger = get_logger("main")


def create_app() -> FastAPI:
    """构建 FastAPI 应用并挂载 Gradio。"""
    app = FastAPI(
        title="企业私有知识库问答平台",
        description="基于 LangChain + LangGraph 的 Agentic 混合检索 RAG 平台",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")

    @app.get("/api")
    def api_root() -> dict:
        return {"message": "Enterprise RAG API", "docs": "/docs"}

    # 将 Gradio 挂载到根路径
    demo = build_ui()
    app = gr.mount_gradio_app(app, demo, path="/")
    return app


def _warmup_models() -> None:
    """后台预热 Embedding / Rerank 模型，避免首次上传或问答时才加载导致卡顿。"""
    try:
        from rag.vector_store import warmup as warmup_embed
        from rag.retriever import _get_reranker

        logger.info("开始预热模型...")
        warmup_embed()
        _get_reranker()
        logger.info("模型预热完成")
    except Exception as exc:
        logger.warning("模型预热失败（不影响后续按需加载）：%s", exc)


def _open_browser() -> None:
    url = f"http://{settings.api_host}:{settings.api_port}/"
    logger.info("浏览器访问地址：%s", url)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    app = create_app()
    # 后台预热模型，不阻塞服务启动
    threading.Thread(target=_warmup_models, daemon=True).start()
    if settings.auto_open_browser:
        # 延迟 1.5s 打开浏览器，确保服务已起
        threading.Timer(1.5, _open_browser).start()
    logger.info("服务启动中：http://%s:%d", settings.api_host, settings.api_port)
    logger.info("API 文档： http://%s:%d/docs", settings.api_host, settings.api_port)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


if __name__ == "__main__":
    main()
