"""Main FastAPI application entry point for the AI service."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.logging_config import setup_logging
from app.middleware.error_handler import setup_error_handlers
from app.routers.chatbot import router as chatbot_router
from app.routers.cv import router as cv_router
from app.routers.candidates import router as candidates_router
from app.routers.matching import router as matching_router
from app.services.chatbot_service import get_chatbot
from app.services.conversation_service import ensure_connection as ensure_conversation_db_connection

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
setup_logging(PROJECT_ROOT / "app.log")

logger = logging.getLogger(__name__)

app = FastAPI(title="AI CV-Job Matcher", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cv_router, prefix="/cv", tags=["CV"])
app.include_router(matching_router, prefix="/matching", tags=["Matching"])
app.include_router(chatbot_router)
app.include_router(candidates_router, prefix="/candidates", tags=["Candidates"])

static_path = PROJECT_ROOT / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info("app.static.mounted path=%s", static_path)
else:
    logger.warning("app.static.missing path=%s", static_path)

setup_error_handlers(app)
logger.info("app.error_handlers.ready")


@app.on_event("startup")
async def startup_event() -> None:
    import asyncio

    await ensure_conversation_db_connection()

    if os.getenv("SKIP_MODEL_PRELOAD", "0") == "1":
        logger.info("startup.preload.skipped skip_model_preload=1")
        return

    async def _preload_models() -> None:
        try:
            pipeline = get_chatbot().pipeline
            logger.info("startup.preload.begin")
            _ = pipeline.embedding.model
            logger.info("startup.preload.embedding.ok device=%s", pipeline.embedding.device)
            logger.info("startup.preload.generation.ok device=%s dtype=%s", pipeline.generation.device, pipeline.generation.dtype)
            logger.info("startup.preload.complete health=%s", pipeline.health())
        except Exception as exc:
            logger.exception("startup.preload.failed error=%s", exc)
            logger.warning("startup.preload.fallback_lazy_load")

    await _preload_models()

    if os.getenv("ENABLE_RAG_WARMUP", "0") != "1":
        logger.info("startup.warmup.skipped enable_rag_warmup=0")
        return

    async def _sync_rag_in_background() -> None:
        try:
            result = await get_chatbot().force_sync()
            logger.info("startup.sync.complete result=%s", result)
        except Exception as exc:
            logger.exception("startup.sync.failed error=%s", exc)

    asyncio.create_task(_sync_rag_in_background())
    logger.info("startup.sync.scheduled")


@app.get("/")
async def root() -> dict:
    return {"message": "CV Matching API is running!"}


@app.get("/chat")
async def chat_page():
    try:
        with open(PROJECT_ROOT / "static" / "index.html", encoding="utf-8") as file:
            from fastapi.responses import HTMLResponse

            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        return {
            "message": "Chat UI not found. Please run the frontend separately.",
            "api_docs": "/docs",
        }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(get_chatbot().pipeline.prometheus_metrics())
