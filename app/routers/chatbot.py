"""Chatbot routes powered by the text-only RAG pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.models.chatbot import (
    ChatbotResponseMode,
    RetrievalBenchmarkRequest,
    ChatbotMessageRequest,
    ChatbotMessageResponse,
    Conversation,
    ConversationMessage,
    HealthCheckResponse,
    QueryRequest,
    RAGObservability,
    ReloadRAGRequest,
    RenameConversationRequest,
    RenameConversationResponse,
    StructuredChatResponse,
    StructuredJobsResponse,
)
from app.services import conversation_service
from app.services.chatbot_service import get_chatbot
from app.utils.pdf_parser import extract_text_from_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["chatbot"], responses={404: {"description": "Not found"}})


def _rag_metadata(payload: dict) -> RAGObservability:
    return RAGObservability(
        sources=payload.get("sources", []),
        sync=payload.get("sync", {}),
        retrieval=payload.get("retrieval", {}),
        generation=payload.get("generation", {}),
        latencyMs=payload.get("latencyMs"),
        promptPreview=payload.get("promptPreview"),
    )


def _build_structured_response(payload: dict):
    data = payload.get("data") or {}
    jobs = data.get("jobs") or []
    if isinstance(jobs, list) and jobs:
        return StructuredJobsResponse(
            items=[
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "company": item.get("company"),
                    "location": item.get("location") or None,
                    "salary": item.get("salary") or None,
                    "skills": item.get("skills") or [],
                    "jobType": item.get("jobType"),
                    "score": item.get("score"),
                    "url": item.get("url"),
                }
                for item in jobs
            ],
            total=int(data.get("total") or len(jobs)),
        ) # type: ignore
    return None


async def _handle_text_message(
    message: str,
    conversation_id: str | None,
    extra_context: str | None,
    retrieval_profile: str | None,
    response_mode: ChatbotResponseMode,
) -> ChatbotMessageResponse:
    service = get_chatbot()

    if conversation_id:
        conv = await conversation_service.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        resolved_conversation_id = conv.id
    else:
        conv = await conversation_service.create_conversation(title=message[:50])
        resolved_conversation_id = conv.id

    await conversation_service.add_message(
        conversation_id=resolved_conversation_id,
        role="user",
        content=message,
    )

    rag_result = await service.chat(
        user_message=message,
        conversation_id=resolved_conversation_id,
        extra_context=extra_context,
        retrieval_profile=retrieval_profile,
    )
    # record assistant message in conversation store
    message_content = rag_result.get("message", {}).get("content") if isinstance(rag_result, dict) else None
    # build lightweight sources list for storage
    data_jobs = (rag_result.get("data") or {}).get("jobs") if isinstance(rag_result, dict) else None
    total_jobs_found = len(data_jobs) if isinstance(data_jobs, list) else int((rag_result.get("data") or {}).get("total") or 0) if isinstance(rag_result, dict) else 0
    if total_jobs_found > 0:
        prefix = f"Tim thay {total_jobs_found} cong viec phu hop."
        if message_content and not str(message_content).lower().startswith("tim thay"):
            message_content = f"{prefix} {message_content}".strip()
        elif not message_content:
            message_content = prefix
    sources_for_db = []
    if isinstance(data_jobs, list):
        for j in data_jobs:
            sources_for_db.append({"entityType": "job", "sourceId": j.get("id"), "title": j.get("title"), "company": j.get("company"), "url": j.get("url")})

    assistant_msg = await conversation_service.add_message(
        conversation_id=resolved_conversation_id,
        role="assistant",
        content=message_content or "",
        sources=sources_for_db,
        detected_intent=(rag_result.get("data") or {}).get("intent") if isinstance(rag_result, dict) else None,
    )

    # Build response model instance
    structured_payload = _build_structured_response(rag_result) if isinstance(rag_result, dict) else None

    resp_payload = {
        "success": rag_result.get("success", True),
        "version": rag_result.get("version", "2.1.0"),
        "conversationId": resolved_conversation_id,
        "message": assistant_msg.model_dump() if hasattr(assistant_msg, "model_dump") else assistant_msg.dict() if hasattr(assistant_msg, "dict") else assistant_msg,
        "data": {
            **(rag_result.get("data") or {}),
            "totalJobsFound": total_jobs_found,
        },
        "structured": structured_payload,
        "retrieval": rag_result.get("retrieval"),
        "meta": rag_result.get("meta"),
    }

    return ChatbotMessageResponse.parse_obj(resp_payload)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    try:
        logger.info("chatbot.health.request")
        details = get_chatbot().health()
        logger.info("chatbot.health.ok details=%s", details)
        return HealthCheckResponse(status="ok")
    except Exception as exc:
        logger.exception("chatbot.health.failed error=%s", exc)
        return HealthCheckResponse(status="error")


@router.get("/info")
async def get_chat_info() -> dict:
    return {
        "version": "2.2.0",
        "name": "RAG Job Chatbot",
        "description": "Text-only RAG chatbot with separate ingest pipeline and observability",
        "pipeline": ["retrieval", "generation"],
        "ingest": ["backend_sync", "embedding", "indexing"],
        "llm": "Qwen (Transformers, CUDA preload)",
        "vectorStore": "ChromaDB",
        "realtimeSync": "manual/background only, no sync during chat",
    }


@router.post("/message", response_model=ChatbotMessageResponse)
async def send_chat_message(req: ChatbotMessageRequest) -> ChatbotMessageResponse:
    """Text-only chat endpoint. No file upload is handled here."""
    try:
        logger.info(
            "chatbot.message.request conversation_id=%s extra_context=%s retrieval_profile=%s",
            req.conversationId,
            bool(req.extraContext),
            req.retrievalProfile,
        )
        return await _handle_text_message(
            message=req.message.strip(),
            conversation_id=req.conversationId,
            extra_context=req.extraContext.strip() if req.extraContext else None,
            retrieval_profile=req.retrievalProfile,
            response_mode=req.responseMode,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chatbot.message.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query", response_model=ChatbotMessageResponse)
async def query_rag(req: QueryRequest) -> ChatbotMessageResponse:
    try:
        return await _handle_text_message(
            message=req.message.strip(),
            conversation_id=req.conversationId,
            extra_context=req.extraContext.strip() if req.extraContext else None,
            retrieval_profile=req.retrievalProfile,
            response_mode=req.responseMode,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chatbot.query.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))





@router.post("/sync")
async def force_rag_sync(req: ReloadRAGRequest) -> dict:
    """Alias of /reload kept for backwards compatibility.
    Accepts same request body and returns the same wrapper response.
    """
    try:
        result = await get_chatbot().pipeline.ensure_index_fresh(force=req.force)
        return {
            "success": True,
            "message": "RAG data reloaded from backend successfully",
            "syncResult": result,
        }
    except Exception as exc:
        logger.exception("chatbot.sync.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=f"RAG sync failed: {exc}")


@router.get("/sync/status")
async def get_rag_sync_status() -> dict:
    try:
        return get_chatbot().health()
    except Exception as exc:
        logger.exception("chatbot.sync_status.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/context/upload")
async def upload_context_pdf(file: UploadFile = File(...)) -> dict:
    """Separate endpoint for PDF extraction, not part of /chatbot/message."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF file is too large")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            temp_path = tmp.name
        text = extract_text_from_pdf(temp_path).strip()
        return {
            "filename": file.filename,
            "characters": len(text),
            "extractedContext": text[:4000],
        }
    except Exception as exc:
        logger.exception("chatbot.context_upload.failed filename=%s error=%s", file.filename, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/retrieval-debug")
async def retrieval_debug(query: str, profile: str | None = None) -> dict:
    try:
        return await get_chatbot().pipeline.retrieval_debug(query, profile_name=profile)
    except Exception as exc:
        logger.exception("chatbot.retrieval_debug.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/retrieval-profiles")
async def list_retrieval_profiles() -> dict:
    try:
        return get_chatbot().pipeline.retrieval_profiles()
    except Exception as exc:
        logger.exception("chatbot.retrieval_profiles.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/retrieval-benchmark")
async def run_retrieval_benchmark(req: RetrievalBenchmarkRequest) -> dict:
    try:
        return await get_chatbot().pipeline.benchmark_retrieval([item.model_dump() for item in req.cases])
    except Exception as exc:
        logger.exception("chatbot.retrieval_benchmark.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/indexed-jobs")
async def list_indexed_jobs(page: int = 1, limit: int = 20) -> dict:
    try:
        return await get_chatbot().pipeline.list_indexed_jobs(page=page, limit=limit)
    except Exception as exc:
        logger.exception("chatbot.indexed_jobs.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/metrics")
async def chatbot_metrics() -> dict:
    try:
        return get_chatbot().pipeline.metrics()
    except Exception as exc:
        logger.exception("chatbot.metrics.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/indexed-companies")
async def list_indexed_companies(page: int = 1, limit: int = 20) -> dict:
    try:
        return await get_chatbot().pipeline.list_indexed_companies(page=page, limit=limit)
    except Exception as exc:
        logger.exception("chatbot.indexed_companies.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/indexed-companies/{company_id}")
async def get_indexed_company_detail(company_id: str) -> dict:
    try:
        return await get_chatbot().pipeline.get_indexed_company_detail(company_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("chatbot.indexed_company_detail.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# Note: `/internal/rag-stats` removed as duplicate of `/metrics`.


@router.get("/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    try:
        return PlainTextResponse(get_chatbot().pipeline.prometheus_metrics())
    except Exception as exc:
        logger.exception("chatbot.prometheus_metrics.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/conversation", response_model=List[Conversation])
async def list_conversations() -> List[Conversation]:
    try:
        return await conversation_service.get_conversations()
    except Exception as exc:
        logger.exception("chatbot.conversations.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/conversation", response_model=Conversation)
async def create_conversation_endpoint() -> Conversation:
    try:
        return await conversation_service.create_conversation(title="Cuoc tro chuyen moi")
    except Exception as exc:
        logger.exception("chatbot.conversation_create.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/conversation/{conversation_id}/message", response_model=List[ConversationMessage])
async def get_conversation_messages(conversation_id: str) -> List[ConversationMessage]:
    try:
        conv = await conversation_service.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return await conversation_service.get_messages(conversation_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chatbot.conversation_messages.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/conversation/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str) -> dict:
    try:
        deleted = await conversation_service.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"success": True, "message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chatbot.conversation_delete.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/conversation/rename", response_model=RenameConversationResponse)
async def rename_conversation_endpoint(req: RenameConversationRequest) -> RenameConversationResponse:
    try:
        updated_conv = await conversation_service.rename_conversation(conv_id=req.conversationId, new_title=req.newTitle)
        if not updated_conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return RenameConversationResponse(
            conversationId=req.conversationId,
            newTitle=req.newTitle,
            updatedAt=updated_conv.updateAt,
            success=True,
            message="Da doi ten doan chat thanh cong",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chatbot.conversation_rename.failed error=%s", exc)
        raise HTTPException(status_code=500, detail=f"Loi doi ten doan chat: {exc}")
