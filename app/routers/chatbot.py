"""Chatbot routes powered by a pure RAG pipeline."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.chatbot import (
    Conversation,
    ConversationMessage,
    HealthCheckResponse,
    QueryRequest,
    ReloadRAGRequest,
    RenameConversationRequest,
    RenameConversationResponse,
    SendMessageResponse,
)
from app.services import conversation_service
from app.services.chatbot_service import get_chatbot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["chatbot"], responses={404: {"description": "Not found"}})


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    try:
        service = get_chatbot()
        details = service.health()
        logger.info("Chatbot health: %s", details)
        return HealthCheckResponse(status="ok")
    except Exception as exc:
        logger.exception("Health check failed: %s", exc)
        return HealthCheckResponse(status="error")


@router.get("/info")
async def get_chat_info() -> dict:
    return {
        "version": "2.0.0",
        "name": "RAG Job Chatbot",
        "description": "Pure RAG chatbot with backend API ingestion + Chroma retrieval + Qwen generation",
        "pipeline": ["ingestion", "embedding", "retrieval", "generation"],
        "llm": "Qwen (Transformers, CUDA)",
        "vectorStore": "ChromaDB",
        "realtimeSync": "auto on interval + manual /chatbot/sync",
    }


@router.post("/sync")
async def force_rag_sync() -> dict:
    try:
        service = get_chatbot()
        return await service.force_sync()
    except Exception as exc:
        logger.exception("Manual sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"RAG sync failed: {exc}")


@router.post("/reload")
async def reload_rag_from_backend(req: ReloadRAGRequest) -> dict:
    """Fetch latest jobs/companies from backend and rebuild the RAG index."""
    try:
        service = get_chatbot()
        result = await service.pipeline.ensure_index_fresh(force=req.force)
        return {
            "success": True,
            "message": "RAG data reloaded from backend successfully",
            "reload": result,
        }
    except Exception as exc:
        logger.exception("Manual reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"RAG reload failed: {exc}")


@router.get("/sync/status")
async def get_rag_sync_status() -> dict:
    try:
        service = get_chatbot()
        return service.health()
    except Exception as exc:
        logger.exception("Sync status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query")
async def query_rag(req: QueryRequest) -> dict:
    try:
        service = get_chatbot()
        if req.conversationId:
            conv = conversation_service.get_conversation(req.conversationId)
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = conv.id
        else:
            conv = conversation_service.create_conversation(title=req.message[:50])
            conversation_id = conv.id

        conversation_service.add_message(conversation_id=conversation_id, role="user", content=req.message)
        rag_result = await service.chat(
            user_message=req.message,
            conversation_id=conversation_id,
            extra_context=req.extraContext,
        )
        assistant_msg = conversation_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=rag_result["bot_response"],
            sources=rag_result.get("sources"),
            detected_intent=rag_result.get("detected_intent"),
        )
        return {
            "message": assistant_msg,
            "conversationId": conversation_id,
            "rag": {
                "sources": rag_result.get("sources", []),
                "sync": rag_result.get("sync", {}),
                "retrieval": rag_result.get("retrieval", {}),
                "generation": rag_result.get("generation", {}),
                "latencyMs": rag_result.get("latencyMs"),
                "promptPreview": rag_result.get("promptPreview"),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("query_rag failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/retrieval-debug")
async def retrieval_debug(query: str) -> dict:
    try:
        service = get_chatbot()
        return await service.pipeline.retrieval_debug(query)
    except Exception as exc:
        logger.exception("retrieval_debug failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/indexed-jobs")
async def list_indexed_jobs(page: int = 1, limit: int = 20) -> dict:
    """List unique jobs currently present in the RAG Chroma index."""
    try:
        service = get_chatbot()
        return await service.pipeline.list_indexed_jobs(page=page, limit=limit)
    except Exception as exc:
        logger.exception("list_indexed_jobs failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/message")
async def send_chat_message(
    message: str = Form(..., description="Nội dung tin nhắn (bắt buộc)"),
    conversationId: Optional[str] = Form(None, description="ID cuộc hội thoại (tùy chọn)"),
    extraContext: Optional[str] = Form(None, description="Context từ CV hoặc tài liệu (tùy chọn, dạng text)"),
) -> SendMessageResponse:
    try:
        service = get_chatbot()

        if conversationId:
            conv = conversation_service.get_conversation(conversationId)
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = conv.id
        else:
            conv = conversation_service.create_conversation(title=message[:50])
            conversation_id = conv.id

        # extraContext được pass trực tiếp từ client (client tự extract PDF nếu cần)

        conversation_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )

        rag_result = await service.chat(
            user_message=message,
            conversation_id=conversation_id,
            extra_context=extraContext,
        )

        assistant_msg = conversation_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=rag_result["bot_response"],
            sources=rag_result.get("sources"),
            detected_intent=rag_result.get("detected_intent"),
        )

        return SendMessageResponse(
            message=assistant_msg,
            conversationId=conversation_id,
            rag={
                "sources": rag_result.get("sources", []),
                "sync": rag_result.get("sync", {}),
                "retrieval": rag_result.get("retrieval", {}),
                "generation": rag_result.get("generation", {}),
                "latencyMs": rag_result.get("latencyMs"),
            },
        )


    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("send_chat_message failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))




@router.get("/conversation", response_model=List[Conversation])
async def list_conversations() -> List[Conversation]:
    try:
        return conversation_service.get_conversations()
    except Exception as exc:
        logger.error("Error listing conversations: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/conversation", response_model=Conversation)
async def create_conversation_endpoint() -> Conversation:
    try:
        return conversation_service.create_conversation(title="Cuoc tro chuyen moi")
    except Exception as exc:
        logger.error("Error creating conversation: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/conversation/{conversation_id}/message", response_model=List[ConversationMessage])
async def get_conversation_messages(conversation_id: str) -> List[ConversationMessage]:
    try:
        conv = conversation_service.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation_service.get_messages(conversation_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting messages: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/conversation/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str) -> dict:
    try:
        deleted = conversation_service.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"success": True, "message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting conversation: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/conversation/rename", response_model=RenameConversationResponse)
async def rename_conversation_endpoint(req: RenameConversationRequest) -> RenameConversationResponse:
    try:
        updated_conv = conversation_service.rename_conversation(conv_id=req.conversationId, new_title=req.newTitle)
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
        logger.error("Error renaming conversation: %s", exc)
        raise HTTPException(status_code=500, detail=f"Loi doi ten doan chat: {exc}")
