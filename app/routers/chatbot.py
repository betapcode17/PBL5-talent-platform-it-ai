# app/routers/chatbot.py
"""
Chatbot Endpoints - FastAPI routes for RAG chatbot.
Provides REST API for chat interactions.
"""

import logging
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request  # type: ignore
from typing import Optional, List

from app.services.chatbot_service import get_chatbot, get_tool_aware_chatbot
from app.services import conversation_service
from app.models.chatbot import (
    HealthCheckResponse,
    Conversation, ConversationMessage, SendMessageRequest, SendMessageResponse,
    RenameConversationRequest, RenameConversationResponse,
)
from app.utils.pdf_parser import extract_text_from_pdf
from app.services.db_utils import insert_cv_record
import os
import tempfile

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"],
    responses={404: {"description": "Not found"}},
)



@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint
    
    Response:
    ```json
    {
        "status": "ok",
        "timestamp": "...",
        "version": "1.0.0",
        "services": {
            "chroma": true,
            "ollama": true,
            "database": true
        }
    }
    ```
    """
    try:
        # Check each service quickly
        chatbot = get_chatbot()
        
        # If we got here, services are working
        return HealthCheckResponse(
            status="ok",
            services={ # type: ignore
                "chroma": True,
                "ollama": True,
                "database": True
            }
        )
        
    except Exception as e:
        logger.error(f" Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            services={ # type: ignore
                "chroma": False,
                "ollama": False,
                "database": False
            }
        )


@router.get("/info")
async def get_chat_info():
    """
    Get chatbot information
    
    Response:
    ```json
    {
        "version": "1.0.0",
        "name": "RAG CV-Job Chatbot",
        "description": "...",
        "context_types": ["jobs", "cv", "matching", "career"],
        "features": [...]
    }
    ```
    """
    return {
        "version": "1.0.0",
        "name": "RAG CV-Job Chatbot",
        "description": "AI-powered chatbot for job matching and CV improvement",
        "context_types": ["jobs", "cv", "matching", "career"],
        "features": [
            "RAG (Retrieval-Augmented Generation)",
            "Multi-context conversations",
            "Chat history management",
            "Job-CV matching",
            "Career advice",
            "Real-time document retrieval"
        ],
        "collection_types": ["jobs", "cvs"],
        "model": "Ollama (Local LLM)"
    }




@router.post("/message")
async def send_chat_message(
    message: str = Form(...),
    conversationId: Optional[str] = Form(None),
    attachments: Optional[List[UploadFile]] = File(None),  # 🔥 FIX QUAN TRỌNG
):
    try:
        logger.info(f"[DEBUG] message={message}, conversationId={conversationId}")

        file_ids = []

     
        attachments = attachments or []

        logger.info(f"[FILE] Attachments received: {len(attachments)} items")

        for file in attachments:
            if not file.filename:
                continue

            if not file.filename.lower().endswith(".pdf"):
                logger.warning(f"[FILE] Skip non-PDF: {file.filename}")
                continue

            try:
                file_data = await file.read()

                if len(file_data) > 10 * 1024 * 1024:
                    logger.warning(f"[FILE] Too large: {file.filename}")
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_data)
                    temp_path = tmp.name

                logger.info(f"[FILE] Saved: {temp_path}")

                # Debug PDF content
                try:
                    text = extract_text_from_pdf(temp_path)
                    logger.info(f"[PDF] Content preview: {text[:300]}")
                except Exception as e:
                    logger.warning(f"[PDF] Read error: {e}")

                file_ids.append(temp_path)

            except Exception as e:
                logger.error(f"[FILE ERROR] {file.filename}: {e}")

        logger.info(f"[FILE] Final file_ids: {file_ids}")

        # =============================
        # ✅ CHATBOT
        # =============================
        chatbot = get_tool_aware_chatbot()

        # Create / get conversation
        if conversationId:
            conv = conversation_service.get_conversation(conversationId)
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = conv.id
        else:
            conv = conversation_service.create_conversation(title=message[:50])
            conversation_id = conv.id

        # Save user message
        conversation_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )

        # Call chatbot
        result = await chatbot.chat_with_tools(
            user_message=message,
            conversation_id=conversation_id,
            file_ids=file_ids if file_ids else None,
        )

        # Save bot message
        assistant_msg = conversation_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["bot_response"],
            sources=result.get("sources"),
            detected_intent=result.get("detected_intent"),
        )

        if not assistant_msg:
            raise HTTPException(status_code=500, detail="Không lưu được tin nhắn")

        return SendMessageResponse(
            message=assistant_msg,
            conversationId=conversation_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] send_chat_message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation", response_model=List[Conversation])
async def list_conversations() -> List[Conversation]:
    """
    List all AI chatbot conversations, most recent first.
    Maps to frontend: getConversationsApi
    """
    try:
        return conversation_service.get_conversations()
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation", response_model=Conversation)
async def create_conversation_endpoint() -> Conversation:
    """
    Create a new empty conversation.
    Maps to frontend: createConversationApi
    """
    try:
        return conversation_service.create_conversation(title="Cuộc trò chuyện mới")
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/message", response_model=List[ConversationMessage])
async def get_conversation_messages(conversation_id: str) -> List[ConversationMessage]:
    """
    Get all messages for a conversation.
    Maps to frontend: getMessagesApi
    """
    try:
        conv = conversation_service.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation_service.get_messages(conversation_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """
    Delete a conversation and all its messages.
    Maps to frontend: deleteConversation
    """
    try:
        deleted = conversation_service.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"success": True, "message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation/rename", response_model=RenameConversationResponse)
async def rename_conversation_endpoint(req: RenameConversationRequest) -> RenameConversationResponse:
    """
    Rename a conversation.
    Maps to frontend: renameConversation
    
    Request:
    ```json
    {
        "conversationId": "uuid-here",
        "newTitle": "New conversation title"
    }
    ```
    
    Response:
    ```json
    {
        "conversationId": "uuid-here",
        "newTitle": "New conversation title",
        "updatedAt": "2026-04-19T10:30:00.000Z",
        "success": true,
        "message": "Đã đổi tên đoạn chat thành công"
    }
    ```
    """
    try:
        updated_conv = conversation_service.rename_conversation(
            conv_id=req.conversationId,
            new_title=req.newTitle
        )
        
        if not updated_conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return RenameConversationResponse(
            conversationId=req.conversationId,
            newTitle=req.newTitle,
            updatedAt=updated_conv.updateAt,
            success=True,
            message="Đã đổi tên đoạn chat thành công"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi đổi tên đoạn chat: {str(e)}")
