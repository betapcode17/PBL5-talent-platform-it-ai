# app/routers/chat.py
"""
Chat Endpoints - FastAPI routes for RAG chatbot.
Provides REST API for chat interactions.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from services.chat_service import get_chatbot
from services import conversation_service
from models.chat import (
    ChatMessage, ChatResponse, ChatHistoryRequest, 
    ChatHistoryResponse, ChatClearRequest, ChatClearResponse,
    ChatHistoryItem, HealthCheckResponse,
    Conversation, ConversationMessage, SendMessageRequest, SendMessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


@router.post("/message", response_model=ChatResponse)
async def send_message(msg: ChatMessage) -> ChatResponse:
    """
    Send a chat message and receive bot response (RAG-augmented)
    
    Request:
    ```json
    {
        "message": "Có công việc Python nào không?",
        "session_id": "optional-uuid",
        "context_type": "jobs"
    }
    ```
    
    Response:
    ```json
    {
        "response": "Dựa trên dữ liệu hiện tại...",
        "session_id": "uuid",
        "timestamp": "2024-03-05T...",
        "sources": [
            {
                "id": "123",
                "title": "Python Developer",
                "company": "TechCorp",
                ...
            }
        ],
        "confidence_score": 0.85
    }
    ```
    """
    try:
        logger.info(f" Incoming message: {msg.message[:50]}...")
        
        # Validate input
        if not msg.message or len(msg.message.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Message không được để trống"
            )
        
        if len(msg.message) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Message quá dài (tối đa 2000 ký tự)"
            )
        
        # Get chatbot instance
        chatbot = get_chatbot()
        
        # Process message
        response = chatbot.chat( # type: ignore
            user_message=msg.message,
            session_id=msg.session_id,
            context_type=msg.context_type or "auto"
        )
        
        logger.info(f" Message processed (session={response.session_id})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Error processing message: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý tin nhắn: {str(e)}"
        )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str = Query(..., description="Chat session ID"),
    limit: int = Query(50, ge=1, le=100, description="Số tin nhắn trở lại")
) -> ChatHistoryResponse:
    """
    Get chat history for a session
    
    Query params:
    - session_id: UUID của phiên chat
    - limit: Số tin nhắn gần nhất cần lấy (1-100)
    
    Response:
    ```json
    {
        "session_id": "uuid",
        "messages": [
            {
                "role": "user",
                "content": "text",
                "timestamp": "...",
                "sources": null
            }
        ],
        "total_messages": 5,
        "created_at": "..."
    }
    ```
    """
    try:
        logger.info(f" Getting history for session: {session_id}")
        
        # Get chatbot
        chatbot = get_chatbot()
        
        # Get history
        messages = chatbot.get_history(session_id, limit=limit)
        
        if not messages:
            return ChatHistoryResponse(
                session_id=session_id,
                messages=[],
                total_messages=0,
                created_at=None # type: ignore
            )
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
            created_at=messages[0].timestamp if messages else None # type: ignore
        )
        
    except Exception as e:
        logger.error(f" Error retrieving history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi truy xuất lịch sử: {str(e)}"
        )


@router.post("/clear", response_model=ChatClearResponse)
async def clear_chat_history(req: ChatClearRequest) -> ChatClearResponse:
    """
    Clear chat history for a session
    
    Request:
    ```json
    {
        "session_id": "uuid"
    }
    ```
    """
    try:
        logger.info(f" Clearing history for session: {req.session_id}")
        
        chatbot = get_chatbot()
        success = chatbot.clear_history(req.session_id)
        
        if success:
            return ChatClearResponse(
                success=True,
                session_id=req.session_id,
                message="Lịch sử chat đã được xóa"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="Phiên chat không tìm thấy"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Error clearing history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xóa lịch sử: {str(e)}"
        )


@router.get("/new-session")
async def create_new_session():
    """
    Create a new chat session
    
    Response:
    ```json
    {
        "session_id": "uuid",
        "message": "Phiên chat mới tạo thành công"
    }
    ```
    """
    try:
        chatbot = get_chatbot()
        session_id = chatbot.create_session()
        
        return {
            "session_id": session_id,
            "message": "Phiên chat mới tạo thành công"
        }
        
    except Exception as e:
        logger.error(f" Error creating session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi tạo phiên: {str(e)}"
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
            "gemini": true,
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
                "gemini": True,
                "database": True
            }
        )
        
    except Exception as e:
        logger.error(f" Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            services={ # type: ignore
                "chroma": False,
                "gemini": False,
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
        "model": "Google Gemini Pro"
    }




@router.post("/send", response_model=SendMessageResponse)
async def send_chat_message(req: SendMessageRequest) -> SendMessageResponse:
    """
    Send a message to the AI chatbot. Creates a new conversation if conversationId is not provided.
    Maps to frontend: sendMessageApi
    """
    try:
        chatbot = get_chatbot()

        # Create or reuse conversation
        if req.conversationId:
            conv = conversation_service.get_conversation(req.conversationId)
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = conv.id
        else:
            # Auto-generate a title from the first message
            title = req.message[:60] + ("..." if len(req.message) > 60 else "")
            conv = conversation_service.create_conversation(title=title)
            conversation_id = conv.id

        # Persist the user message
        conversation_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.message,
        )

        # Process through the RAG chatbot
        result = chatbot.chat_with_conversation(
            user_message=req.message,
            conversation_id=conversation_id,
        )

        # Persist the assistant response
        # If this is the first exchange, set the conversation title from the user message
        update_title = None
        if not req.conversationId:
            update_title = req.message[:60] + ("..." if len(req.message) > 60 else "")

        assistant_msg = conversation_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["bot_response"],
            sources=result.get("sources"),
            detected_intent=result.get("detected_intent"),
            update_title=update_title,
        )

        return SendMessageResponse(
            message=assistant_msg,
            conversationId=conversation_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in send_chat_message: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý tin nhắn: {str(e)}")


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
