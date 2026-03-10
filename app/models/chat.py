# app/models/chat.py
"""
Chat data models for RAG chatbot.
Định nghĩa các Pydantic models cho chat messages, responses, và history.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ChatRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """User chat message input"""
    message: str = Field(..., min_length=1, max_length=2000, description="Nội dung tin nhắn")
    session_id: Optional[str] = Field(None, description="ID phiên chat")
    context_type: Optional[str] = Field("auto", description="Loại context: auto, jobs, cv, matching, career")


class ChatResponse(BaseModel):
    """Chat response output"""
    response: str = Field(..., description="Trả lời từ chatbot")
    session_id: str = Field(..., description="ID phiên chat")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Thời gian phản hồi")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Tài liệu được truy xuất (job IDs, titles, etc)"
    )
    detected_intent: Optional[str] = Field(
        None,
        description="Intent được phát hiện: jobs, cv, matching, career, default"
    )
    confidence_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Độ tin cậy của câu trả lời"
    )


class ChatHistoryItem(BaseModel):
    """Một mục trong lịch sử chat"""
    role: ChatRole = Field(..., description="user hoặc assistant")
    content: str = Field(..., description="Nội dung tin nhắn")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved documents")


class ChatHistoryRequest(BaseModel):
    """Request để lấy lịch sử chat"""
    session_id: str = Field(..., description="ID phiên chat")
    limit: int = Field(50, ge=1, le=100, description="Số lượng tin nhắn gần nhất")


class ChatHistoryResponse(BaseModel):
    """Response lịch sử chat"""
    session_id: str
    messages: List[ChatHistoryItem]
    total_messages: int
    created_at: datetime


class ChatClearRequest(BaseModel):
    """Request xóa chat history"""
    session_id: str = Field(..., description="ID phiên chat cần xóa")


class ChatClearResponse(BaseModel):
    """Response xóa chat history"""
    success: bool
    session_id: str
    message: str


class RetrievedDocument(BaseModel):
    """Document được truy xuất từ ChromaDB"""
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None  # Similarity score


class RAGContext(BaseModel):
    """Context từ RAG retrieval"""
    query: str
    documents: List[RetrievedDocument]
    total_results: int


class ChatSession(BaseModel):
    """Chat session metadata"""
    session_id: str
    created_at: datetime
    last_message_at: datetime
    message_count: int
    context_type: str = "jobs"
    metadata: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---- AI Chatbot Conversation Models (matching frontend interfaces) ----

class ConversationMessage(BaseModel):
    """A single message in the AI chatbot conversation (matches frontend ChatMessage)"""
    id: str
    conversationId: str
    role: str  # 'user' | 'assistant'
    content: str
    createdAt: datetime


class Conversation(BaseModel):
    """AI chatbot conversation (matches frontend Conversation interface)"""
    id: str
    title: str
    lastMessage: Optional[str] = None
    createdAt: datetime
    updateAt: datetime


class SendMessageRequest(BaseModel):
    """Request to send a message to the AI chatbot"""
    conversationId: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=2000)


class SendMessageResponse(BaseModel):
    """Response after sending a message to the AI chatbot"""
    message: ConversationMessage
    conversationId: str
    version: str = "1.0.0"
    services: Dict[str, bool] = Field(
        default_factory=lambda: {
            "chroma": True,
            "gemini": True,
            "database": True
        }
    )
