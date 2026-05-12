# app/models/chatbot.py
"""
Chatbot data models for RAG chatbot.
Định nghĩa các Pydantic models cho chat messages, responses, và history.
"""

from pydantic import BaseModel, Field # type: ignore
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum
from typing import Union


# --- New production-ready response schemas ---


class JobItem(BaseModel):
    id: Optional[str]
    title: Optional[str]
    company: Optional[str]
    location: Optional[str]
    salary: Optional[str]
    skills: Optional[List[str]] = Field(default_factory=list)
    jobType: Optional[str]
    score: Optional[float]
    url: Optional[str]


class StructuredJobsResponse(BaseModel):
    type: Literal["jobs"] = Field("jobs")
    items: List[JobItem] = Field(default_factory=list)
    total: int = 0


class RetrievalSummary(BaseModel):
    profile: Optional[str]
    topScore: Optional[float]
    count: int = 0
    fallbackTriggered: bool = False


class GenerationMeta(BaseModel):
    model: Optional[str]
    latencyMs: Optional[float]
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None


class RetrievalMeta(BaseModel):
    latencyMs: Optional[float]
    rerankLatencyMs: Optional[float]
    contextPackingLatencyMs: Optional[float]


class MetaResponse(BaseModel):
    latencyMs: Optional[float]
    retrieval: Optional[RetrievalMeta]
    generation: Optional[GenerationMeta]


class ChatRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatbotResponseMode(str, Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"


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
    sources: Optional[List[Dict[str, Any]]] = None
    detectedIntent: Optional[str] = None


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
    file_ids: Optional[List[str]] = None  # Optional: list of uploaded CV IDs to analyze with


class SendMessageResponse(BaseModel):
    """Response after sending a message to the AI chatbot"""
    message: ConversationMessage
    conversationId: str
    version: str = "1.0.0"
    rag: Optional[Dict[str, Any]] = None
    services: Dict[str, bool] = Field(
        default_factory=lambda: {
            "chroma": True,
            "qwen": True,
            "database": True
        }
    )


class QueryRequest(BaseModel):
    """JSON request model for RAG query testing."""
    message: str = Field(..., min_length=1, max_length=4000)
    conversationId: Optional[str] = None
    extraContext: Optional[str] = None
    retrievalProfile: Optional[str] = Field(default=None, description="Optional retrieval profile override")
    responseMode: ChatbotResponseMode = Field(default=ChatbotResponseMode.TEXT, description="Response format mode")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Tim job backend Python tai HCM"
            }
        }
    }


class ChatbotMessageRequest(BaseModel):
    """Canonical text-only chat contract for /chatbot/message."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message text")
    conversationId: Optional[str] = Field(default=None, description="Existing conversation id")
    extraContext: Optional[str] = Field(default=None, description="Optional extra plain-text context")
    retrievalProfile: Optional[str] = Field(default=None, description="Optional retrieval profile override")
    responseMode: ChatbotResponseMode = Field(default=ChatbotResponseMode.TEXT, description="Response format mode")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Tim job backend Python tai HCM"
            }
        }
    }


class RAGSourceItem(BaseModel):
    entityType: Optional[str] = None
    sourceId: Optional[str] = None
    id: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    salary: Optional[str] = None
    skills: Optional[str] = None
    industry: Optional[str] = None
    url: Optional[str] = None
    distance: Optional[float] = None
    score: Optional[float] = None
    chunkId: Optional[str] = None


class RAGObservability(BaseModel):
    sync: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    generation: Dict[str, Any] = Field(default_factory=dict)
    latencyMs: Optional[float] = None
    promptPreview: Optional[str] = None
    sources: List[RAGSourceItem] = Field(default_factory=list)


class StructuredChatResponse(BaseModel):
    summary: str
    skills: str
    salary: str
    location: str
    nextStep: str


class ChatbotMessageResponse(BaseModel):
    """Canonical response contract for text-only chat."""
    success: bool = True
    version: str = "2.1.0"
    conversationId: str
    message: ConversationMessage
    data: Optional[Dict[str, Any]] = None
    structured: Optional[Union[StructuredJobsResponse, Dict[str, Any]]] = None
    retrieval: Optional[RetrievalSummary] = None
    meta: Optional[MetaResponse] = None
    rag: Optional[Dict[str, Any]] = None
    responseMode: ChatbotResponseMode = ChatbotResponseMode.TEXT


class ReloadRAGRequest(BaseModel):
    """Request model for manual RAG reload/reindex."""
    force: bool = Field(default=True, description="Force fetch latest data from backend and rebuild index")


class RetrievalEvaluationCase(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    expectedTerms: List[str] = Field(default_factory=list, description="Expected terms for simple quality scoring")
    profile: str = Field(default="balanced", description="Retrieval profile name")


class RetrievalBenchmarkRequest(BaseModel):
    cases: List[RetrievalEvaluationCase] = Field(..., min_length=1, max_length=100)


class RenameConversationRequest(BaseModel):
    """Request to rename a conversation"""
    conversationId: str = Field(..., description="ID của đoạn chat")
    newTitle: str = Field(..., min_length=1, max_length=200, description="Tên mới cho đoạn chat")


class RenameConversationResponse(BaseModel):
    """Response after renaming a conversation"""
    conversationId: str
    newTitle: str
    updatedAt: datetime
    success: bool = True
    message: str = "Đã đổi tên đoạn chat thành công"
