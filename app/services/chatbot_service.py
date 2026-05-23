"""Pure RAG chatbot service.

This module intentionally removes tool-calling agent orchestration and
routes all responses through retrieval-augmented generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services import conversation_service

logger = logging.getLogger(__name__)


class ChatbotRAGService:
    """Conversation-aware wrapper around RAG pipeline."""

    def __init__(self) -> None:
        from app.services.rag import get_rag_pipeline

        self.pipeline = get_rag_pipeline()

    async def chat(
        self,
        user_message: str,
        conversation_id: str,
        extra_context: Optional[str] = None,
        retrieval_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        history = await conversation_service.get_recent_history(conversation_id=conversation_id, max_turns=6)
        logger.info(
            "chatbot.service.chat conversation_id=%s history_turns=%s extra_context=%s retrieval_profile=%s",
            conversation_id,
            len(history),
            bool(extra_context),
            retrieval_profile,
        )
        result = await self.pipeline.answer(
            user_message.strip(),
            conversation_history=history,
            extra_context=extra_context.strip() if extra_context else None,
            profile_name=retrieval_profile,
        )
        return result

    async def force_sync(self) -> Dict[str, Any]:
        result = await self.pipeline.ensure_index_fresh(force=True)
        return {
            "success": True,
            "message": "RAG data reloaded from backend successfully",
            "syncResult": result,
        }

    def health(self) -> Dict[str, Any]:
        return self.pipeline.health()


_chatbot_singleton: Optional[ChatbotRAGService] = None


def get_chatbot() -> ChatbotRAGService:
    global _chatbot_singleton
    if _chatbot_singleton is None:
        _chatbot_singleton = ChatbotRAGService()
    return _chatbot_singleton
