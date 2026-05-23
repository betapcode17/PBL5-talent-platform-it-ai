# app/services/conversation_service.py
"""Conversation persistence service for the AI chatbot.
Stores conversations and messages in PostgreSQL.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.models.chatbot import Conversation, ConversationMessage
from app.services.pg_db import close_pg_pool, execute, fetch, fetchrow, init_pg_pool, run_migration_sql

logger = logging.getLogger(__name__)


async def ensure_connection() -> None:
    """Validate PostgreSQL availability and create chatbot tables."""
    await init_pg_pool()
    await run_migration_sql()


async def create_conversation(title: str = "", seeker_id: Optional[int] = None) -> Conversation:
    """Create a new AI conversation and return it."""
    now = datetime.utcnow()
    row = await fetchrow(
        """
        INSERT INTO conversations (title, created_at, updated_at)
        VALUES ($1, $2, $3)
        RETURNING id
        """,
        title,
        now,
        now,
    )
    conversation_id = str(row["id"]) # type: ignore

    if seeker_id is not None:
        await execute(
            "INSERT INTO conversation_sessions (conversation_id, session_key) VALUES ($1, $2)",
            int(conversation_id),
            f"seeker:{seeker_id}",
        ) # type: ignore

    logger.info("[OK] Created conversation: %s", conversation_id)
    return Conversation(
        id=conversation_id,
        title=title,
        lastMessage=None,
        createdAt=now,
        updateAt=now,
    )


async def get_conversations(seeker_id: Optional[int] = None, limit: int = 50) -> List[Conversation]:
    """List conversations ordered by most recently updated."""
    if seeker_id is not None:
        rows = await fetch(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at
            FROM conversations c
            JOIN conversation_sessions s ON s.conversation_id = c.id
            WHERE s.session_key = $1
            ORDER BY c.updated_at DESC
            LIMIT $2
            """,
            f"seeker:{seeker_id}",
            limit,
        )
    else:
        rows = await fetch(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (
                     SELECT m.content
                     FROM messages m
                     WHERE m.conversation_id = c.id
                     ORDER BY m.created_at DESC
                     LIMIT 1
                   ) AS last_message
            FROM conversations c
            ORDER BY c.updated_at DESC
            LIMIT $1
            """,
            limit,
        )

    conversations: List[Conversation] = []
    for row in rows:
        conversations.append(
            Conversation(
                id=str(row["id"]),
                title=row.get("title") or "",
                lastMessage=row.get("last_message"),
                createdAt=row.get("created_at"), # type: ignore
                updateAt=row.get("updated_at"), # type: ignore
            )
        )
    return conversations


async def get_conversation(conv_id: str) -> Optional[Conversation]:
    """Get a single conversation by id."""
    row = await fetchrow(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at,
               (
                 SELECT m.content
                 FROM messages m
                 WHERE m.conversation_id = c.id
                 ORDER BY m.created_at DESC
                 LIMIT 1
               ) AS last_message
        FROM conversations c
        WHERE c.id = $1
        """,
        int(conv_id),
    )
    if not row:
        return None

    return Conversation(
        id=str(row["id"]),
        title=row.get("title") or "",
        lastMessage=row.get("last_message"),
        createdAt=row.get("created_at"), # type: ignore
        updateAt=row.get("updated_at"), # type: ignore
    )


async def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation and its messages. Returns True if deleted."""
    result = await execute("DELETE FROM conversations WHERE id = $1", int(conv_id))
    deleted = result.startswith("DELETE 1") # type: ignore
    if deleted:
        logger.info("[OK] Deleted conversation: %s", conv_id)
    return deleted


async def _update_conversation(conv_id: str, last_message: str, title: Optional[str] = None) -> None:
    """Update conversation metadata after a new message."""
    now = datetime.utcnow()
    if title:
        await execute(
            "UPDATE conversations SET title = $1, updated_at = $2 WHERE id = $3",
            title,
            now,
            int(conv_id),
        )
    else:
        await execute(
            "UPDATE conversations SET updated_at = $1 WHERE id = $2",
            now,
            int(conv_id),
        )


async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    detected_intent: Optional[str] = None,
    update_title: Optional[str] = None,
) -> ConversationMessage:
    """Insert a message and update the conversation's metadata."""
    now = datetime.utcnow()
    msg_id = str(uuid4())
    meta = {
        "sources": sources,
        "detected_intent": detected_intent,
    }
    await execute(
        """
        INSERT INTO messages (conversation_id, role, content, metadata, created_at)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        """,
        int(conversation_id),
        role,
        content,
        json.dumps(meta),
        now,
    )
    try:
        await _update_conversation(conversation_id, content, title=update_title)
    except Exception as exc:
        logger.error("[ERROR] Error updating conversation: %s", exc)

    message = ConversationMessage(
        id=msg_id,
        conversationId=conversation_id,
        role=role,
        content=content,
        createdAt=now,
        sources=sources,
        detectedIntent=detected_intent,
    )
    logger.info("[OK] Added message: %s", msg_id)
    return message


async def get_messages(conversation_id: str, limit: int = 100) -> List[ConversationMessage]:
    """Get messages for a conversation, ordered chronologically."""
    rows = await fetch(
        """
        SELECT id, conversation_id, role, content, metadata, created_at
        FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at ASC
        LIMIT $2
        """,
        int(conversation_id),
        limit,
    )
    messages: List[ConversationMessage] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        messages.append(
            ConversationMessage(
                id=str(row["id"]),
                conversationId=str(row["conversation_id"]),
                role=row["role"],
                content=row["content"],
                createdAt=row.get("created_at"), # type: ignore 
                sources=(metadata or {}).get("sources"),
                detectedIntent=(metadata or {}).get("detected_intent"),
            )
        )
    return messages


async def get_recent_history(conversation_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
    """Get recent messages as simple dicts for building LLM conversation context."""
    rows = await fetch(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        int(conversation_id),
        max_turns * 2,
    )
    rows = list(rows)
    rows.reverse()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


async def rename_conversation(conv_id: str, new_title: str) -> Optional[Conversation]:
    """Rename a conversation and return the updated object."""
    existing = await get_conversation(conv_id)
    if not existing:
        logger.warning("[WARNING] Conversation not found: %s", conv_id)
        return None
    now = datetime.utcnow()
    await execute(
        "UPDATE conversations SET title = $1, updated_at = $2 WHERE id = $3",
        new_title,
        now,
        int(conv_id),
    )
    logger.info("[OK] Renamed conversation %s to: '%s'", conv_id, new_title)
    return Conversation(
        id=existing.id,
        title=new_title,
        lastMessage=existing.lastMessage,
        createdAt=existing.createdAt,
        updateAt=now,
    )


async def close_connection() -> None:
    """Close PostgreSQL connection pool."""
    await close_pg_pool()
    logger.info("[OK] PostgreSQL connection closed")
