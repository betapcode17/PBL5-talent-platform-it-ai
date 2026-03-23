# app/services/conversation_service.py
"""
Conversation persistence service for the AI chatbot.
Stores conversations and messages in PostgreSQL (AiConversation / AiMessage tables).
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import psycopg2
import psycopg2.extras

from config import DATABASE_URL
from app.models.chatbot import Conversation, ConversationMessage

logger = logging.getLogger(__name__)


def _get_conn():
    return psycopg2.connect(DATABASE_URL)



def create_conversation(title: str = "", seeker_id: Optional[int] = None) -> Conversation:
    """Create a new AI conversation and return it."""
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO "AiConversation" (id, title, created_at, updated_at, seeker_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (conv_id, title, now, now, seeker_id),
            )
        conn.commit()
    return Conversation(
        id=conv_id,
        title=title,
        lastMessage=None,
        createdAt=now,
        updateAt=now,
    )


def get_conversations(seeker_id: Optional[int] = None, limit: int = 50) -> List[Conversation]:
    """List conversations ordered by most recently updated."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if seeker_id is not None:
                cur.execute(
                    """
                    SELECT id, title, last_message, created_at, updated_at
                    FROM "AiConversation"
                    WHERE seeker_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (seeker_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, title, last_message, created_at, updated_at
                    FROM "AiConversation"
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
    return [
        Conversation(
            id=r["id"],
            title=r["title"],
            lastMessage=r["last_message"],
            createdAt=r["created_at"],
            updateAt=r["updated_at"],
        )
        for r in rows
    ]


def get_conversation(conv_id: str) -> Optional[Conversation]:
    """Get a single conversation by id."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, last_message, created_at, updated_at
                FROM "AiConversation"
                WHERE id = %s
                """,
                (conv_id,),
            )
            r = cur.fetchone()
    if not r:
        return None
    return Conversation(
        id=r["id"],
        title=r["title"],
        lastMessage=r["last_message"],
        createdAt=r["created_at"],
        updateAt=r["updated_at"],
    )


def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation and its messages (CASCADE). Returns True if deleted."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'DELETE FROM "AiConversation" WHERE id = %s',
                (conv_id,),
            )
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def _update_conversation(conn, conv_id: str, last_message: str, title: Optional[str] = None):
    """Update conversation metadata after a new message."""
    now = datetime.utcnow()
    with conn.cursor() as cur:
        if title:
            cur.execute(
                """
                UPDATE "AiConversation"
                SET last_message = %s, updated_at = %s, title = %s
                WHERE id = %s
                """,
                (last_message[:200], now, title, conv_id),
            )
        else:
            cur.execute(
                """
                UPDATE "AiConversation"
                SET last_message = %s, updated_at = %s
                WHERE id = %s
                """,
                (last_message[:200], now, conv_id),
            )


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    detected_intent: Optional[str] = None,
    update_title: Optional[str] = None,
) -> ConversationMessage:
    """Insert a message and update the conversation's last_message / updated_at."""
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow()
    sources_json = json.dumps(sources, default=str) if sources else None

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO "AiMessage" (id, conversation_id, role, content, sources, detected_intent, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (msg_id, conversation_id, role, content, sources_json, detected_intent, now),
            )
        _update_conversation(conn, conversation_id, content, title=update_title)
        conn.commit()

    return ConversationMessage(
        id=msg_id,
        conversationId=conversation_id,
        role=role,
        content=content,
        createdAt=now,
    )


def get_messages(conversation_id: str, limit: int = 100) -> List[ConversationMessage]:
    """Get messages for a conversation, ordered chronologically."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM "AiMessage"
                WHERE conversation_id = %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            rows = cur.fetchall()
    return [
        ConversationMessage(
            id=r["id"],
            conversationId=r["conversation_id"],
            role=r["role"],
            content=r["content"],
            createdAt=r["created_at"],
        )
        for r in rows
    ]


def get_recent_history(conversation_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
    """Get recent messages as simple dicts for building LLM conversation context."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content
                FROM "AiMessage"
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (conversation_id, max_turns * 2),
            )
            rows = cur.fetchall()
    # Reverse to chronological order
    rows.reverse()
    return [{"role": r["role"], "content": r["content"]} for r in rows]
