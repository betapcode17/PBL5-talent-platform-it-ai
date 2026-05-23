# app/services/pg_conversation_service.py
"""Conversation persistence implemented on PostgreSQL using asyncpg.
This mirrors the earlier MongoDB-backed API but stores conversations and messages
in relational tables.
"""
from typing import List, Dict, Any, Optional
import json
import logging
from app.services.pg_db import init_pg_pool, run_migration_sql, fetch, fetchrow, execute

async def ensure_connection():
    await init_pg_pool()
    await run_migration_sql()

async def create_conversation(title: Optional[str] = None) -> int:
    row = await fetchrow("INSERT INTO conversations (title) VALUES ($1) RETURNING id", title)
    return int(row['id']) # type: ignore

async def add_message(conversation_id: int, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    meta_json = json.dumps(metadata) if metadata else None
    row = await fetchrow(
        "INSERT INTO messages (conversation_id, role, content, metadata) VALUES ($1,$2,$3,$4) RETURNING id",
        conversation_id, role, content, meta_json
    )
    return int(row['id']) # type: ignore

async def get_messages(conversation_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    rows = await fetch("SELECT id, role, content, metadata, created_at FROM messages WHERE conversation_id=$1 ORDER BY created_at ASC LIMIT $2", conversation_id, limit)
    return [dict(r) for r in rows]

async def get_recent_history(conversation_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    rows = await fetch("SELECT role, content, metadata, created_at FROM messages WHERE conversation_id=$1 ORDER BY created_at DESC LIMIT $2", conversation_id, limit)
    return [dict(r) for r in rows]

async def rename_conversation(conversation_id: int, new_title: str) -> None:
    await execute("UPDATE conversations SET title=$1, updated_at=now() WHERE id=$2", new_title, conversation_id)

async def delete_conversation(conversation_id: int) -> None:
    await execute("DELETE FROM conversations WHERE id=$1", conversation_id)

async def close_connection():
    from app.services.pg_db import close_pg_pool
    await close_pg_pool()
