# app/services/conversation_service.py
"""
Conversation persistence service for the AI chatbot.
Stores conversations and messages in MongoDB (Atlas Cloud).
Replaces PostgreSQL storage with MongoDB NoSQL storage.
"""

from __future__ import annotations

import json
import logging
import uuid
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
except ImportError:  # pragma: no cover - optional runtime dependency
    MongoClient = None  # type: ignore[assignment]
    ServerSelectionTimeoutError = ConnectionFailure = Exception  # type: ignore[assignment]

from app.config import MONGODB_URL, MONGODB_DATABASE
from app.models.chatbot import Conversation, ConversationMessage

logger = logging.getLogger(__name__)

# MongoDB Client (Global connection pool)
_client: Optional[Any] = None
_db = None


def _get_db():
    """Get MongoDB database instance with connection pooling."""
    global _client, _db
    if _db is None:
        if MongoClient is None:
            logger.warning("[WARNING] pymongo not installed; using in-memory mock storage")
            _db = MockDB()
            return _db

        try:
            _client = MongoClient(
                MONGODB_URL,
                serverSelectionTimeoutMS=5000,  # Reduced for faster fallback
                socketTimeoutMS=5000,
                connectTimeoutMS=5000,
                retryWrites=False,
                tlsAllowInvalidCertificates=os.getenv("MONGODB_ALLOW_INVALID_CERTIFICATES", "false").lower() == "true",
            )
            # Verify connection
            _client.admin.command('ping')
            _db = _client[MONGODB_DATABASE]
            logger.info(f"[OK] Connected to MongoDB: {MONGODB_DATABASE}")
            
            # Create indexes for better query performance
            _create_indexes()
        except Exception as e:
            logger.warning(f"[WARNING] MongoDB not available: {e}")
            logger.warning("[FALLBACK] Using in-memory mock storage")
            # Use mock database for testing
            _db = MockDB()
    return _db


class MockDB:
    """Mock MongoDB for testing without real MongoDB."""
    def __init__(self):
        self._collections: Dict[str, MockCollection] = {}

    def __getitem__(self, key):
        if key not in self._collections:
            self._collections[key] = MockCollection()
        return self._collections[key]


class MockCollection:
    """In-memory collection with a subset of pymongo behavior."""

    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    def insert_one(self, doc):
        self.docs.append(dict(doc))
        logger.info(f"[MOCK] Inserted message: {doc.get('id', 'unknown')}")
        return type('Result', (), {'inserted_id': doc.get('_id')})()
    
    def find_one(self, query):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return dict(doc)
        return None
    
    def find(self, query=None):
        matched = [
            dict(doc)
            for doc in self.docs
            if all(doc.get(k) == v for k, v in (query or {}).items())
        ]
        return MockCursor(matched)
    
    def update_one(self, filter_q, update_doc):
        modified = 0
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in filter_q.items()):
                if "$set" in update_doc:
                    doc.update(update_doc["$set"])
                modified += 1
                break
        return type('Result', (), {'modified_count': modified})()

    def delete_one(self, query):
        deleted = 0
        for idx, doc in enumerate(self.docs):
            if all(doc.get(k) == v for k, v in query.items()):
                self.docs.pop(idx)
                deleted = 1
                break
        return type('Result', (), {'deleted_count': deleted})()

    def delete_many(self, query):
        kept = []
        deleted = 0
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                deleted += 1
            else:
                kept.append(doc)
        self.docs = kept
        return type('Result', (), {'deleted_count': deleted})()
    
    def create_index(self, keys):
        logger.info(f"[MOCK] Index: {keys}")
        return None


class MockCursor:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs

    def sort(self, key: str, direction: int):
        reverse = direction == -1
        self.docs.sort(key=lambda doc: doc.get(key) or datetime.min, reverse=reverse)
        return self

    def limit(self, limit: int):
        self.docs = self.docs[:limit]
        return self

    def __iter__(self):
        return iter(self.docs)


def _create_indexes():
    """Create indexes for MongoDB collections."""
    try:
        db = _get_db()
        
        # Conversation indexes
        conversations = db["AiConversation"]
        conversations.create_index("created_at")
        conversations.create_index("updated_at")
        conversations.create_index([("seeker_id", 1), ("updated_at", -1)])
        
        # Message indexes
        messages = db["AiMessage"]
        messages.create_index("conversation_id")
        messages.create_index("created_at")
        
        logger.info("[OK] MongoDB indexes created")
    except Exception as e:
        logger.error(f"[WARNING] Failed to create indexes: {e}")


def create_conversation(title: str = "", seeker_id: Optional[int] = None) -> Conversation:
    """Create a new AI conversation and return it."""
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    db = _get_db()
    conversations = db["AiConversation"]
    
    conv_doc = {
        "_id": conv_id,
        "id": conv_id,
        "title": title,
        "last_message": None,
        "created_at": now,
        "updated_at": now,
        "seeker_id": seeker_id,
    }
    
    conversations.insert_one(conv_doc)
    logger.info(f"[OK] Created conversation: {conv_id}")
    
    return Conversation(
        id=conv_id,
        title=title,
        lastMessage=None,
        createdAt=now,
        updateAt=now,
    )


def get_conversations(seeker_id: Optional[int] = None, limit: int = 50) -> List[Conversation]:
    """List conversations ordered by most recently updated."""
    db = _get_db()
    conversations = db["AiConversation"]
    
    try:
        if seeker_id is not None:
            docs = list(
                conversations.find({"seeker_id": seeker_id})
                .sort("updated_at", -1)
                .limit(limit)
            )
        else:
            docs = list(
                conversations.find({})
                .sort("updated_at", -1)
                .limit(limit)
            )
        
        return [
            Conversation(
                id=doc["id"],
                title=doc.get("title", ""),
                lastMessage=doc.get("last_message"),
                createdAt=doc.get("created_at"), # type: ignore
                updateAt=doc.get("updated_at"), # type: ignore
            )
            for doc in docs
        ]
    except Exception as e:
        logger.error(f"[ERROR] Error getting conversations: {e}")
        return []


def get_conversation(conv_id: str) -> Optional[Conversation]:
    """Get a single conversation by id."""
    db = _get_db()
    conversations = db["AiConversation"]
    
    try:
        doc = conversations.find_one({"_id": conv_id})
        if not doc:
            return None
        
        return Conversation(
            id=doc["id"],
            title=doc.get("title", ""),
            lastMessage=doc.get("last_message"),
            createdAt=doc.get("created_at"), # type: ignore
            updateAt=doc.get("updated_at"), # type: ignore
        )
    except Exception as e:
        logger.error(f"[ERROR] Error getting conversation: {e}")
        return None


def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation and its messages (CASCADE). Returns True if deleted."""
    db = _get_db()
    conversations = db["AiConversation"]
    messages = db["AiMessage"]
    
    try:
        # Delete all messages for this conversation
        messages.delete_many({"conversation_id": conv_id})
        
        # Delete the conversation
        result = conversations.delete_one({"_id": conv_id})
        
        deleted = result.deleted_count > 0 # type: ignore
        if deleted:
            logger.info(f"[OK] Deleted conversation: {conv_id}")
        return deleted
    except Exception as e:
        logger.error(f"[ERROR] Error deleting conversation: {e}")
        return False


def _update_conversation(conv_id: str, last_message: str, title: Optional[str] = None):
    """Update conversation metadata after a new message."""
    now = datetime.utcnow()
    db = _get_db()
    conversations = db["AiConversation"]
    
    try:
        update_doc = {
            "last_message": last_message[:200],
            "updated_at": now,
        }
        if title:
            update_doc["title"] = title
        
        conversations.update_one(
            {"_id": conv_id},
            {"$set": update_doc}
        )
    except Exception as e:
        logger.error(f"[ERROR] Error updating conversation: {e}")


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
    
    db = _get_db()
    messages = db["AiMessage"]
    
    msg_doc = {
        "_id": msg_id,
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "sources": sources,
        "detected_intent": detected_intent,
        "created_at": now,
    }
    
    try:
        messages.insert_one(msg_doc)
    except Exception as e:
        logger.error(f"[ERROR] Error inserting message: {e}")
        raise
    
    try:
        _update_conversation(conversation_id, content, title=update_title)
        logger.info(f"[OK] Added message: {msg_id}")
    except Exception as e:
        logger.error(f"[ERROR] Error updating conversation: {e}")
        # Don't raise here - message was inserted, just update failed
    
    # Always return the message
    msg = ConversationMessage(
        id=msg_id,
        conversationId=conversation_id,
        role=role,
        content=content,
        createdAt=now,
        sources=sources,
        detectedIntent=detected_intent,
    )
    logger.info(f"[OK] Returning ConversationMessage: {msg.id}")
    return msg



def get_messages(conversation_id: str, limit: int = 100) -> List[ConversationMessage]:
    """Get messages for a conversation, ordered chronologically."""
    db = _get_db()
    messages = db["AiMessage"]
    
    try:
        docs = list(
            messages.find({"conversation_id": conversation_id})
            .sort("created_at", 1)
            .limit(limit)
        )
        
        return [
            ConversationMessage(
                id=doc["id"],
                conversationId=doc["conversation_id"],
                role=doc["role"],
                content=doc["content"],
                createdAt=doc.get("created_at"), # type: ignore
                sources=doc.get("sources"),
                detectedIntent=doc.get("detected_intent"),
            )
            for doc in docs
        ]
    except Exception as e:
        logger.error(f"[ERROR] Error getting messages: {e}")
        return []


def get_recent_history(conversation_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
    """Get recent messages as simple dicts for building LLM conversation context."""
    db = _get_db()
    messages = db["AiMessage"]
    
    try:
        docs = list(
            messages.find({"conversation_id": conversation_id})
            .sort("created_at", -1)
            .limit(max_turns * 2)
        )
        
        # Reverse to chronological order
        docs.reverse()
        return [{"role": doc["role"], "content": doc["content"]} for doc in docs]
    except Exception as e:
        logger.error(f"[ERROR] Error getting recent history: {e}")
        return []


def rename_conversation(conv_id: str, new_title: str) -> Optional[Conversation]:
    """
    Rename a conversation.
    
    Args:
        conv_id: Conversation ID
        new_title: New title for the conversation
        
    Returns:
        Updated Conversation object, or None if conversation not found
    """
    db = _get_db()
    conversations = db["AiConversation"]
    
    try:
        # Check if conversation exists
        doc = conversations.find_one({"_id": conv_id})
        if not doc:
            logger.warning(f"[WARNING] Conversation not found: {conv_id}")
            return None
        
        # Update title
        now = datetime.utcnow()
        conversations.update_one(
            {"_id": conv_id},
            {"$set": {"title": new_title, "updated_at": now}}
        )
        
        logger.info(f"[OK] Renamed conversation {conv_id} to: '{new_title}'")
        
        # Return updated conversation
        return Conversation(
            id=doc["id"],
            title=new_title,
            lastMessage=doc.get("last_message"),
            createdAt=doc.get("created_at"), # type: ignore
            updateAt=now,
        )
    except Exception as e:
        logger.error(f"[ERROR] Error renaming conversation: {e}")
        return None


def close_connection():
    """Close MongoDB connection pool."""
    global _client
    if _client:
        _client.close()
        logger.info("[OK] MongoDB connection closed")
