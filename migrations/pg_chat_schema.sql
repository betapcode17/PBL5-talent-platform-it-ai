-- migrations/pg_chat_schema.sql
-- Schema for chatbot persistence in PostgreSQL

-- Conversations table: one row per conversation
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Messages table: messages belonging to conversations
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL, -- 'user' | 'assistant' | 'system' etc.
    content TEXT NOT NULL,
    metadata JSONB, -- optional structured metadata (e.g. model, usage, embeddings ref)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created_at ON messages(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- Optional: table for session or user mapping (if needed)
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    session_key TEXT UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
