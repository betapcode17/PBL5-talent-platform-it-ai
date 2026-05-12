"""SQLite FTS5 store for hybrid retrieval alongside ChromaDB."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .schemas import ChunkRecord
from .settings import RAG_FULLTEXT_DB_PATH

logger = logging.getLogger(__name__)


class RAGFullTextStore:
    """Lightweight full-text index using SQLite FTS5."""

    def __init__(self) -> None:
        self.db_path = Path(RAG_FULLTEXT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts
                USING fts5(chunk_id UNINDEXED, text, metadata_json UNINDEXED, tokenize='unicode61');
                """
            )
            conn.commit()

    def rebuild_index(self, chunks: Sequence[ChunkRecord]) -> int:
        with self._connect() as conn:
            conn.execute("DELETE FROM rag_chunks_fts")
            conn.executemany(
                "INSERT INTO rag_chunks_fts (chunk_id, text, metadata_json) VALUES (?, ?, ?)",
                [(c.chunk_id, c.text, json.dumps(c.metadata, ensure_ascii=False)) for c in chunks],
            )
            conn.commit()
        return len(chunks)

    def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        if not query_text.strip():
            return []

        safe_query = self._to_safe_match_query(query_text)
        if not safe_query:
            return []

        sql = (
            "SELECT chunk_id, text, metadata_json, bm25(rag_chunks_fts) AS rank "
            "FROM rag_chunks_fts WHERE rag_chunks_fts MATCH ? "
            "ORDER BY rank ASC LIMIT ?"
        )
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (safe_query, top_k)).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("fts.query.failed query=%r error=%s", query_text[:120], exc)
            return []

        results: List[Dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            # bm25() in SQLite FTS5 is lower-is-better; convert to positive relevance
            rank = float(row["rank"] if row["rank"] is not None else 0.0)
            relevance = max(0.0, min(1.0, 1.0 / (1.0 + abs(rank))))
            results.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "text": str(row["text"]),
                    "metadata": metadata,
                    "fulltext_score": relevance,
                }
            )
        return results

    @staticmethod
    def _to_safe_match_query(query_text: str) -> str:
        # FTS5 MATCH is sensitive to punctuation/syntax; reduce to token AND query.
        tokens = [t for t in re.findall(r"\w+", query_text.lower(), flags=re.UNICODE) if len(t) >= 2]
        if not tokens:
            return ""
        return " AND ".join(tokens)
