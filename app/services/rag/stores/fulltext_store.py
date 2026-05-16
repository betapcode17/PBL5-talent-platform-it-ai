"""SQLite FTS5-backed fulltext store implementation for fallback lexical retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Iterable, List, Optional

from sqlite3 import connect, Connection

from ..schemas_and_settings.settings import FULLTEXT_DB_PATH # type: ignore

logger = logging.getLogger(__name__)


class RAGFullTextStore:
	def __init__(self, path: Optional[str] = None) -> None:
		self._path = path or FULLTEXT_DB_PATH
		self._conn: Connection = connect(self._path)

	def upsert_chunks(self, chunks: Iterable[Any]) -> None:
		cur = self._conn.cursor()
		cur.execute(
			"CREATE TABLE IF NOT EXISTS documents (chunk_id TEXT PRIMARY KEY, text TEXT, metadata TEXT, fulltext TEXT)"
		)
		for ch in chunks:
			chunk_id = ch.chunk_id
			text = ch.text
			metadata = ch.metadata or {}
			cur.execute(
				"INSERT OR REPLACE INTO documents (chunk_id, text, metadata, fulltext) VALUES (?, ?, ?, ?)",
				(chunk_id, text, str(metadata), text),
			)
		self._conn.commit()

	def query(self, query: str, top_k: int = 10) -> Generator[Dict[str, Any], None, None]:
		cur = self._conn.cursor()
		# naive FTS: compute simple LIKE-based scores and normalize
		pattern = f"%{query}%"
		rows = cur.execute("SELECT chunk_id, text, metadata FROM documents WHERE fulltext LIKE ? LIMIT ?", (pattern, top_k)).fetchall()
		for chunk_id, text, metadata_str in rows:
			yield {"chunk_id": chunk_id, "text": text, "metadata": eval(metadata_str or "{}"), "fulltext_score": 1.0}

	def get(self, limit: int = 100, offset: int = 0, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		cur = self._conn.cursor()
		q = "SELECT chunk_id, text, metadata FROM documents"
		rows = cur.execute(q).fetchmany(limit)
		metadatas = [eval(r[2] or "{}") for r in rows]
		documents = [r[1] for r in rows]
		return {"metadatas": metadatas, "documents": documents}

	def stats(self) -> Dict[str, Any]:
		cur = self._conn.cursor()
		count = cur.execute("SELECT COUNT(1) FROM documents").fetchone()[0]
		return {"fulltextRows": count}

__all__ = ["RAGFullTextStore"]
