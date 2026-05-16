"""Chroma-backed vector store implementation residing in stores package.

This file contains the concrete RAGVectorStore that was previously at the
top-level of the package. It keeps the same public class name but uses the
internal chroma client and expects metadata values to be validated by the
ingestion layer (e.g. `skills` non-empty lists).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from ..schemas_and_settings.settings import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME # type: ignore

logger = logging.getLogger(__name__)


class RAGVectorStore:
	def __init__(self) -> None:
		import chromadb

		self._client = chromadb.PersistentClient(CHROMA_PERSIST_DIRECTORY)
		self._collection = self._client.get_or_create_collection(CHROMA_COLLECTION_NAME)

	def upsert_chunks(self, chunks: Iterable[Any], embeddings_provider: Any) -> None:
		ids = []
		docs = []
		metadatas = []
		for ch in chunks:
			ids.append(ch.chunk_id)
			docs.append(ch.text)
			md = ch.metadata or {}
			# important: chroma requires list-valued metadata fields to be non-empty lists
			if isinstance(md.get("skills"), list) and not md.get("skills"):
				md["skills"] = ["general"]
			metadatas.append(md)

		# Batch-encode all documents once to utilize GPU batching
		embeddings = []
		if embeddings_provider:
			# embeddings_provider.encode accepts an iterable of texts and will
			# internally batch according to its configured batch size.
			embeddings = embeddings_provider.encode(docs)

		if embeddings:
			self._collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
		else:
			self._collection.add(ids=ids, documents=docs, metadatas=metadatas)

	def query(self, query_embedding: List[float], top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		# Chroma expects `where` to be None when no filtering is required.
		# Passing an empty dict causes chroma to validate an empty query
		# object and raise errors like: "have exactly one operator, got {}".
		where_param = where if where else None
		results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where_param)
		return results # type: ignore

	def get(self, limit: int = 100, offset: int = 0, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		return self._collection.get(limit=limit, offset=offset, where=where) # type: ignore

	def count(self) -> int:
		return self._collection.count()

	def stats(self) -> Dict[str, Any]:
		return {"collectionName": CHROMA_COLLECTION_NAME}

__all__ = ["RAGVectorStore"]
