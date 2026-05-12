"""ChromaDB-backed vector store utilities for the RAG pipeline."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import chromadb # type: ignore
from chromadb.api.models.Collection import Collection # type: ignore
from chromadb.config import Settings # type: ignore

from .schemas import ChunkRecord
from .settings import RAG_COLLECTION_NAME, RAG_VECTOR_DIR

logger = logging.getLogger(__name__)


class RAGVectorStore:
    """Thin wrapper around ChromaDB collection operations."""

    def __init__(self) -> None:
        persist_path = Path(RAG_VECTOR_DIR)
        persist_path.mkdir(parents=True, exist_ok=True)
        self.persist_path = persist_path
        self.client = self._create_client(persist_path)
        self.collection: Collection = self._get_or_create_collection()
        logger.info("RAG vector store ready at %s (%s)", persist_path, RAG_COLLECTION_NAME)

    def _create_client(self, persist_path: Path) -> chromadb.ClientAPI: # type: ignore
        settings = Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True)
        try:
            return chromadb.PersistentClient(path=str(persist_path), settings=settings)
        except Exception as exc:
            backup_path = Path(tempfile.gettempdir()) / "pbl5_rag_chroma" / f"{persist_path.name}_{int(time.time())}"
            logger.warning("Chroma store init failed at %s: %s. Switching to %s", persist_path, exc, backup_path)
            if persist_path.exists():
                try:
                    workspace_backup = persist_path.with_name(f"{persist_path.name}_corrupt_{int(time.time())}")
                    shutil.move(str(persist_path), str(workspace_backup))
                except Exception as move_exc:
                    logger.warning("Unable to move corrupt store %s: %s", persist_path, move_exc)
            backup_path.mkdir(parents=True, exist_ok=True)
            self.persist_path = backup_path
            return chromadb.PersistentClient(path=str(backup_path), settings=settings)

    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: Sequence[ChunkRecord], embeddings: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=[list(e) for e in embeddings],
            documents=documents,
            metadatas=metadatas, # type: ignore
        )
        return len(chunks)

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        ) # type: ignore

    def get(
        self,
        limit: int,
        offset: int = 0,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.collection.get(
            limit=limit,
            offset=offset,
            where=where,
            include=["documents", "metadatas"],
        )  # type: ignore

    def reset_collection(self) -> int:
        previous_count = self.collection.count()
        try:
            self.client.delete_collection(name=RAG_COLLECTION_NAME)
        except Exception:
            logger.debug("Collection %s did not exist during reset", RAG_COLLECTION_NAME)
        self.collection = self._get_or_create_collection()
        return previous_count

    def count(self) -> int:
        return self.collection.count()

    def stats(self) -> Dict[str, Any]:
        return {
            "collection": RAG_COLLECTION_NAME,
            "count": self.collection.count(),
            "persist_directory": str(self.persist_path),
        }
