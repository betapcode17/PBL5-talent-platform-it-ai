"""Embedding service (relocated to infra package)."""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np # type: ignore
import torch  # type: ignore
from sentence_transformers import SentenceTransformer   # type: ignore

from ..schemas_and_settings.settings import EMBED_BATCH_SIZE, EMBED_DEVICE, EMBED_MODEL_NAME, HF_LOCAL_FILES_ONLY, QWEN_ENFORCE_CUDA

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Encodes text into dense vectors with optional caching (lazy-loaded)."""

    def __init__(self) -> None:
        self.device = self._pick_device()
        if QWEN_ENFORCE_CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is False")

        self._model: Optional[SentenceTransformer] = None  # Lazy load
        self.batch_size = EMBED_BATCH_SIZE if self.device == "cuda" else min(EMBED_BATCH_SIZE, 8)
        self._cache: Dict[str, np.ndarray] = {}
        logger.info("Embedding service initialized on %s (lazy-loading model on first use)", self.device)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model on first access."""
        if self._model is None:
            logger.info("rag.model_load.embedding.begin model=%s device=%s", EMBED_MODEL_NAME, self.device)
            try:
                self._model = SentenceTransformer(EMBED_MODEL_NAME, device=self.device, local_files_only=HF_LOCAL_FILES_ONLY)
            except Exception as exc:
                logger.exception("rag.model_load.embedding.failed model=%s device=%s error=%s", EMBED_MODEL_NAME, self.device, exc)
                raise
            logger.info("rag.model_load.embedding.ok model=%s device=%s", EMBED_MODEL_NAME, self.device)
        return self._model

    @staticmethod
    def _pick_device() -> str:
        if EMBED_DEVICE in {"cpu", "cuda"}:
            return EMBED_DEVICE

        if not torch.cuda.is_available():
            return "cpu"

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram_gb < 2:
            logger.warning(
                "Embedding model will run on CPU because detected GPU VRAM is only %.2f GiB",
                total_vram_gb,
            )
            return "cpu"
        return "cuda"

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        outputs: List[np.ndarray] = []
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        for idx, text in enumerate(text_list):
            key = self._cache_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                outputs.append(cached)
            else:
                outputs.append(np.array([], dtype=np.float32))
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            encoded = self.model.encode(
                miss_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for i, emb in enumerate(encoded):
                idx = miss_indices[i]
                outputs[idx] = emb.astype(np.float32)
                self._cache[self._cache_key(text_list[idx])] = outputs[idx]

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [arr.tolist() for arr in outputs]
