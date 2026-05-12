"""Optional cross-encoder reranking for high-quality retrieval (requires extra GPU memory)."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from .schemas import RetrievedChunk
from .settings import CROSS_ENCODER_BATCH_SIZE, CROSS_ENCODER_MODEL, RAG_CROSS_ENCODER_THRESHOLD

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Optional cross-encoder reranking for improved result quality.
    
    Cross-encoders directly score query-document pairs, offering better quality than
    bi-encoders (semantic search) but at higher computational cost.
    
    Usage:
        reranker = CrossEncoderReranker()
        if reranker.available:
            items = reranker.rerank(query, items)
    """
    
    def __init__(self):
        """Initialize cross-encoder if available."""
        self.available = False
        self.model: Optional[CrossEncoder] = None # type: ignore
        self.device = "cuda"
        self.batch_size = CROSS_ENCODER_BATCH_SIZE
        
        if CrossEncoder is None:
            logger.warning("sentence-transformers not available; cross-encoder reranking disabled")
            return
        
        try:
            self.model = CrossEncoder(CROSS_ENCODER_MODEL, device=self.device)
            self.available = True
            logger.info("cross-encoder.initialized model=%s device=%s", CROSS_ENCODER_MODEL, self.device)
        except Exception as e:
            logger.warning("cross-encoder.init_failed error=%s", e)
            self.available = False
    
    def rerank(self, query: str, items: List[RetrievedChunk], top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Rerank items using cross-encoder scores.
        
        Args:
            query: Search query
            items: Retrieved items to rerank
            top_k: Optional limit on results (default: keep all)
            
        Returns:
            Reranked items, sorted by cross-encoder score
        """
        if not self.available or not items:
            return items
        
        start = time.perf_counter()
        
        # Prepare pairs: (query, doc_text) for each item
        pairs = [[query, item.text] for item in items]
        
        # Get cross-encoder scores
        try:
            ce_scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)  # type: ignore
            
            # Update items with cross-encoder score
            for i, item in enumerate(items):
                score = float(ce_scores[i]) if i < len(ce_scores) else 0.0
                # Normalize score to [0, 1] if needed
                item.cross_encoder_score = max(0.0, min(1.0, (score + 1) / 2))  # Assuming [-1, 1] range
            
            # Sort by cross-encoder score
            reranked = sorted(items, key=lambda x: x.cross_encoder_score, reverse=True) # type: ignore
            
            # Filter by threshold if specified
            if RAG_CROSS_ENCODER_THRESHOLD > 0:
                reranked = [x for x in reranked if x.cross_encoder_score >= RAG_CROSS_ENCODER_THRESHOLD]
            
            # Apply top_k limit
            if top_k:
                reranked = reranked[:top_k]
            
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.debug(
                "cross-encoder.rerank items=%d filtered=%d kept=%d latency_ms=%s batch_size=%d",
                len(items),
                len(items) - len(reranked),
                len(reranked),
                latency_ms,
                self.batch_size,
            )
            return reranked
            
        except Exception as e:
            logger.error("cross-encoder.rerank_failed error=%s", e)
            return items
