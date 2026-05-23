"""Test retrieval improvements: BM25, configurable weights, cross-encoder."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
except ImportError:  # pragma: no cover - fallback for local manual run
    pytest = None

from app.services.rag.rerank_presets import (
    BALANCED_WEIGHTS,
    BM25_FIRST_WEIGHTS,
    HYBRID_STRICT_WEIGHTS,
    SEMANTIC_FIRST_WEIGHTS,
    get_weights,
)
from app.services.rag.score_and_boost.metadata_boost import build_boost_plan, compute_metadata_scores
from app.services.rag.score_and_boost.scoring import ScoreComponents, compose_hybrid_score
from app.services.rag.schemas import RetrievedChunk


class TestRerankWeights:
    """Test reranking weight presets."""
    
    def test_balanced_preset(self):
        """BALANCED preset should be default."""
        weights = get_weights("balanced")
        assert abs(weights.total_weight - 1.0) < 0.01
        assert weights.semantic_weight == 0.35
        assert weights.bm25_weight == 0.30
    
    def test_semantic_first_preset(self):
        """SEMANTIC_FIRST preset prioritizes vector search."""
        weights = get_weights("semantic-first")
        assert weights.semantic_weight > weights.bm25_weight
        assert weights.semantic_weight == 0.50
    
    def test_bm25_first_preset(self):
        """BM25_FIRST preset prioritizes keyword matching."""
        weights = get_weights("bm25-first")
        assert weights.bm25_weight > weights.semantic_weight
        assert weights.bm25_weight == 0.50
    
    def test_hybrid_strict_preset(self):
        """HYBRID_STRICT preset has equal semantic + BM25."""
        weights = get_weights("hybrid")
        assert abs(weights.semantic_weight - weights.bm25_weight) < 0.01
    
    def test_weights_normalize(self):
        """All presets should normalize to ~1.0 total weight."""
        for preset_name in ["balanced", "semantic-first", "bm25-first", "hybrid"]:
            weights = get_weights(preset_name)
            assert abs(weights.total_weight - 1.0) < 0.01, f"{preset_name} total weight: {weights.total_weight}"
    
    def test_unknown_preset_defaults_to_balanced(self):
        """Unknown preset should fall back to BALANCED."""
        weights = get_weights("unknown-preset")
        assert abs(weights.total_weight - 1.0) < 0.01


class TestRetrievedChunkMutable:
    """Test that RetrievedChunk is now mutable for cross-encoder support."""
    
    def test_retrieved_chunk_is_mutable(self):
        """RetrievedChunk should be mutable (not frozen)."""
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="Sample job posting",
            metadata={"title": "Python Dev"},
            distance=0.2,
            rerank_score=0.85,
            source_key="job_123",
        )
        
        # Should be able to set cross_encoder_score after creation
        chunk.cross_encoder_score = 0.92
        assert chunk.cross_encoder_score == 0.92
    
    def test_retrieved_chunk_default_cross_encoder_none(self):
        """cross_encoder_score should default to None."""
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="Sample job posting",
            metadata={"title": "Python Dev"},
            distance=0.2,
            rerank_score=0.85,
            source_key="job_123",
        )
        
        assert chunk.cross_encoder_score is None


class TestBM25Integration:
    """Test BM25 integration in retrieval service."""
    
    def test_bm25_import(self):
        """rank-bm25 should be importable."""
        try:
            from rank_bm25 import BM25Okapi
            assert BM25Okapi is not None
        except ImportError:
            if pytest is not None:
                pytest.skip("rank-bm25 not installed; run: pip install rank-bm25")
            return
    
    def test_bm25_scoring(self):
        """Test BM25 scoring on sample corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            if pytest is not None:
                pytest.skip("rank-bm25 not installed")
            return
        
        # Create sample corpus
        corpus = [
            ["python", "developer", "role"],
            ["java", "backend", "engineer"],
            ["python", "data", "scientist"],
        ]
        
        # Create BM25 index
        bm25 = BM25Okapi(corpus)
        
        # Query
        query_terms = ["python", "developer"]
        scores = bm25.get_scores(query_terms)
        
        # First document should have highest score
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]


class TestCrossEncoderImport:
    """Test cross-encoder optional dependency."""
    
    def test_cross_encoder_optional(self):
        """Cross-encoder should be optional."""
        from app.services.rag.cross_encoder import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        # Should not raise error, but available might be False if not installed
        assert isinstance(reranker.available, bool)


class TestRetreivalServiceInitialization:
    """Test RAGRetrievalService initializes with new weights."""
    
    @patch('app.services.rag.retrieval.RAGVectorStore')
    @patch('app.services.rag.retrieval.EmbeddingService')
    def test_retrieval_service_loads_weights(self, mock_embedding, mock_vector_store):
        """RAGRetrievalService should load weights from preset."""
        from app.services.rag.retrieval import RAGRetrievalService
        
        service = RAGRetrievalService(mock_vector_store, mock_embedding)
        
        # Should have loaded weights
        assert hasattr(service, 'weights')
        assert service.weights is not None
        assert abs(service.weights.total_weight - 1.0) < 0.01


class TestJobSignalScoring:
    """Test backend-shaped job fields contribute to job-search ranking."""

    def test_backend_fields_score_location_salary_job_type_and_recency(self):
        terms = {"backend", "python", "hcm", "luong", "fulltime"}
        metadata = {
            "title": "Backend Python Engineer",
            "company": "ACME",
            "description": "Build APIs for a fast-growing product team.",
            "location": "Ho Chi Minh City",
            "salary": "25-35m",
            "job_type": "Toàn thời gian",
            "updated_at": "2026-04-19T14:23:28.203Z",
        }

        scores = compute_metadata_scores(terms, metadata, build_boost_plan(terms))

        assert scores.location > 0.0
        assert scores.salary > 0.0
        assert scores.job_type > 0.0
        assert scores.recency > 0.0

        rerank = compose_hybrid_score(
            {
                "semantic": 1.0,
                "bm25": 0.0,
                "title": 0.4,
                "company": 0.1,
                "description": 0.3,
                "location": 0.5,
                "salary": 0.4,
                "job_type": 0.4,
                "recency": 0.3,
                "category": 0.0,
                "entity_bias": 0.0,
                "fulltext": 0.0,
            },
            ScoreComponents(
                semantic=0.55,
                bm25=0.0,
                title=scores.title,
                company=scores.company,
                description=scores.description,
                category=0.0,
                location=scores.location,
                salary=scores.salary,
                job_type=scores.job_type,
                recency=scores.recency,
                entity_bias=0.0,
                fulltext=0.0,
            ),
            include_fields=["semantic", "title", "description", "location", "salary", "job_type", "recency"],
        )

        assert rerank > 0.55

    def test_job_search_profile_exposes_backend_weights(self):
        from app.services.rag.retrieval import RAGRetrievalService

        profile = RAGRetrievalService._resolve_runtime_profile("backend python hcm", None)
        weights = profile.weights()

        assert weights["location"] > 0.0
        assert weights["salary"] > 0.0
        assert weights["job_type"] > 0.0
        assert weights["recency"] > 0.0


def run_manual_tests():
    """Run manual validation tests (print output for review)."""
    print("\n" + "="*60)
    print("RETRIEVAL IMPROVEMENTS VALIDATION")
    print("="*60)
    
    # Test 1: Weight presets
    print("\n[Test 1] Weight Presets")
    print("-" * 60)
    for preset_name in ["balanced", "semantic-first", "bm25-first", "hybrid"]:
        weights = get_weights(preset_name)
        print(f"\n{preset_name.upper()}:")
        print(f"  Semantic:  {weights.semantic_weight:.2f}")
        print(f"  BM25:      {weights.bm25_weight:.2f}")
        print(f"  Title:     {weights.title_weight:.2f}")
        print(f"  Company:   {weights.company_weight:.2f}")
        print(f"  Skills:    {weights.skills_weight:.2f}")
        print(f"  Category:  {weights.category_weight:.2f}")
        print(f"  Entity:    {weights.entity_bias_weight:.2f}")
        print(f"  TOTAL:     {weights.total_weight:.4f} ✓" if abs(weights.total_weight - 1.0) < 0.01 else f"  TOTAL:     {weights.total_weight:.4f} ✗")
    
    # Test 2: BM25 availability
    print("\n[Test 2] BM25 Library Availability")
    print("-" * 60)
    try:
        from rank_bm25 import BM25Okapi
        print("✓ rank-bm25 installed and importable")
        
        # Quick BM25 test
        corpus = [
            ["python", "developer"],
            ["java", "engineer"],
            ["python", "scientist"],
        ]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(["python", "developer"])
        print(f"✓ BM25 scoring works: {scores}")
    except ImportError:
        print("✗ rank-bm25 NOT installed. Install with: pip install rank-bm25")
    
    # Test 3: Cross-encoder availability
    print("\n[Test 3] Cross-Encoder Support")
    print("-" * 60)
    try:
        from sentence_transformers import CrossEncoder
        print("✓ sentence-transformers installed (cross-encoder available)")
    except ImportError:
        print("⚠ sentence-transformers might need update for cross-encoder support")
    
    # Test 4: Config from environment
    print("\n[Test 4] Configuration from Environment")
    print("-" * 60)
    preset = os.getenv("RAG_RERANK_PRESET", "balanced")
    print(f"RAG_RERANK_PRESET={preset}")
    print(f"RAG_SEMANTIC_WEIGHT={os.getenv('RAG_SEMANTIC_WEIGHT', '(not set)')}")
    print(f"RAG_BM25_WEIGHT={os.getenv('RAG_BM25_WEIGHT', '(not set)')}")
    print(f"RAG_USE_CROSS_ENCODER={os.getenv('RAG_USE_CROSS_ENCODER', 'false')}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Run pytest if called with pytest
    if len(sys.argv) > 1 and sys.argv[1] == "pytest":
        if pytest is None:
            print("pytest is not installed. Running manual tests instead.")
            run_manual_tests()
        else:
            pytest.main([__file__, "-v"])
    else:
        # Otherwise run manual tests
        run_manual_tests()
