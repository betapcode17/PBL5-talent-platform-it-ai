#!/usr/bin/env python3
"""Simple validation script for retrieval improvements (no pytest required)."""

import os
import sys


def validate_all():
    """Validate all retrieval improvements."""
    print("\n" + "="*70)
    print("RETRIEVAL IMPROVEMENTS VALIDATION")
    print("="*70)
    
    # Test 1: Import rerank presets
    print("\n[Test 1] Rerank Presets Import")
    print("-" * 70)
    try:
        from app.services.rag.rerank_presets import get_weights, PRESETS
        print("✓ rerank_presets module imported successfully")
        print(f"✓ Available presets: {', '.join(PRESETS.keys())}")
    except Exception as e:
        print(f"✗ Failed to import rerank_presets: {e}")
        return False
    
    # Test 2: Validate weight presets
    print("\n[Test 2] Weight Presets Validation")
    print("-" * 70)
    for preset_name in ["balanced", "semantic-first", "bm25-first", "hybrid"]:
        try:
            weights = get_weights(preset_name)
            total = weights.total_weight
            status = "✓" if abs(total - 1.0) < 0.01 else "✗"
            print(f"{status} {preset_name:20} | S:{weights.semantic_weight:.2f} BM25:{weights.bm25_weight:.2f} Total:{total:.4f}")
        except Exception as e:
            print(f"✗ {preset_name}: {e}")
            return False
    
    # Test 3: BM25 library availability
    print("\n[Test 3] BM25 Library Availability")
    print("-" * 70)
    try:
        from rank_bm25 import BM25Okapi
        print("✓ rank-bm25 installed and importable")
        
        # Quick BM25 test
        corpus = [["python", "developer"], ["java", "engineer"], ["python", "scientist"]]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(["python"])
        print(f"✓ BM25 scoring works | Scores: {[f'{s:.2f}' for s in scores]}")
    except ImportError:
        print("✗ rank-bm25 NOT installed")
        print("  Install with: pip install rank-bm25")
        return False
    except Exception as e:
        print(f"✗ BM25 error: {e}")
        return False
    
    # Test 4: RetrievedChunk mutability
    print("\n[Test 4] RetrievedChunk Mutability")
    print("-" * 70)
    try:
        from app.services.rag.schemas import RetrievedChunk
        chunk = RetrievedChunk(
            chunk_id="test",
            text="Sample",
            metadata={},
            distance=0.1,
            rerank_score=0.9,
            source_key="s1",
        )
        chunk.cross_encoder_score = 0.95
        assert chunk.cross_encoder_score == 0.95
        print("✓ RetrievedChunk is mutable")
        print(f"✓ cross_encoder_score can be set: {chunk.cross_encoder_score}")
    except Exception as e:
        print(f"✗ RetrievedChunk mutability failed: {e}")
        return False
    
    # Test 5: Cross-encoder optional support
    print("\n[Test 5] Cross-Encoder Support (Optional)")
    print("-" * 70)
    try:
        from app.services.rag.cross_encoder import CrossEncoderReranker
        print("✓ CrossEncoderReranker module imported")
        
        reranker = CrossEncoderReranker()
        print(f"  - Available: {reranker.available}")
        print(f"  - Optional dependency: {'sentence-transformers' if not reranker.available else 'loaded'}")
    except Exception as e:
        print(f"✗ Cross-encoder module error: {e}")
        return False
    
    # Test 6: Settings with new config
    print("\n[Test 6] Settings Configuration")
    print("-" * 70)
    try:
        from app.services.rag.settings import (
            RAG_RERANK_PRESET,
            RAG_SEMANTIC_WEIGHT,
            RAG_BM25_WEIGHT,
            RAG_USE_CROSS_ENCODER,
        )
        print(f"✓ RAG_RERANK_PRESET={RAG_RERANK_PRESET}")
        print(f"  - RAG_SEMANTIC_WEIGHT={RAG_SEMANTIC_WEIGHT}")
        print(f"  - RAG_BM25_WEIGHT={RAG_BM25_WEIGHT}")
        print(f"  - RAG_USE_CROSS_ENCODER={RAG_USE_CROSS_ENCODER}")
    except Exception as e:
        print(f"✗ Settings import error: {e}")
        return False
    
    # Test 7: Retrieval service initialization (mock)
    print("\n[Test 7] RAG Retrieval Service Initialization")
    print("-" * 70)
    try:
        from unittest.mock import MagicMock
        from app.services.rag.retrieval import RAGRetrievalService
        
        mock_vs = MagicMock()
        mock_emb = MagicMock()
        
        service = RAGRetrievalService(mock_vs, mock_emb)
        assert hasattr(service, 'weights')
        print(f"✓ RAGRetrievalService initialized with weights")
        print(f"  - Preset loaded: {service.weights.semantic_weight:.2f} semantic")
        print(f"  - BM25 weight: {service.weights.bm25_weight:.2f}")
    except Exception as e:
        print(f"✗ Retrieval service init error: {e}")
        return False
    
    # Test 8: Documentation exists
    print("\n[Test 8] Documentation")
    print("-" * 70)
    doc_path = "docs/RETRIEVAL_IMPROVEMENTS.md"
    if os.path.exists(doc_path):
        print(f"✓ {doc_path} created")
        with open(doc_path) as f:
            lines = len(f.readlines())
        print(f"  - {lines} lines of documentation")
    else:
        print(f"✗ {doc_path} not found")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE ✓")
    print("="*70)
    print("\nNext steps:")
    print("  1. Test with actual queries:")
    print("     python -m pytest test_chatbot_message.py -v")
    print("\n  2. Try different presets:")
    print("     RAG_RERANK_PRESET=semantic-first python app/main.py")
    print("     RAG_RERANK_PRESET=bm25-first python app/main.py")
    print("\n  3. Monitor retrieval quality:")
    print("     tail -f logs/rag_retrieval_metrics.log")
    print()
    return True


if __name__ == "__main__":
    try:
        success = validate_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
