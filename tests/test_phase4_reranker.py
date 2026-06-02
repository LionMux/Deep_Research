"""
Phase 4 Tests — BGE Reranker v2 Integration.

Tests reranker integration without downloading actual models:
  - SciRAGReranker initialization and config
  - Score computation logic (mocked)
  - Rerank pipeline
  - Integration with IterativeSynthesizer
  - Integration with SciRAGPipeline

Run: python tests/test_phase4_reranker.py
"""

import sys

sys.path.insert(0, '.')

from scirag.config import SciRAGConfig
from scirag.reranker import SciRAGReranker


def test_reranker_init():
    """Test reranker initialization without loading model."""
    print("\n=== Test: Reranker Init ===")
    r = SciRAGReranker(
        model_name="BAAI/bge-reranker-base",
        device="cpu",
        batch_size=4,
        max_length=256,
        normalize=True,
    )
    assert r.model_name == "BAAI/bge-reranker-base"
    assert r.device == "cpu"
    assert r.batch_size == 4
    assert r.max_length == 256
    assert r.normalize is True
    assert r._loaded is False
    print("  ✓ Reranker init OK")


def test_reranker_status():
    """Test status report."""
    print("\n=== Test: Reranker Status ===")
    r = SciRAGReranker()
    status = r.status()
    assert status["model_name"] == "BAAI/bge-reranker-base"
    assert status["device"] == "cpu"
    assert status["loaded"] is False
    print(f"  ✓ Status: {status}")


def test_sigmoid():
    """Test sigmoid normalization."""
    print("\n=== Test: Sigmoid ===")
    r = SciRAGReranker()
    # High score -> near 1
    assert r._sigmoid(10.0) > 0.99
    # Zero -> 0.5
    assert abs(r._sigmoid(0.0) - 0.5) < 0.01
    # Low score -> near 0
    assert r._sigmoid(-10.0) < 0.01
    print("  ✓ Sigmoid normalization OK")


def test_rerank_pipeline():
    """Test rerank pipeline with mocked scores."""
    print("\n=== Test: Rerank Pipeline ===")
    r = SciRAGReranker()

    # Mock compute_scores to avoid model loading
    r._loaded = True  # Pretend model is loaded
    def mock_compute_scores(query, docs):
        # Simple heuristic: longer docs get higher scores
        return [len(d) * 0.1 for d in docs]
    r.compute_scores = mock_compute_scores

    docs = [
        {"text": "Short doc.", "id": 1},
        {"text": "This is a much longer document with more content.", "id": 2},
        {"text": "Medium doc here.", "id": 3},
    ]

    ranked = r.rerank("test query", docs, top_n=2, text_key="text")
    assert len(ranked) == 2
    # Longest doc should be first
    assert ranked[0][0]["id"] == 2
    assert ranked[0][1] > ranked[1][1]
    print(f"  ✓ Rerank pipeline OK (top: id={ranked[0][0]['id']}, score={ranked[0][1]:.2f})")


def test_rerank_chunks():
    """Test chunk reranking with score injection."""
    print("\n=== Test: Rerank Chunks ===")
    r = SciRAGReranker()
    r._loaded = True
    def mock_compute_scores(query, docs):
        return [i * 0.5 for i in range(len(docs))]
    r.compute_scores = mock_compute_scores

    chunks = [
        {"text": "Doc 1", "doc_id": "a"},
        {"text": "Doc 2", "doc_id": "b"},
        {"text": "Doc 3", "doc_id": "c"},
    ]

    result = r.rerank_chunks("q", chunks, top_k=2)
    assert len(result) == 2
    assert "rerank_score" in result[0]
    # Scores should be descending
    assert result[0]["rerank_score"] >= result[1]["rerank_score"]
    print(f"  ✓ Chunk reranking OK (scores: {[c['rerank_score'] for c in result]})")


def test_config_integration():
    """Test config has reranker options."""
    print("\n=== Test: Config Integration ===")
    c = SciRAGConfig()
    assert hasattr(c, "reranker_enabled")
    assert hasattr(c, "reranker_model")
    assert hasattr(c, "reranker_batch_size")
    assert c.reranker_enabled is True
    assert c.reranker_model == "BAAI/bge-reranker-base"
    assert c.reranker_batch_size == 8
    print("  ✓ Config integration OK")


def test_synthesizer_accepts_reranker():
    """Test IterativeSynthesizer accepts reranker parameter."""
    print("\n=== Test: Synthesizer Reranker ===")
    # We can't init full synthesizer without API key, but we can test the signature
    import inspect

    from scirag.iterative_synthesizer import IterativeSynthesizer
    sig = inspect.signature(IterativeSynthesizer.__init__)
    params = list(sig.parameters.keys())
    assert "reranker" in params
    print("  ✓ Synthesizer accepts reranker parameter")


def test_pipeline_reranker_property():
    """Test pipeline has reranker property."""
    print("\n=== Test: Pipeline Reranker ===")
    from scirag.pipeline import SciRAGPipeline

    # Check that reranker property exists
    assert hasattr(SciRAGPipeline, "reranker")
    print("  ✓ Pipeline has reranker property")


def run_all_tests():
    """Run all Phase 4 tests."""
    print("=" * 60)
    print("Phase 4 BGE Reranker Tests")
    print("=" * 60)

    tests = [
        test_reranker_init,
        test_reranker_status,
        test_sigmoid,
        test_rerank_pipeline,
        test_rerank_chunks,
        test_config_integration,
        test_synthesizer_accepts_reranker,
        test_pipeline_reranker_property,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed == 0:
        print("All Phase 4 tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
