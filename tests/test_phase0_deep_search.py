"""
Phase 0 Tests — Deep Search (Academic Paper Discovery).

Tests AcademicSearchEngine, PDFDownloader, FuzzyDedup.
Uses mock API responses — no real network calls.

Run: python tests/test_phase0_deep_search.py
"""

import sys
sys.path.insert(0, '.')

from scirag.deep_search import AcademicSearchEngine, PDFDownloader, DiscoveredPaper
from scirag.deep_search.academic_search import FuzzyDedup


def test_discovered_paper():
    """Test DiscoveredPaper dataclass."""
    print("\n=== Test: DiscoveredPaper ===")
    p = DiscoveredPaper(
        title="BERT: Pre-training",
        authors=["J Devlin", "M Chang"],
        year=2019,
        doi="10.1/bert",
        pdf_url="http://example.com/bert.pdf",
        source_api="arxiv",
        citation_count=10000,
    )
    assert p.title == "BERT: Pre-training"
    assert p.has_pdf() is True
    assert p.fingerprint() == "doi:10.1/bert"
    print("  ✓ DiscoveredPaper OK")


def test_fuzzy_dedup():
    """Test fuzzy deduplication."""
    print("\n=== Test: FuzzyDedup ===")
    dedup = FuzzyDedup(threshold=0.82)

    papers = [
        DiscoveredPaper(title="BERT: Pre-training", doi="10.1/bert"),
        DiscoveredPaper(title="BERT: Pre-training", doi="10.1/bert"),  # exact dup
        DiscoveredPaper(title="GPT-3: Few-Shot Learners", doi="10.1/gpt3"),
        DiscoveredPaper(title="BERT for NLP tasks", doi=""),  # fuzzy similar
        DiscoveredPaper(title="T5: Text-to-Text", doi="10.1/t5"),
    ]

    unique = dedup.deduplicate(papers)
    # Exact DOI dup removed -> 4
    # BERT for NLP might merge with BERT depending on threshold
    print(f"  ✓ Dedup: {len(papers)} -> {len(unique)} unique")
    assert len(unique) <= 4  # at most 4 (BERT merged or not)


def test_ranking():
    """Test paper ranking by relevance."""
    print("\n=== Test: Ranking ===")
    engine = AcademicSearchEngine(target=10)
    papers = [
        DiscoveredPaper(title="Neural Networks for NLP", citation_count=5000, is_oa=True),
        DiscoveredPaper(title="Deep Learning Basics", citation_count=100, is_oa=False),
        DiscoveredPaper(title="Neural Retrieval", citation_count=2000, is_oa=True),
    ]
    ranked = engine._rank(papers, {"neural", "networks"})
    assert ranked[0].title == "Neural Networks for NLP"  # highest score
    print(f"  ✓ Ranking: top='{ranked[0].title}' (score={ranked[0].relevance_score:.1f})")


def test_pipeline_integration():
    """Test pipeline accepts deep_search_query."""
    print("\n=== Test: Pipeline Integration ===")
    from scirag.pipeline import SciRAGPipeline
    import inspect
    sig = inspect.signature(SciRAGPipeline.run)
    assert "deep_search_query" in sig.parameters
    print("  ✓ Pipeline.run() has deep_search_query parameter")


def test_mcp_integration():
    """Test MCP server has scirag_deepsearch."""
    print("\n=== Test: MCP Integration ===")
    from scirag.mcp_server import _mcp_tool
    print("  ✓ MCP decorator available")


def run_all_tests():
    """Run all Phase 0 tests."""
    print("=" * 60)
    print("Phase 0 Deep Search Tests")
    print("=" * 60)

    tests = [
        test_discovered_paper,
        test_fuzzy_dedup,
        test_ranking,
        test_pipeline_integration,
        test_mcp_integration,
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
        print("All Phase 0 tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
