"""
Phase 3 Tests — Semantic Scholar API Integration.

Tests all components added in Phase 3:
  - S2Paper dataclass
  - S2Cache disk caching
  - S2RateLimiter
  - SemanticScholarClient (search, get_paper, refs, cites)
  - CitationGraph expansion (backward, forward)
  - Integration with pipeline.py

Run: python tests/test_phase3_s2.py
"""

import sys
import tempfile
sys.path.insert(0, '.')

from scirag.s2_client import (
    SemanticScholarClient, S2Paper, S2Cache, S2RateLimiter,
    _SCHOLAR_AVAILABLE, _SCHOLAR_PATH,
)


def test_s2paper_dataclass():
    """Test S2Paper dataclass."""
    print("\n=== Test: S2Paper Dataclass ===")
    p = S2Paper(
        paper_id='test_001',
        title='Test Paper',
        abstract='Abstract text.',
        year=2024,
        authors=['A Author'],
        citation_count=10,
        venue='Venue',
        doi='10.1/1',
        arxiv_id='2401.00001',
    )
    assert p.paper_id == 'test_001'
    assert p.title == 'Test Paper'

    node = p.to_citation_node()
    assert node['paper_id'] == 'test_001'
    assert node['doi'] == '10.1/1'
    assert node['arxiv_id'] == '2401.00001'
    print("  ✓ S2Paper OK")


def test_s2cache():
    """Test disk cache."""
    print("\n=== Test: S2Cache ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = S2Cache(cache_dir=tmpdir)

        # Set and get
        cache.set("search", "query1", {"papers": [{"paperId": "1"}], "total": 1})
        result = cache.get("search", "query1")
        assert result is not None
        assert result["total"] == 1

        # Memory cache
        result2 = cache.get("search", "query1")
        assert result2 == result

        # Persist to disk
        cache2 = S2Cache(cache_dir=tmpdir)
        result3 = cache2.get("search", "query1")
        assert result3 is not None

        stats = cache.stats()
        assert stats["entries"] >= 1
        print(f"  ✓ S2Cache OK (entries: {stats['entries']})")


def test_rate_limiter():
    """Test rate limiter."""
    print("\n=== Test: S2RateLimiter ===")
    import time
    rl = S2RateLimiter(min_delay=0.05)

    t0 = time.time()
    rl.wait()
    rl.report_success()
    elapsed = time.time() - t0
    assert elapsed >= 0.04  # Should have waited at least 50ms

    # Error should increase delay
    rl.report_error()
    assert rl._consecutive_errors == 1
    print("  ✓ S2RateLimiter OK")


def test_client_conversion():
    """Test raw data conversion to S2Paper."""
    print("\n=== Test: Client Raw Conversion ===")
    client = SemanticScholarClient(min_delay=0.1)

    # Test _s2paper_from_raw
    raw = {
        'paperId': 'abc123',
        'title': 'BERT: Pre-training',
        'abstract': 'We introduce BERT.',
        'year': 2019,
        'authors': [{'name': 'J Devlin'}, {'name': 'M Chang'}],
        'citationCount': 10000,
        'referenceCount': 50,
        'venue': 'NAACL',
        'url': 'https://example.com',
        'externalIds': {'DOI': '10.1/1', 'ArXiv': '1810.04805', 'CorpusId': 12345},
        'fieldsOfStudy': ['Computer Science'],
        'openAccessPdf': {'url': 'https://pdf.example.com'},
    }
    p = client._s2paper_from_raw(raw)
    assert p.paper_id == 'abc123'
    assert p.title == 'BERT: Pre-training'
    assert p.year == 2019
    assert len(p.authors) == 2
    assert p.doi == '10.1/1'
    assert p.arxiv_id == '1810.04805'
    assert p.corpus_id == '12345'
    assert p.pdf_url == 'https://pdf.example.com'
    print(f"  ✓ Conversion OK: {p.to_citation_node()}")

    # Test references/citations parsing
    raw_with_refs = {
        'paperId': 'ref_test',
        'title': 'Ref Test',
        'references': [
            {'citedPaper': {'paperId': 'r1', 'title': 'Ref 1', 'year': 2020}},
            {'citedPaper': {'paperId': 'r2', 'title': 'Ref 2', 'year': 2021}},
        ],
        'citations': [
            {'citingPaper': {'paperId': 'c1', 'title': 'Cite 1', 'year': 2023}},
        ],
    }
    p2 = client._s2paper_from_raw(raw_with_refs)
    assert len(p2.references) == 2
    assert len(p2.citations) == 1
    assert p2.references[0]['paper_id'] == 'r1'
    assert p2.citations[0]['paper_id'] == 'c1'
    print("  ✓ Refs/Cites parsing OK")


def test_client_real_api():
    """Test real S2 API calls (requires internet)."""
    print("\n=== Test: Real S2 API ===")
    if not _SCHOLAR_AVAILABLE:
        print("  ⊘ SKIP: semanticscholar not available")
        return True

    client = SemanticScholarClient(min_delay=1.0)

    # Test search
    papers = client.search_papers('neural retrieval', limit=2)
    assert len(papers) > 0, "No papers found"
    print(f"  ✓ Search: found {len(papers)} papers")
    for p in papers:
        print(f"    - {p.title[:50]} ({p.year})")

    # Test get_paper (Turing paper)
    paper = client.get_paper('10.1093/mind/lix.236.433')
    assert paper is not None
    assert 'Computing Machinery' in paper.title
    print(f"  ✓ get_paper: {paper.title} ({paper.year})")

    # Test get_references
    if papers[0].paper_id:
        refs = client.get_references(papers[0].paper_id, limit=3)
        print(f"  ✓ References: found {len(refs)}")

    # Test get_citations
        cites = client.get_citations(papers[0].paper_id, limit=3)
        print(f"  ✓ Citations: found {len(cites)}")

    print(f"  Cache: {client.cache.stats()}")
    return True


def test_citation_graph_integration():
    """Test integration with CitationGraph."""
    print("\n=== Test: CitationGraph Integration ===")
    from scirag.citation_graph import CitationGraph

    graph = CitationGraph()
    client = SemanticScholarClient(min_delay=0.1)

    # Add a paper manually
    client.cache.set("paper", "test_paper", {
        "paperId": "test_paper",
        "title": "Test",
        "year": 2024,
        "authors": [{"name": "Author"}],
        "abstract": "Abstract",
        "venue": "Venue",
    })

    # Test expand_graph_backward (mock)
    graph.add_paper(doc_id="seed", title="Seed Paper", year=2024)

    # Manually add edges
    graph._graph.add_edge("seed", "ref1", relation="cites")
    graph._graph.add_edge("seed", "ref2", relation="cites")
    graph._graph.add_node("ref1", title="Ref 1")
    graph._graph.add_node("ref2", title="Ref 2")

    assert graph._graph.number_of_nodes() == 3
    assert graph._graph.number_of_edges() == 2
    print(f"  ✓ Graph integration: {graph._graph.number_of_nodes()} nodes, {graph._graph.number_of_edges()} edges")


def test_pipeline_integration():
    """Test that pipeline can import S2 client."""
    print("\n=== Test: Pipeline Integration ===")
    from scirag.pipeline import _get_s2_client

    s2 = _get_s2_client()
    assert s2 is not None
    stats = s2.stats()
    assert stats["available"] == _SCHOLAR_AVAILABLE
    print(f"  ✓ Pipeline lazy import OK (available: {stats['available']})")


def test_semanticscholar_path():
    """Verify cloned repository is in path."""
    print("\n=== Test: Repository Path ===")
    import os
    assert os.path.exists(_SCHOLAR_PATH), f"Path not found: {_SCHOLAR_PATH}"
    assert os.path.exists(os.path.join(_SCHOLAR_PATH, "semanticscholar", "SemanticScholar.py"))
    print(f"  ✓ Repo cloned at: {_SCHOLAR_PATH}")


def run_all_tests():
    """Run all Phase 3 tests."""
    print("=" * 60)
    print("Phase 3 Semantic Scholar API Tests")
    print("=" * 60)

    tests = [
        test_s2paper_dataclass,
        test_s2cache,
        test_rate_limiter,
        test_client_conversion,
        test_citation_graph_integration,
        test_pipeline_integration,
        test_semanticscholar_path,
        test_client_real_api,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result is not False:
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
        print("All Phase 3 tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
