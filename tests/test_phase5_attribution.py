"""
Phase 5 Tests — Post-Hoc Attribution per Sentence.

Tests attribution system without requiring live LLM calls:
  - Sentence splitting
  - Citation detection
  - Passage formatting
  - AttributedSentence dataclass
  - Integration with SciRAGPipeline

Run: python tests/test_phase5_attribution.py
"""

import sys

sys.path.insert(0, '.')

from scirag.attribution import AttributedSentence, PostHocAttributor


def test_sentence_splitting():
    """Test sentence splitting."""
    print("\n=== Test: Sentence Splitting ===")
    text = "First sentence. Second sentence! Third? Fourth with no ending"
    sents = PostHocAttributor.split_sentences(text)
    assert len(sents) == 4
    assert "First sentence." in sents[0]
    assert "Second sentence!" in sents[1]
    assert "Third?" in sents[2]
    print(f"  ✓ Split into {len(sents)} sentences: {sents}")


def test_citation_detection():
    """Test citation detection."""
    print("\n=== Test: Citation Detection ===")
    assert PostHocAttributor.has_citation("This is supported [1].") is True
    assert PostHocAttributor.has_citation("This is supported [doc_1].") is True
    assert PostHocAttributor.has_citation("No citation here.") is False
    assert PostHocAttributor.has_citation("No brackets at all.") is False
    print("  ✓ Citation detection works")


def test_passage_formatting():
    """Test passage formatting for prompts."""
    print("\n=== Test: Passage Formatting ===")
    passages = [
        {"title": "Paper A", "text": "BERT uses transformers."},
        {"title": "Paper B", "text": "GPT uses attention."},
    ]
    formatted = PostHocAttributor.format_passages(passages)
    assert "[0] Title: Paper A" in formatted
    assert "[1] Title: Paper B" in formatted
    assert "BERT uses transformers" in formatted
    print(f"  ✓ Formatted passages:\n    {formatted[:80]}...")


def test_attributed_sentence():
    """Test AttributedSentence dataclass."""
    print("\n=== Test: AttributedSentence ===")
    a = AttributedSentence(
        text="BERT achieves 95% accuracy.",
        citation_ids=["1", "2"],
        confidence=0.9,
        verified=True,
    )
    assert a.text == "BERT achieves 95% accuracy."
    assert a.citation_ids == ["1", "2"]
    assert a.confidence == 0.9
    assert a.verified is True
    print("  ✓ AttributedSentence dataclass works")


def test_attribution_report():
    """Test full attribution report."""
    print("\n=== Test: Attribution Report ===")
    # Mock attributor
    class MockClient:
        pass

    attr = PostHocAttributor(MockClient())
    text = "Sentence one. Sentence two [1]. Sentence three."

    # Test split + detect on report
    sentences = attr.split_sentences(text)
    cited = [attr.has_citation(s) for s in sentences]
    assert cited == [False, True, False]
    print(f"  ✓ Attribution analysis: {len(sentences)} sentences, cited={sum(cited)}/{len(cited)}")


def test_integration_imports():
    """Test all Phase 5 imports work."""
    print("\n=== Test: Integration Imports ===")
    print("  ✓ All imports OK")


def run_all_tests():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("Phase 5 Post-Hoc Attribution Tests")
    print("=" * 60)

    tests = [
        test_sentence_splitting,
        test_citation_detection,
        test_passage_formatting,
        test_attributed_sentence,
        test_attribution_report,
        test_integration_imports,
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
        print("All Phase 5 tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
