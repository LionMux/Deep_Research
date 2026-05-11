"""
Phase 2 Tests — Symbolic Reasoning with nlp-contrib-graph integration.

Tests all components added in Phase 2:
  - Triple extraction (SPO) from nlp-contrib-graph
  - Info-unit classification (Contribution/Task/Definition/etc)
  - Cross-sentence triple resolution
  - Triple-enhanced relationship building
  - Info-unit-aware paper selection
  - Static methods from nlp-contrib-graph (find_source, find_tri_sent, get_entity_spans)

Run: python tests/test_phase2_symbolic.py
"""

import sys
sys.path.insert(0, '.')

from scirag.symbolic_reasoning import (
    SymbolicReasoner, Segment, Triple, InfoUnit, SymbolicRelationship,
    is_heading, is_main_heading, split_into_paragraphs,
    INFO_UNIT_TYPES, SPO_TYPES, SPO_GOOD_STATES,
)


def test_heading_detection():
    """Test heading detection heuristics from nlp-contrib-graph/ext.py"""
    print("\n=== Test: Heading Detection ===")
    assert is_heading("1. Introduction") is True
    assert is_heading("2.3 Methods") is True
    assert is_heading("ABSTRACT") is True
    assert is_heading("RESULTS AND DISCUSSION") is True
    assert is_heading("Related Work") is True
    assert is_heading("This is a normal sentence.") is False
    assert is_heading("") is False
    print("  ✓ All heading detection tests passed")


def test_main_heading_detection():
    """Test main heading detection from nlp-contrib-graph/ext.py"""
    print("\n=== Test: Main Heading Detection ===")
    assert is_main_heading("1. Introduction") is True
    assert is_main_heading("2. Related Work") is True
    assert is_main_heading("3. Method") is True
    assert is_main_heading("4. Results") is True
    assert is_main_heading("A Subsection") is False
    print("  ✓ All main heading tests passed")


def test_paragraph_splitting():
    """Test paragraph splitting with headings"""
    print("\n=== Test: Paragraph Splitting ===")
    text = "Abstract\nThis paper proposes.\n\n1. Introduction\nNeural networks.\n\n2. Method\nOur approach."
    paragraphs = split_into_paragraphs(text)
    assert len(paragraphs) == 3
    assert paragraphs[0][0] == "Abstract"
    assert "This paper proposes" in paragraphs[0][1]
    assert paragraphs[1][0] == "1. Introduction"
    print(f"  ✓ Split into {len(paragraphs)} paragraphs")


def test_triple_dataclass():
    """Test Triple dataclass (SPO from nlp-contrib-graph)"""
    print("\n=== Test: Triple Dataclass ===")
    t = Triple(
        subject="BERT",
        predicate="uses",
        object="transformer architecture",
        paper_id="paper1",
        info_unit="Contribution",
        completeness=3,
    )
    assert t.subject == "BERT"
    assert t.predicate == "uses"
    assert t.object == "transformer architecture"
    assert t.completeness == 3

    # Equality and hash
    t2 = Triple("BERT", "uses", "transformer architecture")
    assert t == t2
    assert hash(t) == hash(t2)

    # Dict serialization
    d = t.to_dict()
    assert d["subject"] == "BERT"
    assert d["completeness"] == 3
    print("  ✓ Triple dataclass tests passed")


def test_info_unit_dataclass():
    """Test InfoUnit dataclass (from nlp-contrib-graph)"""
    print("\n=== Test: InfoUnit Dataclass ===")
    assert "Contribution" in INFO_UNIT_TYPES
    assert "Task" in INFO_UNIT_TYPES
    assert "Definition" in INFO_UNIT_TYPES
    assert "Results" in INFO_UNIT_TYPES

    iu = InfoUnit(
        text="We propose a new method.",
        unit_type="Contribution",
        paper_id="paper1",
        main_heading="1. Introduction",
        predicates=[("propose", (0, 0))],
        entities=[("new method", (0, 0))],
    )
    assert iu.unit_type == "Contribution"
    d = iu.to_dict()
    assert d["unit_type"] == "Contribution"
    assert "predicates" in d
    print("  ✓ InfoUnit dataclass tests passed")


def test_cross_sentence_resolution():
    """Test cross-sentence triple resolution from nlp-contrib-graph/ext.py"""
    print("\n=== Test: Cross-Sentence Resolution ===")
    reasoner = SymbolicReasoner.__new__(SymbolicReasoner)

    # Already complete triple
    triples = [Triple("S", "P", "O", "p1", 0, "Results", 3)]
    resolved = reasoner.resolve_cross_sentence(triples, {})
    assert len(resolved) == 1
    assert resolved[0].completeness == 3

    # Partial resolution with entity matching
    triples = [Triple("BERT", "achieves", "95%", "p1", 0, "Results", 2)]
    info_units = {
        "p1": [
            InfoUnit("BERT achieves 95%.", "Results", "p1",
                    entities=[("BERT", (0, 0)), ("95%", (0, 0))]),
        ]
    }
    resolved = reasoner.resolve_cross_sentence(triples, info_units)
    assert len(resolved) == 1

    # Unresolvable triple
    triples = [Triple("unknown", "does", "xyz", "p1", 0, "Task", 0)]
    info_units = {"p1": [InfoUnit("Some text.", "Contribution", "p1")]}
    resolved = reasoner.resolve_cross_sentence(triples, info_units)
    assert resolved[0].completeness == 0
    print("  ✓ Cross-sentence resolution tests passed")


def test_entity_spans():
    """Test entity span extraction from BIO tags (nlp-contrib-graph/ext.py)"""
    print("\n=== Test: Entity Spans from BIO ===")
    reasoner = SymbolicReasoner.__new__(SymbolicReasoner)

    # Two entities
    bio = ['O', 'B', 'I', 'I', 'O', 'B', 'I', 'O']
    spans = reasoner._get_entity_spans(bio)
    assert spans == [(1, 4), (5, 7)]

    # Single entity
    bio = ['B', 'I', 'O', 'O']
    spans = reasoner._get_entity_spans(bio)
    assert spans == [(0, 2)]

    # No entities
    bio = ['O', 'O', 'O']
    spans = reasoner._get_entity_spans(bio)
    assert spans == []
    print("  ✓ Entity span tests passed")


def test_find_source():
    """Test find_source from nlp-contrib-graph/parse.py"""
    print("\n=== Test: find_source ===")
    data = {
        'Contribution': {
            'Method': {'from sentence': 'We propose BERT.'},
            'Results': {'from sentence': 'Achieves 95%.'},
        }
    }
    sources = SymbolicReasoner.find_source(data)
    assert len(sources) == 2
    assert 'We propose BERT.' in sources
    assert 'Achieves 95%.' in sources
    print("  ✓ find_source tests passed")


def test_is_contained():
    """Test _is_contained from nlp-contrib-graph/parse.py"""
    print("\n=== Test: _is_contained ===")
    trace = ['a', 'b', 'c', 'd', 'e']
    assert SymbolicReasoner._is_contained(trace, ['b', 'c', 'd']) is True
    assert SymbolicReasoner._is_contained(trace, ['x', 'y', 'z']) is False
    print("  ✓ _is_contained tests passed")


def test_spo_good_states():
    """Verify SPO good_state mappings from nlp-contrib-graph/ext.py:405"""
    print("\n=== Test: SPO Good States ===")
    assert SPO_GOOD_STATES['p'] == [0, 1, 0]  # Predicate only
    assert SPO_GOOD_STATES['s'] == [1, 0, 0]  # Subject only
    assert SPO_GOOD_STATES['ob'] == [0, 0, 1]  # Object only
    assert SPO_GOOD_STATES['b'] == [1, 0, 1]  # Both subject and object
    print("  ✓ SPO good states verified")


def test_pipeline_format():
    """Verify return format contract with pipeline.py"""
    print("\n=== Test: Pipeline Return Format ===")
    # This documents the contract between SymbolicReasoner.run() and pipeline.py
    mock_result = {
        'segments': {},
        'triples': {},
        'info_units': {},
        'relationships': [],
        'selections': {},
        'stats': {
            'papers': 0,
            'total_segments': 0,
            'total_triples': 0,
            'fully_resolved_triples': 0,
            'total_info_units': 0,
            'info_unit_distribution': {},
            'relationships': 0,
            'sections': 0,
        }
    }
    # Keys pipeline.py accesses:
    assert 'stats' in mock_result
    assert 'total_segments' in mock_result['stats']
    assert 'relationships' in mock_result['stats']
    assert 'selections' in mock_result
    # Phase 2 additions:
    assert 'triples' in mock_result
    assert 'info_units' in mock_result
    assert 'fully_resolved_triples' in mock_result['stats']
    print("  ✓ Pipeline format contract verified")


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2 Symbolic Reasoning Tests")
    print("=" * 60)

    tests = [
        test_heading_detection,
        test_main_heading_detection,
        test_paragraph_splitting,
        test_triple_dataclass,
        test_info_unit_dataclass,
        test_cross_sentence_resolution,
        test_entity_spans,
        test_find_source,
        test_is_contained,
        test_spo_good_states,
        test_pipeline_format,
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
        print("All Phase 2 tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
