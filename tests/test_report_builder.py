"""Tests for the structured report builder."""

from scirag.report_builder import (
    ReportBuilder,
    confidence_label,
    renumber_citations,
)


def test_confidence_label_thresholds():
    assert confidence_label(0.95) == "high"
    assert confidence_label(0.6) == "medium"
    assert confidence_label(0.2) == "low"
    assert confidence_label(None) == "unknown"


def test_renumber_by_first_appearance():
    text = "A [dpr]. B [bm25]. C [dpr] again. D [reranker]."
    new_text, refs = renumber_citations(text)
    assert "[1]" in new_text and "[2]" in new_text and "[3]" in new_text
    # dpr first -> 1, bm25 -> 2, reranker -> 3; dpr reused as 1.
    assert new_text == "A [1]. B [2]. C [1] again. D [3]."
    assert [(r.number, r.key) for r in refs] == [(1, "dpr"), (2, "bm25"), (3, "reranker")]


def test_unsupported_flag_preserved():
    text = "Solid claim [dpr]. Shaky claim [unsupported]."
    new_text, refs = renumber_citations(text)
    assert "[unsupported]" in new_text
    assert "[1]" in new_text
    assert len(refs) == 1


def test_title_map_used_in_references():
    text = "Claim [dpr]."
    _, refs = renumber_citations(text, title_map={"dpr": "Dense Passage Retrieval"})
    assert refs[0].title == "Dense Passage Retrieval"
    assert "Dense Passage Retrieval" in refs[0].render()
    assert "`dpr`" in refs[0].render()


def test_build_full_report():
    body = "## Intro\n\nDense retrieval beats BM25 [dpr]. Unverified bit [unsupported]."
    verification = {"total": 2, "supported": 1, "unsupported": 1, "faithfulness": 0.5}
    out = ReportBuilder().build(
        query="dense vs sparse retrieval",
        body_text=body,
        title_map={"dpr": "Dense Passage Retrieval"},
        verification=verification,
        tree_confidence=0.7,
    )
    md = out["markdown"]
    assert md.startswith("# Research Report: dense vs sparse retrieval")
    assert "**Confidence:** medium" in md
    assert "faithfulness 50%" in md
    assert "## References" in md
    assert "1. Dense Passage Retrieval (`dpr`)" in md
    assert out["num_references"] == 1
    assert out["confidence"] == "medium"
    # Unsupported caution note present.
    assert "unsupported" in md.lower()


def test_build_report_no_citations():
    out = ReportBuilder().build(query="q", body_text="A plain sentence with no cites.")
    assert out["num_references"] == 0
    assert "No sources were cited" in out["markdown"]
    assert out["confidence"] == "unknown"
