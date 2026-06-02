"""Tests for deterministic per-sentence claim verification (fail-closed)."""

from scirag.claim_verifier import (
    UNSUPPORTED_FLAG,
    ClaimVerifier,
    split_sentences,
)

PASSAGES = [
    {
        "document_id": "dpr",
        "text": (
            "Dense passage retrieval learns embeddings with a dual encoder and "
            "improves retrieval accuracy over BM25 on open-domain question answering."
        ),
    },
    {
        "document_id": "bm25",
        "text": (
            "BM25 is a probabilistic lexical ranking function using term frequency "
            "and inverse document frequency over an inverted index."
        ),
    },
]


def test_split_sentences():
    s = split_sentences("First claim here. Second claim follows! Third?")
    assert len(s) == 3


def test_supported_sentence_detected():
    v = ClaimVerifier(min_support=0.35)
    report = v.verify_text(
        "Dense passage retrieval improves accuracy over BM25.", PASSAGES
    )
    assert report.total == 1
    assert report.supported == 1
    assert report.unsupported == 0
    assert report.faithfulness == 1.0
    assert report.sentences[0].best_doc_id == "dpr"


def test_unsupported_sentence_flagged():
    v = ClaimVerifier(min_support=0.35)
    report = v.verify_text(
        "Quantum entanglement enables faster than light teleportation networks.",
        PASSAGES,
    )
    assert report.total == 1
    assert report.unsupported == 1
    assert report.faithfulness == 0.0
    assert report.sentences[0].supported is False
    assert report.sentences[0].best_doc_id is None


def test_fail_closed_annotation_flags_unsupported():
    v = ClaimVerifier(min_support=0.35)
    text = (
        "Dense passage retrieval improves accuracy over BM25. "
        "Unicorns optimize photosynthesis via blockchain consensus."
    )
    report = v.verify_text(text, PASSAGES)
    annotated = v.annotate(report, fail_closed=True)
    # Unsupported sentence is flagged; supported one gets a citation.
    assert UNSUPPORTED_FLAG in annotated
    assert "[dpr]" in annotated
    assert annotated.count(UNSUPPORTED_FLAG) == 1


def test_fail_open_does_not_flag():
    v = ClaimVerifier(min_support=0.35)
    report = v.verify_text("Unicorns optimize photosynthesis via blockchain.", PASSAGES)
    annotated = v.annotate(report, fail_closed=False)
    assert UNSUPPORTED_FLAG not in annotated


def test_existing_citation_preserved():
    v = ClaimVerifier(min_support=0.35)
    report = v.verify_text(
        "Dense passage retrieval improves accuracy over BM25 [dpr].", PASSAGES
    )
    annotated = v.annotate(report)
    # Already cited -> not double-cited.
    assert annotated.count("[dpr]") == 1


def test_short_sentence_treated_as_non_claim():
    v = ClaimVerifier(min_support=0.35)
    report = v.verify_text("Results.", PASSAGES)
    # Too short to be a claim -> not counted, passes through.
    assert report.total == 0
    assert report.sentences[0].supported is True
