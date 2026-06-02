"""Tests for the evaluation harness (metrics + retrieval runner)."""

from pathlib import Path

from scirag.claim_verifier import ClaimVerifier
from scirag.evaluation import (
    Benchmark,
    citation_precision,
    evaluate_retrieval,
    faithfulness,
    mrr,
    precision_at_k,
    recall_at_k,
)

BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "retrieval_benchmark.json"
)


def test_recall_and_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["b", "z"], k=4) == 0.5
    assert recall_at_k(retrieved, ["a"], k=1) == 1.0
    assert recall_at_k(retrieved, ["d"], k=2) == 0.0
    assert precision_at_k(retrieved, ["a", "b"], k=2) == 1.0
    assert precision_at_k(retrieved, ["a"], k=2) == 0.5


def test_mrr():
    assert mrr(["a", "b", "c"], ["b"]) == 0.5
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    assert mrr(["a", "b", "c"], ["z"]) == 0.0


def test_benchmark_loads():
    b = Benchmark.from_json(str(BENCHMARK_PATH))
    assert b.corpus == "samples/papers"
    assert len(b.queries) == 3
    assert b.queries[0].relevant_doc_ids


def test_evaluate_retrieval_sparse_is_perfect_and_deterministic():
    b = Benchmark.from_json(str(BENCHMARK_PATH))
    # Resolve corpus relative to repo root regardless of CWD.
    corpus = str(Path(__file__).resolve().parent.parent / "samples" / "papers")
    res1 = evaluate_retrieval(b, corpus_dir=corpus, mode="sparse", top_k=3)
    res2 = evaluate_retrieval(b, corpus_dir=corpus, mode="sparse", top_k=3)
    # BM25 lexical retrieval should put each gold doc in the top results.
    assert res1["recall@3"] == 1.0
    assert res1["mrr"] == 1.0
    assert res1 == res2  # deterministic


def test_citation_precision_and_faithfulness():
    passages = [
        {"document_id": "dpr", "text": "dense passage retrieval improves accuracy over bm25"},
    ]
    cv = ClaimVerifier(min_support=0.35)
    report = cv.verify_text(
        "Dense passage retrieval improves accuracy over BM25 [dpr]. "
        "Unrelated nonsense claim about lunar cheese mining [dpr].",
        passages,
    )
    # One cited sentence supported, one cited but unsupported -> precision 0.5.
    assert citation_precision(report) == 0.5
    assert 0.0 <= faithfulness(report) <= 1.0
