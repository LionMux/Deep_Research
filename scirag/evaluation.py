"""
Reproducible evaluation harness for SciRAG (Phase 6).

Provides small, dependency-light metric functions and a runner that measures
retrieval quality on a fixed benchmark so "quality" is measurable and regressions
are caught in CI.

Metrics
-------
- ``recall_at_k``    : fraction of relevant docs retrieved in the top-k.
- ``precision_at_k`` : fraction of top-k that are relevant.
- ``mrr``            : mean reciprocal rank of the first relevant hit.
- ``citation_precision`` / ``faithfulness`` : computed from a
  :class:`scirag.claim_verifier.VerificationReport`.

The retrieval runner defaults to ``sparse`` (BM25) mode so results are
deterministic and do not require embedding model downloads or API keys.

CLI
---
    python -m scirag.evaluation --benchmark benchmarks/retrieval_benchmark.json \
        --mode sparse --top-k 3
"""

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of ``relevant`` ids present in the first ``k`` ``retrieved`` ids."""
    rel = set(relevant)
    if not rel:
        return 0.0
    topk = list(retrieved)[:k]
    hits = sum(1 for r in rel if r in topk)
    return hits / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the first ``k`` ``retrieved`` ids that are relevant."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    topk = list(retrieved)[:k]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r in rel)
    return hits / len(topk)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant hit (0 if none)."""
    rel = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in rel:
            return 1.0 / (i + 1)
    return 0.0


def citation_precision(report) -> float:
    """Of sentences carrying a citation, the fraction that are supported.

    ``report`` is a :class:`scirag.claim_verifier.VerificationReport`.
    """
    cited = [s for s in report.sentences if s.cited_ids]
    if not cited:
        return 1.0
    supported = sum(1 for s in cited if s.supported)
    return supported / len(cited)


def faithfulness(report) -> float:
    """Fraction of citation-worthy claim sentences supported by some passage."""
    return report.faithfulness


# --------------------------------------------------------------------------- #
# Benchmark types
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkQuery:
    id: str
    query: str
    relevant_doc_ids: List[str] = field(default_factory=list)


@dataclass
class Benchmark:
    corpus: str
    queries: List[BenchmarkQuery] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "Benchmark":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        queries = [
            BenchmarkQuery(
                id=q["id"],
                query=q["query"],
                relevant_doc_ids=q.get("relevant_doc_ids", []),
            )
            for q in data.get("queries", [])
        ]
        return cls(corpus=data.get("corpus", ""), queries=queries)


# --------------------------------------------------------------------------- #
# Retrieval evaluation runner
# --------------------------------------------------------------------------- #
def build_retriever_for_corpus(corpus_dir: str, mode: str = "sparse"):
    """Index ``corpus_dir`` and return a HybridRetriever in ``mode``.

    Kept import-local so importing this module (e.g. for metric functions in
    tests) does not pull in heavy retrieval dependencies.
    """
    from .embedder import SimpleEmbedder
    from .faiss_store import Chunk, FAISSVectorStore
    from .hybrid_retriever import HybridRetriever
    from .ingestion import ingest_directory

    chunks = ingest_directory(corpus_dir)
    objs = [
        Chunk(
            id=c.get("id", ""),
            document_id=c.get("document_id", ""),
            text=c.get("text", ""),
            metadata=c.get("metadata", {}),
        )
        for c in chunks
    ]
    embedder = SimpleEmbedder()
    tmp = tempfile.mkdtemp(prefix="scirag_eval_")
    store = FAISSVectorStore(dim=embedder.dim, db_path=str(Path(tmp) / "meta.db"))
    # Sparse mode needs no embeddings; dense/hybrid do.
    embeddings = None if mode == "sparse" else embedder.encode([o.text for o in objs])
    if embeddings is None:
        # add_chunks requires embeddings to register chunk metadata, so pass zeros.
        embeddings = [[0.0] * embedder.dim for _ in objs]
    store.add_chunks(objs, embeddings)
    return HybridRetriever(store, embedder, mode=mode)


def evaluate_retrieval(
    benchmark: Benchmark,
    corpus_dir: Optional[str] = None,
    mode: str = "sparse",
    top_k: int = 3,
) -> Dict:
    """Run retrieval for every benchmark query and aggregate metrics."""
    corpus = corpus_dir or benchmark.corpus
    retriever = build_retriever_for_corpus(corpus, mode=mode)

    per_query = []
    recalls, precisions, rrs = [], [], []
    for q in benchmark.queries:
        chunks = retriever.retrieve(q.query, top_k=top_k)
        # De-duplicate retrieved doc ids preserving order.
        seen, retrieved_docs = set(), []
        for c in chunks:
            if c.document_id not in seen:
                seen.add(c.document_id)
                retrieved_docs.append(c.document_id)

        r = recall_at_k(retrieved_docs, q.relevant_doc_ids, top_k)
        p = precision_at_k(retrieved_docs, q.relevant_doc_ids, top_k)
        rr = mrr(retrieved_docs, q.relevant_doc_ids)
        recalls.append(r)
        precisions.append(p)
        rrs.append(rr)
        per_query.append({
            "id": q.id,
            "retrieved": retrieved_docs,
            "relevant": q.relevant_doc_ids,
            "recall": round(r, 3),
            "precision": round(p, 3),
            "rr": round(rr, 3),
        })

    n = max(len(benchmark.queries), 1)
    return {
        "mode": mode,
        "top_k": top_k,
        "num_queries": len(benchmark.queries),
        f"recall@{top_k}": round(sum(recalls) / n, 3),
        f"precision@{top_k}": round(sum(precisions) / n, 3),
        "mrr": round(sum(rrs) / n, 3),
        "per_query": per_query,
    }


def _main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="SciRAG retrieval evaluation harness")
    parser.add_argument(
        "--benchmark", default="benchmarks/retrieval_benchmark.json",
        help="Path to benchmark JSON.",
    )
    parser.add_argument("--corpus", default=None, help="Override corpus dir from benchmark.")
    parser.add_argument("--mode", default="sparse", choices=["sparse", "dense", "hybrid"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    benchmark = Benchmark.from_json(args.benchmark)
    results = evaluate_retrieval(benchmark, corpus_dir=args.corpus, mode=args.mode, top_k=args.top_k)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
