"""Tests for hybrid (dense + BM25 + RRF) retrieval."""

import tempfile
from pathlib import Path

from scirag.faiss_store import Chunk, FAISSVectorStore
from scirag.hybrid_retriever import (
    DENSE,
    HYBRID,
    SPARSE,
    HybridRetriever,
    reciprocal_rank_fusion,
)


class FakeEmbedder:
    """Deterministic embedder mapping known text to fixed 4-d vectors."""

    _MAP = {
        "dense passage retrieval": [1.0, 0.0, 0.0, 0.0],
        "inverted index lexical": [0.0, 1.0, 0.0, 0.0],
        "cross encoder reranking": [0.0, 0.0, 1.0, 0.0],
    }

    def encode(self, texts):
        out = []
        for t in texts:
            out.append(self._MAP.get(t, [0.25, 0.25, 0.25, 0.25]))
        return out

    @property
    def dim(self):
        return 4


def _build_store(tmpdir):
    store = FAISSVectorStore(dim=4, db_path=str(Path(tmpdir) / "meta.db"))
    chunks = [
        Chunk(id="a", document_id="A", text="dense passage retrieval with neural embeddings"),
        Chunk(id="b", document_id="B", text="bm25 sparse inverted index lexical matching"),
        Chunk(id="c", document_id="C", text="cross encoder reranking of candidates"),
        Chunk(id="d", document_id="D", text="citation graph network expansion"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    store.add_chunks(chunks, embeddings)
    return store


def test_rrf_basic():
    # Item 7 is rank-0 in one list and rank-1 in the other -> highest fused score.
    fused = reciprocal_rank_fusion([[7, 1, 2], [3, 7, 4]], k=60)
    assert fused[0][0] == 7
    ids = [i for i, _ in fused]
    assert set(ids) == {1, 2, 3, 4, 7}


def test_dense_mode_returns_embedding_match():
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        r = HybridRetriever(store, FakeEmbedder(), mode=DENSE)
        # Query embedding aligns with chunk A.
        out = r.retrieve("dense passage retrieval", top_k=1)
        assert out and out[0].document_id == "A"


def test_sparse_mode_returns_lexical_match():
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        r = HybridRetriever(store, FakeEmbedder(), mode=SPARSE)
        # Lexical query overlaps chunk B regardless of embeddings.
        out = r.retrieve("inverted index lexical", top_k=1)
        assert out and out[0].document_id == "B"


def test_hybrid_fuses_both_signals():
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        r = HybridRetriever(store, FakeEmbedder(), mode=HYBRID)
        # Embedding favours A (vector match); lexical favours B (token match).
        out = r.retrieve("inverted index lexical", top_k=4)
        docs = [c.document_id for c in out]
        assert "A" in docs and "B" in docs
        # B should rank ahead of A here since both signals point at it more.
        assert docs.index("B") <= docs.index("A")


def test_hybrid_no_query_terms_falls_back_to_dense():
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        r = HybridRetriever(store, FakeEmbedder(), mode=HYBRID)
        # Query has no lexical overlap; should still return dense results.
        out = r.retrieve("dense passage retrieval", top_k=2)
        assert out and out[0].document_id == "A"
