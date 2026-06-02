"""
Hybrid retriever for SciRAG.

Combines dense (FAISS / embedding) retrieval with sparse lexical (BM25)
retrieval and fuses the two rankings with Reciprocal Rank Fusion (RRF).

Modes:
    - "hybrid" (default): dense + BM25, fused with RRF.
    - "dense":  embedding similarity only.
    - "sparse": BM25 lexical only.

The class keeps the historical constructor signature
``HybridRetriever(vector_store, embedder)`` and ``retrieve(query, top_k, ...)``
so existing callers keep working; the new behaviour is additive.
"""

import logging
import re
from typing import List, Optional, Tuple

from .embedder import SimpleEmbedder
from .faiss_store import Chunk, FAISSVectorStore

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep missing
    BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed; falling back to dense-only retrieval.")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Valid retrieval modes.
DENSE = "dense"
SPARSE = "sparse"
HYBRID = "hybrid"


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric word tokenizer used for BM25."""
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: List[List[int]], k: int = 60
) -> List[Tuple[int, float]]:
    """Fuse several ranked id-lists into one with Reciprocal Rank Fusion.

    For each ranking, an item at 0-based rank ``r`` contributes ``1 / (k + r + 1)``
    to its fused score. Returns ``(id, score)`` pairs sorted by descending score.

    Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
    individual rank learning methods" (SIGIR 2009).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """Hybrid retriever: FAISS dense + BM25 sparse, fused with RRF."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: SimpleEmbedder,
        mode: str = HYBRID,
        rrf_k: int = 60,
        min_score: float = 0.0,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.mode = mode
        self.rrf_k = rrf_k
        # Minimum dense similarity for a chunk to survive in dense-only mode.
        self.min_score = min_score

        # Lazily-built BM25 state.
        self._bm25: Optional["BM25Okapi"] = None
        self._bm25_ids: List[int] = []
        self._bm25_corpus_size: int = -1

    # ------------------------------------------------------------------ #
    # BM25 (sparse) index
    # ------------------------------------------------------------------ #
    def _ensure_bm25(self) -> bool:
        """Build/refresh the BM25 index over the current corpus.

        Returns True if a usable BM25 index is available.
        """
        if not _BM25_AVAILABLE:
            return False

        corpus = self.vector_store.all_chunks()
        if not corpus:
            return False

        # Rebuild only when the corpus changed (cheap identity check on size).
        if self._bm25 is not None and self._bm25_corpus_size == len(corpus):
            return True

        tokenized = [_tokenize(chunk.text) for _, chunk in corpus]
        self._bm25_ids = [idx for idx, _ in corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus_size = len(corpus)
        logger.debug("Built BM25 index over %d chunks", len(corpus))
        return True

    def _sparse_ranking(self, query: str, top_k: int) -> List[int]:
        """Return store-ids ranked by BM25 score (best first)."""
        if not self._ensure_bm25() or self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort descending without numpy dependency here.
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return [self._bm25_ids[i] for i in ranked[:top_k] if scores[i] > 0]

    # ------------------------------------------------------------------ #
    # Dense index
    # ------------------------------------------------------------------ #
    def _dense_results(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return (store-id, similarity) pairs from the dense index."""
        embeddings = self.embedder.encode([query])
        if len(embeddings) == 0:
            return []
        return self.vector_store.search(embeddings[0], top_k=top_k)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        section_name: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[Chunk]:
        """Retrieve the top-k chunks for ``query``.

        ``mode`` overrides the instance default ("hybrid"/"dense"/"sparse").
        ``section_name`` keeps only chunks whose section header matches.
        """
        mode = mode or self.mode
        # Over-fetch so fusion / filtering has candidates to work with.
        fetch_k = max(top_k * 4, top_k)

        ordered_ids: List[int]

        if mode == SPARSE:
            ordered_ids = self._sparse_ranking(query, fetch_k)
        elif mode == DENSE:
            dense = self._dense_results(query, fetch_k)
            ordered_ids = [idx for idx, score in dense if score >= self.min_score]
        else:  # HYBRID
            dense = self._dense_results(query, fetch_k)
            dense_ids = [idx for idx, _ in dense]
            sparse_ids = self._sparse_ranking(query, fetch_k)
            if sparse_ids and dense_ids:
                fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k)
                ordered_ids = [item_id for item_id, _ in fused]
            else:
                # One side empty (e.g. BM25 unavailable): use whichever we have.
                ordered_ids = dense_ids or sparse_ids

        chunks: List[Chunk] = []
        seen: set[int] = set()
        for idx in ordered_ids:
            if idx in seen:
                continue
            seen.add(idx)
            chunk = self.vector_store.get_chunk(idx)
            if chunk is None:
                continue
            chunk.embedding = None  # Don't carry embeddings around.
            if section_name and section_name.lower() not in (chunk.section_header or "").lower():
                continue
            chunks.append(chunk)
            if len(chunks) >= top_k:
                break

        return chunks
