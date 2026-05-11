"""
Simple hybrid retriever for SciRAG.
Replaces retrieval.hybrid_retriever from old architecture.
"""

import logging
from typing import List, Optional

from .embedder import SimpleEmbedder
from .faiss_store import FAISSVectorStore, Chunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever: FAISS dense + optional BM25 sparse."""

    def __init__(self, vector_store: FAISSVectorStore, embedder: SimpleEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10, section_name: Optional[str] = None) -> List[Chunk]:
        """Retrieve top-k chunks for query."""
        embeddings = self.embedder.encode([query])
        if len(embeddings) == 0:
            return []

        results = self.vector_store.search(embeddings[0], top_k=top_k * 2)
        chunks = []
        for idx, score in results:
            chunk = self.vector_store.get_chunk(idx)
            if chunk:
                chunk.embedding = None  # Don't carry embeddings around
                chunks.append(chunk)

        if section_name:
            chunks = [c for c in chunks if section_name.lower() in (c.section_header or "").lower()]

        return chunks[:top_k]
