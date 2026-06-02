from __future__ import annotations

import logging
from typing import List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def filter_chunks_by_doc_ids(chunks: Sequence[object], doc_ids: Set[str]) -> List[object]:
    """
    Filter already-loaded chunk-like objects by their document_id.
    chunk is expected to have attribute/get 'document_id' and optionally 'text' / 'metadata'.
    """
    filtered: List[object] = []
    for c in chunks:
        did = getattr(c, "document_id", None)
        if not did and hasattr(c, "get"):
            did = c.get("document_id")
        if did and did in doc_ids:
            filtered.append(c)
    return filtered


def bind_retriever_to_chunks_subset(
    *,
    all_chunks: Sequence[object],
    base_retriever: object,
    doc_ids: Set[str],
) -> Tuple[List[object], object]:
    """
    Build a temporary retriever over only the evidence subset chunks.

    We intentionally avoid changing HybridRetriever internals:
    - filter subset chunks
    - create a fresh Embedder + FAISSVectorStore
    - return a retriever compatible with IterativeSynthesizer (expects .retrieve(query, top_k))
    """
    subset_chunks = filter_chunks_by_doc_ids(all_chunks, doc_ids)
    if not subset_chunks:
        return [], base_retriever

    # Lazy imports to avoid heavy startup cost.
    from scirag.embedder import EmbedderFactory
    from scirag.faiss_store import Chunk, FAISSVectorStore
    from scirag.hybrid_retriever import HybridRetriever

    embedder = EmbedderFactory.create(prefer_bge=False)
    store = FAISSVectorStore()

    chunk_objects: List[Chunk] = []
    for c in subset_chunks:
        # Convert chunk-like object/dict into faiss_store.Chunk.
        did = getattr(c, "document_id", "") if not hasattr(c, "get") else c.get("document_id", "")
        text = getattr(c, "text", "") if not hasattr(c, "get") else c.get("text", "")
        metadata = getattr(c, "metadata", None) if hasattr(c, "metadata") else (c.get("metadata", {}) if hasattr(c, "get") else {})
        chunk_objects.append(
            Chunk(
                id=getattr(c, "id", "") if not hasattr(c, "get") else c.get("id", ""),
                document_id=did,
                text=text,
                start_pos=getattr(c, "start_pos", 0) if not hasattr(c, "get") else c.get("start_pos", 0),
                end_pos=getattr(c, "end_pos", 0) if not hasattr(c, "get") else c.get("end_pos", 0),
                section_header=metadata.get("section_header", ""),
                metadata=metadata,
            )
        )

    texts = [c.text for c in chunk_objects]
    embeddings = embedder.encode(texts)
    store.add_chunks(chunk_objects, embeddings)

    retriever = HybridRetriever(vector_store=store, embedder=embedder)
    return subset_chunks, retriever
