"""
Simple FAISS vector store for SciRAG.
Replaces vector_store.faiss_store from old architecture.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore
    logging.getLogger(__name__).warning("faiss-cpu not installed. Using numpy fallback for search.")

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Document chunk."""
    id: str = ""
    document_id: str = ""
    text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    section_header: str = ""
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class FAISSVectorStore:
    """FAISS-CPU vector store with SQLite metadata."""

    def __init__(self, dim: int = 384, db_path: str = "./data/vector_index/metadata.db"):
        self.dim = dim
        self.db_path = db_path
        self._index = None  # faiss.Index or None
        self._embeddings: Optional[np.ndarray] = None  # numpy fallback
        self._chunks: Dict[int, Chunk] = {}
        self._next_id = 0
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT,
                document_id TEXT,
                text TEXT,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _ensure_index(self):
        if _FAISS_AVAILABLE and self._index is None:
            self._index = faiss.IndexFlatIP(self.dim)  # type: ignore
            logger.info(f"Created FAISS index (dim={self.dim})")

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None):
        self._ensure_index()
        if not chunks:
            return

        if embeddings is None:
            logger.warning("No embeddings provided, skipping add")
            return

        embs = np.array(embeddings, dtype=np.float32)
        # L2 normalize for cosine similarity via IP
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid div by zero
        embs = embs / norms

        if _FAISS_AVAILABLE and self._index is not None:
            self._index.add(embs)
        else:
            # numpy fallback
            if self._embeddings is None:
                self._embeddings = embs
            else:
                self._embeddings = np.vstack([self._embeddings, embs])

        conn = sqlite3.connect(self.db_path)
        for i, chunk in enumerate(chunks):
            idx = self._next_id + i
            self._chunks[idx] = chunk
            conn.execute(
                "INSERT OR REPLACE INTO chunks (id, chunk_id, document_id, text, metadata) VALUES (?, ?, ?, ?, ?)",
                (idx, chunk.id, chunk.document_id, chunk.text, json.dumps(chunk.metadata)),
            )
        conn.commit()
        conn.close()
        self._next_id += len(chunks)
        logger.info(f"Added {len(chunks)} chunks to index (total: {self._next_id})")

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        self._ensure_index()
        if self._next_id == 0:
            return []

        q = np.array([query_embedding], dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        if _FAISS_AVAILABLE and self._index is not None:
            scores, indices = self._index.search(q, min(top_k, self._next_id))
            return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
        else:
            # numpy fallback: cosine similarity
            if self._embeddings is None or len(self._embeddings) == 0:
                return []
            scores = (self._embeddings @ q.T).flatten()
            top_idx = np.argsort(-scores)[:top_k]
            return [(int(i), float(scores[i])) for i in top_idx]

    def get_chunk(self, idx: int) -> Optional[Chunk]:
        return self._chunks.get(idx)

    def all_chunks(self) -> List[Tuple[int, Chunk]]:
        """Return all (index, chunk) pairs ordered by index.

        Used by sparse/lexical retrievers (e.g. BM25) that need the full
        corpus text rather than just dense-search hits.
        """
        return [(idx, self._chunks[idx]) for idx in sorted(self._chunks.keys())]

    def __len__(self) -> int:
        return self._next_id

    def load_index(self, index_path: str, db_path: str):
        """Load FAISS index and metadata from disk."""
        if _FAISS_AVAILABLE:
            self._index = faiss.read_index(index_path)  # type: ignore
            self.dim = self._index.d  # type: ignore
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT id, chunk_id, document_id, text, metadata FROM chunks")
        for row in cursor:
            self._chunks[row[0]] = Chunk(
                id=row[1], document_id=row[2], text=row[3], metadata=json.loads(row[4] or "{}"),
            )
        self._next_id = max(self._chunks.keys()) + 1 if self._chunks else 0
        conn.close()
        logger.info(f"Loaded index: {self._next_id} chunks")

    def save_index(self, index_path: str, db_path: str):
        """Save FAISS index and metadata to disk."""
        if _FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, index_path)  # type: ignore
        # SQLite already persisted
        logger.info(f"Saved index to {index_path}")
