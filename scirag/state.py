"""
Shared server state for SciRAG.

Unifies state management between api.py (HTTP/MCP hybrid) and mcp_server.py
(FastMCP standalone) so that indexing, search, synthesis, and extraction
all operate on the same in-memory state when running in a single process.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("scirag-state")

# Singleton server state -------------------------------------------------------
_server_state = {
    "initialized": False,
    "papers_dir": None,
    "chunks": [],
    "retriever": None,
    "client": None,
    "config": None,
    "graph": None,
    "classifier": None,
    "symbolic": None,
    "last_query": None,
    "last_result": None,
    "api_calls": 0,
    "index_time": 0,
    "download_errors": [],
}


def _get_symbolic_reasoner(client, config):
    """Lazy loader for SymbolicReasoner to avoid heavy imports at module load."""
    from scirag.symbolic_reasoning import SymbolicReasoner
    return SymbolicReasoner(client, config)


def _ensure_init(papers_dir: Optional[str] = None) -> dict:
    """Initialize (or re-use) the shared server state."""
    state = _server_state
    if state["initialized"] and (papers_dir is None or state["papers_dir"] == papers_dir):
        return state

    papers_dir = papers_dir or os.environ.get("SCIRAG_PAPERS_DIR", "./papers")
    papers_path = Path(papers_dir).expanduser().resolve()

    if not papers_path.exists():
        papers_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created papers directory: {papers_path}")

    from scirag.citation_graph import CitationGraph
    from scirag.config import SciRAGConfig
    from scirag.document_classifier import DocumentClassifier
    from scirag.llm_client import PrimaryClient

    state["papers_dir"] = str(papers_path)
    # Важно: все настройки берем из SciRAGConfig (подхватывает .env)
    state["config"] = SciRAGConfig()
    state["config"].data_dir = str(papers_path.parent / ".scirag_data")
    state["config"].ensure_dirs()

    state["client"] = PrimaryClient.from_config(state["config"])

    # Precheck (primary + gemini) — не должны блокировать старт.
    # Если HF настроен, вообще не трогаем PRIMARY readiness/health.
    hf_configured = bool(
        getattr(state["config"], "hf_api_token", "") and
        getattr(state["config"], "hf_base_url", "") and
        getattr(state["config"], "hf_model_synthesis", "")
    )
    if not hf_configured:
        try:
            state["client"].precheck_or_raise()
        except Exception as e:
            logger.warning(f"SciRAG client precheck failed (non-fatal): {e}")

    state["graph"] = CitationGraph(state["config"].citation_graph_path)
    state["classifier"] = DocumentClassifier(keywords=state["config"].classification_keywords)
    state["symbolic"] = _get_symbolic_reasoner(state["client"], state["config"])
    state["initialized"] = True
    logger.info(f"SciRAG state initialized: {papers_path}")
    return state


def _build_retriever(state: dict):
    """Build (or re-use) the HybridRetriever from indexed chunks."""
    if state["retriever"] is not None:
        return state["retriever"]
    if not state["chunks"]:
        return None

    from scirag.embedder import EmbedderFactory
    from scirag.faiss_store import Chunk, FAISSVectorStore
    from scirag.hybrid_retriever import HybridRetriever

    logger.info(f"Building retriever from {len(state['chunks'])} chunks...")
    embedder = EmbedderFactory.create(prefer_bge=False)
    store = FAISSVectorStore()

    chunk_objects = []
    for c in state["chunks"]:
        chunk_objects.append(Chunk(
            id=c.get("id", ""),
            document_id=c.get("document_id", ""),
            text=c.get("text", ""),
            start_pos=c.get("start_pos", 0),
            end_pos=c.get("end_pos", 0),
            section_header=c.get("metadata", {}).get("section_header", ""),
            metadata=c.get("metadata", {}),
        ))

    texts = [c.text for c in chunk_objects]
    embeddings = embedder.encode(texts)
    store.add_chunks(chunk_objects, embeddings)
    state["retriever"] = HybridRetriever(vector_store=store, embedder=embedder)
    return state["retriever"]


def _load_chunks(papers_dir: str) -> List[dict]:
    """Load and chunk all documents in directory."""
    from scirag.ingestion import ingest_directory
    logger.info(f"Loading documents from: {papers_dir}")
    return ingest_directory(papers_dir, pattern="*")


def do_index(papers_dir: str) -> dict:
    """Index papers and build retriever."""
    t0 = time.time()
    state = _ensure_init(papers_dir)
    if "error" in state:
        return state

    chunks = _load_chunks(state["papers_dir"])
    state["chunks"] = chunks
    state["graph"].build_from_chunks(chunks)
    state["retriever"] = _build_retriever(state)
    state["index_time"] = time.time() - t0

    return {
        "status": "indexed",
        "papers_dir": state["papers_dir"],
        "chunks": len(chunks),
        "graph": state["graph"].stats,
        "time_sec": round(state["index_time"], 1),
    }


def do_search(query: str, top_k: int = 10) -> dict:
    """Search indexed chunks."""
    state = _ensure_init()
    if "error" in state:
        return state
    if not state["retriever"]:
        return {"error": "No index. Run scirag_index first."}

    chunks = state["retriever"].retrieve(query, top_k=top_k)
    results = []
    for c in chunks[:top_k]:
        tag = state["classifier"].get_primary_tag(c.text) if state["classifier"] else "?"
        results.append({
            "document_id": c.document_id,
            "section": c.metadata.get("section_header", "")[:60] if hasattr(c, "metadata") else "",
            "text_preview": c.text[:300],
            "tag": tag,
        })
    return {"query": query, "results": results, "total_found": len(chunks)}


def do_status() -> dict:
    """Return current server status."""
    state = _ensure_init()
    if "error" in state:
        return state

    papers_path = Path(state["papers_dir"]) if state["papers_dir"] else None
    pdf_count = 0
    txt_count = 0
    if papers_path and papers_path.exists():
        pdf_count = len(list(papers_path.rglob("*.pdf")))
        txt_count = len(list(papers_path.rglob("*.txt"))) + len(list(papers_path.rglob("*.md")))

    return {
        "initialized": state["initialized"],
        "version": "TreeNode v2 + Symbolic Reasoning",
        "papers_dir": state["papers_dir"],
        "pdf_files_found": pdf_count,
        "txt_files_found": txt_count,
        "chunks_indexed": len(state["chunks"]),
        "citation_graph": state["graph"].stats if state["graph"] else {},
        "retriever_ready": state["retriever"] is not None,
        "last_query": state["last_query"],
        "api_key_set": bool(
            (getattr(state["config"], "primary_api_key", "") or "")
            or (getattr(state["config"], "gemini_api_key", "") or "")
        )
        if state["config"]
        else False,
        "download_errors": state.get("download_errors", []),
    }
