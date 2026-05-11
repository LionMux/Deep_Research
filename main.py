#!/usr/bin/env python3
"""
SciRAG — Unified CLI Entry Point.

Usage:
    python main.py query "Your research question"
    python main.py index /path/to/papers
    python main.py search "query"
    python main.py api          # Start FastAPI server
    python main.py mcp          # Start MCP server

Examples:
    python main.py query "What are the latest methods for dense passage retrieval?"
    python main.py index ./papers
    python main.py api --port 8000
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scirag")


def _load_env():
    """Load API key from environment."""
    api_key = os.environ.get("MOONSHOT_API_KEY", os.environ.get("KIMI_API_KEY", ""))
    if not api_key:
        logger.warning("No API key found. Set MOONSHOT_API_KEY or KIMI_API_KEY")
    return api_key


def cmd_query(args):
    """Run full SciRAG pipeline for a query."""
    from scirag.llm_client import PrimaryClient
    from scirag.config import SciRAGConfig
    from scirag.pipeline import SciRAGPipeline
    from scirag.citation_graph import CitationGraph

    # Все настройки берем из .env через SciRAGConfig (PRIMARY_*/GEMINI_* + бэквард с KIMI_*)
    config = SciRAGConfig()
    client = PrimaryClient.from_config(config)
    client.precheck_or_raise()

    graph = CitationGraph()
    pipeline = SciRAGPipeline(client, config, graph)

    # Use empty chunks or load from directory
    chunks = []
    if args.papers_dir:
        from scirag.ingestion import ingest_directory
        chunks = ingest_directory(args.papers_dir)
        logger.info(f"Loaded {len(chunks)} chunks from {args.papers_dir}")

    result = pipeline.run(args.query, chunks)

    print("\n" + "=" * 60)
    print("SCIRAG RESEARCH REPORT")
    print("=" * 60)
    print(result.get("report", ""))
    print("\n" + "=" * 60)
    print("CITATIONS")
    print("=" * 60)
    for c in result.get("citations", []):
        print(f"  [{c.get('number', '?')}] {c.get('text', '')[:80]}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to {args.output}")


def cmd_index(args):
    """Index papers into vector store."""
    from scirag.ingestion import ingest_directory
    from scirag.embedder import EmbedderFactory
    from scirag.faiss_store import FAISSVectorStore

    chunks = ingest_directory(args.papers_dir)
    logger.info(f"Ingested {len(chunks)} chunks")

    embedder = EmbedderFactory.create()
    store = FAISSVectorStore()
    texts = [c["text"] for c in chunks]
    embs = embedder.encode(texts)
    store.add_chunks(
        [{"id": c["id"], "document_id": c["document_id"], "text": c["text"],
          "metadata": c.get("metadata", {})} for c in chunks],
        embs,
    )

    # Save
    os.makedirs("./data/vector_index", exist_ok=True)
    store.save_index("./data/vector_index/faiss.index", "./data/vector_index/metadata.db")
    logger.info("Index saved to ./data/vector_index/")


def cmd_search(args):
    """Search indexed papers."""
    from scirag.embedder import EmbedderFactory
    from scirag.faiss_store import FAISSVectorStore
    from scirag.hybrid_retriever import HybridRetriever

    embedder = EmbedderFactory.create()
    store = FAISSVectorStore()
    store.load_index("./data/vector_index/faiss.index", "./data/vector_index/metadata.db")
    retriever = HybridRetriever(store, embedder)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nTop-{args.top_k} results for: {args.query}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.document_id}")
        print(f"    {r.text[:200]}...")


def cmd_api(args):
    """Start FastAPI server."""
    import uvicorn
    from scirag.api import app
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_mcp(args):
    """Start MCP server."""
    from scirag.mcp_server import mcp, _MCP_AVAILABLE
    if _MCP_AVAILABLE:
        mcp.run(transport="stdio")
    else:
        logger.error("MCP SDK not installed. Run: pip install mcp")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="SciRAG — Scientific Research Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # query
    q = subparsers.add_parser("query", help="Run full research pipeline")
    q.add_argument("query", help="Research question")
    q.add_argument("--papers-dir", "-d", help="Directory with papers")
    q.add_argument("--output", "-o", help="Save JSON output to file")

    # index
    idx = subparsers.add_parser("index", help="Index papers into vector store")
    idx.add_argument("papers_dir", help="Directory with papers")

    # search
    s = subparsers.add_parser("search", help="Search indexed papers")
    s.add_argument("query", help="Search query")
    s.add_argument("--top-k", "-k", type=int, default=5)

    # api
    a = subparsers.add_parser("api", help="Start FastAPI server")
    a.add_argument("--host", default="0.0.0.0")
    a.add_argument("--port", "-p", type=int, default=8000)

    # mcp
    m = subparsers.add_parser("mcp", help="Start MCP server")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    commands = {
        "query": cmd_query,
        "index": cmd_index,
        "search": cmd_search,
        "api": cmd_api,
        "mcp": cmd_mcp,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
