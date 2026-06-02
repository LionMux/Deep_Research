#!/usr/bin/env python3
"""
SciRAG CLI — standalone command-line interface.

Usage:
    # Index papers
    scirag index ./papers/

    # Search
    scirag search "neural retrieval methods" --top-k 10

    # Full synthesis
    scirag synthesize "neural retrieval methods" --sections 6

    # Extract facts
    scirag extract "DPR performance on MS MARCO"

    # Verify a fact
    scirag verify "DPR achieves 41.5% MRR@10" --doc Karpukhin2020

    # Status
    scirag status

    # Install MCP server for Claude Code
    scirag install-claude
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("scirag-cli")


def _get_env_or_ask(key: str, prompt: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        val = input(f"{prompt}: ").strip()
    return val


def cmd_index(args):
    """Index papers directory."""
    from scirag.mcp_server import scirag_index
    result = scirag_index(papers_dir=args.dir)
    data = json.loads(result)
    if "error" in data:
        logger.error(f"Error: {data['error']}")
        return 1
    logger.info(f"Indexed: {data['chunks']} chunks in {data['time_sec']}s")
    logger.info(f"Graph: {data['graph']}")
    return 0


def cmd_search(args):
    """Search indexed papers."""
    from scirag.mcp_server import scirag_search
    result = scirag_search(query=args.query, top_k=args.top_k)
    data = json.loads(result)
    if "error" in data:
        logger.error(f"Error: {data['error']}")
        return 1
    logger.info(f"Query: {data['query']}")
    logger.info(f"Found: {data['total_found']} chunks")
    for i, r in enumerate(data['results'], 1):
        logger.info(f"\n  [{i}] [{r['tag']}] {r['document_id']}")
        logger.info(f"      {r['text_preview'][:200]}...")
    return 0


def cmd_synthesize(args):
    """Full synthesis pipeline."""
    from scirag.mcp_server import scirag_synthesize
    logger.info(f"Synthesizing: {args.query}")
    result = scirag_synthesize(query=args.query, max_sections=args.sections)
    data = json.loads(result)
    if "error" in data:
        logger.error(f"Error: {data['error']}")
        return 1

    # Save to file
    output_path = args.output or f"scirag_{args.query.replace(' ', '_')[:30]}.md"
    with open(output_path, "w") as f:
        f.write(data["final_text"])

    stats = data["stats"]
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Synthesis complete: {stats['total_sections']} sections")
    logger.info(f"Facts: {stats['verified_facts']}/{stats['total_facts']} verified")
    logger.info(f"Confidence: {stats['avg_confidence']:.0%}")
    logger.info(f"Time: {stats['time_sec']}s")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"{'=' * 50}")
    return 0


def cmd_extract(args):
    """Extract facts."""
    from scirag.mcp_server import scirag_extract
    result = scirag_extract(query=args.query, top_k=args.top_k)
    data = json.loads(result)
    if "error" in data:
        logger.error(f"Error: {data['error']}")
        return 1
    logger.info(f"Query: {data['query']}")
    logger.info(f"Facts extracted: {data['count']}")
    for f in data['facts']:
        logger.info(f"  - {f['text']} [{f['source']}]")
    return 0


def cmd_verify(args):
    """Verify a fact."""
    from scirag.mcp_server import scirag_verify
    result = scirag_verify(fact=args.fact, source_doc_id=args.doc)
    data = json.loads(result)
    if "error" in data:
        logger.error(f"Error: {data['error']}")
        return 1
    status = "VERIFIED" if data["verified"] else "NOT VERIFIED"
    logger.info(f"Fact: {args.fact}")
    logger.info(f"Result: {status} (confidence: {data['confidence']:.2f})")
    logger.info(f"Method: {data['method']}")
    return 0


def cmd_status(args):
    """Check status."""
    from scirag.mcp_server import scirag_status
    result = scirag_status()
    data = json.loads(result)
    logger.info(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_install_claude(args):
    """Install MCP server for Claude Code."""
    install_dir = Path.home() / ".config" / "claude"
    install_dir.mkdir(parents=True, exist_ok=True)

    # Get python path
    python_path = sys.executable
    server_path = Path(__file__).parent / "mcp_server.py"

    config = {
        "mcpServers": {
            "scirag": {
                "command": python_path,
                "args": [str(server_path)],
                "env": {
                    "KIMI_API_KEY": _get_env_or_ask("KIMI_API_KEY", "Enter your Kimi API key"),
                    "SCIRAG_PAPERS_DIR": _get_env_or_ask("SCIRAG_PAPERS_DIR", "Enter papers directory path"),
                },
            }
        }
    }

    config_path = install_dir / "mcp_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Claude Code MCP config written to: {config_path}")
    logger.info("Restart Claude Code to use SciRAG tools.")
    logger.info("Then ask: 'Search my papers for neural retrieval methods'")
    return 0


def cmd_query(args):
    """Run the full SciRAG research pipeline for a query."""
    from scirag.citation_graph import CitationGraph
    from scirag.config import SciRAGConfig
    from scirag.llm_client import PrimaryClient
    from scirag.pipeline import SciRAGPipeline

    config = SciRAGConfig()
    client = PrimaryClient.from_config(config)
    client.precheck_or_raise()

    graph = CitationGraph()
    pipeline = SciRAGPipeline(client, config, graph)

    chunks = []
    if args.papers_dir:
        from scirag.ingestion import ingest_directory
        chunks = ingest_directory(args.papers_dir)
        logger.info(f"Loaded {len(chunks)} chunks from {args.papers_dir}")

    result = pipeline.run(args.query, chunks)

    print(result.get("report_markdown") or result.get("final_text", ""))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON to {args.output}")
    return 0


def cmd_api(args):
    """Start the FastAPI server."""
    import uvicorn

    from scirag.api import app
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_mcp(args):
    """Start the MCP server (stdio transport)."""
    from scirag.mcp_server import _MCP_AVAILABLE, mcp
    if not _MCP_AVAILABLE:
        logger.error("MCP SDK not installed. Run: pip install mcp")
        return 1
    mcp.run(transport="stdio")
    return 0


def cmd_eval(args):
    """Run the retrieval evaluation harness on a fixed benchmark."""
    from scirag.evaluation import Benchmark, evaluate_retrieval

    benchmark = Benchmark.from_json(args.benchmark)
    results = evaluate_retrieval(
        benchmark, corpus_dir=args.corpus, mode=args.mode, top_k=args.top_k
    )
    print(json.dumps(results, indent=2))
    return 0


def cmd_orchestrate(args):
    """Run the local-LLM orchestration loop (LM Studio / Ollama)."""
    from scirag.llm_client import KimiClient
    logger.info("Checking local LLM...")
    llm = KimiClient(
        api_key="",
        fallback_url="http://localhost:1234/v1",
        fallback_model="meta-llama-3.1-8b-instruct",
    )
    try:
        test = llm.chat([{"role": "user", "content": "Say hi"}], max_tokens=10)
        logger.info(f"Local LLM OK: {test.strip()[:60]}")
    except Exception as e:
        logger.error(f"Local LLM not available: {e}")
        print("ERROR: Start LM Studio / Ollama and load a model, or let SciRAG auto-start it.")
        return 1

    from scirag.state import _ensure_init, do_index, do_search
    logger.info(f"Initializing SciRAG (papers: {args.index})")
    state = _ensure_init(args.index)
    if not state["retriever"]:
        logger.info("Indexing papers...")
        idx = do_index(args.index)
        logger.info(f"Indexed {idx['chunks']} chunks")

    def retriever_fn(query: str, top_k: int):
        return do_search(query, top_k=top_k).get("results", [])

    from scirag.orchestrator import LocalLLMOrchestrator
    orch = LocalLLMOrchestrator(
        llm_client=llm, retriever_fn=retriever_fn, max_extra_retrievals=args.max_extra
    )
    print("=" * 70)
    print(f"RESEARCH: {args.query}")
    print("=" * 70)
    result = orch.research(args.query, top_k=args.top_k)
    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["answer"])
    print(
        f"\nStats: {result['iterations']} iteration(s), "
        f"{result['total_chunks']} chunks retrieved"
    )
    if result.get("forced"):
        print("Note: Max iterations reached, forced final answer.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SciRAG — Scientific Iterative Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scirag index ./my_papers/
  scirag search "neural retrieval" --top-k 15
  scirag query "dense passage retrieval methods" --papers-dir ./samples/papers
  scirag synthesize "dense passage retrieval methods" --sections 8 -o report.md
  scirag verify "DPR achieves 41.5% MRR" --doc Karpukhin2020
  scirag eval --mode sparse --top-k 3
  scirag api --port 8000
  scirag mcp
  scirag install-claude
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # query (full research pipeline)
    p_query = sub.add_parser("query", help="Run the full research pipeline")
    p_query.add_argument("query", help="Research question")
    p_query.add_argument("--papers-dir", "-d", default=None, help="Directory with papers")
    p_query.add_argument("-o", "--output", help="Save JSON result to file")

    # index
    p_index = sub.add_parser("index", help="Index papers directory")
    p_index.add_argument("dir", help="Path to papers directory")

    # search
    p_search = sub.add_parser("search", help="Search indexed papers")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=10)

    # synthesize
    p_syn = sub.add_parser("synthesize", help="Full synthesis pipeline")
    p_syn.add_argument("query", help="Research question")
    p_syn.add_argument("--sections", type=int, default=6)
    p_syn.add_argument("-o", "--output", help="Output file path")

    # extract
    p_ext = sub.add_parser("extract", help="Extract structured facts")
    p_ext.add_argument("query", help="Query for fact extraction")
    p_ext.add_argument("--top-k", type=int, default=5)

    # verify
    p_ver = sub.add_parser("verify", help="Verify a fact")
    p_ver.add_argument("fact", help="Fact to verify")
    p_ver.add_argument("--doc", default="", help="Specific document ID")

    # orchestrate (local LLM)
    p_orch = sub.add_parser("orchestrate", help="Local-LLM orchestration loop")
    p_orch.add_argument("query", help="Research question")
    p_orch.add_argument("--index", default="./samples/papers", help="Papers directory to index")
    p_orch.add_argument("--top-k", type=int, default=8, help="Chunks per retrieval")
    p_orch.add_argument("--max-extra", type=int, default=2, help="Max extra retrievals")

    # eval
    p_eval = sub.add_parser("eval", help="Run the retrieval evaluation harness")
    p_eval.add_argument("--benchmark", default="benchmarks/retrieval_benchmark.json")
    p_eval.add_argument("--corpus", default=None, help="Override corpus dir from benchmark")
    p_eval.add_argument("--mode", default="sparse", choices=["sparse", "dense", "hybrid"])
    p_eval.add_argument("--top-k", type=int, default=3)

    # api
    p_api = sub.add_parser("api", help="Start the FastAPI server")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", "-p", type=int, default=8000)

    # mcp
    sub.add_parser("mcp", help="Start the MCP server (stdio)")

    # status
    sub.add_parser("status", help="Check pipeline status")

    # install-claude
    sub.add_parser("install-claude", help="Install MCP server for Claude Code")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "query": cmd_query,
        "index": cmd_index,
        "search": cmd_search,
        "synthesize": cmd_synthesize,
        "extract": cmd_extract,
        "verify": cmd_verify,
        "orchestrate": cmd_orchestrate,
        "eval": cmd_eval,
        "api": cmd_api,
        "mcp": cmd_mcp,
        "status": cmd_status,
        "install-claude": cmd_install_claude,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
