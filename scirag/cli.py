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


def main():
    parser = argparse.ArgumentParser(
        description="SciRAG — Scientific Iterative Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scirag index ./my_papers/
  scirag search "neural retrieval" --top-k 15
  scirag synthesize "dense passage retrieval methods" --sections 8 -o report.md
  scirag verify "DPR achieves 41.5% MRR" --doc Karpukhin2020
  scirag install-claude
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command")

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

    # status
    sub.add_parser("status", help="Check pipeline status")

    # install-claude
    sub.add_parser("install-claude", help="Install MCP server for Claude Code")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "index": cmd_index,
        "search": cmd_search,
        "synthesize": cmd_synthesize,
        "extract": cmd_extract,
        "verify": cmd_verify,
        "status": cmd_status,
        "install-claude": cmd_install_claude,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
