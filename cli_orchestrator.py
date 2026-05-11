#!/usr/bin/env python3
"""
SciRAG Local LLM Orchestrator — Entry point (Variant A).

Usage:
    python cli_orchestrator.py "Your research question"
    python cli_orchestrator.py --index ./papers "Your research question"

Flow:
    1. Ensure local LLM is running (auto-start via LM Studio / Ollama)
    2. Index papers (if --index provided)
    3. Retrieve relevant chunks via SciRAG
    4. Local LLM decides: enough info? (max 2 extra retrievals)
    5. Print final markdown answer
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scirag-orchestrator")


def main():
    parser = argparse.ArgumentParser(description="SciRAG + Local LLM Orchestrator")
    parser.add_argument("query", help="Research question")
    parser.add_argument("--index", default="./papers", help="Papers directory to index")
    parser.add_argument("--top-k", type=int, default=8, help="Chunks per retrieval")
    parser.add_argument("--max-extra", type=int, default=2, help="Max extra retrievals")
    args = parser.parse_args()

    # Ensure local LLM is available
    logger.info("Checking local LLM...")
    from scirag.llm_client import KimiClient
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
        sys.exit(1)

    # Initialize SciRAG state
    from scirag.state import _ensure_init, do_index, do_search
    logger.info(f"Initializing SciRAG (papers: {args.index})")
    state = _ensure_init(args.index)

    if not state["retriever"]:
        logger.info("Indexing papers...")
        idx = do_index(args.index)
        logger.info(f"Indexed {idx['chunks']} chunks")

    def retriever_fn(query: str, top_k: int):
        results = do_search(query, top_k=top_k)
        return results.get("results", [])

    # Run orchestrator
    from scirag.orchestrator import LocalLLMOrchestrator
    orch = LocalLLMOrchestrator(llm_client=llm, retriever_fn=retriever_fn, max_extra_retrievals=args.max_extra)

    print("=" * 70)
    print(f"RESEARCH: {args.query}")
    print("=" * 70)

    result = orch.research(args.query, top_k=args.top_k)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["answer"])
    print("\n" + "=" * 70)
    print(f"Stats: {result['iterations']} iteration(s), {result['total_chunks']} chunks retrieved")
    if result.get("forced"):
        print("Note: Max iterations reached, forced final answer.")
    print("=" * 70)


if __name__ == "__main__":
    main()
