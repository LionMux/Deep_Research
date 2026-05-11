"""SciRAG MCP HTTP server — independent, no mcp_server imports."""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scirag-api")

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except ImportError:
    print("ERROR: starlette not installed. Run: pip install starlette uvicorn", file=sys.stderr)
    sys.exit(1)

# SciRAG direct imports (avoid mcp_server which has strict stdout requirements)
from scirag.llm_client import KimiClient
from scirag.config import SciRAGConfig
from scirag.outline_generator import OutlineGenerator
from scirag.tree_node import TreeNode, GapCriticTree
from scirag.citation_graph import CitationGraph
from scirag.document_classifier import DocumentClassifier
from scirag.iterative_synthesizer import IterativeSynthesizer, Fact
from scirag.verifier import FactVerifier
from scirag.embedder import EmbedderFactory
from scirag.faiss_store import FAISSVectorStore, Chunk
from scirag.hybrid_retriever import HybridRetriever

# Shared state (singleton across api.py and mcp_server.py when in same process)
from scirag.state import (
    _server_state,
    _ensure_init,
    _build_retriever,
    _load_chunks,
    do_index,
    do_search,
    do_status,
)

try:
    from scirag.deep_search import AcademicSearchEngine, PDFDownloader
    _DEEP_SEARCH_AVAILABLE = True
except ImportError:
    _DEEP_SEARCH_AVAILABLE = False


# ===== Synthesis / Extraction / Verification (local to api.py state) =====

def do_synthesize(query: str, max_sections: int = 6) -> dict:
    state = _ensure_init()
    if "error" in state:
        return state
    if not state["retriever"]:
        return {"error": "No index. Run scirag_index first."}

    t0 = time.time()
    config = state["config"]
    client = state["client"]

    outline_gen = OutlineGenerator(client, config)
    synthesizer = IterativeSynthesizer(client, config, state["graph"])
    verifier = FactVerifier(client, config)

    root = outline_gen.generate(query, max_sections=max_sections)

    # Skip critique and symbolic reasoning when using local fallback (too slow)
    using_local = getattr(client, '_use_fallback', False)
    if not using_local:
        root = outline_gen.critique_and_expand(root, query)
    else:
        logger.info("Local fallback detected: skipping critique for speed")

    papers_for_symbolic = []
    seen_docs = set()
    for c in state["chunks"]:
        did = c.get("document_id", "")
        if did and did not in seen_docs:
            papers_for_symbolic.append({
                "id": did,
                "title": c.get("metadata", {}).get("title", did),
                "text": c.get("text", ""),
            })
            seen_docs.add(did)

    symbolic_result = None
    if papers_for_symbolic and not using_local:
        try:
            symbolic_result = state["symbolic"].run(
                papers=papers_for_symbolic,
                query=query,
                outline=root,
            )
        except Exception as e:
            logger.warning(f"Symbolic reasoning failed: {e}")
    elif using_local:
        logger.info("Local fallback detected: skipping symbolic reasoning for speed")

    root = synthesizer.synthesize(root, state["retriever"])

    all_leaves = root.get_leaves()
    all_facts = [{"text": f.text, "source_doc": f.source_doc}
                 for leaf in all_leaves for f in leaf.facts]

    source_chunks = []
    seen_docs = set()
    for c in state["chunks"]:
        if c.get("document_id") not in seen_docs:
            source_chunks.append(c)
            seen_docs.add(c.get("document_id"))

    verify_results = verifier.verify_batch(all_facts, source_chunks)

    fact_idx = 0
    for leaf in all_leaves:
        for f in leaf.facts:
            if fact_idx < len(verify_results):
                v = verify_results[fact_idx]
                f.verified = v["verified"]
                f.confidence = v["confidence"]
                fact_idx += 1

    final_text = synthesizer.compile_final(root)

    from scirag.tree_node import GapCriticTree
    gap_critic = GapCriticTree(
        llm_generate=lambda n, ctx: "",
        llm_evaluate=lambda n, ctx: 0.0,
        max_depth=config.max_iterations,
    )
    tree_stats = gap_critic.tree_stats(root)

    total_facts = sum(len(leaf.facts) for leaf in all_leaves)
    verified_facts = sum(1 for leaf in all_leaves for f in leaf.facts if f.verified)

    result = {
        "query": query,
        "final_text": final_text,
        "tree": {
            "sections": len(root.children),
            "leaves": len(all_leaves),
            "max_depth": tree_stats["max_depth"],
            "with_gaps": tree_stats["with_gaps"],
        },
        "symbolic_reasoning": {
            "segments": symbolic_result.get("stats", {}).get("total_segments", 0) if symbolic_result else 0,
            "relationships": symbolic_result.get("stats", {}).get("relationships", 0) if symbolic_result else 0,
        },
        "stats": {
            "total_time_sec": round(time.time() - t0, 1),
            "total_sections": len(root.children),
            "total_leaves": len(all_leaves),
            "total_facts": total_facts,
            "verified_facts": verified_facts,
            "verification_rate": f"{verified_facts}/{total_facts} = {verified_facts/max(total_facts,1)*100:.0f}%",
            "avg_confidence": round(sum(leaf.confidence for leaf in all_leaves) / max(len(all_leaves), 1), 2),
        },
    }
    state["last_result"] = result
    return result


def do_extract(query: str, top_k: int = 5) -> dict:
    state = _ensure_init()
    if "error" in state:
        return state
    if not state["retriever"]:
        return {"error": "No index. Run scirag_index first."}

    config = state["config"]
    client = state["client"]
    synthesizer = IterativeSynthesizer(client, config, state["graph"])

    from scirag.tree_node import TreeNode
    node = TreeNode(title=query, description=query, keywords=[query])
    chunks = state["retriever"].retrieve(query, top_k=top_k)
    facts = synthesizer._extract_facts(chunks, node)

    return {
        "query": query,
        "facts": [f.to_dict() for f in facts],
        "count": len(facts),
    }


def do_verify(fact: str, source_doc_id: str = "") -> dict:
    state = _ensure_init()
    if "error" in state:
        return state

    verifier = FactVerifier(state["client"], state["config"])
    source_chunks = [c for c in state["chunks"]
                     if not source_doc_id or c.get("document_id") == source_doc_id]

    if not source_chunks:
        return {"error": f"Source document '{source_doc_id}' not found"}

    combined_source = " ".join(c.get("text", "") for c in source_chunks)[:5000]
    result = verifier.verify(fact, combined_source, source_doc_id)
    return result


def do_deepsearch(query, target=50, download=True):
    if not _DEEP_SEARCH_AVAILABLE:
        return {"error": "Deep search not available"}
    engine = AcademicSearchEngine(target=target)
    
    # Run async search in a separate thread to avoid asyncio.run() conflict
    # with the running uvicorn event loop
    result_holder = [None]
    def _run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder[0] = loop.run_until_complete(engine.search(query))
        finally:
            loop.close()
    t = threading.Thread(target=_run_async)
    t.start()
    t.join()
    papers, stats = result_holder[0]
    
    result = {
        "query": query,
        "stats": stats,
        "papers": [
            {
                "title": p.title,
                "authors": p.authors[:3],
                "year": p.year,
                "doi": p.doi,
                "url": p.url,
                "pdf_url": p.pdf_url,
                "source_api": p.source_api,
                "citations": p.citation_count,
                "relevance": round(p.relevance_score, 2),
            } for p in papers[:target]
        ],
    }
    if download and papers:
        state = _ensure_init()
        dl = PDFDownloader(papers_dir=state["papers_dir"])
        # Also fix PDFDownloader sync call
        def _run_download():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                download_limit = min(len(papers), target)
                return loop.run_until_complete(dl.download(papers[:download_limit]))
            finally:
                loop.close()
        t2 = threading.Thread(target=_run_download)
        t2.start()
        t2.join()
        # Note: download returns nothing meaningful currently, skip counting
    return result


# ===== HTTP handlers =====

async def health(request):
    return JSONResponse({"status": "ok", "version": "2.0"})


async def mcp_handle(request):
    # MCP over JSON-RPC: usually POST with JSON body.
    # Your client does GET /mcp for probing, so handle it gracefully.
    if request.method == "GET":
        accept = (request.headers.get("accept") or "").lower()
        if "text/event-stream" in accept:
            from starlette.responses import StreamingResponse

            async def _event_stream():
                yield b": mcp sse probe\n\n"
                # More compatible MCP SSE payload: JSON-RPC shaped "result"
                yield b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":null,\"result\":{}}\n\n"
                # Keep connection open (some clients expect long-lived SSE)
                while True:
                    yield b"event: heartbeat\ndata: {}\n\n"
                    await asyncio.sleep(15)

            return StreamingResponse(
                _event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return JSONResponse({
            "status": "ok",
            "path": "/mcp",
            "hint": "Send JSON-RPC 2.0 POST to /mcp with {id, method, params}.",
            "supported_jsonrpc_methods": ["initialize", "tools/list", "tools/call"],
            "version": "2.0",
        })

    try:
        # Be defensive: some MCP clients (especially when misconfigured for SSE/stream)
        # may POST empty/non-JSON bodies. Return JSON-RPC error instead of 500.
        raw = await request.body()
        try:
            body = json.loads(raw.decode("utf-8") if raw else "")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Invalid JSON in request body"},
                },
                status_code=400,
            )

        method = body.get("method", "")
        params = body.get("params", {})
        req_id = body.get("id")

        # Notifications have no id — just acknowledge
        if req_id is None:
            return JSONResponse({})

        def ok(result):
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})

        def ok_text(text):
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": text}]}})

        def err(msg, code=-32000):
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}})

    except Exception as e:
        logger.exception("mcp_handle crashed")
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}},
            status_code=500,
        )

    try:
        if method == "initialize":
            return ok({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "scirag", "version": "2.0"},
            })

        elif method == "tools/list":
            return ok({
                "tools": [
                    {
                        "name": "scirag_status",
                        "description": "Check SciRAG pipeline status and statistics.",
                        "inputSchema": {"type": "object", "properties": {}}
                    },
                    {
                        "name": "scirag_index",
                        "description": "Index all academic papers in a directory for SciRAG retrieval.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "papers_dir": {"type": "string", "description": "Path to directory containing PDF/DOCX/TXT papers"}
                            },
                            "required": ["papers_dir"]
                        }
                    },
                    {
                        "name": "scirag_search",
                        "description": "Search indexed papers for chunks relevant to a research query.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer", "default": 10}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "scirag_synthesize",
                        "description": "Full SciRAG synthesis: outline -> retrieval -> extraction -> synthesis -> verification.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_sections": {"type": "integer", "default": 6}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "scirag_deepsearch",
                        "description": "Deep search for academic papers across OpenAlex, CrossRef, ArXiv, Semantic Scholar.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "target": {"type": "integer", "default": 50},
                                "download": {"type": "boolean", "default": True}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "scirag_extract",
                        "description": "Extract structured facts from papers relevant to a query.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer", "default": 5}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "scirag_verify",
                        "description": "Verify a factual claim against indexed source documents.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "fact": {"type": "string"},
                                "source_doc_id": {"type": "string", "default": ""}
                            },
                            "required": ["fact"]
                        }
                    },
                ]
            })

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            if tool_name == "scirag_status":
                return ok_text(json.dumps(do_status(), indent=2))
            elif tool_name == "scirag_index":
                return ok_text(json.dumps(do_index(tool_args.get("papers_dir", "./papers")), indent=2))
            elif tool_name == "scirag_search":
                return ok_text(json.dumps(do_search(tool_args.get("query", ""), tool_args.get("top_k", 10)), indent=2))
            elif tool_name == "scirag_synthesize":
                return ok_text(json.dumps(do_synthesize(tool_args.get("query", ""), tool_args.get("max_sections", 6)), indent=2))
            elif tool_name == "scirag_deepsearch":
                return ok_text(json.dumps(do_deepsearch(
                    tool_args.get("query", ""),
                    tool_args.get("target", 50),
                    tool_args.get("download", True)
                ), indent=2))
            elif tool_name == "scirag_extract":
                return ok_text(json.dumps(do_extract(tool_args.get("query", ""), tool_args.get("top_k", 5)), indent=2))
            elif tool_name == "scirag_verify":
                return ok_text(json.dumps(do_verify(tool_args.get("fact", ""), tool_args.get("source_doc_id", "")), indent=2))
            else:
                return err(f"Unknown tool: {tool_name}", -32601)

        elif method == "scirag_status":
            return ok_text(json.dumps(do_status(), indent=2))
        elif method == "scirag_index":
            return ok_text(json.dumps(do_index(params.get("papers_dir", "./papers")), indent=2))
        elif method == "scirag_search":
            return ok_text(json.dumps(do_search(params.get("query", ""), params.get("top_k", 10)), indent=2))
        elif method == "scirag_synthesize":
            return ok_text(json.dumps(do_synthesize(params.get("query", ""), params.get("max_sections", 6)), indent=2))
        elif method == "scirag_extract":
            return ok_text(json.dumps(do_extract(params.get("query", ""), params.get("top_k", 5)), indent=2))
        elif method == "scirag_verify":
            return ok_text(json.dumps(do_verify(params.get("fact", ""), params.get("source_doc_id", "")), indent=2))
        elif method == "scirag_deepsearch":
            return ok_text(json.dumps(do_deepsearch(params.get("query", ""), params.get("target", 50), params.get("download", True)), indent=2))
        else:
            return err(f"Unknown method: {method}", -32601)

    except Exception as e:
        logger.exception("API error")
        return err(str(e))


# Create app
app = Starlette(
    debug=False,
    routes=[
        Route("/", health),
        Route("/health", health),
        Route("/mcp", mcp_handle, methods=["GET", "POST"]),
    ],
)


if __name__ == "__main__":
    import uvicorn
    import traceback
    print("=" * 60)
    print("SciRAG HTTP server starting...")
    print("=" * 60)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        print(f"FATAL: Server failed to start: {e}")
        traceback.print_exc()
        sys.exit(1)
