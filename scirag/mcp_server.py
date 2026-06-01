#!/usr/bin/env python3
"""
SciRAG MCP Server — Windows-safe UTF-8 version.

This file MUST keep stdout as clean UTF-8 JSON-RPC only.
All logs and diagnostics go to stderr.
"""

# ========== STEP 1: IMPORTS ==========
import io
import sys


# NOTE:
# Не переназначаем sys.stdout/sys.stderr при импорте модуля — pytest-у capture
# это ломает (ValueError: I/O operation on closed file).
# Вместо этого настраиваем UTF-8 streams только при запуске как main.
def _configure_utf8_streams() -> None:
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='strict',
            line_buffering=True,
        )

    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True,
        )

# ========== STEP 2: LOGGING TO STDERR ONLY ==========
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("scirag-mcp")

# ========== STEP 3: IMPORTS WITH GUARDED STDOUT ==========
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports for MCP SDK
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    logger.error("mcp SDK not installed. Run: pip install mcp")
    sys.exit(1)

# SciRAG imports (these may print; stdout is now locked)
from scirag.iterative_synthesizer import IterativeSynthesizer
from scirag.outline_generator import OutlineGenerator
from scirag.state import (
    _build_retriever,
    _ensure_init,
    _load_chunks,
    do_status,
)
from scirag.tree_node import GapCriticTree, TreeNode
from scirag.verifier import FactVerifier

_symbolic_reasoner_cls = None

def _get_symbolic_reasoner(client, config):
    global _symbolic_reasoner_cls
    if _symbolic_reasoner_cls is None:
        from scirag.symbolic_reasoning import SymbolicReasoner
        _symbolic_reasoner_cls = SymbolicReasoner
    return _symbolic_reasoner_cls(client, config)

# ========== STEP 4: CREATE MCP SERVER ==========
mcp = FastMCP("scirag")

# Strict tool exposure: keep only two FULL-report UX endpoints.
_allowed_tools = {"scirag_full_report_by_query", "scirag_full_report_refine"}
try:
    for tool_name in [
        "scirag_verify",
        "scirag_status",
        "scirag_theme_open",
        "scirag_theme_followup",
        "scirag_full_report",
        "scirag_followup_report",
        "scirag_index",
        "scirag_search",
        "scirag_synthesize",
        "scirag_deepsearch",
        "scirag_extract",
        "scirag_full_report",
    ]:
        if tool_name not in _allowed_tools:
            mcp.remove_tool(tool_name)
except Exception:
    # Non-fatal: tool removal may differ across mcp SDK versions.
    pass

# Backward-compatible alias expected by tests
_mcp_tool = mcp.tool

# ========== STEP 5: STATE ==========
# NOTE: _server_state, _ensure_init, _build_retriever, _load_chunks are imported
# from scirag.state so that api.py (HTTP server) and mcp_server.py (FastMCP)
# share the same in-memory state when running in a single process.


# ========== MCP TOOLS ==========

def scirag_index(papers_dir: str) -> str:
    """Index all academic papers in a directory for SciRAG retrieval."""
    t0 = time.time()
    state = _ensure_init(papers_dir)
    if "error" in state:
        return json.dumps(state, indent=2)

    chunks = _load_chunks(state["papers_dir"])
    state["chunks"] = chunks
    state["graph"].build_from_chunks(chunks)
    retriever = _build_retriever(state)
    state["index_time"] = time.time() - t0
    state["retriever"] = retriever

    return json.dumps({
        "status": "indexed",
        "papers_dir": state["papers_dir"],
        "chunks": len(chunks),
        "graph": state["graph"].stats,
        "time_sec": round(state["index_time"], 1),
    }, indent=2)


def scirag_search(query: str, top_k: int = 10) -> str:
    """Search indexed papers for chunks relevant to a research query."""
    state = _ensure_init()
    if "error" in state:
        return json.dumps(state, indent=2)
    if not state["retriever"]:
        return json.dumps({"error": "No index. Run scirag_index first."}, indent=2)

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

    state["last_query"] = query
    return json.dumps({
        "query": query,
        "results": results,
        "total_found": len(chunks),
    }, indent=2)


def scirag_synthesize(query: str, max_sections: int = 6) -> str:
    """Full SciRAG synthesis: outline -> retrieval -> extraction -> synthesis -> verification."""
    state = _ensure_init()
    if "error" in state:
        return json.dumps(state, indent=2)
    if not state["retriever"]:
        return json.dumps({"error": "No index. Run scirag_index first."}, indent=2)

    t0 = time.time()
    config = state["config"]
    client = state["client"]

    outline_gen = OutlineGenerator(client, config)
    synthesizer = IterativeSynthesizer(client, config, state["graph"])
    verifier = FactVerifier(client, config)

    root = outline_gen.generate(query, max_sections=max_sections)
    root = outline_gen.critique_and_expand(root, query)

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
    if papers_for_symbolic:
        try:
            symbolic_result = state["symbolic"].run(
                papers=papers_for_symbolic,
                query=query,
                outline=root,
            )
        except Exception as e:
            logger.warning(f"Symbolic reasoning failed: {e}")

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
    return json.dumps(result, indent=2)


def scirag_deepsearch(query: str, target: int = 50, download: bool = True) -> str:
    """Deep search for academic papers across OpenAlex, CrossRef, ArXiv, Semantic Scholar."""
    try:
        from scirag.deep_search import AcademicSearchEngine, PDFDownloader

        logger.info(f"MCP deepsearch: '{query}' (target: {target})")
        engine = AcademicSearchEngine(target=target)
        papers, stats = engine.search_sync(query)

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
                }
                for p in papers[:target]
            ],
        }

        if download and papers:
            state = _ensure_init()
            dl = PDFDownloader(papers_dir=state["papers_dir"])
            download_limit = min(len(papers), target)
            downloaded = dl.download_sync(papers[:download_limit])
            result["downloaded"] = len(downloaded)

        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Deepsearch error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def scirag_extract(query: str, top_k: int = 5) -> str:
    """Extract structured facts from papers relevant to a query."""
    state = _ensure_init()
    if "error" in state:
        return json.dumps(state, indent=2)
    if not state["retriever"]:
        return json.dumps({"error": "No index. Run scirag_index first."}, indent=2)

    config = state["config"]
    client = state["client"]
    synthesizer = IterativeSynthesizer(client, config, state["graph"])

    node = TreeNode(title=query, description=query, keywords=[query])
    chunks = state["retriever"].retrieve(query, top_k=top_k)
    facts = synthesizer._extract_facts(chunks, node)

    return json.dumps({
        "query": query,
        "facts": [f.to_dict() for f in facts],
        "count": len(facts),
    }, indent=2)


def scirag_verify(fact: str, source_doc_id: str = "") -> str:
    """Verify a factual claim against indexed source documents."""
    state = _ensure_init()
    if "error" in state:
        return json.dumps(state, indent=2)

    verifier = FactVerifier(state["client"], state["config"])
    source_chunks = [c for c in state["chunks"]
                     if not source_doc_id or c.get("document_id") == source_doc_id]

    if not source_chunks:
        return json.dumps({
        "error": f"Source document '{source_doc_id}' not found"
    }, indent=2)

    combined_source = " ".join(c.get("text", "") for c in source_chunks)[:5000]
    result = verifier.verify(fact, combined_source, source_doc_id)

    return json.dumps(result, indent=2)


def scirag_status() -> str:
    """Check SciRAG pipeline status and statistics."""
    return json.dumps(do_status(), indent=2)


# =========================
# Theme materialization MCP
# =========================

def scirag_theme_open(
    query: str,
    used_doc_ids: List[str],
    base_papers_dir: Optional[str] = None,
    resume: bool = True,
) -> str:
    """
    Create/open theme materialization from an evidence set.

    - theme_id derived from query deterministically
    - materialize used/remaining into: papers/themes/<theme_id>/{used,remaining,meta.json}
    """
    try:
        from scirag.theme_materialize import theme_materialize_from_query
        state = _ensure_init()
        base_dir = base_papers_dir or state.get("papers_dir") or "./papers"

        res = theme_materialize_from_query(
            base_papers_dir=base_dir,
            query=query,
            used_doc_ids=used_doc_ids,
            resume=resume,
        )
        return json.dumps(res, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def scirag_theme_followup(
    theme_id: str,
    query: str,
    used_doc_ids_add: List[str],
    base_papers_dir: Optional[str] = None,
    resume: bool = True,
) -> str:
    """
    Incrementally update a theme after follow-up expansion.

    Contract (minimal):
    - reads prev used_doc_ids from meta.json under papers/themes/<theme_id>/meta.json
    - merges with used_doc_ids_add (dedupe)
    - re-materializes used/remaining idempotently with resume=True/False
    """
    try:
        from scirag.theme_materialize import (
            _read_meta,  # type: ignore[attr-defined]
            theme_materialize_from_query,
        )
        from scirag.theme_materialize import theme_id as compute_theme_id

        state = _ensure_init()
        base_dir = base_papers_dir or state.get("papers_dir") or "./papers"

        computed = compute_theme_id(query)
        if theme_id != computed:
            return json.dumps(
                {
                    "error": "theme_id mismatch: provided theme_id doesn't match theme_id(query). Please pass consistent theme_id/query.",
                    "provided_theme_id": theme_id,
                    "computed_theme_id": computed,
                },
                indent=2,
            )

        theme_dir = Path("papers") / "themes" / theme_id
        meta_path = theme_dir / "meta.json"
        prev_meta = _read_meta(meta_path)
        prev_used = prev_meta.get("used_doc_ids", []) if isinstance(prev_meta, dict) else []

        merged = list({str(x) for x in (prev_used + list(used_doc_ids_add or [])) if x})

        res = theme_materialize_from_query(
            base_papers_dir=base_dir,
            query=query,
            used_doc_ids=merged,
            resume=resume,
        )

        res["theme_id"] = theme_id
        return json.dumps(res, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# =========================
# Full/Follow-up report MCP (UX-friendly wrappers)
# =========================

# report_id -> internal evidence-session mapping (in-process)
# report_id is returned to the user; the client never handles evidence sets directly.
_REPORT_STORE: Dict[str, dict] = {}


def _make_report_id(prefix: str) -> str:
    # Always unique, even for rapid successive calls.
    import uuid
    return f"{prefix}_{uuid.uuid4().hex}"


@mcp.tool()
def scirag_full_report_by_query(
    query: str,
    target_papers: int = 70,
    coverage_mode: str = "balanced",
    style: str = "engineer",
    max_sections: int = 6,
    verification_level: str = "basic",
    freshness: str = "local_only",
    session_store: str = "in_memory",
    random_seed: Optional[int] = None,
) -> str:
    """
    UX wrapper:
    - returns report_id for a FULL report
    - client doesn't pass evidence/used/remaining, server hides it
    """
    try:
        # 1) run existing full report (returns JSON string)
        full_json = scirag_full_report(
            query=query,
            target_papers=target_papers,
            coverage_mode=coverage_mode,
            style=style,
            max_sections=max_sections,
            verification_level=verification_level,
            freshness=freshness,
            session_store=session_store,
            random_seed=random_seed,
        )
        full_data = json.loads(full_json)

        # 2) mapping
        session_id = full_data.get("session_id")
        if not session_id:
            return json.dumps({"error": "full report returned no session_id"}, indent=2)

        from scirag.theme_materialize import theme_id as compute_theme_id

        base_query = query
        t_id = compute_theme_id(base_query)

        report_id = _make_report_id(prefix="full")

        _REPORT_STORE[report_id] = {
            "kind": "full",
            "parent_report_id": None,
            "session_id": session_id,
            "base_query": base_query,
            "theme_id": t_id,
        }

        # 3) return minimal payload + mark as FULL
        return json.dumps(
            {
                "report_id": report_id,
                "report_type": "full",
                "final_markdown": full_data.get("final_markdown", ""),
                "session_id": session_id,
                "theme_id": t_id,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def scirag_full_report_refine(
    report_id: str,
    clarification_text: str,
    max_sections: int = 2,
    evidence_target: str = "expand_if_needed",
    extra_papers_per_iteration: int = 10,
    max_iterations: int = 2,
    verification_level: str = "basic",
) -> str:
    """
    UX wrapper:
    - returns NEW report_id on each refinement
    - refinement is incremental and NOT a full report
    - uses evidence session referenced by report_id
    """
    try:
        rec = _REPORT_STORE.get(report_id)
        if not rec:
            return json.dumps({"error": f"Unknown report_id: {report_id}"}, indent=2)

        session_id = rec["session_id"]
        base_query = rec["base_query"]
        t_id = rec["theme_id"]

        # 1) run existing follow-up restricted by session's evidence set
        follow_json = scirag_followup_report(
            session_id=session_id,
            followup_query=clarification_text,
            max_sections=max_sections,
            evidence_target=evidence_target,
            extra_papers_per_iteration=extra_papers_per_iteration,
            max_iterations=max_iterations,
            verification_level=verification_level,
        )
        follow_data = json.loads(follow_json)

        new_report_id = _make_report_id(prefix="followup")

        # 2) update topic-folders incrementally (efficient)
        # We derive "added" by comparing prev used_doc_ids to new used_doc_ids in meta.json.
        # Note: scirag_theme_followup reads prev used from meta.json, merges with used_doc_ids_add.
        from scirag.theme_materialize import theme_id as _compute_theme_id  # ensure base_query stable
        computed = _compute_theme_id(base_query)
        if computed != t_id:
            # fail-safe: update with recomputed theme_id
            t_id = computed

        theme_dir = Path("papers") / "themes" / t_id
        meta_path = theme_dir / "meta.json"
        from scirag.theme_materialize import _read_meta  # type: ignore[attr-defined]
        _read_meta(meta_path)

        # follow-up response doesn't directly return used_doc_ids list in current implementation.
        # fallback: use follow-up session selected_doc_ids as ground truth by reusing session_id.
        # easiest: re-run minimal access via session store logic isn't exposed here;
        # so we instead conservatively pass "no additions" if we can't detect.
        # However: follow-up_report internally updates session["selected_doc_ids"].
        # We'll approximate added by:
        # - if follow_data contains used_evidence.added_paper_ids (it does in current code path)
        added_ids = []
        used_evidence = follow_data.get("used_evidence", {}) if isinstance(follow_data, dict) else {}
        if isinstance(used_evidence, dict):
            added_ids = used_evidence.get("added_paper_ids", []) or []

        # ensure list
        used_doc_ids_add = [str(x) for x in added_ids]

        # materialize update
        _ = scirag_theme_followup(
            theme_id=t_id,
            query=base_query,
            used_doc_ids_add=used_doc_ids_add,
            base_papers_dir=None,
            resume=True,
        )

        _REPORT_STORE[new_report_id] = {
            "kind": "followup",
            "parent_report_id": report_id,
            "session_id": session_id,
            "base_query": base_query,
            "theme_id": t_id,
        }

        return json.dumps(
            {
                "report_id": new_report_id,
                "report_type": "followup_incremental",
                "parent_report_id": report_id,
                "final_markdown": follow_data.get("final_markdown", ""),
                "added_paper_ids_count": len(used_doc_ids_add),
                "theme_id": t_id,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# =========================
# Full/Follow-up report MCP
# =========================

# Lazy module-level singletons for session management (kept in-process)
from scirag.deep_search.coverage_planner import CandidatePaper, select_evidence_papers
from scirag.deep_search.evidence_binder import bind_retriever_to_chunks_subset
from scirag.deep_search.mcp_contracts import (
    FollowupReportResponse,
    FullReportResponse,
    make_new_session_id,
)
from scirag.deep_search.session_store import (
    DiskPaths,
    DiskSessionStore,
    HybridSessionStore,
    InMemorySessionStore,
)
from scirag.deep_search.sufficiency_evaluator import default_sufficiency_policy, evaluate_information_sufficiency

_SESSION_STORE: Optional[HybridSessionStore] = None


def _hash_doc_ids(doc_ids: List[str]) -> str:
    import hashlib
    # deterministic fingerprint independent of order
    joined = "\n".join(sorted([str(x) for x in doc_ids if x is not None]))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _get_session_store(state) -> HybridSessionStore:
    global _SESSION_STORE
    if _SESSION_STORE is not None:
        return _SESSION_STORE

    papers_root = Path(state["papers_dir"]) if state.get("papers_dir") else Path("./papers")
    disk_root = str(papers_root.parent / "scirag_data" / "sessions")
    mem = InMemorySessionStore()
    disk = DiskSessionStore(DiskPaths(root_dir=disk_root))
    _SESSION_STORE = HybridSessionStore(memory=mem, disk=disk)
    return _SESSION_STORE


def _chunk_candidates_from_retrieval(query: str, candidates_chunks: List[object], state: dict) -> List[CandidatePaper]:
    """
    Convert retrieved chunk objects into paper-level candidates expected by coverage_planner.
    We use the first seen chunk per document_id as representative (title is also best-effort).
    """
    by_doc: Dict[str, CandidatePaper] = {}
    for c in candidates_chunks:
        did = getattr(c, "document_id", None)
        if not did:
            if hasattr(c, "get"):
                did = c.get("document_id")
        if not did:
            continue

        if did in by_doc:
            continue

        # best-effort title
        title = did
        meta = getattr(c, "metadata", None) if hasattr(c, "metadata") else (c.get("metadata", {}) if hasattr(c, "get") else {})
        if isinstance(meta, dict):
            title = meta.get("title", did) or did
        # classify tag by representative chunk text
        text = getattr(c, "text", "") if hasattr(c, "text") else (c.get("text", "") if hasattr(c, "get") else "")
        primary_tag = state["classifier"].get_primary_tag(text) if state.get("classifier") else "T"

        by_doc[did] = CandidatePaper(
            paper_id=did,
            title=title,
            tags={"primary_tag": primary_tag},
            score=None,
            relevance_reasons=[f"derived_from_retrieval_for_query:{query}"],
            year=None,
        )

    # preserve original retrieval order (by_doc preserves insertion order in py3.7+)
    return list(by_doc.values())


def _build_evidence_session_payload(
    *,
    session_id: str,
    query: str,
    evidence_paper_ids: List[str],
    selected_doc_ids: List[str],
    coverage_report: dict,
    config_snapshot: dict,
) -> dict:
    return {
        "session_id": session_id,
        "query": query,
        "created_at": time.time(),
        "evidence_papers": evidence_paper_ids,
        "evidence_paper_source": ["local_index"] * len(evidence_paper_ids),
        "selected_doc_ids": selected_doc_ids,
        "selected_chunk_ids": [],
        "coverage_report": coverage_report,
        "config_snapshot": config_snapshot,
        "synthesis": {},
        "last_sufficiency": {},
    }


def _run_report_synthesis_on_evidence(
    *,
    query: str,
    max_sections: int,
    verification_level: str,
    evidence_doc_ids: List[str],
    state: dict,
) -> dict:
    """
    Restrict synthesis by building a temporary retriever over only evidence subset chunks.
    """
    all_chunks = state["chunks"]
    doc_ids_set = set(evidence_doc_ids)

    subset_chunks, restricted_retriever = bind_retriever_to_chunks_subset(
        all_chunks=all_chunks,
        base_retriever=state["retriever"],
        doc_ids=doc_ids_set,
    )

    if not subset_chunks:
        return {
            "error": "No chunks found for selected evidence papers. Evidence subset is empty.",
            "restricted_chunks": 0,
        }

    t0 = time.time()
    config = state["config"]
    client = state["client"]

    outline_gen = OutlineGenerator(client, config)
    synthesizer = IterativeSynthesizer(client, config, state["graph"])
    verifier = FactVerifier(client, config)

    # Outline is based on query; actual factual extraction is restricted via restricted_retriever
    root = outline_gen.generate(query, max_sections=max_sections)
    root = outline_gen.critique_and_expand(root, query)

    # Synthesize restricted
    root = synthesizer.synthesize(root, restricted_retriever)

    # Verification is also restricted by building source_chunks from subset doc_ids
    all_leaves = root.get_leaves()
    all_facts = [{"text": f.text, "source_doc": f.source_doc} for leaf in all_leaves for f in leaf.facts]

    source_chunks = []
    seen_docs = set()
    for c in all_chunks:
        did = c.get("document_id") if hasattr(c, "get") else getattr(c, "document_id", None)
        if not did:
            continue
        if did in doc_ids_set and did not in seen_docs:
            source_chunks.append(c)
            seen_docs.add(did)

    verify_results = verifier.verify_batch(all_facts, source_chunks)
    fact_idx = 0
    for leaf in all_leaves:
        for f in leaf.facts:
            if fact_idx < len(verify_results):
                v = verify_results[fact_idx]
                f.verified = v.get("verified", False)
                f.confidence = v.get("confidence", 0.0)
                fact_idx += 1

    final_text = synthesizer.compile_final(root)

    gap_critic = GapCriticTree(
        llm_generate=lambda n, ctx: "",
        llm_evaluate=lambda n, ctx: 0.0,
        max_depth=config.max_iterations,
    )
    tree_stats = gap_critic.tree_stats(root)

    total_facts = sum(len(leaf.facts) for leaf in all_leaves)
    verified_facts = sum(1 for leaf in all_leaves for f in leaf.facts if getattr(f, "verified", False))

    # Best-effort: evaluator loop relies on these fields
    stats = {
        "total_time_sec": round(time.time() - t0, 1),
        "total_sections": len(root.children),
        "total_leaves": len(all_leaves),
        "total_facts": total_facts,
        "verified_facts": verified_facts,
        "verification_rate": f"{verified_facts}/{total_facts} = {verified_facts/max(total_facts,1)*100:.0f}%",
        "avg_confidence": round(sum(getattr(leaf, "confidence", 0.0) for leaf in all_leaves) / max(len(all_leaves), 1), 2),
        "with_gaps": tree_stats.get("with_gaps", 0),
        "max_depth": tree_stats.get("max_depth", 0),
    }

    return {
        "query": query,
        "final_text": final_text,
        "tree": {
            "sections": len(root.children),
            "leaves": len(all_leaves),
            "max_depth": tree_stats.get("max_depth", 0),
            "with_gaps": tree_stats.get("with_gaps", 0),
        },
        "stats": stats,
    }


def scirag_full_report(
    query: str,
    target_papers: int = 70,
    coverage_mode: str = "balanced",
    style: str = "engineer",
    max_sections: int = 6,
    verification_level: str = "basic",
    freshness: str = "local_only",
    session_store: str = "in_memory",
    random_seed: Optional[int] = None,
) -> str:
    """
    MCP full report with evidence set of ~70 papers.
    Stores selected papers in session for follow-up.
    """
    t0 = time.time()
    state = _ensure_init()
    if "error" in state:
        return json.dumps(state, indent=2)
    if not state["retriever"]:
        return json.dumps({"error": "No index. Run scirag_index first."}, indent=2)

    # Step 1: retrieve candidate chunks for planning (local only)
    candidates_chunks = state["retriever"].retrieve(query, top_k=max(200, target_papers * 4))
    candidates = _chunk_candidates_from_retrieval(query, candidates_chunks, state)

    # Step 2: select evidence papers with planner
    selected, clusters = select_evidence_papers(
        candidates=candidates,
        target_papers=target_papers,
        coverage_mode=coverage_mode,  # type: ignore[arg-type]
        seed=random_seed,
    )
    evidence_paper_ids = [p.paper_id for p in selected]
    selected_doc_ids = evidence_paper_ids[:]  # we use doc_ids==paper_ids for local index

    coverage_report = {
        "overall": {
            "total_papers_selected": len(evidence_paper_ids),
            "clusters": len(clusters),
            "diversity_metrics": {
                "tag_counts": {k: sum(1 for p in selected if (p.tags or {}).get("primary_tag") == k) for k in ["T", "E", "M", "A"]},
            },
        },
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "cluster_label": c.cluster_label,
                "paper_ids": c.paper_ids,
                "budget_alloc": c.budget_alloc,
                "coverage_notes": c.coverage_notes,
                "gaps_estimate": c.gaps_estimate,
            }
            for c in clusters
        ],
    }

    session_id = make_new_session_id(prefix="full")
    policy = default_sufficiency_policy()
    evidence_version = _hash_doc_ids(evidence_paper_ids)

    config_snapshot = {
        "target_papers": target_papers,
        "coverage_mode": coverage_mode,
        "style": style,
        "max_sections": max_sections,
        "verification_level": verification_level,
        "freshness": freshness,
        "session_store": session_store,
        "random_seed": random_seed,
        "sufficiency_policy": policy,
        "evidence_version": evidence_version,
    }

    session_store_obj = _get_session_store(state)
    session_payload = _build_evidence_session_payload(
        session_id=session_id,
        query=query,
        evidence_paper_ids=evidence_paper_ids,
        selected_doc_ids=selected_doc_ids,
        coverage_report=coverage_report,
        config_snapshot=config_snapshot,
    )
    session_store_obj.put(session_payload)  # persist before follow-up

    # Step 3: run synthesis restricted to evidence
    synth_res = _run_report_synthesis_on_evidence(
        query=query,
        max_sections=max_sections,
        verification_level=verification_level,
        evidence_doc_ids=selected_doc_ids,
        state=state,
    )
    if "error" in synth_res:
        return json.dumps({"error": synth_res["error"], "session_id": session_id}, indent=2)

    # Step 4: run sufficiency evaluation (mostly to populate audit; follow-up will do auto-expand)
    gaps_summary = ["gap_proxy_with_gaps_count"] if synth_res["tree"].get("with_gaps") else []
    suff_passed, suff_checks = evaluate_information_sufficiency(
        verification_summary={
            "verified_facts": synth_res["stats"]["verified_facts"],
            "total_facts": synth_res["stats"]["total_facts"],
            "with_gaps": synth_res["tree"]["with_gaps"],
        },
        attribution_summary=None,
        gaps_summary=gaps_summary,
        sufficiency_policy=policy,
    )

    session_payload["synthesis"] = {
        "final_text": synth_res["final_text"],
        "stats": synth_res["stats"],
    }
    session_payload["last_sufficiency"] = {
        "passed": suff_passed,
        **suff_checks,
    }
    session_store_obj.put(session_payload)

    result: FullReportResponse = {
        "session_id": session_id,
        "query": query,
        "final_markdown": synth_res["final_text"],
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "tags": p.tags,
                "score": p.score,
                "relevance_reasons": p.relevance_reasons,
                "cluster_id": (p.tags or {}).get("primary_tag") if p.tags else None,
            }
            for p in selected
        ],
        "coverage_report": coverage_report,
        "verification": {
            "level": verification_level,
            "verified_facts": synth_res["stats"]["verified_facts"],
            "total_facts": synth_res["stats"]["total_facts"],
            "verification_rate": synth_res["stats"].get("verification_rate"),
        },
        "extraction": {
            "facts_total": synth_res["stats"]["total_facts"],
            "evidence_types": {},
        },
        "sufficiency": {
            "passed": suff_passed,
            "checks": suff_checks,
        },
        "audit": {
            "selected_papers_count": len(selected),
            "restricted_doc_ids": len(selected_doc_ids),
            "chunk_candidates_top_k": len(candidates_chunks),
            "synthesis_time_sec": synth_res["stats"]["total_time_sec"],
            "auto_expansions": [],
            "style": style,
            "freshness": freshness,
            "seed": random_seed,
            "total_time_sec": round(time.time() - t0, 1),
        },
    }

    return json.dumps(result, indent=2)


def scirag_followup_report(
    session_id: str,
    followup_query: str,
    max_sections: int = 2,
    evidence_target: str = "expand_if_needed",
    extra_papers_per_iteration: int = 10,
    max_iterations: int = 2,
    verification_level: str = "basic",
) -> str:
    """
    Follow-up report constrained to the session's evidence set.
    If sufficiency fails, expand evidence by +10 papers and re-run.
    """
    state = _ensure_init()
    if "error" in state:
        return json.dumps(state, indent=2)
    if not state["retriever"]:
        return json.dumps({"error": "No index. Run scirag_index first."}, indent=2)

    store = _get_session_store(state)

    try:
        session: dict = store.get(session_id)
    except Exception as e:
        return json.dumps({"error": f"Session not found: {session_id}", "details": str(e)}, indent=2)

    base_doc_ids: List[str] = session.get("selected_doc_ids", session.get("evidence_papers", [])) or []
    if not base_doc_ids:
        return json.dumps({"error": "Session has no evidence papers to follow up on.", "session_id": session_id}, indent=2)

    # Use policy stored in session for strict reproducibility
    thresholds_policy = session.get("config_snapshot", {}).get("sufficiency_policy") or default_sufficiency_policy()

    iterations_done = 0
    papers_added: List[str] = []
    audit_iterations: List[dict] = []

    current_doc_ids = base_doc_ids[:]

    for it in range(max_iterations + 1):
        iterations_done = it + 1
        # 1) run synthesis restricted to current evidence subset
        synth_res = _run_report_synthesis_on_evidence(
            query=followup_query,
            max_sections=max_sections,
            verification_level=verification_level,
            evidence_doc_ids=current_doc_ids,
            state=state,
        )
        if "error" in synth_res:
            return json.dumps({"error": synth_res["error"], "session_id": session_id}, indent=2)

        gaps_summary = ["gap_proxy_with_gaps_count"] if synth_res["tree"].get("with_gaps") else []
        passed, checks = evaluate_information_sufficiency(
            verification_summary={
                "verified_facts": synth_res["stats"]["verified_facts"],
                "total_facts": synth_res["stats"]["total_facts"],
                "with_gaps": synth_res["tree"]["with_gaps"],
            },
            attribution_summary=None,
            gaps_summary=gaps_summary,
            sufficiency_policy=thresholds_policy,
        )

        audit_iterations.append({
            "iteration": it,
            "evidence_doc_ids_count": len(current_doc_ids),
            "evidence_version": _hash_doc_ids(current_doc_ids),
            "verified_facts": synth_res["stats"]["verified_facts"],
            "total_facts": synth_res["stats"]["total_facts"],
            "with_gaps": synth_res["tree"]["with_gaps"],
            "passed": passed,
            "expanded_this_iteration": False,
            "added_paper_ids": [],
        })

        # stop if passed or policy doesn't allow expand
        if passed or evidence_target == "same_set":
            # update session last_sufficiency and synthesis snapshot for traceability
            session["last_sufficiency"] = {"passed": passed, **checks}
            session["synthesis"] = session.get("synthesis", {})
            store.put(session)
            result: FollowupReportResponse = {
                "session_id": session_id,
                "followup_query": followup_query,
                "final_markdown": synth_res["final_text"],
                "used_evidence": {
                    "paper_ids_total": len(current_doc_ids),
                    "added_paper_ids": papers_added,
                },
                "papers_added_this_turn": len(papers_added),
                "sufficiency": {"passed": passed, "checks": checks, "iterations": iterations_done, "gaps_summary": gaps_summary},
                "verification": {
                    "level": verification_level,
                    "verified_facts": synth_res["stats"]["verified_facts"],
                    "total_facts": synth_res["stats"]["total_facts"],
                    "attribution_coverage": None,
                },
                "audit": {
                    "iteration_plans": audit_iterations,
                    "retrieval_queries": [followup_query],
                },
            }
            return json.dumps(result, indent=2)

        # else: expand by +10 papers (avoid duplicates)
        # retrieve candidates for follow-up query from full index
        candidate_chunks = state["retriever"].retrieve(followup_query, top_k=max(250, (len(current_doc_ids) // 2) + 200))
        candidate_papers = _chunk_candidates_from_retrieval(followup_query, candidate_chunks, state)

        existing = set(current_doc_ids)
        filtered_candidates = [p for p in candidate_papers if p.paper_id not in existing]

        extra_target = extra_papers_per_iteration
        selected_extra, _clusters = select_evidence_papers(
            candidates=filtered_candidates,
            target_papers=extra_target,
            coverage_mode=session.get("config_snapshot", {}).get("coverage_mode", "balanced"),  # type: ignore[arg-type]
            seed=None,
        )

        added_ids = [p.paper_id for p in selected_extra]
        if not added_ids:
            # forced stop because can't expand
            session["last_sufficiency"] = {"passed": passed, **checks}
            store.put(session)
            result: FollowupReportResponse = {
                "session_id": session_id,
                "followup_query": followup_query,
                "final_markdown": synth_res["final_text"],
                "used_evidence": {"paper_ids_total": len(current_doc_ids), "added_paper_ids": papers_added},
                "papers_added_this_turn": len(papers_added),
                "sufficiency": {"passed": passed, "checks": checks, "iterations": iterations_done, "gaps_summary": gaps_summary},
                "verification": {
                    "level": verification_level,
                    "verified_facts": synth_res["stats"]["verified_facts"],
                    "total_facts": synth_res["stats"]["total_facts"],
                    "attribution_coverage": None,
                },
                "audit": {"iteration_plans": audit_iterations + [{"forced_stop": "no_new_papers_found"}], "retrieval_queries": [followup_query]},
            }
            return json.dumps(result, indent=2)

        papers_added.extend(added_ids)
        current_doc_ids.extend(added_ids)

        # Update audit entry for this iteration with expansion details
        audit_iterations[-1]["expanded_this_iteration"] = True
        audit_iterations[-1]["added_paper_ids"] = added_ids

        session["selected_doc_ids"] = current_doc_ids
        session["evidence_papers"] = current_doc_ids
        session["coverage_report"] = session.get("coverage_report", {})
        store.put(session)

    # forced stop after max_iterations
    # run one last synth for response (already last computed in loop)
    passed = audit_iterations[-1]["passed"] if audit_iterations else False
    last = audit_iterations[-1] if audit_iterations else {}
    result: FollowupReportResponse = {
        "session_id": session_id,
        "followup_query": followup_query,
        "final_markdown": "",
        "used_evidence": {"paper_ids_total": len(current_doc_ids), "added_paper_ids": papers_added},
        "papers_added_this_turn": len(papers_added),
        "sufficiency": {"passed": passed, "checks": {}, "iterations": iterations_done, "gaps_summary": ["forced_stop_max_iterations"]},
        "verification": {"level": verification_level, "verified_facts": last.get("verified_facts", 0), "total_facts": last.get("total_facts", 0), "attribution_coverage": None},
        "audit": {"iteration_plans": audit_iterations, "retrieval_queries": [followup_query]},
    }
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    _configure_utf8_streams()

    # Keep MCP tools strictly limited to the two FULL-report UX endpoints.
    allowed = {"scirag_full_report_by_query", "scirag_full_report_refine"}

    # FastMCP exposes remove_tool(); remove all other registered tools.
    try:
        for tool_name in [
            "scirag_verify",
            "scirag_status",
            "scirag_theme_open",
            "scirag_theme_followup",
            "scirag_full_report",
            "scirag_followup_report",
        ]:
            if tool_name not in allowed:
                mcp.remove_tool(tool_name)
    except Exception:
        # Non-fatal: if remove_tool signature/behavior changes in the MCP SDK,
        # tools may remain, but server must still start.
        pass

    mcp.run
