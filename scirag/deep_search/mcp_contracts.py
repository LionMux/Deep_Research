from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict


CoverageMode = Literal["balanced", "diverse", "rigorous"]
StyleMode = Literal["engineer", "literature_review"]

VerificationLevel = Literal["off", "basic", "strict"]
Freshness = Literal["local_only", "auto_fallback"]

SessionStoreMode = Literal["in_memory", "disk"]

EvidenceTarget = Literal["same_set", "expand_if_needed"]
SufficiencyMode = Literal["verified_facts", "gap_coverage", "llm_judge", "hybrid"]

ExtractionTags = Literal["T", "E", "M", "A"]


class FullReportRequest(TypedDict, total=False):
    # required
    query: str

    # defaults
    target_papers: int
    coverage_mode: CoverageMode
    style: StyleMode
    max_sections: int
    verification_level: VerificationLevel
    freshness: Freshness
    session_store: SessionStoreMode
    random_seed: Optional[int]


class EvidencePaperSource(TypedDict, total=False):
    # kept for future extensibility
    source: Literal["local_index", "web_deepsearch", "s2_expansion"]


class PaperReference(TypedDict, total=False):
    paper_id: str
    title: str
    year: int | None
    tags: Dict[str, ExtractionTags | None]
    score: float | None
    relevance_reasons: List[str]
    cluster_id: str | None


class CoverageCluster(TypedDict, total=False):
    cluster_id: str
    cluster_label: str
    paper_ids: List[str]
    budget_alloc: int
    coverage_notes: List[str]
    gaps_estimate: List[str]


class FullReportSession(TypedDict, total=False):
    session_id: str
    query: str
    created_at: float

    # evidence
    evidence_papers: List[str]
    evidence_paper_source: List[Literal["local_index", "web_deepsearch", "s2_expansion"]]
    selected_doc_ids: List[str]
    selected_chunk_ids: List[str]  # optional but planned

    coverage_report: Dict
    config_snapshot: Dict

    synthesis: Dict
    last_sufficiency: Dict


class FollowupReportRequest(TypedDict, total=False):
    session_id: str
    followup_query: str

    max_sections: int
    evidence_target: EvidenceTarget
    extra_papers_per_iteration: int
    max_iterations: int
    verification_level: VerificationLevel

    sufficiency_policy: Dict


class FullReportResponse(TypedDict, total=False):
    session_id: str
    query: str
    final_markdown: str

    papers: List[PaperReference]
    coverage_report: Dict

    verification: Dict
    extraction: Dict
    sufficiency: Dict

    audit: Dict


class FollowupReportResponse(TypedDict, total=False):
    session_id: str
    followup_query: str
    final_markdown: str

    used_evidence: Dict
    papers_added_this_turn: int

    sufficiency: Dict
    verification: Dict
    audit: Dict


def make_new_session_id(prefix: str = "sess") -> str:
    # deterministic format, good for logs/audit trail
    return f"{prefix}_{int(time.time() * 1000)}"


def now_epoch() -> float:
    return time.time()


# =========================
# Theme materialization MCP
# =========================

ThemeOpenRequest = TypedDict(
    "ThemeOpenRequest",
    {
        # required
        "query": str,
        "used_doc_ids": List[str],

        # defaults
        "base_papers_dir": str,
        "resume": bool,
    },
    total=False,
)


class ThemeOpenResponse(TypedDict, total=False):
    theme_id: str
    theme: str

    base_papers_dir: str
    used_doc_ids_count: int
    remaining_doc_ids_count: int

    used_dir: str
    remaining_dir: str

    status: str
    created_used_links: int
    removed_from_remaining: int
    skipped_missing_source: int

    meta: Dict


class ThemeFollowupRequest(TypedDict, total=False):
    theme_id: str
    query: str
    used_doc_ids_add: List[str]

    base_papers_dir: str
    resume: bool


class ThemeFollowupResponse(TypedDict, total=False):
    theme_id: str
    theme: str

    used_doc_ids_count: int
    remaining_doc_ids_count: int

    used_dir: str
    remaining_dir: str

    status: str
    created_used_links: int
    removed_from_remaining: int
    skipped_missing_source: int

    # Resume/progress bookkeeping for auditability
    prev_meta: Dict
    next_meta: Dict


# =========================
# UX-friendly wrappers MCP
# =========================

class FullReportByQueryRequest(TypedDict, total=False):
    query: str

    target_papers: int
    coverage_mode: CoverageMode
    style: StyleMode
    max_sections: int
    verification_level: VerificationLevel
    freshness: Freshness
    session_store: SessionStoreMode
    random_seed: Optional[int]


class FullReportByQueryResponse(TypedDict, total=False):
    report_id: str
    report_type: Literal["full"]

    final_markdown: str
    session_id: str
    theme_id: str


class FullReportRefineRequest(TypedDict, total=False):
    report_id: str
    clarification_text: str

    max_sections: int
    evidence_target: EvidenceTarget
    extra_papers_per_iteration: int
    max_iterations: int
    verification_level: VerificationLevel


class FullReportRefineResponse(TypedDict, total=False):
    report_id: str
    report_type: Literal["followup_incremental"]
    parent_report_id: str

    final_markdown: str
    added_paper_ids_count: int
    theme_id: str
