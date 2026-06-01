from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .mcp_contracts import CoverageMode, ExtractionTags


@dataclass(frozen=True)
class CandidatePaper:
    paper_id: str
    title: str
    tags: Dict[str, ExtractionTags | None]
    score: float | None
    relevance_reasons: List[str]
    # optional for better clustering
    year: int | None = None


@dataclass(frozen=True)
class CoverageClusterModel:
    cluster_id: str
    cluster_label: str
    paper_ids: List[str]
    budget_alloc: int
    coverage_notes: List[str]
    gaps_estimate: List[str]


def _safe_seed(seed: int | None) -> int:
    return seed if seed is not None else 1337


def _tag_of_candidate(c: CandidatePaper) -> ExtractionTags:
    tag = c.tags.get("primary_tag") if c.tags else None
    return tag if tag in ("T", "E", "M", "A") else "T"


def _get_tag_counts(papers: List[CandidatePaper]) -> Dict[ExtractionTags, int]:
    counts: Dict[ExtractionTags, int] = {"T": 0, "E": 0, "M": 0, "A": 0}
    for p in papers:
        counts[_tag_of_candidate(p)] += 1
    return counts


def _diversity_budget_by_mode(
    coverage_mode: CoverageMode,
    target_papers: int,
) -> Dict[ExtractionTags, int]:
    tags: List[ExtractionTags] = ["T", "E", "M", "A"]
    if target_papers <= 0:
        return {t: 0 for t in tags}

    # balanced: roughly uniform
    if coverage_mode == "balanced":
        base = target_papers // len(tags)
        rem = target_papers % len(tags)
        budget = {t: base for t in tags}
        for t in tags[:rem]:
            budget[t] += 1
        return budget

    # diverse: encourage exploration -> still uniform but slightly more weight to underrepresented
    if coverage_mode == "diverse":
        base = max(1, target_papers // (len(tags) + 1))
        remaining = target_papers - base * len(tags)
        budget = {t: base for t in tags}
        # distribute remaining in round-robin
        tags_cycle = tags[:]
        i = 0
        while remaining > 0:
            budget[tags_cycle[i % len(tags_cycle)]] += 1
            remaining -= 1
            i += 1
        return budget

    # rigorous: equal but allocate "at least" 1 to all and bias to T/E a bit
    # (heuristic: T/E often contain methods & evaluations)
    if coverage_mode == "rigorous":
        budget = {t: 1 for t in tags}
        spent = len(tags)
        if target_papers < spent:
            # trim deterministically
            over = spent - target_papers
            for t in ["A", "M", "E", "T"]:
                if over <= 0:
                    break
                if budget[t] > 0:
                    budget[t] -= 1
                    over -= 1
            return budget
        remaining = target_papers - spent
        # bias order: T, E, M, A
        order: List[ExtractionTags] = ["T", "E", "M", "A"]
        i = 0
        while remaining > 0:
            budget[order[i % len(order)]] += 1
            remaining -= 1
            i += 1
        return budget

    # fallback
    return _diversity_budget_by_mode("balanced", target_papers)


def select_evidence_papers(
    candidates: List[CandidatePaper],
    target_papers: int,
    coverage_mode: CoverageMode = "balanced",
    seed: int | None = None,
) -> Tuple[List[CandidatePaper], List[CoverageClusterModel]]:
    """
    Select evidence papers with coverage/diversity constraints over T/E/M/A primary tags.

    Notes:
    - input candidates are expected to be already ranked by relevance (higher score earlier).
    - deterministic under seed.
    - fallback if not enough candidates per tag: fill remaining via global rank-diversity.
    """
    rng = random.Random(_safe_seed(seed))
    if target_papers <= 0:
        return [], []

    if not candidates:
        return [], []

    # Partition by tag while keeping original order (assumed by caller relevance rank)
    by_tag: Dict[ExtractionTags, List[CandidatePaper]] = {"T": [], "E": [], "M": [], "A": []}
    for c in candidates:
        by_tag[_tag_of_candidate(c)].append(c)

    budget = _diversity_budget_by_mode(coverage_mode, target_papers)

    selected: List[CandidatePaper] = []
    clusters: Dict[ExtractionTags, CoverageClusterModel] = {}

    # First pass: take up to budget per tag with a mild diversification (skip near-duplicates by paper_id)
    for tag in ["T", "E", "M", "A"]:
        want = budget.get(tag, 0)
        pool = by_tag[tag]
        if want <= 0 or not pool:
            continue

        seen_ids = set()
        picked: List[CandidatePaper] = []
        # Diversify by taking every k-th element after shuffle of indices buckets for stability
        # but keep determinism.
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        indices.sort()  # preserve relevance-ish order but deterministic shuffle then stable sort
        for idx in indices:
            c = pool[idx]
            if c.paper_id in seen_ids:
                continue
            picked.append(c)
            seen_ids.add(c.paper_id)
            if len(picked) >= want:
                break

        if picked:
            cluster_id = f"cluster_{tag}"
            clusters[tag] = CoverageClusterModel(
                cluster_id=cluster_id,
                cluster_label=f"PrimaryTag:{tag}",
                paper_ids=[p.paper_id for p in picked],
                budget_alloc=want,
                coverage_notes=[f"Budget allocation for tag {tag}"],
                gaps_estimate=[],
            )
            selected.extend(picked)

    # Second pass: fill remainder using global rank-diversity round-robin across tags that still have pool
    if len(selected) < target_papers:
        remaining = target_papers - len(selected)
        already_ids = {p.paper_id for p in selected}
        # pointer per tag
        pointers: Dict[ExtractionTags, int] = {t: 0 for t in ["T", "E", "M", "A"]}
        for t in ["T", "E", "M", "A"]:
            # advance pointers past already picked ids
            pool = by_tag[t]
            while pointers[t] < len(pool) and pool[pointers[t]].paper_id in already_ids:
                pointers[t] += 1

        tag_cycle: List[ExtractionTags] = ["T", "E", "M", "A"]
        i = 0
        while remaining > 0:
            tag = tag_cycle[i % len(tag_cycle)]
            pool = by_tag[tag]
            pidx = pointers[tag]
            if pidx >= len(pool):
                i += 1
                # check if all exhausted
                if all(pointers[t] >= len(by_tag[t]) for t in tag_cycle):
                    break
                continue
            c = pool[pidx]
            pointers[tag] += 1
            if c.paper_id in already_ids:
                i += 1
                continue
            selected.append(c)
            already_ids.add(c.paper_id)
            remaining -= 1
            i += 1

    # Rebuild clusters with actual chosen ids
    chosen_by_tag: Dict[ExtractionTags, List[CandidatePaper]] = {"T": [], "E": [], "M": [], "A": []}
    for p in selected:
        chosen_by_tag[_tag_of_candidate(p)].append(p)

    final_clusters: List[CoverageClusterModel] = []
    for tag in ["T", "E", "M", "A"]:
        chosen_ids = [p.paper_id for p in chosen_by_tag[tag]]
        if not chosen_ids:
            continue
        # preserve budget alloc from planner even if filled less
        want = budget.get(tag, 0)
        cluster_id = f"cluster_{tag}"
        final_clusters.append(
            CoverageClusterModel(
                cluster_id=cluster_id,
                cluster_label=f"PrimaryTag:{tag}",
                paper_ids=chosen_ids,
                budget_alloc=want,
                coverage_notes=[f"Selected {len(chosen_ids)} for tag {tag}"],
                gaps_estimate=_estimate_gaps_placeholder(tag, len(chosen_ids), want),
            )
        )

    return selected, final_clusters


def _estimate_gaps_placeholder(tag: ExtractionTags, selected: int, budget_alloc: int) -> List[str]:
    """
    Minimal gap proxy until we have deeper semantic gap scoring:
    if selected < budget_alloc, produce a note.
    """
    gaps: List[str] = []
    if budget_alloc > 0 and selected < budget_alloc:
        gaps.append(
            f"Under-covered tag {tag}: selected {selected} vs budget {budget_alloc}"
        )
    return gaps
