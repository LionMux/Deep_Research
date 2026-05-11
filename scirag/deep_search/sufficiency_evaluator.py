from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

SufficiencyPolicyMode = Literal["verified_facts", "gap_coverage", "llm_judge", "hybrid", "hybrid_v2"]


def _safe_ratio(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return float(n) / float(d)


def _proxy_coverage_proxy(with_gaps: int) -> float:
    """
    Convert with_gaps (smaller is better) into a [0,1] proxy coverage.
    with_gaps==0 => 1.0
    with_gaps==1 => 0.5
    with_gaps==2 => 0.333...
    """
    if with_gaps <= 0:
        return 1.0
    return 1.0 / (1.0 + float(with_gaps))


def evaluate_information_sufficiency(
    *,
    verification_summary: Dict[str, Any],
    attribution_summary: Optional[Dict[str, Any]],
    gaps_summary: List[str],
    sufficiency_policy: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether follow-up evidence is sufficient.

    verification_summary expected keys (best-effort):
    - verified_facts: int
    - total_facts: int
    - with_gaps: int (smaller is better; from GapCriticTree stats proxy)

    attribution_summary optional:
    - attribution_coverage: float in [0,1]
    """
    verified_facts = int(verification_summary.get("verified_facts", 0))
    total_facts = int(verification_summary.get("total_facts", 0))
    verified_ratio = _safe_ratio(verified_facts, total_facts)

    with_gaps = int(verification_summary.get("with_gaps", 0))
    coverage_proxy = _proxy_coverage_proxy(with_gaps)

    attribution_coverage = None
    if attribution_summary and isinstance(attribution_summary, dict):
        attribution_coverage = attribution_summary.get("attribution_coverage")
        if attribution_coverage is not None:
            try:
                attribution_coverage = float(attribution_coverage)
            except Exception:
                attribution_coverage = None

    mode: SufficiencyPolicyMode = sufficiency_policy.get("mode", "hybrid")  # type: ignore[assignment]
    thresholds: Dict[str, Any] = sufficiency_policy.get("thresholds", {}) or {}

    # Shared thresholds (legacy + hybrid_v2)
    min_verified_facts_abs: int = int(thresholds.get("min_verified_facts", 8))
    min_verified_ratio: float = float(thresholds.get("min_coverage_sentences", 0.55))
    max_with_gaps: int = int(thresholds.get("max_gaps", 2))

    # hybrid_v2-specific thresholds (optional)
    min_total_facts_for_decision: int = int(thresholds.get("min_sentence_pool", 12))
    min_coverage_proxy: float = float(thresholds.get("min_coverage_proxy", 0.55))
    consistency_required_iters: int = int(thresholds.get("consistency_required_iters", 1))

    # We can't compute iter streak inside this function (no history input),
    # but we expose the requirement in audit for the caller.
    # Caller may choose to apply consistency.
    llm_judge_accept_if: str = thresholds.get("llm_judge_accept_if", "enough")  # placeholder

    # Build audit checks
    checks: Dict[str, Any] = {
        "verified_facts": verified_facts,
        "total_facts": total_facts,
        "verified_ratio": verified_ratio,
        "with_gaps": with_gaps,
        "coverage_proxy": coverage_proxy,
        "attribution_coverage": attribution_coverage,
        "policy_mode": mode,
        "thresholds": {
            "min_verified_facts_abs": min_verified_facts_abs,
            "min_verified_ratio": min_verified_ratio,
            "max_with_gaps": max_with_gaps,
            "min_total_facts_for_decision": min_total_facts_for_decision,
            "min_coverage_proxy": min_coverage_proxy,
            "consistency_required_iters": consistency_required_iters,
            "llm_judge_accept_if": llm_judge_accept_if,
        },
        "gaps_summary": gaps_summary[:],
    }

    # Gate conditions
    gate_verified_count = verified_facts >= min_verified_facts_abs
    gate_verified_ratio = verified_ratio >= min_verified_ratio
    gate_with_gaps = with_gaps <= max_with_gaps
    gate_total_facts_pool = total_facts >= min_total_facts_for_decision
    gate_coverage_proxy = coverage_proxy >= min_coverage_proxy

    # Default behavior (legacy)
    if mode in ("verified_facts", "gap_coverage", "llm_judge", "hybrid"):
        if mode == "verified_facts":
            passed = gate_verified_count and gate_verified_ratio and gate_with_gaps
            checks["passed_reason"] = "verified_facts_gate"
        elif mode == "gap_coverage":
            passed = gate_coverage_proxy and gate_with_gaps
            checks["passed_reason"] = "gap_coverage_gate"
        elif mode == "llm_judge":
            passed = gate_verified_count and gate_verified_ratio and gate_with_gaps
            checks["passed_reason"] = "llm_judge_fallback_hybrid_gate"
        else:  # hybrid
            passed = gate_verified_count and gate_coverage_proxy and gate_with_gaps
            checks["passed_reason"] = "hybrid_gate"
        checks["passed"] = passed
        return passed, checks

    # hybrid_v2: multi-criteria score + stricter "minimum pool"
    # (passed should be harder to reach for tiny evidence sets)
    if mode == "hybrid_v2":
        passed_reason_parts: List[str] = []
        insufficient_pool = not gate_total_facts_pool

        score_weights: Dict[str, float] = sufficiency_policy.get("score_weights", {}) or {}
        w_verified_ratio = float(score_weights.get("w_verified_ratio", 0.45))
        w_with_gaps = float(score_weights.get("w_with_gaps", 0.35))
        w_coverage_proxy = float(score_weights.get("w_coverage_proxy", 0.20))

        # normalize to sum=1 (best-effort)
        s = w_verified_ratio + w_with_gaps + w_coverage_proxy
        if s <= 0:
            w_verified_ratio, w_with_gaps, w_coverage_proxy = 0.45, 0.35, 0.20
            s = 1.0
        w_verified_ratio /= s
        w_with_gaps /= s
        w_coverage_proxy /= s

        verified_ratio_score = min(max(verified_ratio, 0.0), 1.0)
        # gap score: with_gaps==0 -> 1, with_gaps==max_with_gaps -> ~0.5 (heuristic)
        # cap for stability
        gap_score = 1.0 / (1.0 + float(with_gaps))
        coverage_proxy_score = coverage_proxy  # already in [0,1]

        suff_score = (
            w_verified_ratio * verified_ratio_score
            + w_with_gaps * gap_score
            + w_coverage_proxy * coverage_proxy_score
        )

        checks["sufficiency_score"] = suff_score
        checks["gates"] = {
            "gate_verified_count": gate_verified_count,
            "gate_verified_ratio": gate_verified_ratio,
            "gate_with_gaps": gate_with_gaps,
            "gate_total_facts_pool": gate_total_facts_pool,
            "gate_coverage_proxy": gate_coverage_proxy,
        }

        passed = (
            gate_verified_count
            and gate_verified_ratio
            and gate_with_gaps
            and gate_total_facts_pool
            and gate_coverage_proxy
        )

        if insufficient_pool:
            passed_reason_parts.append("pool_too_small")
        else:
            if not gate_verified_count:
                passed_reason_parts.append("verified_facts_below_min")
            if not gate_verified_ratio:
                passed_reason_parts.append("verified_ratio_below_min")
            if not gate_with_gaps:
                passed_reason_parts.append("with_gaps_exceeds_max")
            if not gate_coverage_proxy:
                passed_reason_parts.append("coverage_proxy_below_min")

        checks["passed_reason"] = "hybrid_v2_gate:" + ("_".join(passed_reason_parts) if passed_reason_parts else "all_gates_passed")
        checks["passed"] = passed
        return passed, checks

    # Unknown mode fallback
    passed = gate_verified_count and gate_verified_ratio and gate_with_gaps
    checks["passed_reason"] = "unknown_mode_fallback"
    checks["passed"] = passed
    return passed, checks


def default_sufficiency_policy() -> Dict[str, Any]:
    # Legacy-compatible defaults (but we shift to hybrid_v2 for stricter behavior)
    return {
        "mode": "hybrid_v2",
        "thresholds": {
            "min_verified_facts": 8,
            "min_coverage_sentences": 0.55,
            "max_gaps": 2,
            # v2 additions
            "min_sentence_pool": 12,
            "min_coverage_proxy": 0.55,
            "consistency_required_iters": 1,
            "llm_judge_accept_if": "enough",
        },
        "score_weights": {
            "w_verified_ratio": 0.45,
            "w_with_gaps": 0.35,
            "w_coverage_proxy": 0.20,
        },
    }
