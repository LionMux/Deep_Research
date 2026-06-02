"""
Structured scientific report builder for SciRAG (Phase 7).

Turns a synthesized report body (which carries inline ``[doc_id]`` / ``[key]``
citation markers, possibly with ``[unsupported]`` flags from the claim verifier)
into a clean, sectioned scientific report:

  - inline citations renumbered to sequential ``[1], [2], ...`` by first appearance,
  - a numbered **References** list resolving each number to a source,
  - a **confidence** header derived from verification faithfulness,
  - unsupported-claim markers preserved and summarised.

Everything here is deterministic and LLM-free so it is testable and CI-stable.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A citation marker token: [something]. We exclude the literal unsupported flag.
_CITE_RE = re.compile(r"\[([A-Za-z0-9_][A-Za-z0-9_.\-]*)\]")
_UNSUPPORTED = "unsupported"


@dataclass
class Reference:
    number: int
    key: str
    title: str

    def render(self) -> str:
        if self.title and self.title != self.key:
            return f"{self.number}. {self.title} (`{self.key}`)"
        return f"{self.number}. `{self.key}`"


def confidence_label(faithfulness: Optional[float]) -> str:
    """Map a faithfulness ratio (0..1) to a human confidence flag."""
    if faithfulness is None:
        return "unknown"
    if faithfulness >= 0.8:
        return "high"
    if faithfulness >= 0.5:
        return "medium"
    return "low"


def renumber_citations(
    text: str, title_map: Optional[Dict[str, str]] = None
) -> Tuple[str, List[Reference]]:
    """Replace ``[key]`` markers with sequential ``[N]`` by first appearance.

    The literal ``[unsupported]`` flag is left untouched. Returns the rewritten
    text and the ordered list of :class:`Reference` entries.
    """
    title_map = title_map or {}
    order: List[str] = []
    numbering: Dict[str, int] = {}

    def _assign(key: str) -> int:
        if key not in numbering:
            order.append(key)
            numbering[key] = len(order)
        return numbering[key]

    def _sub(match: "re.Match") -> str:
        key = match.group(1)
        if key.lower() == _UNSUPPORTED:
            return match.group(0)
        return f"[{_assign(key)}]"

    new_text = _CITE_RE.sub(_sub, text)
    references = [
        Reference(number=numbering[key], key=key, title=title_map.get(key, key))
        for key in order
    ]
    return new_text, references


class ReportBuilder:
    """Assemble a structured, citable scientific report (markdown)."""

    def build(
        self,
        query: str,
        body_text: str,
        title_map: Optional[Dict[str, str]] = None,
        verification: Optional[Dict] = None,
        tree_confidence: Optional[float] = None,
    ) -> Dict:
        """Build the structured report.

        Returns a dict with ``markdown`` (the full report), ``references``
        (list of dicts), ``confidence``, and ``num_references``.
        """
        numbered_body, references = renumber_citations(body_text, title_map)

        faithfulness = None
        unsupported = 0
        if verification and verification.get("total"):
            faithfulness = verification.get("faithfulness")
            unsupported = verification.get("unsupported", 0)

        confidence = confidence_label(faithfulness)

        lines: List[str] = []
        lines.append(f"# Research Report: {query}".rstrip())
        lines.append("")

        # Confidence / provenance header.
        header_bits = [f"**Confidence:** {confidence}"]
        if faithfulness is not None:
            header_bits.append(f"faithfulness {faithfulness * 100:.0f}%")
        if unsupported:
            header_bits.append(f"{unsupported} unsupported claim(s) flagged")
        if tree_confidence is not None:
            header_bits.append(f"synthesis confidence {tree_confidence * 100:.0f}%")
        lines.append("> " + " · ".join(header_bits))
        lines.append("")

        # Body (already sectioned by the synthesizer).
        lines.append(numbered_body.strip())
        lines.append("")

        # References.
        lines.append("## References")
        lines.append("")
        if references:
            for ref in references:
                lines.append(ref.render())
        else:
            lines.append("_No sources were cited in this report._")
        lines.append("")

        if unsupported:
            lines.append(
                f"> ⚠️ {unsupported} sentence(s) marked `[unsupported]` are not backed "
                "by any retrieved source and should be treated with caution."
            )

        markdown = "\n".join(lines).rstrip() + "\n"
        return {
            "markdown": markdown,
            "references": [
                {"number": r.number, "key": r.key, "title": r.title} for r in references
            ],
            "num_references": len(references),
            "confidence": confidence,
        }
