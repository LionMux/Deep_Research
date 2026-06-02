"""
Deterministic per-sentence claim verification for SciRAG (Phase 5b).

Unlike :class:`scirag.attribution.PostHocAttributor`, which relies on an LLM,
this module verifies each sentence of a generated report against the retrieved
source passages using a fast, offline lexical-overlap metric. It produces a
structured :class:`VerificationReport` and supports a **fail-closed** policy:
sentences that no source passage supports are flagged rather than silently
presented as established fact.

The two layers are complementary:
  - ``PostHocAttributor`` *inserts* citations using an LLM (best quality, needs API).
  - ``ClaimVerifier`` *audits* support deterministically (always available, used
    for the verification report and fail-closed gating).
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Small English stopword set so support scoring focuses on content words.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in into is it its of on or that
    the their this to was were will with which we our these those than then can
    such not but also using used use over more most via per very may might
    """.split()
)

# Marker appended to unsupported sentences under the fail-closed policy.
UNSUPPORTED_FLAG = "[unsupported]"


def _content_tokens(text: str) -> List[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple punctuation-based splitter)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _citation_ids(sentence: str) -> List[str]:
    """Extract existing citation tokens like [3] or [doc_id] from a sentence."""
    return re.findall(r"\[([A-Za-z0-9_.\-]+)\]", sentence)


@dataclass
class SentenceVerdict:
    """Verification verdict for a single sentence."""
    text: str
    supported: bool
    support_score: float
    best_doc_id: Optional[str] = None
    cited_ids: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Aggregate verification result for a report."""
    sentences: List[SentenceVerdict] = field(default_factory=list)
    total: int = 0
    supported: int = 0
    unsupported: int = 0
    # Fraction of citation-worthy sentences supported by some passage.
    faithfulness: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "faithfulness": round(self.faithfulness, 3),
            "sentences": [asdict(s) for s in self.sentences],
        }


class ClaimVerifier:
    """Audit each sentence of a report against source passages, offline.

    Support score is token *containment*: the fraction of a sentence's content
    tokens that also appear in a passage. A sentence is "supported" if its best
    passage scores at or above ``min_support`` (and the sentence is long enough
    to be citation-worthy).
    """

    def __init__(self, min_support: float = 0.35, min_tokens: int = 4):
        self.min_support = min_support
        # Sentences with fewer content tokens are treated as non-claims and
        # always pass (e.g. headings, transitions).
        self.min_tokens = min_tokens

    def _support(self, sent_tokens: List[str], passage_tokens: set) -> float:
        if not sent_tokens:
            return 0.0
        hits = sum(1 for t in sent_tokens if t in passage_tokens)
        return hits / len(sent_tokens)

    def verify_text(self, text: str, passages: List[Dict]) -> VerificationReport:
        """Build a :class:`VerificationReport` for ``text`` given ``passages``.

        ``passages`` items use the documented chunk schema: at least ``text``
        and ``document_id`` (``title`` optional).
        """
        sentences = split_sentences(text)
        # Pre-tokenize passages once.
        passage_tokens = [
            (p.get("document_id") or p.get("title") or f"doc{i}", set(_content_tokens(p.get("text", ""))))
            for i, p in enumerate(passages)
        ]

        verdicts: List[SentenceVerdict] = []
        claim_total = 0
        claim_supported = 0

        for sent in sentences:
            tokens = _content_tokens(sent)
            cited = _citation_ids(sent)

            if len(tokens) < self.min_tokens:
                # Not a citation-worthy claim; pass through without penalty.
                verdicts.append(SentenceVerdict(
                    text=sent, supported=True, support_score=1.0, cited_ids=cited,
                ))
                continue

            best_doc: Optional[str] = None
            best_score = 0.0
            for doc_id, ptoks in passage_tokens:
                score = self._support(tokens, ptoks)
                if score > best_score:
                    best_score = score
                    best_doc = doc_id

            supported = best_score >= self.min_support
            claim_total += 1
            if supported:
                claim_supported += 1

            verdicts.append(SentenceVerdict(
                text=sent,
                supported=supported,
                support_score=round(best_score, 3),
                best_doc_id=best_doc if supported else None,
                cited_ids=cited,
            ))

        report = VerificationReport(
            sentences=verdicts,
            total=claim_total,
            supported=claim_supported,
            unsupported=claim_total - claim_supported,
            faithfulness=(claim_supported / claim_total) if claim_total else 1.0,
        )
        logger.info(
            "Claim verification: %d/%d claims supported (faithfulness=%.0f%%)",
            report.supported, report.total, report.faithfulness * 100,
        )
        return report

    def annotate(self, report: VerificationReport, fail_closed: bool = True) -> str:
        """Render the report's sentences back into text.

        With ``fail_closed=True`` (default), unsupported claim sentences get an
        ``[unsupported]`` flag appended so they are not presented as established
        fact. Supported sentences without an explicit citation get their best
        source id appended as ``[doc_id]``.
        """
        out: List[str] = []
        for v in report.sentences:
            sent = v.text
            if v.supported:
                if v.best_doc_id and not v.cited_ids:
                    sent = f"{sent} [{v.best_doc_id}]"
            elif fail_closed:
                sent = f"{sent} {UNSUPPORTED_FLAG}"
            out.append(sent)
        return " ".join(out)
