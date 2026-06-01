"""
Firecrawl deep-research adapter.

Takes a single markdown report produced by `open-deep-research-w-firecrawl`
and converts it into SciRAG-compatible `chunks` (List[dict]).
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
_SOURCE_LINK_RE = re.compile(r"\[.*?\]\((https?://[^)]+)\)")
_PLAIN_URL_RE = re.compile(r"(https?://[^\s)\]]+)")


def _stable_document_id(query: str) -> str:
    h = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"firecrawl://{h}"


def _extract_sources(markdown: str) -> List[str]:
    """
    Best-effort extraction of URL sources from markdown.

    Firecrawl reports typically contain bullet points like:
      - [Title](https://example.com)
    """
    urls = []
    for m in _SOURCE_LINK_RE.finditer(markdown):
        urls.append(m.group(1).strip())

    if not urls:
        # Fallback: any URL-like tokens
        urls = _PLAIN_URL_RE.findall(markdown)

    # Dedup while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _split_into_sections(markdown: str, max_sections: int) -> List[Tuple[str, str]]:
    """
    Split by markdown headings (#..####). Returns list of (heading, body).
    If no headings are present, returns a single section.
    """
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        text = markdown.strip()
        return [("Firecrawl report", text)] if text else []

    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading_level = len(m.group(1))
        heading_text = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()

        # Heuristic: ignore very small bodies (noise from formatting)
        if len(body) < 30:
            continue

        # Keep top-N sections
        if len(sections) >= max_sections:
            break

        # We store section_header with heading level prefix to preserve context
        section_header = f"{'#' * heading_level} {heading_text}".strip()
        sections.append((section_header, body))

    return sections


def firecrawl_report_to_chunks(
    report_markdown: str,
    query: str,
    max_sections: int = 8,
    max_chunks: int = 12,
) -> List[Dict]:
    """
    Convert Firecrawl markdown report into SciRAG-compatible chunks (List[dict]).

    Chunk dict fields used by SciRAG pipeline:
      - id
      - document_id
      - text
      - start_pos / end_pos
      - metadata: must include at least `section_header` (used by pipeline)
    """
    report_markdown = (report_markdown or "").strip()
    if not report_markdown:
        return []

    document_id = _stable_document_id(query)
    sources = _extract_sources(report_markdown)

    sections = _split_into_sections(report_markdown, max_sections=max_sections)
    if not sections:
        return []

    chunks: List[Dict] = []
    for idx, (section_header, body) in enumerate(sections[:max_chunks], start=1):
        # Chunk id: stable per section index
        chunk_id = f"{document_id}::chunk{idx}"

        # Keep text bounded (FAISS embedding cost)
        # Note: SciRAG chunking size is 512/overlap, so we cap roughly.
        text = body
        if len(text) > 2500:
            text = text[:2500].rsplit("\n", 1)[0]

        metadata = {
            "section_header": section_header,
            "title": section_header.replace("#", "").strip(),
            "sources": sources,
            "firecrawl_query": query,
        }

        chunks.append(
            {
                "id": chunk_id,
                "document_id": document_id,
                "text": text,
                "start_pos": 0,
                "end_pos": 0,
                "metadata": metadata,
            }
        )

    return chunks
