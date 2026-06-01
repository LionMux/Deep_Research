"""
Deep Search — Academic Paper Discovery.

Integrates Deep Research Orchestrator v2 search capabilities into SciRAG.

Features:
  - Multi-API academic search: OpenAlex, CrossRef, ArXiv, Semantic Scholar
  - 200+ results per query
  - Fuzzy deduplication
  - PDF URL enrichment
  - Cascade search with adaptive stopping

Usage:
    from scirag.deep_search import AcademicSearchEngine
    engine = AcademicSearchEngine(target=100)
    papers = engine.search("neural retrieval")
"""

import asyncio
import html
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import httpx

logger = logging.getLogger("deep_search")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class DiscoveredPaper:
    """Paper discovered through academic search APIs."""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    pdf_alternatives: List[str] = field(default_factory=list)
    source_api: str = ""
    citation_count: int = 0
    relevance_score: float = 0.0
    is_oa: bool = False

    def fingerprint(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        return html.unescape(self.title).lower().strip()[:80]

    def has_pdf(self) -> bool:
        return bool(self.pdf_url) or bool(self.pdf_alternatives)


# =============================================================================
# Fuzzy Deduplication
# =============================================================================

class FuzzyDedup:
    STOP_WORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
                   "with", "by", "from", "as", "is", "are", "was", "were", "using",
                   "based", "via"}

    def __init__(self, threshold: float = 0.82):
        self.threshold = threshold

    def _tokenize(self, title: str) -> Set[str]:
        t = html.unescape(title).lower()
        tokens = re.findall(r'[a-z0-9]+', t)
        return set(t for t in tokens if t not in self.STOP_WORDS and len(t) > 2)

    def _similarity(self, t1: str, t2: str) -> float:
        a = self._tokenize(t1)
        b = self._tokenize(t2)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def deduplicate(self, papers: List[DiscoveredPaper]) -> List[DiscoveredPaper]:
        by_doi: Dict[str, DiscoveredPaper] = {}
        by_title: Dict[str, DiscoveredPaper] = {}
        no_doi: List[DiscoveredPaper] = []

        for p in papers:
            # Exact title match (strip year for preprint vs published versions)
            title_key = re.sub(r'\s*\(\d{4}\)\s*', ' ', p.title.lower()).strip()

            if p.doi:
                dk = p.doi.lower().strip()
                if dk in by_doi:
                    self._merge(by_doi[dk], p)
                else:
                    by_doi[dk] = p
            elif title_key in by_title:
                self._merge(by_title[title_key], p)
            else:
                by_title[title_key] = p
                no_doi.append(p)

        # Combine all accepted papers
        all_accepted = list(by_doi.values())

        # Check no_doi against ALL accepted (including DOI papers) by title similarity
        for p in list(no_doi):
            is_dup = False
            for existing in all_accepted:
                if self._similarity(p.title, existing.title) >= self.threshold:
                    self._merge(existing, p)
                    is_dup = True
                    break
            if not is_dup:
                all_accepted.append(p)

        # Second pass: check DOI papers against each other by title (catches preprint vs published)
        result = []
        for p in all_accepted:
            is_dup = False
            for existing in result:
                if self._similarity(p.title, existing.title) >= 0.95:  # Very high threshold for exact title
                    self._merge(existing, p)
                    is_dup = True
                    break
            if not is_dup:
                result.append(p)

        return result

    def _merge(self, into: DiscoveredPaper, other: DiscoveredPaper):
        if not into.pdf_url and other.pdf_url:
            into.pdf_url = other.pdf_url
        into.pdf_alternatives.extend(other.pdf_alternatives)
        into.pdf_alternatives = list(dict.fromkeys(into.pdf_alternatives))
        if other.citation_count > into.citation_count:
            into.citation_count = other.citation_count
        if other.relevance_score > into.relevance_score:
            into.relevance_score = other.relevance_score


# =============================================================================
# API Clients
# =============================================================================

class OpenAlexClient:
    """OpenAlex API client (no key required, polite pool with email)."""
    BASE = "https://api.openalex.org/works"

    def __init__(self, mailto: str = ""):
        self.mailto = mailto

    async def search(self, query: str, limit: int = 200) -> List[DiscoveredPaper]:
        params = {"search": query, "per-page": min(limit, 200), "sort": "relevance_score:desc"}
        if self.mailto:
            params["mailto"] = self.mailto

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(self.BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"OpenAlex error: {e}")
                return []

        papers = []
        for r in data.get("results", []):
            p = DiscoveredPaper(
                title=r.get("display_name", ""),
                authors=[a.get("author", {}).get("display_name", "") for a in r.get("authorships", [])],
                year=r.get("publication_year"),
                abstract=r.get("abstract", ""),
                doi=(r.get("doi") or "").replace("https://doi.org/", ""),
                url=r.get("id", ""),
                source_api="openalex",
                citation_count=r.get("cited_by_count", 0),
                is_oa=r.get("open_access", {}).get("is_oa", False),
            )
            # PDF URL
            oa = r.get("open_access", {})
            if oa.get("oa_url"):
                p.pdf_url = oa["oa_url"]
            elif oa.get("pdf_url"):
                p.pdf_url = oa["pdf_url"]
            papers.append(p)

        logger.info(f"OpenAlex: {len(papers)} papers")
        return papers


class CrossRefClient:
    """CrossRef API client (no key required)."""
    BASE = "https://api.crossref.org/works"

    async def search(self, query: str, limit: int = 100) -> List[DiscoveredPaper]:
        params = {"query": query, "rows": min(limit, 1000), "sort": "relevance", "order": "desc"}
        headers = {"User-Agent": "SciRAG/1.0 (mailto:scirag@example.com)"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(self.BASE, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"CrossRef error: {e}")
                return []

        papers = []
        for item in data.get("message", {}).get("items", []):
            p = DiscoveredPaper(
                title=item.get("title", [""])[0],
                authors=[f"{a.get('given', '')} {a.get('family', '')}".strip()
                         for a in item.get("author", [])],
                year=item.get("published-print", {}).get("date-parts", [[None]])[0][0],
                doi=(item.get("DOI") or "").strip(),
                url=item.get("URL", ""),
                source_api="crossref",
                citation_count=item.get("is-referenced-by-count", 0),
                is_oa=item.get("open_access", {}).get("is_oa", False),
            )
            # PDF via links
            for link in item.get("link", []):
                if "pdf" in link.get("content-type", "").lower():
                    p.pdf_url = link["URL"]
                    break
            papers.append(p)

        logger.info(f"CrossRef: {len(papers)} papers")
        return papers


class ArXivClient:
    """ArXiv API client (no key required)."""
    BASE = "https://export.arxiv.org/api/query"  # HTTPS to avoid 301 redirect

    async def search(self, query: str, limit: int = 200) -> List[DiscoveredPaper]:
        params = {"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 2000),
                  "sortBy": "relevance", "sortOrder": "descending"}

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                resp = await client.get(self.BASE, params=params)
                resp.raise_for_status()
                xml = resp.text
            except Exception as e:
                logger.warning(f"ArXiv error: {e}")
                return []

        import xml.etree.ElementTree as ET
        papers = []
        root = ET.fromstring(xml)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            id_elem = entry.find("atom:id", ns)
            published = entry.find("atom:published", ns)

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)

            # PDF URL
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break
            if not pdf_url and id_elem is not None:
                arxiv_id = id_elem.text.split("/abs/")[-1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            p = DiscoveredPaper(
                title=title.text.strip() if title is not None else "",
                authors=authors,
                year=int(published.text[:4]) if published is not None else None,
                abstract=summary.text.strip() if summary is not None else "",
                url=id_elem.text if id_elem is not None else "",
                pdf_url=pdf_url,
                source_api="arxiv",
                is_oa=True,
            )
            papers.append(p)

        logger.info(f"ArXiv: {len(papers)} papers")
        return papers


class SemanticScholarSearchClient:
    """Semantic Scholar API client (public tier, no key)."""
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    async def search(self, query: str, limit: int = 100) -> List[DiscoveredPaper]:
        params = {"query": query, "limit": min(limit, 100), "fields": "title,authors,year,abstract,doi,citationCount,openAccessPdf"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(2):  # 2 attempts max (public tier is very limited)
                try:
                    resp = await client.get(self.BASE, params=params)
                    if resp.status_code == 429:
                        logger.warning("S2 rate limit hit (public tier exhausted). Skipping S2.")
                        return []  # Skip S2 rather than waiting forever
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if attempt == 1:
                        logger.warning(f"S2 search failed: {e}")
                        return []
                    await asyncio.sleep(3)  # Wait before retry
            else:
                return []

        papers = []
        for r in data.get("data", []):
            oa = r.get("openAccessPdf", {}) or {}
            p = DiscoveredPaper(
                title=r.get("title", ""),
                authors=[a.get("name", "") for a in r.get("authors", [])],
                year=r.get("year"),
                abstract=r.get("abstract", ""),
                doi=r.get("doi", ""),
                url=f"https://www.semanticscholar.org/paper/{r.get('paperId', '')}",
                pdf_url=oa.get("url", "") if isinstance(oa, dict) else "",
                source_api="semanticscholar",
                citation_count=r.get("citationCount", 0),
                is_oa=bool(oa.get("url", "")) if isinstance(oa, dict) else False,
            )
            papers.append(p)

        logger.info(f"SemanticScholar: {len(papers)} papers")
        return papers


# =============================================================================
# Search Engine
# =============================================================================

class AcademicSearchEngine:
    """
    Multi-API academic search engine.

    Searches OpenAlex, CrossRef, ArXiv, Semantic Scholar in parallel.
    Deduplicates results. Returns ranked list of DiscoveredPaper.
    """

    # Auto-expansion for generic queries that don't return relevant ML results
    QUERY_EXPANSION = {
        "deep search": "deep search retrieval augmented generation iterative",
        "neural network": "neural network deep learning",
        "machine learning": "machine learning artificial intelligence",
    }

    def __init__(self, target: int = 100, threshold: float = 8.0, mailto: str = ""):
        self.target = target
        self.threshold = threshold
        self.mailto = mailto
        self.dedup = FuzzyDedup()

    def _expand_query(self, query: str) -> str:
        """Expand generic queries with ML/AI terms for better results."""
        query_lower = query.lower()
        for generic, expanded in self.QUERY_EXPANSION.items():
            if generic in query_lower and len(query.split()) < 4:
                logger.info(f"Query expanded: '{query}' -> '{expanded}'")
                return expanded
        return query

    async def _search_all(self, query: str) -> List[DiscoveredPaper]:
        """Search all APIs in parallel."""
        clients = [
            OpenAlexClient(self.mailto),
            CrossRefClient(),
            ArXivClient(),
            SemanticScholarSearchClient(),
        ]

        tasks = [
            clients[0].search(query, limit=min(self.target * 2, 200)),
            clients[1].search(query, limit=min(self.target * 3, 1000)),
            clients[2].search(query, limit=min(self.target * 2, 200)),
            clients[3].search(query, limit=min(self.target, 100)),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_papers: List[DiscoveredPaper] = []
        for r in results:
            if isinstance(r, list):
                all_papers.extend(r)
            elif isinstance(r, Exception):
                logger.warning(f"API failed: {r}")

        return all_papers

    def _rank(self, papers: List[DiscoveredPaper], query_terms: Set[str]) -> List[DiscoveredPaper]:
        """Rank papers by relevance with strong citation weight."""
        for p in papers:
            score = 0.0
            title_lower = p.title.lower()
            abstract_lower = p.abstract.lower()

            # Term matching in title (strong signal)
            for term in query_terms:
                if term in title_lower:
                    score += 5.0  # Increased from 3.0
                if term in abstract_lower:
                    score += 1.0

            # Citations bonus (very strong signal of quality)
            score += min(p.citation_count / 50.0, 15.0)  # Increased weight

            # Open access bonus
            if p.is_oa:
                score += 1.0

            # PDF available bonus
            if p.has_pdf():
                score += 0.5

            # Source quality bonus
            if p.source_api == "arxiv":
                score += 0.3  # ArXiv tends to be cutting-edge

            p.relevance_score = score

        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers

    async def search(self, query: str) -> Tuple[List[DiscoveredPaper], dict]:
        """
        Full search pipeline with query expansion and sequential API calls.

        Returns:
            (papers, stats) where stats contains timing and counts.
        """
        start = time.time()

        # Expand query if generic
        query = self._expand_query(query)
        logger.info(f"Deep Search: '{query}' (target: {self.target})")

        # Parse query terms
        query_terms = set(re.findall(r'[a-z0-9]+', query.lower()))
        query_terms = {t for t in query_terms if len(t) > 2}

        # Parallel API calls (first 3 in parallel, S2 after short delay)
        all_papers: List[DiscoveredPaper] = []
        api_results = {}

        t0 = time.time()
        oa, cr, ax = await asyncio.gather(
            OpenAlexClient(self.mailto).search(query, limit=min(self.target * 2, 200)),
            CrossRefClient().search(query, limit=min(self.target * 3, 1000)),
            ArXivClient().search(query, limit=min(self.target * 2, 200)),
            return_exceptions=True,
        )
        if isinstance(oa, list):
            all_papers.extend(oa)
        else:
            logger.warning(f"OpenAlex failed: {oa}")
        if isinstance(cr, list):
            all_papers.extend(cr)
        else:
            logger.warning(f"CrossRef failed: {cr}")
        if isinstance(ax, list):
            all_papers.extend(ax)
        else:
            logger.warning(f"ArXiv failed: {ax}")

        api_results['parallel_batch'] = {'count': len(all_papers), 'time': time.time() - t0}
        await asyncio.sleep(1.0)  # Brief delay before S2

        # Semantic Scholar (optional, often rate limited)
        t0 = time.time()
        s2 = await SemanticScholarSearchClient().search(query, limit=min(self.target, 100))
        api_results['semanticscholar'] = {'count': len(s2), 'time': time.time() - t0, 'skipped': len(s2) == 0}
        all_papers.extend(s2)

        logger.info(f"Raw results: {len(all_papers)} papers ({api_results})")

        # Deduplicate
        unique = self.dedup.deduplicate(all_papers)
        logger.info(f"After dedup: {len(unique)} papers")

        # Rank
        ranked = self._rank(unique, query_terms)

        elapsed = time.time() - start
        stats = {
            "query": query,
            "target": self.target,
            "raw_count": len(all_papers),
            "unique_count": len(unique),
            "returned": len(ranked),
            "time": elapsed,
            "with_pdf": sum(1 for p in ranked if p.has_pdf()),
            "open_access": sum(1 for p in ranked if p.is_oa),
            "api_results": api_results,
        }

        logger.info(f"Deep Search done: {stats['unique_count']} unique, {stats['with_pdf']} with PDF, {elapsed:.1f}s")
        return ranked, stats

    def search_sync(self, query: str) -> Tuple[List[DiscoveredPaper], dict]:
        """Synchronous wrapper for search."""
        return asyncio.run(self.search(query))
