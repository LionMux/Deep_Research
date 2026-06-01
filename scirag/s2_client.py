"""
Semantic Scholar API Client — Phase 3: Citation Expansion.

Integrates danielnsilva/semanticscholar (cloned to third_party/semanticscholar)
into SciRAG for paper search, citation graph expansion, and related paper discovery.

Features:
  - Paper search by query
  - Citation/references retrieval (backward & forward exploration)
  - Paper recommendations
  - Disk-based caching (JSON)
  - Rate limiting for public API tier
  - Conversion to CitationGraph format

API Limits (public tier, no key):
  - ~100 requests/5 minutes (shared pool)
  - Rate limiting via exponential backoff
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Add cloned semanticscholar to path
_SCHOLAR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "semanticscholar"
)
if _SCHOLAR_PATH not in os.sys.path:
    os.sys.path.insert(0, _SCHOLAR_PATH)

# Lazy import to avoid loading during module init
try:
    from semanticscholar import SemanticScholar
    _SCHOLAR_AVAILABLE = True
except ImportError:
    _SCHOLAR_AVAILABLE = False
    logger.warning("semanticscholar not available. Install: pip install semanticscholar")


@dataclass
class S2Paper:
    """Lightweight paper record from Semantic Scholar."""
    paper_id: str
    title: str
    abstract: str = ""
    year: int = 0
    authors: List[str] = field(default_factory=list)
    citation_count: int = 0
    reference_count: int = 0
    venue: str = ""
    url: str = ""
    pdf_url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    corpus_id: str = ""
    fields_of_study: List[str] = field(default_factory=list)
    # Citation data
    references: List[dict] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)

    def to_citation_node(self) -> dict:
        """Convert to CitationGraph node format."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "authors": self.authors,
            "abstract": self.abstract,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "url": self.url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "corpus_id": self.corpus_id,
        }


class S2RateLimiter:
    """
    Rate limiter for Semantic Scholar public API tier.
    Public tier: ~100 requests per 5 minutes (shared pool).
    Uses conservative delays to avoid 429 errors.
    """

    def __init__(self, min_delay: float = 1.5, max_retries: int = 5):
        self.min_delay = min_delay
        self.max_retries = max_retries
        self._last_request_time = time.time()
        self._consecutive_errors = 0

    def wait(self):
        """Wait appropriate time before next request."""
        elapsed = time.time() - self._last_request_time
        delay = self.min_delay + (self._consecutive_errors * 2.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def report_success(self):
        """Call after successful request."""
        self._consecutive_errors = max(0, self._consecutive_errors - 1)

    def report_error(self):
        """Call after rate limit error (429)."""
        self._consecutive_errors += 1
        backoff = min(2 ** self._consecutive_errors, 60)
        logger.warning(f"S2 rate limit hit, backing off {backoff}s")
        time.sleep(backoff)


class S2Cache:
    """Disk-based cache for S2 API responses."""

    def __init__(self, cache_dir: str = "./scirag_data/s2_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, dict] = {}

    def _cache_key(self, method: str, query: str) -> str:
        """Generate cache file name."""
        import hashlib
        key = f"{method}:{query}"
        return hashlib.md5(key.encode()).hexdigest()

    def _cache_path(self, method: str, query: str) -> Path:
        return self.cache_dir / f"{self._cache_key(method, query)}.json"

    def get(self, method: str, query: str) -> Optional[dict]:
        """Get cached result."""
        key = f"{method}:{query}"
        if key in self._memory_cache:
            return self._memory_cache[key]

        path = self._cache_path(method, query)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._memory_cache[key] = data
                return data
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def set(self, method: str, query: str, data: dict):
        """Cache result."""
        key = f"{method}:{query}"
        self._memory_cache[key] = data
        path = self._cache_path(method, query)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.warning(f"Failed to write cache: {e}")

    def stats(self) -> dict:
        """Return cache statistics."""
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "entries": len(files),
            "memory_entries": len(self._memory_cache),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


class SemanticScholarClient:
    """
    Semantic Scholar API client for SciRAG citation expansion.

    Integrates semanticscholar library with:
      - Paper search by query
      - Citation/references retrieval
      - Related paper recommendations
      - Disk caching
      - Rate limiting
      - Conversion to CitationGraph format
    """

    # Default fields to request from S2 API (balance detail vs. speed)
    DEFAULT_FIELDS = [
        "paperId", "title", "abstract", "year", "authors",
        "citationCount", "referenceCount", "venue", "url",
        "externalIds", "fieldsOfStudy", "openAccessPdf",
        "publicationDate", "publicationTypes",
    ]

    # Fields for citations/references (lighter weight)
    CITATION_FIELDS = [
        "paperId", "title", "abstract", "year", "authors",
        "citationCount", "venue", "url", "externalIds",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "./scirag_data/s2_cache",
        min_delay: float = 1.5,
        timeout: int = 30,
    ):
        self.rate_limiter = S2RateLimiter(min_delay=min_delay)
        self.cache = S2Cache(cache_dir=cache_dir)
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

        if not _SCHOLAR_AVAILABLE:
            logger.error("semanticscholar library not available. "
                        "Run: pip install semanticscholar or "
                        "add third_party/semanticscholar to PYTHONPATH")

    @property
    def client(self):
        """Lazy initialization of SemanticScholar client."""
        if self._client is None and _SCHOLAR_AVAILABLE:
            self._client = SemanticScholar(
                api_key=self._api_key,
                timeout=self._timeout,
                retry=True,
            )
        return self._client

    def _s2paper_from_raw(self, data: dict) -> S2Paper:
        """Convert raw S2 API response dict to S2Paper."""
        ext_ids = data.get("externalIds", {}) or {}
        pdf_info = data.get("openAccessPdf", {}) or {}

        authors = []
        for a in (data.get("authors") or []):
            if isinstance(a, dict):
                authors.append(a.get("name", ""))
            elif isinstance(a, str):
                authors.append(a)

        refs = []
        for r in (data.get("references") or []):
            cited = r.get("citedPaper", r) if isinstance(r, dict) else {}
            if isinstance(cited, dict):
                refs.append({
                    "paper_id": cited.get("paperId", ""),
                    "title": cited.get("title", ""),
                    "year": cited.get("year", 0),
                })

        cites = []
        for c in (data.get("citations") or []):
            citing = c.get("citingPaper", c) if isinstance(c, dict) else {}
            if isinstance(citing, dict):
                cites.append({
                    "paper_id": citing.get("paperId", ""),
                    "title": citing.get("title", ""),
                    "year": citing.get("year", 0),
                })

        return S2Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            year=data.get("year", 0),
            authors=authors,
            citation_count=data.get("citationCount", 0),
            reference_count=data.get("referenceCount", 0),
            venue=data.get("venue", ""),
            url=data.get("url", ""),
            pdf_url=pdf_info.get("url", "") if isinstance(pdf_info, dict) else "",
            doi=ext_ids.get("DOI", "") if isinstance(ext_ids, dict) else "",
            arxiv_id=ext_ids.get("ArXiv", "") if isinstance(ext_ids, dict) else "",
            corpus_id=str(ext_ids.get("CorpusId", "")) if isinstance(ext_ids, dict) else "",
            fields_of_study=data.get("fieldsOfStudy", []) or [],
            references=refs,
            citations=cites,
        )

    def _s2paper_list_from_raw(self, items: list) -> List[S2Paper]:
        """Convert list of raw S2 API items to S2Paper list."""
        results = []
        for item in items:
            if isinstance(item, dict):
                results.append(self._s2paper_from_raw(item))
            elif hasattr(item, "raw_data"):
                # semanticscholar Paper object
                results.append(self._s2paper_from_raw(item.raw_data))
        return results

    def _api_call_with_retry(self, method_name: str, **kwargs):
        """Execute API call with rate limiting and retry logic."""
        if not self.client:
            return None

        self.rate_limiter.wait()

        for attempt in range(self.rate_limiter.max_retries):
            try:
                method = getattr(self.client, method_name)
                result = method(**kwargs)
                self.rate_limiter.report_success()
                return result
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str:
                    self.rate_limiter.report_error()
                elif attempt < self.rate_limiter.max_retries - 1:
                    logger.warning(f"S2 API error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"S2 API call failed after {attempt+1} attempts: {e}")
                    return None
        return None

    # ========================================================================
    # Public API Methods
    # ========================================================================

    def search_papers(
        self,
        query: str,
        limit: int = 10,
        year: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        fields: Optional[List[str]] = None,
    ) -> List[S2Paper]:
        """
        Search papers by query string.

        Args:
            query: Search query (e.g., "neural retrieval information retrieval")
            limit: Max results (default 10, max 100)
            year: Year filter (e.g., "2020-2023")
            min_citation_count: Minimum citation count filter
            fields: Custom fields to retrieve

        Returns:
            List of S2Paper records
        """
        cache_key = f"search:{query}:{limit}:{year}:{min_citation_count}"
        cached = self.cache.get("search", cache_key)
        if cached:
            logger.debug(f"Cache hit: search '{query}'")
            return [self._s2paper_from_raw(p) for p in cached.get("papers", [])]

        result = self._api_call_with_retry(
            "search_paper",
            query=query,
            limit=min(limit, 100),
            year=year,
            min_citation_count=min_citation_count,
            fields=fields or self.DEFAULT_FIELDS,
        )

        if result is None:
            return []

        # Extract papers from PaginatedResults
        papers = []
        if hasattr(result, "items"):
            for paper_obj in result.items:
                if hasattr(paper_obj, "raw_data"):
                    papers.append(paper_obj.raw_data)
                elif isinstance(paper_obj, dict):
                    papers.append(paper_obj)
        elif isinstance(result, list):
            papers = result

        # Cache raw results
        self.cache.set("search", cache_key, {"papers": papers, "total": getattr(result, "total", len(papers))})

        return self._s2paper_list_from_raw(papers)

    def get_paper(self, paper_id: str, fields: Optional[List[str]] = None) -> Optional[S2Paper]:
        """
        Get paper details by ID (S2PaperId, CorpusId, DOI, ArXivId).

        Args:
            paper_id: Paper identifier
            fields: Custom fields to retrieve

        Returns:
            S2Paper or None
        """
        cached = self.cache.get("paper", paper_id)
        if cached:
            return self._s2paper_from_raw(cached)

        result = self._api_call_with_retry(
            "get_paper",
            paper_id=paper_id,
            fields=fields or self.DEFAULT_FIELDS,
        )

        if result is None:
            return None

        raw = result.raw_data if hasattr(result, "raw_data") else result
        if isinstance(raw, dict):
            self.cache.set("paper", paper_id, raw)
            return self._s2paper_from_raw(raw)
        return None

    def get_references(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None,
    ) -> List[S2Paper]:
        """
        Get papers cited BY the given paper (backward exploration).

        Args:
            paper_id: Paper identifier
            limit: Max references to retrieve (max 1000)
            fields: Custom fields

        Returns:
            List of referenced S2Papers
        """
        cache_key = f"{paper_id}:{limit}"
        cached = self.cache.get("references", cache_key)
        if cached:
            return [self._s2paper_from_raw(p) for p in cached.get("papers", [])]

        result = self._api_call_with_retry(
            "get_paper_references",
            paper_id=paper_id,
            limit=min(limit, 1000),
            fields=fields or self.CITATION_FIELDS,
        )

        if result is None:
            return []

        # Extract papers from PaginatedResults
        papers = []
        if hasattr(result, "items"):
            for ref_obj in result.items:
                # References return {citedPaper: {...}}
                raw = ref_obj
                if hasattr(ref_obj, "raw_data"):
                    raw = ref_obj.raw_data
                if isinstance(raw, dict) and "citedPaper" in raw:
                    papers.append(raw["citedPaper"])
                elif isinstance(raw, dict):
                    papers.append(raw)

        self.cache.set("references", cache_key, {"papers": papers, "total": getattr(result, "total", len(papers))})
        return self._s2paper_list_from_raw(papers)

    def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None,
    ) -> List[S2Paper]:
        """
        Get papers that CITE the given paper (forward exploration).

        Args:
            paper_id: Paper identifier
            limit: Max citations to retrieve (max 1000)
            fields: Custom fields

        Returns:
            List of citing S2Papers
        """
        cache_key = f"{paper_id}:{limit}"
        cached = self.cache.get("citations", cache_key)
        if cached:
            return [self._s2paper_from_raw(p) for p in cached.get("papers", [])]

        result = self._api_call_with_retry(
            "get_paper_citations",
            paper_id=paper_id,
            limit=min(limit, 1000),
            fields=fields or self.CITATION_FIELDS,
        )

        if result is None:
            return []

        # Extract papers from PaginatedResults
        papers = []
        if hasattr(result, "items"):
            for cite_obj in result.items:
                raw = cite_obj
                if hasattr(cite_obj, "raw_data"):
                    raw = cite_obj.raw_data
                if isinstance(raw, dict) and "citingPaper" in raw:
                    papers.append(raw["citingPaper"])
                elif isinstance(raw, dict):
                    papers.append(raw)

        self.cache.set("citations", cache_key, {"papers": papers, "total": getattr(result, "total", len(papers))})
        return self._s2paper_list_from_raw(papers)

    def get_related_papers(
        self,
        paper_id: str,
        limit: int = 20,
        fields: Optional[List[str]] = None,
    ) -> List[S2Paper]:
        """
        Get recommended/related papers.

        Args:
            paper_id: Paper identifier
            limit: Max recommendations (max 500)
            fields: Custom fields

        Returns:
            List of related S2Papers
        """
        cache_key = f"{paper_id}:{limit}"
        cached = self.cache.get("related", cache_key)
        if cached:
            return [self._s2paper_from_raw(p) for p in cached.get("papers", [])]

        result = self._api_call_with_retry(
            "get_recommended_papers",
            paper_id=paper_id,
            limit=min(limit, 500),
            fields=fields or self.DEFAULT_FIELDS,
        )

        if result is None:
            return []

        papers = []
        if isinstance(result, list):
            for p in result:
                if hasattr(p, "raw_data"):
                    papers.append(p.raw_data)
                elif isinstance(p, dict):
                    papers.append(p)

        self.cache.set("related", cache_key, {"papers": papers, "total": len(papers)})
        return self._s2paper_list_from_raw(papers)

    # ========================================================================
    # CitationGraph Integration
    # ========================================================================

    def expand_graph_backward(
        self,
        paper_id: str,
        citation_graph,
        depth: int = 1,
        max_per_level: int = 20,
    ) -> List[S2Paper]:
        """
        Expand citation graph backward (get references).
        Adds found papers to the CitationGraph.

        Args:
            paper_id: Starting paper ID
            citation_graph: CitationGraph instance to expand
            depth: Exploration depth (1 = direct references only)
            max_per_level: Max papers to fetch per level

        Returns:
            List of all discovered S2Papers
        """
        if not paper_id:
            return []

        all_papers: List[S2Paper] = []
        current_ids = {paper_id}

        for level in range(depth):
            next_ids = set()
            for pid in current_ids:
                refs = self.get_references(pid, limit=max_per_level)
                for ref in refs:
                    if ref.paper_id:
                        all_papers.append(ref)
                        next_ids.add(ref.paper_id)
                        # Add to citation graph
                        citation_graph.add_paper(
                            doc_id=ref.paper_id,
                            title=ref.title,
                            year=ref.year,
                            authors=ref.authors,
                            abstract=ref.abstract,
                        )
                        citation_graph._graph.add_edge(pid, ref.paper_id, relation="cites")
                logger.debug(f"Level {level+1}: {pid} -> {len(refs)} references")
            current_ids = next_ids
            if not current_ids:
                break

        logger.info(f"Expanded graph backward: found {len(all_papers)} papers from {paper_id}")
        return all_papers

    def expand_graph_forward(
        self,
        paper_id: str,
        citation_graph,
        depth: int = 1,
        max_per_level: int = 20,
    ) -> List[S2Paper]:
        """
        Expand citation graph forward (get citations / who cites this paper).
        Adds found papers to the CitationGraph.

        Args:
            paper_id: Starting paper ID
            citation_graph: CitationGraph instance to expand
            depth: Exploration depth
            max_per_level: Max papers per level

        Returns:
            List of all discovered S2Papers
        """
        if not paper_id:
            return []

        all_papers: List[S2Paper] = []
        current_ids = {paper_id}

        for level in range(depth):
            next_ids = set()
            for pid in current_ids:
                cites = self.get_citations(pid, limit=max_per_level)
                for cite in cites:
                    if cite.paper_id:
                        all_papers.append(cite)
                        next_ids.add(cite.paper_id)
                        citation_graph.add_paper(
                            doc_id=cite.paper_id,
                            title=cite.title,
                            year=cite.year,
                            authors=cite.authors,
                            abstract=cite.abstract,
                        )
                        citation_graph._graph.add_edge(cite.paper_id, pid, relation="cites")
                logger.debug(f"Level {level+1}: {pid} <- {len(cites)} citations")
            current_ids = next_ids
            if not current_ids:
                break

        logger.info(f"Expanded graph forward: found {len(all_papers)} papers citing {paper_id}")
        return all_papers

    def search_and_expand(
        self,
        query: str,
        citation_graph,
        search_limit: int = 5,
        backward_depth: int = 1,
        forward_depth: int = 0,
        max_per_level: int = 10,
    ) -> List[S2Paper]:
        """
        Search papers by query and expand citation graph.
        Primary entry point for Phase 3.

        Args:
            query: Search query
            citation_graph: CitationGraph to expand
            search_limit: Number of seed papers from search
            backward_depth: Depth of backward exploration
            forward_depth: Depth of forward exploration
            max_per_level: Max papers per expansion level

        Returns:
            List of all discovered papers (seeds + expanded)
        """
        logger.info(f"S2 search & expand: query='{query}', seeds={search_limit}")

        # Step 1: Search for seed papers
        seeds = self.search_papers(query, limit=search_limit)
        if not seeds:
            logger.warning(f"No papers found for query: {query}")
            return []

        logger.info(f"Found {len(seeds)} seed papers")
        for s in seeds:
            logger.info(f"  [{s.paper_id}] {s.title[:80]} ({s.year})")

        # Add seeds to graph
        for seed in seeds:
            if seed.paper_id:
                citation_graph.add_paper(
                    doc_id=seed.paper_id,
                    title=seed.title,
                    year=seed.year,
                    authors=seed.authors,
                    abstract=seed.abstract,
                )

        # Step 2: Backward expansion (references)
        all_papers = list(seeds)
        if backward_depth > 0:
            for seed in seeds:
                if seed.paper_id:
                    refs = self.expand_graph_backward(
                        seed.paper_id, citation_graph,
                        depth=backward_depth, max_per_level=max_per_level,
                    )
                    all_papers.extend(refs)

        # Step 3: Forward expansion (citations)
        if forward_depth > 0:
            for seed in seeds:
                if seed.paper_id:
                    cites = self.expand_graph_forward(
                        seed.paper_id, citation_graph,
                        depth=forward_depth, max_per_level=max_per_level,
                    )
                    all_papers.extend(cites)

        # Deduplicate
        seen = set()
        unique = []
        for p in all_papers:
            if p.paper_id and p.paper_id not in seen:
                seen.add(p.paper_id)
                unique.append(p)

        logger.info(f"Total unique papers: {len(unique)} (seeds: {len(seeds)})")
        return unique

    # ========================================================================
    # Stats
    # ========================================================================

    def stats(self) -> dict:
        """Return client statistics."""
        return {
            "available": _SCHOLAR_AVAILABLE,
            "cache": self.cache.stats(),
            "rate_limiter": {
                "min_delay": self.rate_limiter.min_delay,
                "consecutive_errors": self.rate_limiter._consecutive_errors,
            },
        }
