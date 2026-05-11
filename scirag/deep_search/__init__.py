"""
Deep Search — Academic Paper Discovery for SciRAG.

Integrates Deep Research Orchestrator capabilities:
  - Multi-API academic search (OpenAlex, CrossRef, ArXiv, S2)
  - PDF downloading
  - 200+ results per query
  - Fuzzy deduplication

Usage:
    from scirag.deep_search import AcademicSearchEngine, PDFDownloader
    engine = AcademicSearchEngine(target=100)
    papers, stats = engine.search_sync("your topic")
    downloader = PDFDownloader(papers_dir="./papers")
    downloaded = downloader.download_sync(papers[:10])
"""

from .academic_search import AcademicSearchEngine, DiscoveredPaper
from .downloader import PDFDownloader

__all__ = [
    "AcademicSearchEngine",
    "DiscoveredPaper",
    "PDFDownloader",
]
