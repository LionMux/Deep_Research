"""
PDF Downloader for Deep Search.

Downloads PDFs from discovered papers.
Supports: arXiv, open access URLs, Unpaywall fallback.

Usage:
    from scirag.deep_search import PDFDownloader
    dl = PDFDownloader(papers_dir="./papers")
    downloaded = dl.download(papers[:10])
"""

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("deep_search.downloader")

# Windows reserved characters and names
_WIN_RESERVED = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WIN_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
    "LPT6", "LPT7", "LPT8", "LPT9",
}


def _sanitize_filename(name: str, max_len: int = 120) -> str:
    """Sanitize a string so it is safe to use as a Windows filename."""
    # Replace reserved characters
    safe = _WIN_RESERVED.sub("_", name)
    # Collapse multiple underscores/spaces
    safe = re.sub(r"[_ ]+", "_", safe)
    # Strip leading/trailing dots and spaces
    safe = safe.strip(". ")
    # Avoid reserved names
    if safe.upper() in _WIN_RESERVED_NAMES:
        safe = f"_{safe}"
    # Limit length (keep room for .pdf)
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe


class PDFDownloader:
    """Download PDFs from academic papers."""

    def __init__(self, papers_dir: str = "./papers", timeout: int = 30, max_concurrent: int = 8):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _download_one(self, url: str, filename: str) -> Optional[str]:
        """Download single PDF with retries. Returns path or None."""
        async with self.semaphore:
            # Delay to be polite to APIs (especially ArXiv)
            await asyncio.sleep(0.3)

            for attempt in range(1, 4):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()

                        # Check content type and magic bytes
                        content_type = resp.headers.get("content-type", "").lower()
                        if "pdf" not in content_type and not resp.content.startswith(b"%PDF"):
                            # Some paywalls return HTML with 200 OK
                            if b"<html" in resp.content[:200].lower() or "text/html" in content_type:
                                logger.warning(f"Paywall/HTML returned instead of PDF: {url}")
                                return None
                            logger.warning(f"Not a PDF: {url} (type: {content_type}, magic: {resp.content[:8]})")
                            return None

                        path = self.papers_dir / filename
                        path.write_bytes(resp.content)
                        logger.info(f"Downloaded: {path.name} ({len(resp.content)} bytes)")
                        return str(path)

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    if status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limit {url}, waiting {wait}s (attempt {attempt})")
                        await asyncio.sleep(wait)
                        continue
                    elif status in (403, 401):
                        logger.warning(f"Access denied ({status}) for {url}")
                        return None
                    else:
                        logger.warning(f"HTTP {status} for {url}: {e}")
                        if attempt < 3:
                            await asyncio.sleep(attempt)
                            continue
                        return None
                except Exception as e:
                    logger.warning(f"Download failed {url} (attempt {attempt}): {e}")
                    if attempt < 3:
                        await asyncio.sleep(attempt)
                        continue
                    return None

            return None

    async def download(self, papers: List) -> Dict[str, str]:
        """
        Download PDFs for papers.

        Args:
            papers: List of DiscoveredPaper objects

        Returns:
            {fingerprint: local_path} dict
        """
        tasks = []
        fp_to_url = {}

        for p in papers:
            fp = p.fingerprint()
            url = p.pdf_url or (p.pdf_alternatives[0] if p.pdf_alternatives else None)
            if not url:
                continue

            safe_title = _sanitize_filename(p.title[:80])
            safe_fp = _sanitize_filename(fp[:40])
            filename = f"{safe_title}_{safe_fp}.pdf"
            # Ensure total filename length is safe for Windows (max ~250 chars)
            if len(filename) > 200:
                filename = f"{safe_fp}.pdf"

            fp_to_url[fp] = url
            tasks.append(self._download_one(url, filename))

        results = await asyncio.gather(*tasks)

        downloaded = {}
        for fp, path in zip(fp_to_url.keys(), results):
            if path:
                downloaded[fp] = path

        logger.info(f"Downloaded {len(downloaded)}/{len(tasks)} PDFs")
        return downloaded

    def download_sync(self, papers: List) -> Dict[str, str]:
        """Synchronous wrapper."""
        return asyncio.run(self.download(papers))
