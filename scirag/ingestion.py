"""
Simple document ingestion for SciRAG.
Replaces chunking.pipeline.IngestionPipeline.

Supports: .txt, .md, .pdf (with PyPDF2 fallback), .docx (with python-docx fallback)
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    """Extract text from PDF using PyPDF2 or pdfplumber as fallback."""
    text = ""
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.warning(f"PyPDF2 failed for {path}: {e}")

    # Fallback to pdfplumber if PyPDF2 yielded nothing or failed
    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            logger.warning(f"pdfplumber failed for {path}: {e}")

    if not text.strip():
        logger.error(f"Could not extract text from PDF: {path}")
    else:
        logger.debug(f"Extracted {len(text)} chars from PDF: {path}")
    return text


def _read_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning(f"Failed to read DOCX {path}: {e}")
        return ""


def _chunk_text(text: str, doc_id: str, chunk_size: int = 1000, overlap: int = 100) -> List[dict]:
    """Simple sliding-window chunking."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at sentence
        if end < len(text):
            for sep in ["\n\n", ". ", "\n"]:
                pos = text.rfind(sep, start, end)
                if pos > start + chunk_size // 2:
                    end = pos + len(sep)
                    break
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"{doc_id}_chunk_{len(chunks)}",
                "document_id": doc_id,
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "metadata": {"source": doc_id},
            })
        start = end - overlap if end < len(text) else end
    return chunks


def ingest_directory(papers_dir: str, pattern: str = "*") -> List[dict]:
    """
    Load and chunk all documents in directory.

    Returns list of chunk dicts with keys: id, document_id, text, start_pos, end_pos, metadata
    """
    path = Path(papers_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {papers_dir}")

    all_chunks: List[dict] = []

    for file_path in path.rglob(pattern):
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        doc_id = file_path.stem

        if suffix == ".txt" or suffix == ".md":
            text = _read_txt(str(file_path))
        elif suffix == ".pdf":
            text = _read_pdf(str(file_path))
        elif suffix == ".docx":
            text = _read_docx(str(file_path))
        else:
            continue

        if text.strip():
            chunks = _chunk_text(text, doc_id)
            all_chunks.extend(chunks)
            logger.debug(f"Loaded {file_path}: {len(chunks)} chunks")

    logger.info(f"Ingested {len(all_chunks)} chunks from {papers_dir}")
    return all_chunks
