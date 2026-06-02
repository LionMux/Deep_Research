"""Ingestion smoke test over the committed sample corpus (samples/papers/)."""

from pathlib import Path

from scirag.ingestion import ingest_directory

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples" / "papers"


def test_samples_dir_exists():
    assert SAMPLES_DIR.is_dir(), f"missing sample corpus: {SAMPLES_DIR}"
    md_files = list(SAMPLES_DIR.glob("*.md"))
    assert len(md_files) >= 3, "expected at least 3 sample documents"


def test_ingest_samples_produces_chunks():
    chunks = ingest_directory(str(SAMPLES_DIR))
    assert len(chunks) > 0, "ingestion produced no chunks"

    # Every chunk has the documented schema.
    for c in chunks:
        assert set(["id", "document_id", "text", "metadata"]).issubset(c.keys())
        assert c["text"].strip()

    # Chunks come from more than one source document.
    doc_ids = {c["document_id"] for c in chunks}
    assert len(doc_ids) >= 3
