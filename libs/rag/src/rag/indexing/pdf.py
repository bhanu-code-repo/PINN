"""PDF-to-markdown conversion using pymupdf4llm.

Isolates the PDF dependency so the rest of the library works
without pymupdf installed.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def pdf_to_markdown(path: Path) -> str:
    """Convert a PDF file to markdown text (zero LLM cost).

    Uses pymupdf4llm for font-size-based header detection, table
    preservation, and multi-column layout handling.

    Raises
    ------
    ImportError
        If ``pymupdf4llm`` is not installed.
    """
    try:
        import pymupdf4llm
    except ImportError as exc:
        msg = "pymupdf4llm is required for PDF indexing: pip install pymupdf4llm"
        raise ImportError(msg) from exc

    logger.info("Converting PDF to markdown: {}", path.name)
    return pymupdf4llm.to_markdown(str(path))
