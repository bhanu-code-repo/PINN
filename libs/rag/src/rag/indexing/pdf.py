"""Hybrid PDF-to-markdown conversion.

Two-tier approach to minimize cost:
1. **pymupdf4llm** (free) — handles text-heavy pages via font-size
   header detection, table preservation, multi-column layout.
2. **LLM vision** (paid) — handles complex pages with diagrams,
   flowcharts, graphs, or heavy image content that pymupdf4llm
   can't meaningfully extract.

Page complexity is detected by image-to-text ratio: pages where
images dominate (>40% of page area) or have very little extractable
text are flagged as complex.
"""

from __future__ import annotations

import base64
from pathlib import Path

from loguru import logger

# Threshold: if text has fewer tokens than this, page is "low text"
_MIN_TEXT_TOKENS = 30

# Threshold: if images cover more than this fraction of the page, page is "complex"
_IMAGE_AREA_THRESHOLD = 0.4


def _ensure_pymupdf():
    """Import and return pymupdf, raising a clear error if missing."""
    try:
        import pymupdf
    except ImportError as exc:
        msg = "pymupdf is required for PDF processing: pip install pymupdf4llm"
        raise ImportError(msg) from exc
    return pymupdf


def pdf_to_markdown(path: Path) -> str:
    """Convert a PDF file to markdown using pymupdf4llm (zero LLM cost).

    Simple pass-through for backward compatibility.
    """
    try:
        import pymupdf4llm
    except ImportError as exc:
        msg = "pymupdf4llm is required for PDF indexing: pip install pymupdf4llm"
        raise ImportError(msg) from exc

    logger.info("Converting PDF to markdown: {}", path.name)
    return pymupdf4llm.to_markdown(str(path))


def classify_pages(path: Path) -> list[dict]:
    """Classify each page of a PDF as 'simple' or 'complex'.

    Returns a list of dicts with keys:
    - ``page_num``: 0-based page index
    - ``complexity``: ``"simple"`` or ``"complex"``
    - ``text_tokens``: word count of extractable text
    - ``image_area_ratio``: fraction of page area covered by images

    A page is complex if:
    - It has very little extractable text (<30 tokens), OR
    - Images cover >40% of the page area
    """
    pymupdf = _ensure_pymupdf()
    doc = pymupdf.open(str(path))

    results = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_area = page.rect.width * page.rect.height

        # Extract text and count tokens
        text = page.get_text("text").strip()
        text_tokens = len(text.split())

        # Calculate image area coverage
        image_area = 0.0
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            for rect in rects:
                image_area += rect.width * rect.height

        image_ratio = image_area / page_area if page_area > 0 else 0.0

        is_complex = text_tokens < _MIN_TEXT_TOKENS or image_ratio > _IMAGE_AREA_THRESHOLD

        results.append({
            "page_num": page_num,
            "complexity": "complex" if is_complex else "simple",
            "text_tokens": text_tokens,
            "image_area_ratio": round(image_ratio, 3),
        })

    doc.close()
    return results


def _page_to_base64_image(path: Path, page_num: int) -> str:
    """Render a PDF page as a base64-encoded PNG image."""
    pymupdf = _ensure_pymupdf()
    doc = pymupdf.open(str(path))
    page = doc[page_num]
    # Render at 2x resolution for better LLM vision quality
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("ascii")


async def _convert_complex_page_llm(
    path: Path,
    page_num: int,
    llm_client: object,
) -> str:
    """Convert a complex PDF page to markdown using LLM vision."""
    img_b64 = _page_to_base64_image(path, page_num)

    prompt = (
        f"Convert this PDF page (page {page_num + 1}) to well-structured markdown. "
        "Preserve all text content, headings, lists, and tables. "
        "For diagrams or flowcharts, describe them in a markdown blockquote. "
        "For mathematical equations, use LaTeX notation. "
        "Output ONLY the markdown content, no preamble."
    )

    try:
        result = await llm_client.ask_async(  # type: ignore[union-attr]
            prompt,
            images=[img_b64],
        )
        return result
    except Exception as exc:
        logger.warning(
            "LLM conversion failed for page {}: {}, falling back to pymupdf4llm",
            page_num + 1,
            exc,
        )
        # Fallback: use pymupdf4llm for this page
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(path), pages=[page_num])
        except Exception:
            return f"<!-- Page {page_num + 1}: conversion failed -->"


async def hybrid_pdf_to_markdown(
    path: Path,
    *,
    llm_client: object | None = None,
) -> tuple[str, list[int]]:
    """Convert PDF using hybrid approach: pymupdf4llm + LLM vision.

    Simple pages use pymupdf4llm (free). Complex pages (heavy images,
    diagrams, low text) use LLM vision for better extraction.

    Parameters
    ----------
    path : Path
        Path to the PDF file.
    llm_client : LLMClient or None
        Required for complex page conversion. If None, all pages
        use pymupdf4llm regardless of complexity.

    Returns
    -------
    tuple[str, list[int]]
        (full_markdown, llm_page_numbers) where llm_page_numbers
        is a list of 0-based page indices that used LLM conversion.
    """
    try:
        import pymupdf4llm
    except ImportError as exc:
        msg = "pymupdf4llm is required for PDF indexing: pip install pymupdf4llm"
        raise ImportError(msg) from exc

    classifications = classify_pages(path)
    page_count = len(classifications)

    complex_pages = [c for c in classifications if c["complexity"] == "complex"]
    simple_pages = [c for c in classifications if c["complexity"] == "simple"]

    logger.info(
        "PDF '{}': {} pages total, {} simple, {} complex",
        path.name,
        page_count,
        len(simple_pages),
        len(complex_pages),
    )

    # If no LLM client or no complex pages, use pymupdf4llm for everything
    if not complex_pages or llm_client is None:
        if complex_pages and llm_client is None:
            logger.warning(
                "Complex pages detected but no LLM client provided — "
                "falling back to pymupdf4llm for all pages"
            )
        md = pymupdf4llm.to_markdown(str(path))
        return md, []

    # Hybrid: process simple pages with pymupdf4llm, complex with LLM
    page_markdowns: list[str] = [""] * page_count
    llm_page_nums: list[int] = []

    # Simple pages via pymupdf4llm (batch)
    if simple_pages:
        simple_nums = [c["page_num"] for c in simple_pages]
        simple_md = pymupdf4llm.to_markdown(str(path), pages=simple_nums)
        # pymupdf4llm with pages= returns all selected pages as one string
        # Split by page separator and assign
        parts = simple_md.split("\n-----\n")
        for i, page_num in enumerate(simple_nums):
            if i < len(parts):
                page_markdowns[page_num] = parts[i]
            else:
                page_markdowns[page_num] = ""

    # Complex pages via LLM
    import asyncio

    sem = asyncio.Semaphore(3)  # limit concurrent LLM calls

    async def _process_complex(page_num: int) -> None:
        async with sem:
            md = await _convert_complex_page_llm(path, page_num, llm_client)
            page_markdowns[page_num] = md
            llm_page_nums.append(page_num)

    await asyncio.gather(*[
        _process_complex(c["page_num"]) for c in complex_pages
    ])

    llm_page_nums.sort()

    full_md = "\n\n---\n\n".join(
        f"<!-- Page {i + 1} -->\n{md}" for i, md in enumerate(page_markdowns) if md.strip()
    )

    logger.info(
        "Hybrid conversion complete: {} pages via pymupdf4llm, {} via LLM",
        len(simple_pages),
        len(llm_page_nums),
    )

    return full_md, llm_page_nums


def get_page_count(path: Path) -> int:
    """Get the number of pages in a PDF file."""
    pymupdf = _ensure_pymupdf()
    doc = pymupdf.open(str(path))
    count = len(doc)
    doc.close()
    return count
