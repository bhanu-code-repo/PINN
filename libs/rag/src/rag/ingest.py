"""Document ingestion pipeline — ingest files into the knowledge store.

Ties together the indexer, registry, and store into a single entry point.
Handles deduplication (skip already-indexed files), format detection,
hybrid PDF conversion, and metadata enrichment.

Usage::

    from rag import ingest_file, KnowledgeStore, FileRegistry

    store = KnowledgeStore("./kb")
    registry = FileRegistry("./kb/registry.db")

    result = ingest_file("paper.pdf", store=store, registry=registry)
    print(result.status)  # "indexed", "skipped", or "error"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .indexing import MarkdownIndexer
from .indexing.pdf import get_page_count
from .registry import FileRegistry
from .store import KnowledgeStore


@dataclass
class IngestResult:
    """Result of a file ingestion attempt."""

    status: str  # "indexed", "skipped", "error"
    file_path: str
    file_hash: str
    doc_id: str | None = None
    message: str = ""


def ingest_file(
    path: str | Path,
    *,
    store: KnowledgeStore,
    registry: FileRegistry,
    llm_client: object | None = None,
    hybrid_pdf: bool = False,
    doc_id: str | None = None,
    metadata: dict | None = None,
    force: bool = False,
) -> IngestResult:
    """Ingest a single file into the knowledge store.

    Checks the registry for deduplication, indexes the file, and
    registers it. Supports markdown, text, and PDF files.

    Parameters
    ----------
    path : str or Path
        Path to the file to ingest.
    store : KnowledgeStore
        Where to store the indexed document tree.
    registry : FileRegistry
        SQLite registry for deduplication tracking.
    llm_client : LLMClient or None
        LLM client for hybrid PDF conversion and/or summaries.
    hybrid_pdf : bool
        Use hybrid PDF conversion (pymupdf4llm + LLM vision).
    doc_id : str or None
        Custom document ID. Auto-generated from filename if None.
    metadata : dict or None
        Optional metadata to attach (pde_type, techniques, etc.).
    force : bool
        If True, re-index even if the file hash already exists.

    Returns
    -------
    IngestResult
        Status of the ingestion attempt.
    """
    path = Path(path)

    if not path.exists():
        return IngestResult(
            status="error",
            file_path=str(path),
            file_hash="",
            message=f"File not found: {path}",
        )

    # Check deduplication
    file_hash = FileRegistry.compute_hash(path)

    if not force:
        existing = registry.lookup(file_hash)
        if existing is not None:
            logger.info("File '{}' already indexed (hash={}…), skipping", path.name, file_hash[:12])
            return IngestResult(
                status="skipped",
                file_path=str(path),
                file_hash=file_hash,
                doc_id=existing.doc_id,
                message="File already indexed (unchanged content)",
            )

    # Index the file
    try:
        indexer = MarkdownIndexer(
            min_node_tokens=30,
            enable_thinning=True,
            hybrid_pdf=hybrid_pdf,
        )
        tree = indexer.index_file_sync(path, llm_client=llm_client)
    except Exception as exc:
        logger.error("Failed to index '{}': {}", path.name, exc)
        return IngestResult(
            status="error",
            file_path=str(path),
            file_hash=file_hash,
            message=f"Indexing failed: {exc}",
        )

    # Enrich with metadata
    if metadata:
        if "pde_type" in metadata:
            tree.pde_type = metadata["pde_type"]
        if "techniques" in metadata:
            tree.techniques = metadata["techniques"]
        if "keywords" in metadata:
            tree.keywords = metadata["keywords"]
        if "known_issues" in metadata:
            tree.known_issues = metadata["known_issues"]

    # Add to store
    assigned_id = store.add_document(tree, doc_id=doc_id)
    store.save()

    # Determine converter type and page info
    suffix = path.suffix.lower()
    page_count = None
    llm_pages: list[int] = []

    if suffix == ".pdf":
        page_count = get_page_count(path)
        llm_pages = getattr(tree, "_llm_pages", [])
        converter = "hybrid" if llm_pages else "pymupdf4llm"
    else:
        converter = "text"

    # Register in the file registry
    registry.register(
        path,
        doc_id=assigned_id,
        converter=converter,
        page_count=page_count,
        llm_pages=llm_pages,
    )

    logger.info("Ingested '{}' as doc_id='{}'", path.name, assigned_id)
    return IngestResult(
        status="indexed",
        file_path=str(path),
        file_hash=file_hash,
        doc_id=assigned_id,
        message=f"Successfully indexed ({converter})",
    )
