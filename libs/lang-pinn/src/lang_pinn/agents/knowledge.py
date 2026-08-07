"""PINN knowledge base integration via RAG library.

Loads the pre-built knowledge store and provides BM25 search
to retrieve relevant PDE literature context. No LLM call needed
at retrieval time — pure keyword matching over structured entries.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from rag import KnowledgeStore, SearchEngine

# Default store location (pre-built by data/pinn-knowledge/build_index.py)
_DEFAULT_STORE_DIR = Path(__file__).resolve().parents[5] / "data" / "pinn-knowledge" / "store"


def _flatten_text(nodes: list, parts: list[str]) -> None:
    """Recursively collect text from tree nodes."""
    for node in nodes:
        if node.text:
            parts.append(node.text.strip())
        _flatten_text(node.children, parts)


def load_knowledge(
    store_dir: str | Path | None = None,
) -> tuple[KnowledgeStore, SearchEngine] | None:
    """Load the pre-built knowledge store and build search engine.

    Returns None if the store doesn't exist (e.g. in tests without fixtures).
    """
    store_path = Path(store_dir) if store_dir else _DEFAULT_STORE_DIR
    manifest = store_path / "manifest.json"

    if not manifest.exists():
        logger.debug("No knowledge store at {}, skipping", store_path)
        return None

    store = KnowledgeStore.load(store_path)
    engine = SearchEngine.from_store(store)
    return store, engine


def search_knowledge(
    spec_query: str,
    store: KnowledgeStore,
    engine: SearchEngine,
    *,
    top_k: int = 3,
) -> str:
    """Search the knowledge base and return formatted context.

    Args:
        spec_query: Search query built from PDE features.
        store: The knowledge store to retrieve documents from.
        engine: Pre-built BM25 search engine.
        top_k: Maximum number of documents to retrieve.

    Returns:
        Formatted string with relevant knowledge, or empty string if none found.
    """
    doc_ids = engine.search(spec_query, top_k=top_k)
    if not doc_ids:
        return ""

    sections: list[str] = []
    for doc_id in doc_ids:
        tree = store.get_document(doc_id)
        meta = store.get_metadata(doc_id)

        parts: list[str] = []
        _flatten_text(tree.root_nodes, parts)
        body = "\n".join(parts)

        header = f"### {tree.doc_name}"
        if meta.pde_type:
            header += f" ({meta.pde_type})"
        if meta.techniques:
            header += f"\nTechniques: {', '.join(meta.techniques)}"
        if meta.known_issues:
            header += f"\nKnown issues: {', '.join(meta.known_issues)}"

        sections.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(sections)


def build_search_query(spec) -> str:
    """Build a search query string from a PDESpec."""
    parts = [spec.name, spec.equation]

    if spec.has_high_frequency:
        parts.append("high frequency oscillation spectral bias")
    if spec.has_sharp_gradients:
        parts.append("sharp gradients shock adaptive refinement")
    if spec.has_periodic_bc:
        parts.append("periodic boundary conditions")
    if not spec.is_linear:
        parts.append("nonlinear")
    if spec.is_time_dependent:
        parts.append("time dependent")
    if spec.spatial_dim >= 2:
        parts.append("2D multi-dimensional")

    return " ".join(parts)
