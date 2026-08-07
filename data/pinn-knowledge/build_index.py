"""Build the PINN knowledge base index from source markdown files.

Run this once to create the pre-built index that the PINN Agent uses
at recommendation time. No LLM needed — summaries are extracted from
the structured markdown itself.

Usage::

    uv run python data/pinn-knowledge/build_index.py
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger
from rag import KnowledgeStore, MarkdownIndexer

SOURCES_DIR = Path(__file__).parent / "sources"
STORE_DIR = Path(__file__).parent / "store"


def _extract_metadata(content: str) -> dict:
    """Extract PDE-specific metadata from structured markdown."""
    metadata: dict = {
        "pde_type": "",
        "techniques": [],
        "keywords": [],
        "known_issues": [],
    }

    # Extract equation type from ## Equation Type section
    eq_match = re.search(
        r"## Equation Type\n(.+?)(?=\n##|\Z)", content, re.DOTALL
    )
    if eq_match:
        eq_text = eq_match.group(1).strip()
        first_line = eq_text.split("\n")[0].strip()
        metadata["pde_type"] = first_line

    # Extract techniques from ## Techniques section
    tech_match = re.search(
        r"## Techniques\n(.+?)(?=\n##|\Z)", content, re.DOTALL
    )
    if tech_match:
        for line in tech_match.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                # Extract the technique name (before the colon)
                name = line[2:].split(":")[0].split("(")[0].strip()
                if name:
                    metadata["techniques"].append(name)

    # Extract failure modes as known_issues
    fail_match = re.search(
        r"## Known Failure Modes\n(.+?)(?=\n##|\Z)", content, re.DOTALL
    )
    if fail_match:
        for line in fail_match.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                issue = line[2:].split(":")[0].strip()
                if issue:
                    metadata["known_issues"].append(issue)

    # Build keywords from title + key terms in content
    keywords_set: set[str] = set()
    # Common PDE keywords
    for kw in [
        "burgers", "heat", "wave", "schrodinger", "navier-stokes",
        "advection", "allen-cahn", "helmholtz", "kdv", "poisson",
        "spectral bias", "ansatz", "collocation", "rar", "curriculum",
        "periodic", "inverse", "parametric", "loss weighting",
        "high frequency", "training stability",
    ]:
        if kw in content.lower():
            keywords_set.add(kw)

    metadata["keywords"] = sorted(keywords_set)
    return metadata


def build() -> None:
    """Build the knowledge store from source markdown files."""
    indexer = MarkdownIndexer(
        min_node_tokens=30,
        enable_thinning=True,
    )
    store = KnowledgeStore(STORE_DIR)

    source_files = sorted(SOURCES_DIR.glob("*.md"))
    logger.info("Indexing {} source files", len(source_files))

    for md_file in source_files:
        content = md_file.read_text(encoding="utf-8")
        tree = indexer.index_file_sync(str(md_file))

        # Enrich with PDE-specific metadata
        meta = _extract_metadata(content)
        tree.pde_type = meta["pde_type"]
        tree.techniques = meta["techniques"]
        tree.keywords = meta["keywords"]
        tree.known_issues = meta["known_issues"]

        store.add_document(tree)

    store.save()
    logger.info(
        "Built knowledge store: {} documents in {}",
        len(store.list_documents()),
        STORE_DIR,
    )


if __name__ == "__main__":
    build()
