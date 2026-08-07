"""Shared dataclasses for the RAG library.

All data models live here — a single source of truth for the shapes
that flow through indexing, storage, search, and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """A single node in the document tree."""

    node_id: str  # "0001", "0002", ...
    title: str
    text: str  # full text content under this header
    level: int  # header depth (1 = #, 2 = ##, ...)
    summary: str = ""
    children: list[TreeNode] = field(default_factory=list)


@dataclass
class DocumentTree:
    """Complete indexed document."""

    doc_name: str  # filename stem
    source_path: str  # original file path
    root_nodes: list[TreeNode] = field(default_factory=list)
    # Domain metadata (optional, set by caller)
    pde_type: str = ""
    spatial_dim: int | None = None
    known_issues: list[str] = field(default_factory=list)
    techniques: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class DocumentMetadata:
    """Summary metadata for a single indexed document."""

    doc_id: str
    doc_name: str
    source_path: str
    pde_type: str = ""
    spatial_dim: int | None = None
    techniques: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)
    node_count: int = 0
    total_tokens: int = 0
    indexed_at: str = ""


@dataclass
class RetrievalResult:
    """A single retrieved document section with source attribution."""

    doc_id: str
    doc_name: str
    node_id: str
    title: str
    text: str
    summary: str
