"""Backward-compatible re-exports from refactored modules.

All classes and functions are now in ``rag.models`` and ``rag.indexing``.
This module exists so existing imports continue to work.
"""

from .indexing.indexer import MarkdownIndexer
from .models.schemas import DocumentTree, TreeNode

__all__ = ["DocumentTree", "MarkdownIndexer", "TreeNode"]
