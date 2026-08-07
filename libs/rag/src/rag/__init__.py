"""rag -- structure-based retrieval-augmented generation.

Parses markdown and PDF documents into hierarchical trees, indexes
with BM25 keyword search, and retrieves relevant sections via LLM
reasoning over summaries. No vector embeddings needed.

Quick start::

    from rag import MarkdownIndexer, KnowledgeStore, retrieve

    indexer = MarkdownIndexer()
    tree = indexer.index_file_sync("paper.pdf")   # PDF or Markdown

    store = KnowledgeStore("./kb")
    store.add_document(tree)
    store.save()

    from llm_provider import LLMClient
    results = await retrieve(store, "How to handle shock waves?", LLMClient())
"""

from .indexing import MarkdownIndexer
from .models import DocumentMetadata, DocumentTree, RetrievalResult, TreeNode
from .retrieve import retrieve
from .search import SearchEngine
from .store import KnowledgeStore

__all__ = [
    "DocumentMetadata",
    "DocumentTree",
    "KnowledgeStore",
    "MarkdownIndexer",
    "RetrievalResult",
    "SearchEngine",
    "TreeNode",
    "retrieve",
]
