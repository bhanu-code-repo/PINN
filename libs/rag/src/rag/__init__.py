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

File ingestion with deduplication::

    from rag import ingest_file, KnowledgeStore, FileRegistry

    store = KnowledgeStore("./kb")
    registry = FileRegistry("./kb/registry.db")
    result = ingest_file("paper.pdf", store=store, registry=registry)
"""

from .collections import Collection, CollectionManager
from .indexing import MarkdownIndexer
from .ingest import IngestResult, ingest_file
from .models import DocumentMetadata, DocumentTree, RetrievalResult, TreeNode
from .registry import FileRecord, FileRegistry
from .retrieve import retrieve
from .search import SearchEngine
from .store import KnowledgeStore

__all__ = [
    "Collection",
    "CollectionManager",
    "DocumentMetadata",
    "DocumentTree",
    "FileRecord",
    "FileRegistry",
    "IngestResult",
    "KnowledgeStore",
    "MarkdownIndexer",
    "RetrievalResult",
    "SearchEngine",
    "TreeNode",
    "ingest_file",
    "retrieve",
]
