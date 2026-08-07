"""BM25 keyword search engine over document metadata and summaries.

Provides fast initial filtering before LLM-based reasoning. Adapted
from the vectorless-rag KeywordSearchEngine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from .store import KnowledgeStore


def _collect_node_text(nodes: list, parts: list[str]) -> None:
    """Recursively collect titles, summaries, and text from tree nodes."""
    for node in nodes:
        parts.append(node.title)
        if node.summary:
            parts.append(node.summary)
        if node.text:
            parts.append(node.text)
        _collect_node_text(node.children, parts)


class SearchEngine:
    """In-memory BM25 keyword search over indexed documents.

    Build from a :class:`KnowledgeStore` or manually index documents.
    """

    def __init__(self) -> None:
        self._corpus: dict[str, str] = {}  # doc_id -> searchable_text

    # -- Indexing -----------------------------------------------------------

    def index_document(self, doc_id: str, searchable_text: str) -> None:
        """Add or update a document's searchable text."""
        self._corpus[doc_id] = searchable_text.lower()

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        self._corpus.pop(doc_id, None)

    # -- Search -------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        """Return ranked ``doc_id`` values by BM25 relevance.

        Zero-score results are filtered out.
        """
        if not self._corpus:
            return []

        doc_ids = list(self._corpus.keys())
        documents = list(self._corpus.values())

        tokenized_corpus = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [doc_ids[i] for i in ranked if scores[i] > 0]

        return results[:top_k]

    # -- Factory ------------------------------------------------------------

    @classmethod
    def from_store(cls, store: KnowledgeStore) -> SearchEngine:
        """Build a search engine from all documents in a store.

        Searchable text is constructed from metadata keywords, techniques,
        PDE type, document name, and all node titles and summaries.
        """
        engine = cls()

        for meta in store.list_documents():
            tree = store.get_document(meta.doc_id)

            parts: list[str] = [
                meta.doc_name,
                meta.pde_type,
                " ".join(meta.techniques),
                " ".join(meta.keywords),
                " ".join(meta.known_issues),
            ]

            # Collect titles, summaries, and text from all nodes
            _collect_node_text(tree.root_nodes, parts)



            engine.index_document(meta.doc_id, " ".join(parts))

        logger.debug("Built search engine with {} documents", len(engine._corpus))
        return engine
