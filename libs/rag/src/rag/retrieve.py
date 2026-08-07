"""Two-tier retrieval: BM25 pre-filter then LLM node reasoning.

Adapted from the vectorless-rag RetrievalService. Simplified to two
tiers (no cluster validation) since we operate on a single knowledge
base rather than multiple clusters.

Tier 1: BM25 keyword search narrows to top-K candidate documents.
Tier 2: LLM reasons over summarized trees (titles + summaries only)
        to select relevant node IDs, then fetches full text.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from .models.schemas import RetrievalResult, TreeNode
from .search import SearchEngine
from .store import KnowledgeStore

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def retrieve(
    store: KnowledgeStore,
    query: str,
    llm_client: object,
    *,
    top_k_docs: int = 3,
    top_k_nodes: int = 5,
    search_engine: SearchEngine | None = None,
) -> list[RetrievalResult]:
    """Two-tier retrieval pipeline.

    Parameters
    ----------
    store : KnowledgeStore
        Knowledge store with indexed documents.
    query : str
        Natural-language question.
    llm_client : LLMClient
        LLM client for tier-2 node reasoning.
    top_k_docs : int
        Maximum documents from BM25 pre-filter.
    top_k_nodes : int
        Maximum total nodes to return.
    search_engine : SearchEngine or None
        Pre-built engine; if ``None``, built from store.

    Returns
    -------
    list[RetrievalResult]
        Retrieved sections with source attribution.
    """
    # Tier 1: BM25 pre-filter
    if search_engine is None:
        search_engine = SearchEngine.from_store(store)

    candidate_ids = search_engine.search(query, top_k=top_k_docs)

    if not candidate_ids:
        # Fallback: use all documents if BM25 finds nothing
        all_docs = store.list_documents()
        candidate_ids = [m.doc_id for m in all_docs[:top_k_docs]]
        logger.debug("BM25 returned no results, using first {} docs", len(candidate_ids))

    if not candidate_ids:
        return []

    logger.debug("Tier 1 candidates: {}", candidate_ids)

    # Tier 2: LLM node reasoning per document
    results: list[RetrievalResult] = []

    for doc_id in candidate_ids:
        tree = store.get_document(doc_id)
        summarized = _build_summarized_tree(tree.root_nodes)

        prompt = _build_node_selection_prompt(tree.doc_name, summarized, query)

        try:
            response = await llm_client.ask_async(prompt)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("LLM call failed for {}: {}", doc_id, exc)
            continue

        target_ids = _parse_node_ids(response)
        logger.debug("Tier 2 selected nodes for '{}': {}", doc_id, target_ids)

        if not target_ids:
            continue

        # Fetch full text for selected nodes
        found = _find_nodes_by_ids(tree.root_nodes, set(target_ids))
        for node in found:
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    doc_name=tree.doc_name,
                    node_id=node.node_id,
                    title=node.title,
                    text=node.text,
                    summary=node.summary,
                )
            )

        if len(results) >= top_k_nodes:
            break

    return results[:top_k_nodes]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_summarized_tree(nodes: list[TreeNode]) -> list[dict]:
    """Recursively strip full text, keeping only title + node_id + summary."""
    result: list[dict] = []
    for node in nodes:
        entry: dict = {
            "title": node.title,
            "node_id": node.node_id,
        }
        if node.summary:
            entry["summary"] = node.summary
        if node.children:
            entry["children"] = _build_summarized_tree(node.children)
        result.append(entry)
    return result


def _find_nodes_by_ids(
    nodes: list[TreeNode],
    target_ids: set[str],
) -> list[TreeNode]:
    """Recursive search for nodes matching IDs."""
    found: list[TreeNode] = []
    for node in nodes:
        if node.node_id in target_ids:
            found.append(node)
        found.extend(_find_nodes_by_ids(node.children, target_ids))
    return found


def _build_node_selection_prompt(
    doc_name: str,
    summarized_tree: list[dict],
    query: str,
) -> str:
    """Construct the LLM prompt for tier-2 node selection."""
    tree_json = json.dumps(summarized_tree, indent=2)
    return f"""You are a research assistant analyzing a document about physics-informed neural networks (PINNs) and differential equations.

Document: '{doc_name}'
Structure (titles and summaries only):
{tree_json}

User Query: {query}

Which sections of this document are most relevant to answering the query?
Select the node IDs of the most relevant sections.

Return ONLY a JSON array of node ID strings, e.g.: ["0001", "0003", "0007"]
If no sections are relevant, return: []"""


def _parse_node_ids(response: str) -> list[str]:
    """Extract JSON array of node IDs from LLM response."""
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
    return []
