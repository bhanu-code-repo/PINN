"""Tests for the 2-tier retrieval pipeline."""

import asyncio
from unittest.mock import AsyncMock

from rag.index import DocumentTree, TreeNode
from rag.retrieve import (
    RetrievalResult,
    _build_node_selection_prompt,
    _build_summarized_tree,
    _find_nodes_by_ids,
    _parse_node_ids,
    retrieve,
)
from rag.store import KnowledgeStore


def _make_tree_with_nodes() -> DocumentTree:
    """Create a tree with known structure for testing."""
    child1 = TreeNode(
        node_id="0002",
        title="Methods",
        text="We used a PINN with 4 hidden layers.",
        level=2,
        summary="PINN architecture description",
    )
    child2 = TreeNode(
        node_id="0003",
        title="Results",
        text="The model achieved rel-L2 error of 1e-3 on Burgers equation.",
        level=2,
        summary="Burgers equation results with low error",
    )
    child3 = TreeNode(
        node_id="0004",
        title="Failure Modes",
        text="Sharp gradients cause training instability without RAR.",
        level=2,
        summary="RAR needed for sharp gradient problems",
    )
    root = TreeNode(
        node_id="0001",
        title="PINN for Burgers Equation",
        text="Overview of solving Burgers equation with PINNs.",
        level=1,
        summary="Burgers PINN overview",
        children=[child1, child2, child3],
    )
    return DocumentTree(
        doc_name="burgers-pinn-paper",
        source_path="/tmp/burgers.md",
        root_nodes=[root],
        keywords=["burgers", "pinn", "shock"],
    )


class TestBuildSummarizedTree:
    def test_strips_text(self):
        tree = _make_tree_with_nodes()
        summarized = _build_summarized_tree(tree.root_nodes)

        assert len(summarized) == 1
        root = summarized[0]
        assert "text" not in root
        assert root["node_id"] == "0001"
        assert root["summary"] == "Burgers PINN overview"
        assert len(root["children"]) == 3

    def test_children_have_summaries(self):
        tree = _make_tree_with_nodes()
        summarized = _build_summarized_tree(tree.root_nodes)
        children = summarized[0]["children"]
        assert children[0]["summary"] == "PINN architecture description"
        assert children[2]["summary"] == "RAR needed for sharp gradient problems"


class TestFindNodesByIds:
    def test_finds_root(self):
        tree = _make_tree_with_nodes()
        found = _find_nodes_by_ids(tree.root_nodes, {"0001"})
        assert len(found) == 1
        assert found[0].title == "PINN for Burgers Equation"

    def test_finds_children(self):
        tree = _make_tree_with_nodes()
        found = _find_nodes_by_ids(tree.root_nodes, {"0003", "0004"})
        assert len(found) == 2
        titles = {n.title for n in found}
        assert titles == {"Results", "Failure Modes"}

    def test_finds_nothing_for_unknown_ids(self):
        tree = _make_tree_with_nodes()
        found = _find_nodes_by_ids(tree.root_nodes, {"9999"})
        assert found == []


class TestParseNodeIds:
    def test_clean_json_array(self):
        assert _parse_node_ids('["0001", "0003"]') == ["0001", "0003"]

    def test_json_in_text(self):
        text = 'Based on the structure, relevant nodes are: ["0002", "0004"] for this query.'
        assert _parse_node_ids(text) == ["0002", "0004"]

    def test_empty_array(self):
        assert _parse_node_ids("[]") == []

    def test_no_json(self):
        assert _parse_node_ids("No relevant sections found.") == []

    def test_malformed_json(self):
        assert _parse_node_ids("[invalid json}") == []


class TestBuildPrompt:
    def test_contains_query_and_doc(self):
        tree = _make_tree_with_nodes()
        summarized = _build_summarized_tree(tree.root_nodes)
        prompt = _build_node_selection_prompt("my-paper", summarized, "How to handle shocks?")

        assert "my-paper" in prompt
        assert "How to handle shocks?" in prompt
        assert "0001" in prompt  # node IDs should be in the tree JSON


class TestRetrieveEndToEnd:
    def test_full_pipeline_with_mock_llm(self, tmp_path):
        # Setup store
        store = KnowledgeStore(tmp_path / "kb")
        store.add_document(_make_tree_with_nodes())
        store.save()

        # Mock LLM that returns node IDs
        mock_llm = AsyncMock()
        mock_llm.ask_async.return_value = '["0003", "0004"]'

        results = asyncio.run(retrieve(store, "shock handling", mock_llm))

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Should have found the Results and Failure Modes nodes
        node_ids = {r.node_id for r in results}
        assert "0003" in node_ids or "0004" in node_ids

    def test_empty_store_returns_empty(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        mock_llm = AsyncMock()

        results = asyncio.run(retrieve(store, "anything", mock_llm))
        assert results == []

    def test_llm_failure_graceful(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add_document(_make_tree_with_nodes())
        store.save()

        mock_llm = AsyncMock()
        mock_llm.ask_async.side_effect = RuntimeError("LLM down")

        # Should not raise, just return empty
        results = asyncio.run(retrieve(store, "shock handling", mock_llm))
        assert results == []
