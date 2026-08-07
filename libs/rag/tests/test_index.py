"""Tests for the markdown/PDF indexer."""

import textwrap

from rag.indexing import MarkdownIndexer
from rag.indexing.markdown import (
    build_tree,
    compute_token_counts,
    extract_headers,
    extract_text,
    thin_tree,
)
from rag.models import DocumentTree, TreeNode


def _sample_markdown() -> str:
    return textwrap.dedent("""\
        # Introduction

        This is the intro paragraph.

        ## Background

        Some background text here with details.

        ## Methods

        ### Data Collection

        We collected data from sensors.

        ### Model Architecture

        A neural network with 3 layers.

        # Results

        The model achieved 95% accuracy.

        ## Ablation Study

        Removing layer 2 dropped accuracy to 80%.
    """)


class TestExtractHeaders:
    def test_basic_headers(self):
        nodes, lines = extract_headers(_sample_markdown())
        titles = [n["title"] for n in nodes]
        assert titles == [
            "Introduction",
            "Background",
            "Methods",
            "Data Collection",
            "Model Architecture",
            "Results",
            "Ablation Study",
        ]

    def test_header_levels(self):
        nodes, _ = extract_headers(_sample_markdown())
        levels = [n["level"] for n in nodes]
        assert levels == [1, 2, 2, 3, 3, 1, 2]

    def test_skips_code_block_headers(self):
        md = textwrap.dedent("""\
            # Real Header

            Some text.

            ```python
            # This is a comment, not a header
            ## Also not a header
            ```

            ## Another Real Header

            More text.
        """)
        nodes, _ = extract_headers(md)
        titles = [n["title"] for n in nodes]
        assert titles == ["Real Header", "Another Real Header"]


class TestExtractText:
    def test_text_attached_to_nodes(self):
        nodes, lines = extract_headers(_sample_markdown())
        result = extract_text(nodes, lines)
        assert "intro paragraph" in result[0]["text"]
        assert "background text" in result[1]["text"]

    def test_last_node_gets_remaining_text(self):
        nodes, lines = extract_headers(_sample_markdown())
        result = extract_text(nodes, lines)
        last = result[-1]
        assert "accuracy to 80%" in last["text"]


class TestBuildTree:
    def test_hierarchy(self):
        nodes, lines = extract_headers(_sample_markdown())
        nodes_with_text = extract_text(nodes, lines)
        tree = build_tree(nodes_with_text)

        assert len(tree) == 2
        assert tree[0].title == "Introduction"
        assert tree[1].title == "Results"
        assert len(tree[0].children) == 2
        assert tree[0].children[0].title == "Background"
        assert tree[0].children[1].title == "Methods"
        assert len(tree[0].children[1].children) == 2

    def test_node_ids_sequential(self):
        nodes, lines = extract_headers(_sample_markdown())
        nodes_with_text = extract_text(nodes, lines)
        tree = build_tree(nodes_with_text)

        ids: list[str] = []

        def _collect(nodes: list[TreeNode]) -> None:
            for n in nodes:
                ids.append(n.node_id)
                _collect(n.children)

        _collect(tree)
        assert ids == ["0001", "0002", "0003", "0004", "0005", "0006", "0007"]

    def test_empty_list(self):
        tree = build_tree([])
        assert tree == []


class TestTreeThinning:
    def test_small_nodes_merged(self):
        nodes, lines = extract_headers(_sample_markdown())
        nodes_with_text = extract_text(nodes, lines)
        nodes_with_counts = compute_token_counts(nodes_with_text)
        thinned = thin_tree(nodes_with_counts, min_node_tokens=500)
        assert len(thinned) < len(nodes_with_text)


class TestIndexFile:
    def test_index_markdown_no_llm(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text(_sample_markdown())

        indexer = MarkdownIndexer(enable_thinning=False)
        tree = indexer.index_file_sync(str(md_file))

        assert isinstance(tree, DocumentTree)
        assert tree.doc_name == "test"
        assert len(tree.root_nodes) == 2
        assert tree.root_nodes[0].title == "Introduction"

    def test_index_no_headers(self, tmp_path):
        md_file = tmp_path / "plain.md"
        md_file.write_text("Just some text with no headers at all.\nLine two.")

        indexer = MarkdownIndexer(enable_thinning=False)
        tree = indexer.index_file_sync(str(md_file))

        assert len(tree.root_nodes) == 1
        assert "no headers" in tree.root_nodes[0].text

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        indexer = MarkdownIndexer()
        try:
            indexer.index_file_sync(str(f))
        except ValueError as exc:
            assert "Unsupported" in str(exc)
        else:
            raise AssertionError("Expected ValueError for unsupported format")

    def test_summaries_empty_without_llm(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text(_sample_markdown())

        indexer = MarkdownIndexer(enable_thinning=False)
        tree = indexer.index_file_sync(str(md_file))

        for node in tree.root_nodes:
            assert node.summary == ""
