"""Tests for the knowledge store."""

import json

import pytest
from rag.index import DocumentTree, TreeNode
from rag.store import KnowledgeStore


def _make_tree(name: str = "test-doc", n_nodes: int = 3) -> DocumentTree:
    """Create a simple DocumentTree for testing."""
    children = [
        TreeNode(
            node_id=str(i + 2).zfill(4),
            title=f"Section {i + 1}",
            text=f"Content for section {i + 1}.",
            level=2,
        )
        for i in range(n_nodes - 1)
    ]
    root = TreeNode(
        node_id="0001",
        title="Root",
        text="Root content.",
        level=1,
        children=children,
    )
    return DocumentTree(
        doc_name=name,
        source_path=f"/tmp/{name}.md",
        root_nodes=[root],
        pde_type="Burgers",
        techniques=["RAR", "adaptive weighting"],
        keywords=["shock", "viscosity"],
    )


class TestAddAndGet:
    def test_round_trip(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        tree = _make_tree()
        doc_id = store.add_document(tree)

        loaded = store.get_document(doc_id)
        assert loaded.doc_name == tree.doc_name
        assert len(loaded.root_nodes) == 1
        assert len(loaded.root_nodes[0].children) == 2

    def test_custom_doc_id(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        tree = _make_tree()
        doc_id = store.add_document(tree, doc_id="my-custom-id")
        assert doc_id == "my-custom-id"
        assert store.get_metadata(doc_id).doc_id == "my-custom-id"

    def test_overwrite(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        tree1 = _make_tree("doc1", n_nodes=2)
        tree2 = _make_tree("doc2", n_nodes=4)

        store.add_document(tree1, doc_id="same-id")
        store.add_document(tree2, doc_id="same-id")

        # Should have the second tree's data
        loaded = store.get_document("same-id")
        assert loaded.doc_name == "doc2"
        assert len(store.list_documents()) == 1


class TestRemove:
    def test_remove_existing(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        doc_id = store.add_document(_make_tree())
        store.remove_document(doc_id)
        assert len(store.list_documents()) == 0

    def test_remove_nonexistent_raises(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        with pytest.raises(KeyError):
            store.remove_document("nonexistent")

    def test_structure_file_deleted(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        doc_id = store.add_document(_make_tree())
        structure_file = tmp_path / "kb" / f"{doc_id}_structure.json"
        assert structure_file.exists()
        store.remove_document(doc_id)
        assert not structure_file.exists()


class TestListAndMetadata:
    def test_list_documents(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add_document(_make_tree("doc-a"))
        store.add_document(_make_tree("doc-b"))
        docs = store.list_documents()
        assert len(docs) == 2
        names = {d.doc_name for d in docs}
        assert names == {"doc-a", "doc-b"}

    def test_metadata_fields(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        tree = _make_tree()
        doc_id = store.add_document(tree)
        meta = store.get_metadata(doc_id)

        assert meta.pde_type == "Burgers"
        assert "RAR" in meta.techniques
        assert "shock" in meta.keywords
        assert meta.node_count == 3
        assert meta.total_tokens > 0
        assert meta.indexed_at != ""

    def test_get_metadata_nonexistent_raises(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        with pytest.raises(KeyError):
            store.get_metadata("missing")


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add_document(_make_tree("paper-a"))
        store.add_document(_make_tree("paper-b"))
        store.save()

        # Load into a new instance
        loaded = KnowledgeStore.load(tmp_path / "kb")
        assert len(loaded.list_documents()) == 2

        # Can still retrieve documents
        doc = loaded.get_document("paper-a")
        assert doc.doc_name == "paper-a"

    def test_load_empty_dir(self, tmp_path):
        store = KnowledgeStore.load(tmp_path / "empty-kb")
        assert len(store.list_documents()) == 0

    def test_manifest_is_valid_json(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        store.add_document(_make_tree())
        store.save()

        manifest_path = tmp_path / "kb" / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1


class TestSlugify:
    def test_basic(self):
        assert KnowledgeStore._slugify("My Paper Title") == "my-paper-title"

    def test_special_chars(self):
        assert KnowledgeStore._slugify("Paper (2024) v2.0") == "paper-2024-v20"

    def test_empty_string(self):
        assert KnowledgeStore._slugify("") == "unnamed"
