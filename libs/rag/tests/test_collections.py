"""Tests for collection management with access control."""

import pytest
from rag import CollectionManager


@pytest.fixture
def cm(tmp_path):
    with CollectionManager(tmp_path / "collections.db") as mgr:
        yield mgr


class TestCollectionCRUD:
    def test_create_collection(self, cm):
        c = cm.create_collection("Research Papers", description="Internal research")
        assert c.collection_id == "research-papers"
        assert c.name == "Research Papers"
        assert c.access == "public"
        assert c.doc_count == 0

    def test_create_restricted_collection(self, cm):
        c = cm.create_collection(
            "Confidential",
            access="restricted",
            allowed_groups=["admin", "research"],
        )
        assert c.access == "restricted"
        assert c.allowed_groups == ["admin", "research"]

    def test_create_duplicate_raises(self, cm):
        cm.create_collection("Test")
        with pytest.raises(ValueError, match="already exists"):
            cm.create_collection("Test")

    def test_get_collection(self, cm):
        cm.create_collection("MyCol")
        c = cm.get_collection("mycol")
        assert c.name == "MyCol"

    def test_get_nonexistent_raises(self, cm):
        with pytest.raises(KeyError):
            cm.get_collection("nope")

    def test_list_collections(self, cm):
        cm.create_collection("Alpha")
        cm.create_collection("Beta")
        cols = cm.list_collections()
        assert len(cols) == 2
        assert cols[0].name == "Alpha"  # sorted

    def test_update_collection(self, cm):
        cm.create_collection("Old", description="old desc", access="public")
        updated = cm.update_collection(
            "old", description="new desc", access="restricted", allowed_groups=["eng"]
        )
        assert updated.description == "new desc"
        assert updated.access == "restricted"
        assert updated.allowed_groups == ["eng"]

    def test_delete_collection(self, cm):
        cm.create_collection("ToDelete")
        cm.delete_collection("todelete")
        assert len(cm.list_collections()) == 0

    def test_delete_nonexistent_raises(self, cm):
        with pytest.raises(KeyError):
            cm.delete_collection("nope")


class TestDocumentMapping:
    def test_add_and_list_docs(self, cm):
        cm.create_collection("Col")
        cm.add_document_to_collection("col", "doc-1")
        cm.add_document_to_collection("col", "doc-2")
        doc_ids = cm.get_collection_doc_ids("col")
        assert doc_ids == ["doc-1", "doc-2"]

    def test_add_duplicate_is_idempotent(self, cm):
        cm.create_collection("Col")
        cm.add_document_to_collection("col", "doc-1")
        cm.add_document_to_collection("col", "doc-1")
        assert len(cm.get_collection_doc_ids("col")) == 1

    def test_remove_doc_from_collection(self, cm):
        cm.create_collection("Col")
        cm.add_document_to_collection("col", "doc-1")
        cm.remove_document_from_collection("col", "doc-1")
        assert cm.get_collection_doc_ids("col") == []

    def test_get_document_collections(self, cm):
        cm.create_collection("A")
        cm.create_collection("B")
        cm.add_document_to_collection("a", "doc-1")
        cm.add_document_to_collection("b", "doc-1")
        cols = cm.get_document_collections("doc-1")
        assert set(cols) == {"a", "b"}

    def test_doc_count_in_collection(self, cm):
        cm.create_collection("Col")
        cm.add_document_to_collection("col", "d1")
        cm.add_document_to_collection("col", "d2")
        c = cm.get_collection("col")
        assert c.doc_count == 2

    def test_cascade_delete_removes_mappings(self, cm):
        cm.create_collection("Col")
        cm.add_document_to_collection("col", "doc-1")
        cm.delete_collection("col")
        # doc-1 should no longer be in any collection
        assert cm.get_document_collections("doc-1") == []


class TestAccessControl:
    def test_list_all_without_groups(self, cm):
        cm.create_collection("Public")
        cm.create_collection("Secret", access="restricted", allowed_groups=["admin"])
        # Admin view: no filter
        assert len(cm.list_collections()) == 2

    def test_filter_by_user_groups(self, cm):
        cm.create_collection("Public")
        cm.create_collection("Research Only", access="restricted", allowed_groups=["research"])
        cm.create_collection("Eng Only", access="restricted", allowed_groups=["engineering"])

        # User in "research" group
        visible = cm.list_collections(user_groups=["research"])
        names = {c.name for c in visible}
        assert "Public" in names
        assert "Research Only" in names
        assert "Eng Only" not in names

    def test_accessible_doc_ids(self, cm):
        cm.create_collection("Public")
        cm.create_collection("Secret", access="restricted", allowed_groups=["admin"])
        cm.add_document_to_collection("public", "pub-doc")
        cm.add_document_to_collection("secret", "secret-doc")

        # User without admin group
        accessible = cm.get_accessible_doc_ids(["research"])
        assert "pub-doc" in accessible
        assert "secret-doc" not in accessible

        # Admin user
        accessible = cm.get_accessible_doc_ids(["admin"])
        assert "pub-doc" in accessible
        assert "secret-doc" in accessible
