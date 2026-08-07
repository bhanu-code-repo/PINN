"""Tests for the SQLite file registry."""

from rag.registry import FileRegistry


class TestFileRegistry:
    def test_register_and_lookup(self, tmp_path):
        db = tmp_path / "registry.db"
        f = tmp_path / "test.md"
        f.write_text("# Hello\n\nSome content.")

        with FileRegistry(db) as reg:
            record = reg.register(f, doc_id="hello-doc", converter="text")
            assert record.file_name == "test.md"
            assert record.doc_id == "hello-doc"
            assert record.converter == "text"

            # Lookup by hash
            found = reg.lookup(record.file_hash)
            assert found is not None
            assert found.doc_id == "hello-doc"

    def test_lookup_by_path(self, tmp_path):
        db = tmp_path / "registry.db"
        f = tmp_path / "doc.md"
        f.write_text("content")

        with FileRegistry(db) as reg:
            reg.register(f, doc_id="doc-1")
            found = reg.lookup_by_path(str(f))
            assert found is not None
            assert found.doc_id == "doc-1"

    def test_is_indexed(self, tmp_path):
        db = tmp_path / "registry.db"
        f = tmp_path / "doc.md"
        f.write_text("some content")

        with FileRegistry(db) as reg:
            assert reg.is_indexed(f) is False
            reg.register(f, doc_id="d1")
            assert reg.is_indexed(f) is True

    def test_dedup_same_content(self, tmp_path):
        db = tmp_path / "registry.db"
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("identical content")
        f2.write_text("identical content")

        with FileRegistry(db) as reg:
            r1 = reg.register(f1, doc_id="doc-a")
            r2 = reg.register(f2, doc_id="doc-b")
            # Same hash → update, not duplicate
            assert r1.file_hash == r2.file_hash
            assert reg.count() == 1
            # Updated doc_id
            found = reg.lookup(r1.file_hash)
            assert found.doc_id == "doc-b"

    def test_different_content_different_hash(self, tmp_path):
        db = tmp_path / "registry.db"
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("content alpha")
        f2.write_text("content beta")

        with FileRegistry(db) as reg:
            r1 = reg.register(f1)
            r2 = reg.register(f2)
            assert r1.file_hash != r2.file_hash
            assert reg.count() == 2

    def test_unregister(self, tmp_path):
        db = tmp_path / "registry.db"
        f = tmp_path / "doc.md"
        f.write_text("content")

        with FileRegistry(db) as reg:
            record = reg.register(f)
            assert reg.unregister(record.file_hash) is True
            assert reg.lookup(record.file_hash) is None
            assert reg.count() == 0

    def test_unregister_nonexistent(self, tmp_path):
        db = tmp_path / "registry.db"
        with FileRegistry(db) as reg:
            assert reg.unregister("nonexistent-hash") is False

    def test_list_all(self, tmp_path):
        db = tmp_path / "registry.db"

        with FileRegistry(db) as reg:
            for i in range(3):
                f = tmp_path / f"doc{i}.md"
                f.write_text(f"content {i}")
                reg.register(f, doc_id=f"doc-{i}")

            all_files = reg.list_all()
            assert len(all_files) == 3

    def test_compute_hash_deterministic(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("deterministic content")
        h1 = FileRegistry.compute_hash(f)
        h2 = FileRegistry.compute_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_llm_pages_stored(self, tmp_path):
        db = tmp_path / "registry.db"
        f = tmp_path / "paper.md"
        f.write_text("some content")

        with FileRegistry(db) as reg:
            record = reg.register(
                f, doc_id="paper", converter="hybrid",
                page_count=10, llm_pages=[2, 5, 8],
            )
            assert record.page_count == 10
            assert record.llm_pages == "2,5,8"
            assert record.converter == "hybrid"

    def test_persistence_across_connections(self, tmp_path):
        db = tmp_path / "registry.db"
        f = tmp_path / "doc.md"
        f.write_text("persistent content")

        # First connection
        with FileRegistry(db) as reg:
            reg.register(f, doc_id="p1")

        # Second connection
        with FileRegistry(db) as reg:
            assert reg.count() == 1
            found = reg.lookup_by_path(str(f))
            assert found is not None
            assert found.doc_id == "p1"
