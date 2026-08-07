"""Tests for the document ingestion pipeline."""

from rag.ingest import ingest_file
from rag.registry import FileRegistry
from rag.store import KnowledgeStore


class TestIngestFile:
    def test_ingest_markdown(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nSome content about PINNs.\n\n## Methods\n\nWe used tanh.")

        result = ingest_file(md, store=store, registry=registry)
        assert result.status == "indexed"
        assert result.doc_id is not None
        assert "text" in result.message

        # Verify in store
        docs = store.list_documents()
        assert len(docs) == 1

        # Verify in registry
        assert registry.is_indexed(md)
        registry.close()

    def test_ingest_skips_duplicate(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        md = tmp_path / "test.md"
        md.write_text("# Content\n\nSome text.")

        r1 = ingest_file(md, store=store, registry=registry)
        assert r1.status == "indexed"

        r2 = ingest_file(md, store=store, registry=registry)
        assert r2.status == "skipped"
        assert "already indexed" in r2.message
        registry.close()

    def test_ingest_force_reindex(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        md = tmp_path / "test.md"
        md.write_text("# Content\n\nSome text.")

        r1 = ingest_file(md, store=store, registry=registry)
        assert r1.status == "indexed"

        r2 = ingest_file(md, store=store, registry=registry, force=True)
        assert r2.status == "indexed"
        registry.close()

    def test_ingest_with_metadata(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        md = tmp_path / "burgers.md"
        md.write_text("# Burgers\n\nShock formation in viscous flows.")

        result = ingest_file(
            md, store=store, registry=registry,
            metadata={
                "pde_type": "Nonlinear hyperbolic",
                "techniques": ["RAR", "causal training"],
                "keywords": ["burgers", "shock"],
            },
        )
        assert result.status == "indexed"

        doc = store.get_document(result.doc_id)
        assert doc.pde_type == "Nonlinear hyperbolic"
        assert "RAR" in doc.techniques
        registry.close()

    def test_ingest_custom_doc_id(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        md = tmp_path / "test.md"
        md.write_text("# Test\n\nContent.")

        result = ingest_file(md, store=store, registry=registry, doc_id="custom-id")
        assert result.doc_id == "custom-id"

        rec = registry.lookup(result.file_hash)
        assert rec.doc_id == "custom-id"
        registry.close()

    def test_ingest_file_not_found(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        result = ingest_file(tmp_path / "missing.md", store=store, registry=registry)
        assert result.status == "error"
        assert "not found" in result.message
        registry.close()

    def test_ingest_unsupported_format(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        csv = tmp_path / "data.csv"
        csv.write_text("a,b,c")

        result = ingest_file(csv, store=store, registry=registry)
        assert result.status == "error"
        assert "Unsupported" in result.message or "failed" in result.message.lower()
        registry.close()

    def test_ingest_txt_file(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        txt = tmp_path / "notes.txt"
        txt.write_text("Some plain text notes about neural networks.")

        result = ingest_file(txt, store=store, registry=registry)
        assert result.status == "indexed"
        registry.close()

    def test_ingest_changed_file(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        registry = FileRegistry(tmp_path / "registry.db")

        md = tmp_path / "evolving.md"
        md.write_text("# Version 1\n\nOriginal content.")

        r1 = ingest_file(md, store=store, registry=registry)
        assert r1.status == "indexed"

        # Modify the file
        md.write_text("# Version 2\n\nUpdated content with new findings.")

        r2 = ingest_file(md, store=store, registry=registry)
        assert r2.status == "indexed"  # New hash → indexed again
        assert r2.file_hash != r1.file_hash
        registry.close()
