"""Tests for the BM25 search engine."""

from rag.index import DocumentTree, TreeNode
from rag.search import SearchEngine
from rag.store import KnowledgeStore


def _make_tree(name: str, text: str, **kwargs) -> DocumentTree:
    # BM25 needs documents with enough tokens for meaningful IDF scores,
    # so we repeat the text to ensure sufficient length.
    long_text = " ".join([text] * 5)
    root = TreeNode(
        node_id="0001",
        title=name,
        text=long_text,
        level=1,
        summary=text,
    )
    return DocumentTree(
        doc_name=name,
        source_path=f"/tmp/{name}.md",
        root_nodes=[root],
        **kwargs,
    )


class TestSearchEngine:
    def test_search_returns_ranked(self):
        engine = SearchEngine()
        engine.index_document("burgers", "burgers equation shock viscosity pinn")
        engine.index_document("heat", "heat equation diffusion thermal conduction")
        engine.index_document("wave", "wave equation propagation acoustic speed")

        results = engine.search("viscosity shock formation")
        assert results[0] == "burgers"

    def test_search_empty_corpus(self):
        engine = SearchEngine()
        results = engine.search("anything")
        assert results == []

    def test_search_no_match(self):
        engine = SearchEngine()
        engine.index_document("doc1", "apples oranges bananas")
        results = engine.search("quantum chromodynamics")
        assert results == []

    def test_remove_document(self):
        engine = SearchEngine()
        engine.index_document("doc1", "neural networks deep learning")
        engine.remove_document("doc1")
        results = engine.search("neural networks")
        assert results == []

    def test_top_k_limit(self):
        engine = SearchEngine()
        for i in range(10):
            engine.index_document(f"doc{i}", f"pinn physics neural network experiment {i}")
        results = engine.search("pinn physics", top_k=3)
        assert len(results) <= 3


class TestFromStore:
    def test_builds_from_store(self, tmp_path):
        # BM25 IDF needs >= 3 docs to differentiate (with 2 docs,
        # terms in 1 doc get IDF=0 due to log((N-n+0.5)/(n+0.5)) floor)
        store = KnowledgeStore(tmp_path / "kb")
        store.add_document(
            _make_tree(
                "burgers-paper",
                "Solving Burgers equation with PINNs and sharp gradients.",
                keywords=["burgers", "shock", "viscosity"],
                techniques=["RAR", "adaptive-refinement"],
            )
        )
        store.add_document(
            _make_tree(
                "heat-paper",
                "Heat equation diffusion on a rod with Dirichlet BCs.",
                keywords=["heat", "diffusion", "thermal"],
                techniques=["ansatz", "fourier"],
            )
        )
        store.add_document(
            _make_tree(
                "wave-paper",
                "Wave equation propagation acoustic speed displacement.",
                keywords=["wave", "propagation", "acoustic"],
                techniques=["spectral", "collocation"],
            )
        )
        store.save()

        engine = SearchEngine.from_store(store)
        results = engine.search("burgers viscosity shock")
        assert len(results) > 0
        assert results[0] == "burgers-paper"

    def test_empty_store(self, tmp_path):
        store = KnowledgeStore(tmp_path / "kb")
        engine = SearchEngine.from_store(store)
        assert engine.search("anything") == []
