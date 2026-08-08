"""Tests for PINN knowledge base integration."""

from pathlib import Path

from lang_pinn.agents.knowledge import (
    build_search_query,
    load_knowledge,
    search_knowledge,
)
from lang_pinn.agents.pinn_agent import PINNAgent
from lang_pinn.schemas import PDESpec
from rag import KnowledgeStore
from rag.indexing import MarkdownIndexer


def _make_spec(**kwargs) -> PDESpec:
    defaults = {
        "name": "Test",
        "equation": "u_t = 0",
        "independent_vars": ["t"],
        "dependent_var": "u",
        "order": 1,
        "spatial_dim": 0,
        "domain": {"t": (0.0, 1.0)},
    }
    defaults.update(kwargs)
    return PDESpec(**defaults)


# -- Pre-built store tests (uses actual knowledge base) --------------------

_STORE_DIR = Path(__file__).resolve().parents[3] / "data" / "pinn-knowledge" / "store"


class TestPrebuiltKnowledge:
    """Tests against the actual pre-built knowledge store."""

    def test_load_prebuilt_store(self):
        result = load_knowledge(_STORE_DIR)
        assert result is not None
        store, engine = result
        # At least the 20 curated entries; may have more from user uploads
        assert len(store.list_documents()) >= 20

    def test_search_burgers(self):
        result = load_knowledge(_STORE_DIR)
        assert result is not None
        store, engine = result
        context = search_knowledge("burgers equation shock viscosity", store, engine)
        assert "burgers" in context.lower()
        assert len(context) > 0

    def test_search_heat_equation(self):
        result = load_knowledge(_STORE_DIR)
        assert result is not None
        store, engine = result
        context = search_knowledge("heat equation diffusion thermal", store, engine)
        assert "heat" in context.lower()

    def test_search_no_match_returns_empty(self):
        result = load_knowledge(_STORE_DIR)
        assert result is not None
        store, engine = result
        context = search_knowledge("xylophone zymurgy zeppelin kumquat", store, engine)
        assert context == ""

    def test_agent_with_prebuilt_store(self):
        agent = PINNAgent(knowledge_store_dir=_STORE_DIR)
        spec = _make_spec(
            name="Burgers Equation",
            equation="u_t + u*u_x = nu*u_xx",
            independent_vars=["x", "t"],
            spatial_dim=1,
            has_sharp_gradients=True,
            domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        )
        rec = agent.recommend(spec)
        assert rec.knowledge_context != ""
        assert "burgers" in rec.knowledge_context.lower()


# -- Isolated tests (no pre-built store dependency) ------------------------


class TestBuildSearchQuery:
    def test_basic_query(self):
        spec = _make_spec(name="Heat", equation="u_t = u_xx")
        query = build_search_query(spec)
        assert "Heat" in query
        assert "u_t = u_xx" in query

    def test_high_frequency_adds_keywords(self):
        spec = _make_spec(has_high_frequency=True)
        query = build_search_query(spec)
        assert "spectral bias" in query

    def test_sharp_gradients_adds_keywords(self):
        spec = _make_spec(has_sharp_gradients=True)
        query = build_search_query(spec)
        assert "shock" in query

    def test_periodic_bc_adds_keywords(self):
        spec = _make_spec(has_periodic_bc=True)
        query = build_search_query(spec)
        assert "periodic" in query


class TestLoadKnowledge:
    def test_missing_store_returns_none(self, tmp_path):
        result = load_knowledge(tmp_path / "nonexistent")
        assert result is None


class TestAgentWithoutKnowledge:
    def test_agent_works_without_store(self, tmp_path):
        """Agent should still work if knowledge store doesn't exist."""
        agent = PINNAgent(knowledge_store_dir=tmp_path / "no-store")
        spec = _make_spec()
        rec = agent.recommend(spec)
        assert rec.hidden_layers == 3
        assert rec.knowledge_context == ""


class TestAgentWithFixture:
    def test_agent_retrieves_matching_context(self, tmp_path):
        """Build a mini knowledge store and verify agent retrieves from it."""
        store_dir = tmp_path / "kb"
        store = KnowledgeStore(store_dir)

        indexer = MarkdownIndexer(enable_thinning=False)

        # Create 3 mini markdown docs (BM25 needs >= 3 for meaningful IDF)
        for name, content in [
            ("burgers", "# Burgers\n\n## Equation Type\nNonlinear shock viscosity\n\n"
                        "## Techniques\n- RAR: adaptive refinement\n- Causal training\n\n"
                        "## Known Failure Modes\n- Shock starvation\n"),
            ("heat", "# Heat\n\n## Equation Type\nLinear diffusion thermal\n\n"
                     "## Techniques\n- Ansatz: boundary enforcement\n\n"
                     "## Known Failure Modes\n- Slow convergence\n"),
            ("wave", "# Wave\n\n## Equation Type\nHyperbolic propagation acoustic\n\n"
                     "## Techniques\n- Spectral methods\n\n"
                     "## Known Failure Modes\n- Dispersion error\n"),
        ]:
            md_file = tmp_path / f"{name}.md"
            md_file.write_text(content)
            tree = indexer.index_file_sync(str(md_file))
            store.add_document(tree)

        store.save()

        agent = PINNAgent(knowledge_store_dir=store_dir)
        spec = _make_spec(
            name="Burgers Equation",
            equation="u_t + u*u_x = nu*u_xx",
            independent_vars=["x", "t"],
            spatial_dim=1,
            has_sharp_gradients=True,
            domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        )
        rec = agent.recommend(spec)
        assert rec.knowledge_context != ""
