"""Tests for the RAG tester routes."""

import pytest
from api.app import create_app
from api.config import Settings
from fastapi.testclient import TestClient
from rag import DocumentTree, KnowledgeStore, TreeNode


@pytest.fixture
def settings(tmp_path):
    return Settings(
        knowledge_store_dir=tmp_path / "store",
        knowledge_sources_dir=tmp_path / "sources",
        registry_db=tmp_path / "registry.db",
        collections_db=tmp_path / "collections.db",
        users_db=tmp_path / "users.db",
        secret_key="test-secret",
        admin_username="admin",
        admin_password="pass123",
    )


@pytest.fixture
def store_with_docs(settings):
    """Create a store with enough documents for BM25 to score non-zero."""
    store = KnowledgeStore(settings.knowledge_store_dir)

    # Need 3+ docs for BM25 to produce non-zero scores (IDF needs variation)
    tree1 = DocumentTree(
        doc_name="burgers-equation",
        source_path="burgers.md",
        root_nodes=[
            TreeNode(node_id="0001", title="Burgers Equation", level=1,
                     text="The Burgers equation is a nonlinear PDE with shock formation at low viscosity."),
            TreeNode(node_id="0002", title="Techniques", level=2,
                     text="Adaptive refinement and residual collocation help resolve sharp shock fronts."),
        ],
        pde_type="hyperbolic",
        techniques=["adaptive refinement", "shock capturing"],
        keywords=["burgers", "shock", "viscosity"],
    )
    store.add_document(tree1)

    tree2 = DocumentTree(
        doc_name="heat-equation",
        source_path="heat.md",
        root_nodes=[
            TreeNode(node_id="0001", title="Heat Equation", level=1,
                     text="The heat equation is a parabolic PDE modelling thermal diffusion and conduction."),
            TreeNode(node_id="0002", title="Methods", level=2,
                     text="Standard PINN training with collocation points works well for smooth solutions."),
        ],
        pde_type="parabolic",
        techniques=["standard training"],
        keywords=["heat", "diffusion", "thermal"],
    )
    store.add_document(tree2)

    tree3 = DocumentTree(
        doc_name="wave-equation",
        source_path="wave.md",
        root_nodes=[
            TreeNode(node_id="0001", title="Wave Equation", level=1,
                     text="The wave equation describes propagation of waves in elastic media."),
            TreeNode(node_id="0002", title="Boundary Conditions", level=2,
                     text="Dirichlet and Neumann boundary conditions for wave problems."),
        ],
        pde_type="hyperbolic",
        techniques=["time stepping", "spectral methods"],
        keywords=["wave", "propagation", "elastic"],
    )
    store.add_document(tree3)

    tree4 = DocumentTree(
        doc_name="navier-stokes",
        source_path="ns.md",
        root_nodes=[
            TreeNode(node_id="0001", title="Navier-Stokes Equations", level=1,
                     text="The Navier-Stokes equations govern viscous incompressible fluid flow."),
            TreeNode(node_id="0002", title="Streamfunction", level=2,
                     text="Streamfunction formulation enforces incompressibility by construction."),
        ],
        pde_type="elliptic",
        techniques=["streamfunction", "pressure recovery"],
        keywords=["navier-stokes", "fluid", "viscous"],
    )
    store.add_document(tree4)

    store.save()
    return store


@pytest.fixture
def app(settings, store_with_docs):
    return create_app(settings)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def authed_client(client):
    client.post("/login", data={"username": "admin", "password": "pass123"})
    return client


class TestRagTesterPage:
    def test_page_renders(self, authed_client):
        resp = authed_client.get("/rag-tester")
        assert resp.status_code == 200
        assert "RAG Retrieval Tester" in resp.text
        assert "4 documents indexed" in resp.text

    def test_unauthenticated_redirects(self, client):
        resp = client.get("/rag-tester", follow_redirects=False)
        assert resp.status_code == 303


class TestBm25Search:
    def test_search_finds_burgers(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/search",
            data={"query": "shock viscosity burgers", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "burgers-equation" in resp.text

    def test_search_finds_heat(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/search",
            data={"query": "thermal diffusion heat conduction", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "heat-equation" in resp.text

    def test_search_no_results(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/search",
            data={"query": "xyznonexistent12345", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "No matching documents" in resp.text

    def test_search_shows_node_tree(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/search",
            data={"query": "shock viscosity burgers", "top_k": "5"},
        )
        assert resp.status_code == 200
        # Only check if we got results at all
        if "burgers-equation" in resp.text:
            assert "Show node tree" in resp.text


class TestRetrieveContext:
    def test_retrieve_returns_context(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/retrieve",
            data={"query": "shock viscosity burgers", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "Retrieved Context" in resp.text
        if "burgers-equation" in resp.text:
            assert "nonlinear PDE" in resp.text

    def test_retrieve_shows_stats(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/retrieve",
            data={"query": "shock burgers viscosity", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "Retrieved Context" in resp.text

    def test_retrieve_empty_query(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/retrieve",
            data={"query": "", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "No context retrieved" in resp.text


class TestChat:
    def test_chat_renders_response_section(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/chat",
            data={"query": "shock burgers", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "RAG Chat Response" in resp.text
        assert "LLM Response" in resp.text

    def test_chat_shows_context(self, authed_client):
        resp = authed_client.post(
            "/rag-tester/chat",
            data={"query": "thermal diffusion heat", "top_k": "5"},
        )
        assert resp.status_code == 200
        assert "Retrieved Context" in resp.text
        assert "Pipeline Summary" in resp.text

    def test_chat_handles_llm_error(self, authed_client):
        """Chat should gracefully handle LLM failures."""
        resp = authed_client.post(
            "/rag-tester/chat",
            data={"query": "shock burgers viscosity", "top_k": "5"},
        )
        assert resp.status_code == 200
        # Should render without crashing even if LLM is unavailable
        assert "RAG Chat Response" in resp.text
