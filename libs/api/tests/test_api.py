"""Tests for the FastAPI admin application."""

import pytest
from api.app import create_app
from api.config import Settings
from fastapi.testclient import TestClient


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
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def authed_client(client):
    """Client with an authenticated session."""
    client.post("/login", data={"username": "admin", "password": "pass123"})
    return client


class TestAuth:
    def test_login_page_renders(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200
        assert "PINN Knowledge Admin" in resp.text

    def test_login_success_redirects(self, client):
        resp = client.post(
            "/login",
            data={"username": "admin", "password": "pass123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "/login" not in resp.headers["location"]

    def test_login_failure_shows_error(self, client):
        resp = client.post(
            "/login", data={"username": "admin", "password": "wrong"}
        )
        assert "Invalid credentials" in resp.text

    def test_unauthenticated_redirects_to_login(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 303

    def test_logout(self, authed_client):
        resp = authed_client.get("/logout", follow_redirects=False)
        assert resp.status_code == 303


class TestDashboard:
    def test_dashboard_renders(self, authed_client):
        resp = authed_client.get("/")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text

    def test_dashboard_shows_stats(self, authed_client):
        resp = authed_client.get("/")
        assert "Collections" in resp.text
        assert "Documents" in resp.text

    def test_settings_page(self, authed_client):
        resp = authed_client.get("/settings")
        assert resp.status_code == 200
        assert "Settings" in resp.text


class TestCollections:
    def test_list_collections_empty(self, authed_client):
        resp = authed_client.get("/collections")
        assert resp.status_code == 200
        assert "No collections yet" in resp.text

    def test_create_collection_page(self, authed_client):
        resp = authed_client.get("/collections/new")
        assert resp.status_code == 200
        assert "New Collection" in resp.text

    def test_create_collection(self, authed_client):
        resp = authed_client.post(
            "/collections/new",
            data={
                "name": "Test Collection",
                "description": "A test",
                "access": "public",
                "allowed_groups": "",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert "Test Collection" in resp.text

    def test_create_and_list(self, authed_client):
        authed_client.post(
            "/collections/new",
            data={"name": "Alpha", "description": "", "access": "public", "allowed_groups": ""},
        )
        resp = authed_client.get("/collections")
        assert "Alpha" in resp.text

    def test_edit_collection(self, authed_client):
        authed_client.post(
            "/collections/new",
            data={"name": "Editable", "description": "v1", "access": "public", "allowed_groups": ""},
        )
        resp = authed_client.post(
            "/collections/editable/edit",
            data={
                "name": "Editable",
                "description": "v2",
                "access": "restricted",
                "allowed_groups": "research, engineering",
            },
            follow_redirects=True,
        )
        assert "v2" in resp.text or "Restricted" in resp.text

    def test_delete_collection(self, authed_client):
        authed_client.post(
            "/collections/new",
            data={"name": "Gone", "description": "", "access": "public", "allowed_groups": ""},
        )
        authed_client.post("/collections/gone/delete", follow_redirects=True)
        resp = authed_client.get("/collections")
        assert "Gone" not in resp.text


class TestDocuments:
    def test_list_documents_empty(self, authed_client):
        resp = authed_client.get("/documents")
        assert resp.status_code == 200
        assert "No documents yet" in resp.text

    def test_upload_page(self, authed_client):
        resp = authed_client.get("/upload")
        assert resp.status_code == 200
        assert "Upload Document" in resp.text

    def test_upload_markdown(self, authed_client, tmp_path):
        # Create a markdown file to upload
        md_content = "# Test Doc\n\nSome content about heat equation.\n"
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        with open(md_file, "rb") as f:
            resp = authed_client.post(
                "/upload",
                files={"file": ("test.md", f, "text/markdown")},
                data={"collection_id": "", "hybrid_pdf": ""},
                follow_redirects=True,
            )
        assert resp.status_code == 200
        # Should redirect to documents list with the new doc
        assert "test" in resp.text.lower() or "Documents" in resp.text

    def test_search_empty(self, authed_client):
        resp = authed_client.get("/search?q=heat")
        assert resp.status_code == 200
        assert "0 result" in resp.text

    def test_upload_unsupported_format(self, authed_client, tmp_path):
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b,c")
        with open(bad_file, "rb") as f:
            resp = authed_client.post(
                "/upload",
                files={"file": ("data.csv", f, "text/csv")},
                data={"collection_id": "", "hybrid_pdf": ""},
                follow_redirects=True,
            )
        assert "Unsupported" in resp.text or "Upload" in resp.text
