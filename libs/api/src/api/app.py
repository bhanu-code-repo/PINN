"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from starlette.middleware.sessions import SessionMiddleware

from .config import Settings
from .users import UserManager

_PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    # Ensure at least one admin user exists
    with UserManager(settings.users_db) as mgr:
        mgr.ensure_admin(settings.admin_username, settings.admin_password)

    app = FastAPI(
        title="PINN Knowledge Admin",
        description="Admin interface for managing PINN knowledge collections",
        version="0.1.0",
    )

    # CORS for React frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Session middleware for flash messages and auth
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

    # Store settings on app state
    app.state.settings = settings

    # Templates with markdown filter
    templates = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))

    import markdown as _md
    from markupsafe import Markup

    def _md_filter(text: str) -> Markup:
        """Convert markdown text to HTML."""
        html = _md.markdown(text, extensions=["fenced_code", "tables", "codehilite"])
        return Markup(html)

    templates.env.filters["markdown"] = _md_filter
    app.state.templates = templates

    # Static files
    static_dir = _PACKAGE_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register routes
    from .routes import api_v1, auth, collections, dashboard, documents, rag_tester

    app.include_router(api_v1.router)
    app.include_router(dashboard.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(rag_tester.router)
    app.include_router(auth.router)

    logger.info("PINN Knowledge Admin started (store={})", settings.knowledge_store_dir)
    return app
