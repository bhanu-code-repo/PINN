"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from starlette.middleware.sessions import SessionMiddleware

from .config import Settings

_PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="PINN Knowledge Admin",
        description="Admin interface for managing PINN knowledge collections",
        version="0.1.0",
    )

    # Session middleware for flash messages and auth
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

    # Store settings on app state
    app.state.settings = settings

    # Templates
    templates = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))
    app.state.templates = templates

    # Static files
    static_dir = _PACKAGE_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register routes
    from .routes import auth, collections, dashboard, documents

    app.include_router(dashboard.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(auth.router)

    logger.info("PINN Knowledge Admin started (store={})", settings.knowledge_store_dir)
    return app
