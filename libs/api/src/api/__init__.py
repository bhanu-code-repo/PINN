"""api -- FastAPI admin server for PINN knowledge base management.

Provides REST API endpoints and Jinja2-rendered admin pages for
managing knowledge collections, documents, and access control.

Quick start::

    uvicorn api:create_app --factory --reload
"""

from .app import create_app

__all__ = ["create_app"]
