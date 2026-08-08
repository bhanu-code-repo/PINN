"""Dashboard and settings routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from rag import CollectionManager, FileRegistry, KnowledgeStore

from ..config import Settings
from ..deps import (
    get_collection_manager,
    get_registry,
    get_settings,
    get_store,
    get_templates,
)

router = APIRouter(tags=["dashboard"])


def _require_auth(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse(url=request.url_for("login_page"), status_code=303)
    return None


@router.get("/", name="dashboard")
def dashboard(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
    cm: CollectionManager = Depends(get_collection_manager),
    registry: FileRegistry = Depends(get_registry),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    collections = cm.list_collections()
    documents = store.list_documents()

    stats = {
        "collections": len(collections),
        "documents": len(documents),
        "files": registry.count(),
        "restricted": sum(1 for c in collections if c.access == "restricted"),
    }

    return templates.TemplateResponse(
        request,
        "pages/dashboard.html",
        {
            "active_page": "dashboard",
            "stats": stats,
            "recent_collections": collections[:5],
            "recent_documents": documents[:5],
        },
    )


@router.get("/settings", name="settings_page")
def settings_page(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    settings: Settings = Depends(get_settings),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    return templates.TemplateResponse(
        request,
        "pages/settings.html",
        {
            "active_page": "settings",
            "settings": settings,
        },
    )
