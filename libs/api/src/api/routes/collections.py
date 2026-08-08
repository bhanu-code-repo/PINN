"""Collection management routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from rag import CollectionManager, KnowledgeStore

from ..deps import get_collection_manager, get_store, get_templates

router = APIRouter(prefix="/collections", tags=["collections"])


def _require_auth(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse(url=request.url_for("login_page"), status_code=303)
    return None


def _flash(request: Request, message: str, category: str = "success"):
    request.session["flash"] = {"message": message, "category": category}


# -- List -------------------------------------------------------------------

@router.get("", name="list_collections")
def list_collections(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    collections = cm.list_collections()
    return templates.TemplateResponse(
        request,
        "pages/collections_list.html",
        {"active_page": "collections", "collections": collections},
    )


# -- Create -----------------------------------------------------------------

@router.get("/new", name="create_collection_page")
def create_collection_page(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    return templates.TemplateResponse(
        request,
        "pages/collection_form.html",
        {"active_page": "collections", "collection": None},
    )


@router.post("/new", name="create_collection")
def create_collection(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    access: str = Form("public"),
    allowed_groups: str = Form(""),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    groups = [g.strip() for g in allowed_groups.split(",") if g.strip()]
    try:
        c = cm.create_collection(
            name, description=description, access=access, allowed_groups=groups
        )
        _flash(request, f"Collection '{c.name}' created.")
        return RedirectResponse(
            url=request.url_for("view_collection", collection_id=c.collection_id),
            status_code=303,
        )
    except ValueError as e:
        _flash(request, str(e), "danger")
        return RedirectResponse(
            url=request.url_for("create_collection_page"), status_code=303
        )


# -- View -------------------------------------------------------------------

@router.get("/{collection_id}", name="view_collection")
def view_collection(
    request: Request,
    collection_id: str,
    templates: Jinja2Templates = Depends(get_templates),
    cm: CollectionManager = Depends(get_collection_manager),
    store: KnowledgeStore = Depends(get_store),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    try:
        collection = cm.get_collection(collection_id)
    except KeyError:
        _flash(request, f"Collection '{collection_id}' not found.", "danger")
        return RedirectResponse(url=request.url_for("list_collections"), status_code=303)

    doc_ids = cm.get_collection_doc_ids(collection_id)
    all_docs = store.list_documents()
    documents = [d for d in all_docs if d.doc_id in doc_ids]
    available_docs = [d for d in all_docs if d.doc_id not in doc_ids]

    return templates.TemplateResponse(
        request,
        "pages/collection_detail.html",
        {
            "active_page": "collections",
            "collection": collection,
            "documents": documents,
            "available_docs": available_docs,
        },
    )


# -- Edit -------------------------------------------------------------------

@router.get("/{collection_id}/edit", name="edit_collection_page")
def edit_collection_page(
    request: Request,
    collection_id: str,
    templates: Jinja2Templates = Depends(get_templates),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    try:
        collection = cm.get_collection(collection_id)
    except KeyError:
        _flash(request, f"Collection '{collection_id}' not found.", "danger")
        return RedirectResponse(url=request.url_for("list_collections"), status_code=303)

    return templates.TemplateResponse(
        request,
        "pages/collection_form.html",
        {"active_page": "collections", "collection": collection},
    )


@router.post("/{collection_id}/edit", name="edit_collection")
def edit_collection(
    request: Request,
    collection_id: str,
    name: str = Form(...),
    description: str = Form(""),
    access: str = Form("public"),
    allowed_groups: str = Form(""),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    groups = [g.strip() for g in allowed_groups.split(",") if g.strip()]
    try:
        cm.update_collection(
            collection_id,
            description=description,
            access=access,
            allowed_groups=groups,
        )
        _flash(request, "Collection updated.")
    except KeyError:
        _flash(request, f"Collection '{collection_id}' not found.", "danger")

    return RedirectResponse(
        url=request.url_for("view_collection", collection_id=collection_id),
        status_code=303,
    )


# -- Delete -----------------------------------------------------------------

@router.post("/{collection_id}/delete", name="delete_collection")
def delete_collection(
    request: Request,
    collection_id: str,
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    try:
        cm.delete_collection(collection_id)
        _flash(request, "Collection deleted.")
    except KeyError:
        _flash(request, "Collection not found.", "danger")

    return RedirectResponse(url=request.url_for("list_collections"), status_code=303)


# -- Document membership ---------------------------------------------------

@router.post("/{collection_id}/docs", name="add_doc_to_collection")
def add_doc_to_collection(
    request: Request,
    collection_id: str,
    doc_id: str = Form(""),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    if doc_id:
        cm.add_document_to_collection(collection_id, doc_id)
        _flash(request, "Document added to collection.")

    return RedirectResponse(
        url=request.url_for("view_collection", collection_id=collection_id),
        status_code=303,
    )


@router.post(
    "/{collection_id}/docs/{doc_id}/remove", name="remove_doc_from_collection"
)
def remove_doc_from_collection(
    request: Request,
    collection_id: str,
    doc_id: str,
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    cm.remove_document_from_collection(collection_id, doc_id)
    _flash(request, "Document removed from collection.")

    return RedirectResponse(
        url=request.url_for("view_collection", collection_id=collection_id),
        status_code=303,
    )
