"""Document management routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from rag import (
    CollectionManager,
    FileRegistry,
    KnowledgeStore,
    SearchEngine,
    ingest_file,
)

from ..config import Settings
from ..deps import (
    get_collection_manager,
    get_registry,
    get_search_engine,
    get_settings,
    get_store,
    get_templates,
)

router = APIRouter(tags=["documents"])

_PAGE_SIZE = 10


def _require_auth(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse(url=request.url_for("login_page"), status_code=303)
    return None


def _flash(request: Request, message: str, category: str = "success"):
    request.session["flash"] = {"message": message, "category": category}


# -- List -------------------------------------------------------------------

@router.get("/documents", name="list_documents")
def list_documents(
    request: Request,
    page: int = 1,
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    all_docs = store.list_documents()
    total = len(all_docs)
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * _PAGE_SIZE
    documents = all_docs[start : start + _PAGE_SIZE]

    return templates.TemplateResponse(
        request,
        "pages/documents_list.html",
        {
            "active_page": "documents",
            "documents": documents,
            "page": page,
            "total_pages": total_pages,
            "query": None,
        },
    )


# -- View -------------------------------------------------------------------

@router.get("/documents/{doc_id}", name="view_document")
def view_document(
    request: Request,
    doc_id: str,
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    try:
        doc = store.get_metadata(doc_id)
        tree = store.get_document(doc_id)
    except KeyError:
        _flash(request, f"Document '{doc_id}' not found.", "danger")
        return RedirectResponse(url=request.url_for("list_documents"), status_code=303)

    collections = cm.get_document_collections(doc_id)

    return templates.TemplateResponse(
        request,
        "pages/document_detail.html",
        {
            "active_page": "documents",
            "doc": doc,
            "tree": tree,
            "collections": collections,
        },
    )


# -- Search -----------------------------------------------------------------

@router.get("/search", name="search_documents")
def search_documents(
    request: Request,
    q: str = "",
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    results = []
    if q.strip():
        hits = engine.search(q.strip(), top_k=10)
        for doc_id in hits:
            try:
                meta = store.get_metadata(doc_id)
                results.append(
                    {
                        "doc_id": meta.doc_id,
                        "doc_name": meta.doc_name,
                        "keywords": meta.keywords,
                        "context": "",
                    }
                )
            except KeyError:
                pass

    return templates.TemplateResponse(
        request,
        "pages/search_results.html",
        {
            "active_page": "documents",
            "query": q,
            "results": results,
        },
    )


# -- Upload -----------------------------------------------------------------

@router.get("/upload", name="upload_page")
def upload_page(
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
        "pages/upload.html",
        {"active_page": "upload", "collections": collections},
    )


@router.post("/upload", name="upload_document")
def upload_document(
    request: Request,
    file: UploadFile,
    collection_id: str = Form(""),
    hybrid_pdf: str = Form(""),
    settings: Settings = Depends(get_settings),
    store: KnowledgeStore = Depends(get_store),
    registry: FileRegistry = Depends(get_registry),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    if not file.filename:
        _flash(request, "No file selected.", "danger")
        return RedirectResponse(url=request.url_for("upload_page"), status_code=303)

    ext = Path(file.filename).suffix.lower()
    if ext not in (".md", ".txt", ".pdf"):
        _flash(request, f"Unsupported file type: {ext}", "danger")
        return RedirectResponse(url=request.url_for("upload_page"), status_code=303)

    # Save to sources directory
    sources_dir = settings.knowledge_sources_dir
    sources_dir.mkdir(parents=True, exist_ok=True)
    dest = sources_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    use_hybrid = hybrid_pdf == "1"
    result = ingest_file(
        dest,
        store=store,
        registry=registry,
        hybrid_pdf=use_hybrid,
    )

    if result.status == "skipped":
        _flash(request, f"File already indexed: {result.message}", "warning")
    elif result.status == "error":
        _flash(request, f"Error: {result.message}", "danger")
    else:
        store.save()
        _flash(request, f"Indexed '{file.filename}' as {result.doc_id}")

        # Add to collection if specified
        if collection_id and result.doc_id:
            cm.add_document_to_collection(collection_id, result.doc_id)

    return RedirectResponse(url=request.url_for("list_documents"), status_code=303)


# -- Delete -----------------------------------------------------------------

@router.post("/documents/{doc_id}/delete", name="delete_document")
def delete_document(
    request: Request,
    doc_id: str,
    settings: Settings = Depends(get_settings),
    store: KnowledgeStore = Depends(get_store),
    registry: FileRegistry = Depends(get_registry),
    cm: CollectionManager = Depends(get_collection_manager),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    try:
        store.remove_document(doc_id)
        store.save()
    except KeyError:
        _flash(request, "Document not found.", "danger")
        return RedirectResponse(url=request.url_for("list_documents"), status_code=303)

    # Clean up registry
    for rec in registry.list_all():
        if rec.doc_id == doc_id:
            registry.unregister(rec.file_hash)
            break

    # Clean up source file
    for ext in (".md", ".txt", ".pdf"):
        source = settings.knowledge_sources_dir / f"{doc_id}{ext}"
        if source.exists():
            source.unlink()

    # Remove from all collections
    for coll_id in cm.get_document_collections(doc_id):
        cm.remove_document_from_collection(coll_id, doc_id)

    _flash(request, f"Document '{doc_id}' deleted.")
    return RedirectResponse(url=request.url_for("list_documents"), status_code=303)
