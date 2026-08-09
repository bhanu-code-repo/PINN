"""JSON API endpoints for the React frontend (v1)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel
from rag import CollectionManager, FileRegistry, KnowledgeStore, SearchEngine

from ..config import Settings
from ..deps import (
    get_collection_manager,
    get_registry,
    get_search_engine,
    get_settings,
    get_store,
    get_user_manager,
)
from ..users import UserManager

router = APIRouter(prefix="/api/v1", tags=["api_v1"])


# -- Auth dependency ----------------------------------------------------------

def require_auth(request: Request) -> dict:
    """Raise 401 if not authenticated. Returns session user info."""
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "username": request.session.get("username", ""),
        "groups": request.session.get("groups", []),
        "is_admin": request.session.get("is_admin", False),
    }


# -- Request/Response models --------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 5


class CreateCollectionRequest(BaseModel):
    name: str
    description: str = ""
    access: str = "public"
    allowed_groups: str = ""


class AddDocRequest(BaseModel):
    doc_id: str


# -- Auth endpoints -----------------------------------------------------------

@router.post("/auth/login")
def api_login(
    request: Request,
    body: LoginRequest,
    user_mgr: UserManager = Depends(get_user_manager),
):
    user = user_mgr.authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["authenticated"] = True
    request.session["username"] = user.username
    request.session["groups"] = user.groups
    request.session["is_admin"] = user.is_admin
    return {"username": user.username, "groups": user.groups, "is_admin": user.is_admin}


@router.post("/auth/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/auth/me")
def api_me(user: dict = Depends(require_auth)):
    return user


# -- Dashboard ----------------------------------------------------------------

@router.get("/dashboard/stats")
def api_dashboard_stats(
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
    cm: CollectionManager = Depends(get_collection_manager),
    registry: FileRegistry = Depends(get_registry),
):
    all_docs = store.list_documents()
    collections = cm.list_collections()
    restricted = [c for c in collections if c.access == "restricted"]

    recent_colls = []
    for c in collections[:5]:
        doc_ids = cm.get_collection_doc_ids(c.id)
        recent_colls.append({
            "id": c.id, "name": c.name,
            "access": c.access, "doc_count": len(doc_ids),
        })

    recent_docs = [
        {
            "doc_id": d.doc_id, "doc_name": d.doc_name,
            "node_count": d.node_count, "total_tokens": d.total_tokens,
        }
        for d in all_docs[:5]
    ]

    return {
        "collections": len(collections),
        "documents": len(all_docs),
        "files": registry.count(),
        "restricted": len(restricted),
        "recent_collections": recent_colls,
        "recent_documents": recent_docs,
    }


# -- Collections --------------------------------------------------------------

@router.get("/collections")
def api_list_collections(
    _user: dict = Depends(require_auth),
    cm: CollectionManager = Depends(get_collection_manager),
):
    collections = cm.list_collections()
    result = []
    for c in collections:
        doc_ids = cm.get_collection_doc_ids(c.id)
        result.append({
            "id": c.id, "name": c.name, "description": c.description,
            "access": c.access,
            "allowed_groups": [g.strip() for g in (c.allowed_groups or "").split(",") if g.strip()],
            "doc_count": len(doc_ids),
            "created_at": c.created_at, "updated_at": c.updated_at,
        })
    return result


@router.get("/collections/{coll_id}")
def api_get_collection(
    coll_id: str,
    _user: dict = Depends(require_auth),
    cm: CollectionManager = Depends(get_collection_manager),
    store: KnowledgeStore = Depends(get_store),
):
    try:
        c = cm.get_collection(coll_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Collection not found") from None

    doc_ids = cm.get_collection_doc_ids(coll_id)
    documents = []
    for did in doc_ids:
        try:
            meta = store.get_metadata(did)
            documents.append({"doc_id": meta.doc_id, "doc_name": meta.doc_name})
        except KeyError:
            documents.append({"doc_id": did, "doc_name": did})

    all_docs = store.list_documents()
    available = [
        {"doc_id": d.doc_id, "doc_name": d.doc_name}
        for d in all_docs if d.doc_id not in doc_ids
    ]

    return {
        "collection": {
            "id": c.id, "name": c.name, "description": c.description,
            "access": c.access,
            "allowed_groups": [g.strip() for g in (c.allowed_groups or "").split(",") if g.strip()],
            "created_at": c.created_at, "updated_at": c.updated_at,
        },
        "documents": documents,
        "available_docs": available,
    }


@router.post("/collections")
def api_create_collection(
    body: CreateCollectionRequest,
    _user: dict = Depends(require_auth),
    cm: CollectionManager = Depends(get_collection_manager),
):
    c = cm.create_collection(
        name=body.name, description=body.description,
        access=body.access, allowed_groups=body.allowed_groups,
    )
    return {"id": c.id, "name": c.name, "description": c.description, "access": c.access}


@router.put("/collections/{coll_id}")
def api_update_collection(
    coll_id: str,
    body: CreateCollectionRequest,
    _user: dict = Depends(require_auth),
    cm: CollectionManager = Depends(get_collection_manager),
):
    try:
        c = cm.update_collection(
            coll_id, name=body.name, description=body.description,
            access=body.access, allowed_groups=body.allowed_groups,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Collection not found") from None
    return {"id": c.id, "name": c.name, "description": c.description, "access": c.access}


@router.delete("/collections/{coll_id}")
def api_delete_collection(
    coll_id: str,
    _user: dict = Depends(require_auth),
    cm: CollectionManager = Depends(get_collection_manager),
):
    try:
        cm.delete_collection(coll_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Collection not found") from None
    return {"ok": True}


@router.post("/collections/{coll_id}/docs")
def api_add_doc_to_collection(
    coll_id: str,
    body: AddDocRequest,
    _user: dict = Depends(require_auth),
    cm: CollectionManager = Depends(get_collection_manager),
):
    cm.add_document_to_collection(coll_id, body.doc_id)
    return {"ok": True}


@router.delete("/collections/{coll_id}/docs/{doc_id}")
def api_remove_doc_from_collection(
    coll_id: str,
    doc_id: str,
    _user: dict = Depends(require_auth),
    cm: CollectionManager = Depends(get_collection_manager),
):
    cm.remove_document_from_collection(coll_id, doc_id)
    return {"ok": True}


# -- Documents ----------------------------------------------------------------

_PAGE_SIZE = 10


@router.get("/documents")
def api_list_documents(
    page: int = 1,
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
):
    all_docs = store.list_documents()
    total = len(all_docs)
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * _PAGE_SIZE
    documents = all_docs[start:start + _PAGE_SIZE]

    return {
        "documents": [
            {
                "doc_id": d.doc_id, "doc_name": d.doc_name, "pde_type": d.pde_type,
                "techniques": d.techniques, "keywords": d.keywords,
                "known_issues": d.known_issues, "node_count": d.node_count,
                "total_tokens": d.total_tokens, "indexed_at": d.indexed_at,
            }
            for d in documents
        ],
        "page": page,
        "total_pages": total_pages,
        "total": total,
    }


@router.get("/documents/search")
def api_search_documents(
    q: str = "",
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    results = []
    if q.strip():
        hits = engine.search(q.strip(), top_k=10)
        for doc_id in hits:
            try:
                meta = store.get_metadata(doc_id)
                results.append({
                    "doc_id": meta.doc_id, "doc_name": meta.doc_name,
                    "keywords": meta.keywords, "context": "",
                })
            except KeyError:
                pass
    return results


@router.get("/documents/{doc_id}")
def api_get_document(
    doc_id: str,
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
    cm: CollectionManager = Depends(get_collection_manager),
):
    try:
        doc = store.get_metadata(doc_id)
        tree = store.get_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found") from None

    collections = cm.get_document_collections(doc_id)

    def serialize_node(node):
        return {
            "node_id": node.node_id, "title": node.title,
            "text": node.text, "level": node.level,
            "summary": node.summary,
            "children": [serialize_node(c) for c in node.children],
        }

    return {
        "doc": {
            "doc_id": doc.doc_id, "doc_name": doc.doc_name, "pde_type": doc.pde_type,
            "techniques": doc.techniques, "keywords": doc.keywords,
            "known_issues": doc.known_issues, "node_count": doc.node_count,
            "total_tokens": doc.total_tokens, "indexed_at": doc.indexed_at,
        },
        "tree": {
            "doc_name": tree.doc_name,
            "root_nodes": [serialize_node(n) for n in tree.root_nodes],
        },
        "collections": collections,
    }


@router.post("/documents/upload")
def api_upload_document(
    request: Request,
    _user: dict = Depends(require_auth),
):
    # Upload handled via the existing template route for now
    # React will use the existing /upload endpoint until we refactor
    raise HTTPException(status_code=501, detail="Use multipart upload via /upload")


@router.delete("/documents/{doc_id}")
def api_delete_document(
    doc_id: str,
    _user: dict = Depends(require_auth),
    settings: Settings = Depends(get_settings),
    store: KnowledgeStore = Depends(get_store),
    registry: FileRegistry = Depends(get_registry),
    cm: CollectionManager = Depends(get_collection_manager),
):
    try:
        store.remove_document(doc_id)
        store.save()
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found") from None

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

    return {"ok": True}


# -- RAG Tester ---------------------------------------------------------------

@router.get("/rag/info")
def api_rag_info(
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
):
    return {"doc_count": len(store.list_documents())}


@router.post("/rag/search")
def api_rag_search(
    body: RagQueryRequest,
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    results = []
    if body.query.strip():
        doc_ids = engine.search(body.query.strip(), top_k=body.top_k)
        for doc_id in doc_ids:
            try:
                meta = store.get_metadata(doc_id)
                tree = store.get_document(doc_id)
                node_titles: list[dict] = []
                _collect_node_titles(tree.root_nodes, node_titles)
                results.append({
                    "doc_id": meta.doc_id, "doc_name": meta.doc_name,
                    "pde_type": meta.pde_type, "techniques": meta.techniques,
                    "keywords": meta.keywords, "node_count": meta.node_count,
                    "total_tokens": meta.total_tokens,
                    "node_titles": node_titles[:15],
                })
            except KeyError:
                pass
    return results


@router.post("/rag/retrieve")
def api_rag_retrieve(
    body: RagQueryRequest,
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    context_parts = []
    if body.query.strip():
        doc_ids = engine.search(body.query.strip(), top_k=body.top_k)
        for doc_id in doc_ids:
            try:
                meta = store.get_metadata(doc_id)
                tree = store.get_document(doc_id)
                text = _format_document_context(meta, tree)
                context_parts.append({
                    "doc_id": meta.doc_id, "doc_name": meta.doc_name,
                    "pde_type": meta.pde_type, "text": text,
                })
            except KeyError:
                pass
    return context_parts


@router.post("/rag/chat")
def api_rag_chat(
    body: RagQueryRequest,
    _user: dict = Depends(require_auth),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    context_parts = []
    llm_response = ""
    llm_error = ""

    if body.query.strip():
        doc_ids = engine.search(body.query.strip(), top_k=body.top_k)
        for doc_id in doc_ids:
            try:
                meta = store.get_metadata(doc_id)
                tree = store.get_document(doc_id)
                text = _format_document_context(meta, tree)
                context_parts.append({
                    "doc_id": meta.doc_id, "doc_name": meta.doc_name,
                    "text": text,
                })
            except KeyError:
                pass

        if context_parts:
            full_context = "\n\n---\n\n".join(p["text"] for p in context_parts)
            prompt = (
                "You are a knowledgeable assistant for Physics-Informed Neural Networks (PINNs). "
                "Use the following knowledge base context to answer the user's question. "
                "If the context doesn't contain enough information, say so.\n\n"
                f"## Knowledge Base Context\n\n{full_context}\n\n"
                f"## Question\n\n{body.query}"
            )
            try:
                from llm_provider import LLMClient
                client = LLMClient()
                llm_response = client.ask(prompt)
            except Exception as e:
                logger.warning("LLM call failed: {}", e)
                llm_error = str(e)

    return {
        "query": body.query,
        "context_parts": context_parts,
        "llm_response": llm_response,
        "llm_error": llm_error,
    }


# -- Settings -----------------------------------------------------------------

@router.get("/settings")
def api_settings(
    _user: dict = Depends(require_auth),
    settings: Settings = Depends(get_settings),
):
    return {
        "knowledge_store_dir": str(settings.knowledge_store_dir),
        "knowledge_sources_dir": str(settings.knowledge_sources_dir),
        "registry_db": str(settings.registry_db),
        "collections_db": str(settings.collections_db),
        "users_db": str(settings.users_db),
        "host": settings.host,
        "port": settings.port,
    }


# -- Helpers ------------------------------------------------------------------

def _collect_node_titles(nodes, titles, depth=0):
    for node in nodes:
        titles.append({"title": node.title, "depth": depth, "node_id": node.node_id})
        _collect_node_titles(node.children, titles, depth + 1)


def _format_document_context(meta, tree):
    lines = [f"### {meta.doc_name}"]
    if meta.pde_type:
        lines.append(f"**Type:** {meta.pde_type}")
    if meta.techniques:
        lines.append(f"**Techniques:** {', '.join(meta.techniques)}")
    if meta.known_issues:
        lines.append(f"**Known issues:** {', '.join(meta.known_issues)}")
    lines.append("")
    _render_nodes(tree.root_nodes, lines)
    return "\n".join(lines)


def _render_nodes(nodes, lines, depth=0):
    for node in nodes:
        prefix = "#" * (depth + 4)
        if node.title:
            lines.append(f"{prefix} {node.title}")
        if node.text:
            lines.append(node.text.strip())
            lines.append("")
        _render_nodes(node.children, lines, depth + 1)
