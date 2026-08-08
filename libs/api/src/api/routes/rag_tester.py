"""RAG retrieval tester and chat routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from rag import KnowledgeStore, SearchEngine

from ..deps import get_search_engine, get_store, get_templates

router = APIRouter(prefix="/rag-tester", tags=["rag_tester"])


def _require_auth(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse(url=request.url_for("login_page"), status_code=303)
    return None


# -- Page ---------------------------------------------------------------------

@router.get("", name="rag_tester_page")
def rag_tester_page(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    doc_count = len(store.list_documents())
    return templates.TemplateResponse(
        request,
        "pages/rag_tester.html",
        {"active_page": "rag_tester", "doc_count": doc_count},
    )


# -- BM25 Search (returns partial HTML) --------------------------------------

@router.post("/search", name="rag_search")
def rag_search(
    request: Request,
    query: str = Form(""),
    top_k: int = Form(5),
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    results = []
    if query.strip():
        doc_ids = engine.search(query.strip(), top_k=top_k)
        for doc_id in doc_ids:
            try:
                meta = store.get_metadata(doc_id)
                tree = store.get_document(doc_id)
                # Collect node titles for tree preview
                node_titles = []
                _collect_node_titles(tree.root_nodes, node_titles)
                results.append({
                    "doc_id": meta.doc_id,
                    "doc_name": meta.doc_name,
                    "pde_type": meta.pde_type,
                    "techniques": meta.techniques,
                    "keywords": meta.keywords,
                    "node_count": meta.node_count,
                    "total_tokens": meta.total_tokens,
                    "node_titles": node_titles[:15],  # cap for display
                })
            except KeyError:
                pass

    return templates.TemplateResponse(
        request,
        "partials/rag_search_results.html",
        {"query": query, "results": results, "top_k": top_k},
    )


# -- Retrieve Context (BM25 + full node text) --------------------------------

@router.post("/retrieve", name="rag_retrieve")
def rag_retrieve(
    request: Request,
    query: str = Form(""),
    top_k: int = Form(5),
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    context_parts = []
    if query.strip():
        doc_ids = engine.search(query.strip(), top_k=top_k)
        for doc_id in doc_ids:
            try:
                meta = store.get_metadata(doc_id)
                tree = store.get_document(doc_id)
                # Build formatted context from full tree
                text = _format_document_context(meta, tree)
                context_parts.append({
                    "doc_id": meta.doc_id,
                    "doc_name": meta.doc_name,
                    "pde_type": meta.pde_type,
                    "text": text,
                })
            except KeyError:
                pass

    return templates.TemplateResponse(
        request,
        "partials/rag_context.html",
        {"query": query, "context_parts": context_parts},
    )


# -- Chat (retrieve context + LLM response) ----------------------------------

@router.post("/chat", name="rag_chat")
def rag_chat(
    request: Request,
    query: str = Form(""),
    top_k: int = Form(5),
    templates: Jinja2Templates = Depends(get_templates),
    store: KnowledgeStore = Depends(get_store),
    engine: SearchEngine = Depends(get_search_engine),
):
    redirect = _require_auth(request)
    if redirect:
        return redirect

    context_parts = []
    llm_response = ""
    llm_error = ""

    if query.strip():
        # Build context from BM25 results
        doc_ids = engine.search(query.strip(), top_k=top_k)
        for doc_id in doc_ids:
            try:
                meta = store.get_metadata(doc_id)
                tree = store.get_document(doc_id)
                text = _format_document_context(meta, tree)
                context_parts.append({
                    "doc_id": meta.doc_id,
                    "doc_name": meta.doc_name,
                    "text": text,
                })
            except KeyError:
                pass

        # Send to LLM with context
        if context_parts:
            full_context = "\n\n---\n\n".join(p["text"] for p in context_parts)
            prompt = (
                "You are a knowledgeable assistant for Physics-Informed Neural Networks (PINNs). "
                "Use the following knowledge base context to answer the user's question. "
                "If the context doesn't contain enough information, say so.\n\n"
                f"## Knowledge Base Context\n\n{full_context}\n\n"
                f"## Question\n\n{query}"
            )
            try:
                from llm_provider import LLMClient
                client = LLMClient()
                llm_response = client.ask(prompt)
            except Exception as e:
                logger.warning("LLM call failed: {}", e)
                llm_error = str(e)

    return templates.TemplateResponse(
        request,
        "partials/rag_chat_response.html",
        {
            "query": query,
            "context_parts": context_parts,
            "llm_response": llm_response,
            "llm_error": llm_error,
        },
    )


# -- Helpers ------------------------------------------------------------------

def _collect_node_titles(nodes, titles, depth=0):
    """Recursively collect node titles with depth."""
    for node in nodes:
        titles.append({"title": node.title, "depth": depth, "node_id": node.node_id})
        _collect_node_titles(node.children, titles, depth + 1)


def _format_document_context(meta, tree):
    """Format a document tree into readable context text."""
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
    """Recursively render node text."""
    for node in nodes:
        prefix = "#" * (depth + 4)  # Start at h4
        if node.title:
            lines.append(f"{prefix} {node.title}")
        if node.text:
            lines.append(node.text.strip())
            lines.append("")
        _render_nodes(node.children, lines, depth + 1)
