"""Shared dependencies for route handlers."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates
from rag import CollectionManager, FileRegistry, KnowledgeStore, SearchEngine

from .config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


def get_store(settings: Settings = Depends(get_settings)) -> KnowledgeStore:
    store_dir = settings.knowledge_store_dir
    if not Path(store_dir).exists():
        return KnowledgeStore(store_dir)
    return KnowledgeStore.load(store_dir)


def get_search_engine(store: KnowledgeStore = Depends(get_store)) -> SearchEngine:
    return SearchEngine.from_store(store)


def get_collection_manager(
    settings: Settings = Depends(get_settings),
) -> CollectionManager:
    return CollectionManager(settings.collections_db)


def get_registry(settings: Settings = Depends(get_settings)) -> FileRegistry:
    return FileRegistry(settings.registry_db)
