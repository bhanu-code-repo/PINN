"""JSON-based knowledge store for indexed document trees.

Manages a directory of indexed documents with metadata. Each document
is stored as a JSON file with its hierarchical tree structure.

Layout::

    store_dir/
        manifest.json               # list of DocumentMetadata
        {doc_id}_structure.json      # serialized DocumentTree
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .models.schemas import DocumentMetadata, DocumentTree, TreeNode

# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


class KnowledgeStore:
    """Manages a directory of indexed document trees.

    Parameters
    ----------
    store_dir : str or Path
        Directory for storing manifest and structure files.
        Created if it does not exist.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, DocumentMetadata] = {}

    # -- Document management ------------------------------------------------

    def add_document(
        self,
        tree: DocumentTree,
        *,
        doc_id: str | None = None,
    ) -> str:
        """Index a document tree. Returns the assigned ``doc_id``.

        If ``doc_id`` is None, derives one from ``tree.doc_name``.
        Overwrites if ``doc_id`` already exists (upsert).
        """
        if doc_id is None:
            doc_id = self._slugify(tree.doc_name)

        # Save structure JSON
        structure_path = self.store_dir / f"{doc_id}_structure.json"
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(self._tree_to_dict(tree), f, indent=2, ensure_ascii=False)

        # Build metadata
        node_count = self._count_nodes(tree.root_nodes)
        total_tokens = self._count_tokens_in_tree(tree.root_nodes)

        meta = DocumentMetadata(
            doc_id=doc_id,
            doc_name=tree.doc_name,
            source_path=tree.source_path,
            pde_type=tree.pde_type,
            spatial_dim=tree.spatial_dim,
            techniques=list(tree.techniques),
            keywords=list(tree.keywords),
            known_issues=list(tree.known_issues),
            node_count=node_count,
            total_tokens=total_tokens,
            indexed_at=datetime.now(tz=UTC).isoformat(),
        )
        self._manifest[doc_id] = meta
        logger.info("Added document '{}' ({} nodes)", doc_id, node_count)
        return doc_id

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and its structure file."""
        if doc_id not in self._manifest:
            msg = f"Document '{doc_id}' not found in store"
            raise KeyError(msg)

        structure_path = self.store_dir / f"{doc_id}_structure.json"
        if structure_path.exists():
            structure_path.unlink()

        del self._manifest[doc_id]
        logger.info("Removed document '{}'", doc_id)

    def get_document(self, doc_id: str) -> DocumentTree:
        """Load a ``DocumentTree`` from disk."""
        if doc_id not in self._manifest:
            msg = f"Document '{doc_id}' not found in store"
            raise KeyError(msg)

        structure_path = self.store_dir / f"{doc_id}_structure.json"
        with open(structure_path, encoding="utf-8") as f:
            data = json.load(f)

        return self._dict_to_tree(data)

    def list_documents(self) -> list[DocumentMetadata]:
        """Return all document metadata entries."""
        return list(self._manifest.values())

    def get_metadata(self, doc_id: str) -> DocumentMetadata:
        """Return metadata for a single document."""
        if doc_id not in self._manifest:
            msg = f"Document '{doc_id}' not found in store"
            raise KeyError(msg)
        return self._manifest[doc_id]

    # -- Persistence --------------------------------------------------------

    def save(self) -> None:
        """Write manifest to disk."""
        manifest_path = self.store_dir / "manifest.json"
        data = [asdict(m) for m in self._manifest.values()]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("Saved manifest with {} documents", len(self._manifest))

    @classmethod
    def load(cls, store_dir: str | Path) -> KnowledgeStore:
        """Load an existing store from a directory."""
        store = cls(store_dir)
        manifest_path = store.store_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                meta = DocumentMetadata(**entry)
                store._manifest[meta.doc_id] = meta

            logger.info("Loaded store with {} documents", len(store._manifest))
        else:
            logger.debug("No manifest found at {}, starting empty", manifest_path)

        return store

    # -- Serialization helpers ----------------------------------------------

    def _tree_to_dict(self, tree: DocumentTree) -> dict:
        return {
            "doc_name": tree.doc_name,
            "source_path": tree.source_path,
            "pde_type": tree.pde_type,
            "spatial_dim": tree.spatial_dim,
            "known_issues": tree.known_issues,
            "techniques": tree.techniques,
            "keywords": tree.keywords,
            "root_nodes": [self._node_to_dict(n) for n in tree.root_nodes],
        }

    def _dict_to_tree(self, data: dict) -> DocumentTree:
        return DocumentTree(
            doc_name=data["doc_name"],
            source_path=data["source_path"],
            pde_type=data.get("pde_type", ""),
            spatial_dim=data.get("spatial_dim"),
            known_issues=data.get("known_issues", []),
            techniques=data.get("techniques", []),
            keywords=data.get("keywords", []),
            root_nodes=[self._dict_to_node(n) for n in data.get("root_nodes", [])],
        )

    def _node_to_dict(self, node: TreeNode) -> dict:
        return {
            "node_id": node.node_id,
            "title": node.title,
            "text": node.text,
            "level": node.level,
            "summary": node.summary,
            "children": [self._node_to_dict(c) for c in node.children],
        }

    def _dict_to_node(self, data: dict) -> TreeNode:
        return TreeNode(
            node_id=data["node_id"],
            title=data["title"],
            text=data.get("text", ""),
            level=data.get("level", 1),
            summary=data.get("summary", ""),
            children=[self._dict_to_node(c) for c in data.get("children", [])],
        )

    # -- Utilities ----------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a name to a filesystem-safe slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-") or "unnamed"

    @staticmethod
    def _count_nodes(nodes: list[TreeNode]) -> int:
        count = len(nodes)
        for node in nodes:
            count += KnowledgeStore._count_nodes(node.children)
        return count

    @staticmethod
    def _count_tokens_in_tree(nodes: list[TreeNode]) -> int:
        total = 0
        for node in nodes:
            total += len(node.text.split())
            total += KnowledgeStore._count_tokens_in_tree(node.children)
        return total
