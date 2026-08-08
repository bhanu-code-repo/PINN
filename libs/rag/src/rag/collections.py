"""SQLite-backed collection management with access control.

Collections group documents and control who can access them.
Each collection has an access level (public or restricted) and
a list of allowed groups. Documents can belong to multiple collections.

Schema is intentionally simple — easy to migrate to PostgreSQL later
by swapping the connection string.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS collections (
    collection_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL DEFAULT '',
    access          TEXT NOT NULL DEFAULT 'public',
    allowed_groups  TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_documents (
    collection_id   TEXT NOT NULL,
    doc_id          TEXT NOT NULL,
    added_at        TEXT NOT NULL,
    PRIMARY KEY (collection_id, doc_id),
    FOREIGN KEY (collection_id) REFERENCES collections(collection_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cd_doc_id ON collection_documents(doc_id);
"""


@dataclass
class Collection:
    """A knowledge collection with access control."""

    collection_id: str
    name: str
    description: str
    access: str  # "public" or "restricted"
    allowed_groups: list[str]
    created_at: str
    updated_at: str
    doc_count: int = 0


class CollectionManager:
    """Manages collections and document-to-collection mappings.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        logger.debug("CollectionManager opened at {}", self.db_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CollectionManager:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Slugify ---------------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        import re

        slug = name.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-") or "unnamed"

    # -- Collection CRUD -------------------------------------------------------

    def create_collection(
        self,
        name: str,
        *,
        description: str = "",
        access: str = "public",
        allowed_groups: list[str] | None = None,
    ) -> Collection:
        """Create a new collection. Returns the created collection."""
        collection_id = self._slugify(name)
        now = datetime.now(tz=UTC).isoformat()
        groups_str = ",".join(allowed_groups or [])

        try:
            self._conn.execute(
                "INSERT INTO collections "
                "(collection_id, name, description, access, allowed_groups, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (collection_id, name, description, access, groups_str, now, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as e:
            msg = f"Collection '{name}' already exists"
            raise ValueError(msg) from e

        logger.info("Created collection '{}' (access={})", name, access)
        return self.get_collection(collection_id)

    def get_collection(self, collection_id: str) -> Collection:
        """Get a collection by ID."""
        row = self._conn.execute(
            "SELECT c.*, COUNT(cd.doc_id) as doc_count "
            "FROM collections c "
            "LEFT JOIN collection_documents cd ON c.collection_id = cd.collection_id "
            "WHERE c.collection_id = ? "
            "GROUP BY c.collection_id",
            (collection_id,),
        ).fetchone()
        if row is None:
            msg = f"Collection '{collection_id}' not found"
            raise KeyError(msg)
        return self._row_to_collection(row)

    def list_collections(
        self,
        *,
        user_groups: list[str] | None = None,
    ) -> list[Collection]:
        """List all collections, optionally filtered by user group access.

        If ``user_groups`` is None, returns all collections (admin view).
        Otherwise, returns public collections + restricted ones the user can access.
        """
        rows = self._conn.execute(
            "SELECT c.*, COUNT(cd.doc_id) as doc_count "
            "FROM collections c "
            "LEFT JOIN collection_documents cd ON c.collection_id = cd.collection_id "
            "GROUP BY c.collection_id "
            "ORDER BY c.name",
        ).fetchall()

        collections = [self._row_to_collection(r) for r in rows]

        if user_groups is None:
            return collections

        return [
            c
            for c in collections
            if c.access == "public"
            or any(g in c.allowed_groups for g in user_groups)
        ]

    def update_collection(
        self,
        collection_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        access: str | None = None,
        allowed_groups: list[str] | None = None,
    ) -> Collection:
        """Update collection fields. Only provided fields are changed."""
        existing = self.get_collection(collection_id)
        now = datetime.now(tz=UTC).isoformat()

        new_name = name if name is not None else existing.name
        new_desc = description if description is not None else existing.description
        new_access = access if access is not None else existing.access
        new_groups = (
            ",".join(allowed_groups)
            if allowed_groups is not None
            else ",".join(existing.allowed_groups)
        )

        self._conn.execute(
            "UPDATE collections SET name=?, description=?, access=?, "
            "allowed_groups=?, updated_at=? WHERE collection_id=?",
            (new_name, new_desc, new_access, new_groups, now, collection_id),
        )
        self._conn.commit()
        logger.info("Updated collection '{}'", collection_id)
        return self.get_collection(collection_id)

    def delete_collection(self, collection_id: str) -> None:
        """Delete a collection and its document mappings."""
        self.get_collection(collection_id)  # raises KeyError if not found
        self._conn.execute(
            "DELETE FROM collections WHERE collection_id = ?",
            (collection_id,),
        )
        self._conn.commit()
        logger.info("Deleted collection '{}'", collection_id)

    # -- Document-to-collection mapping ----------------------------------------

    def add_document_to_collection(
        self, collection_id: str, doc_id: str
    ) -> None:
        """Add a document to a collection."""
        now = datetime.now(tz=UTC).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO collection_documents (collection_id, doc_id, added_at) "
                "VALUES (?, ?, ?)",
                (collection_id, doc_id, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass  # already in collection, idempotent

    def remove_document_from_collection(
        self, collection_id: str, doc_id: str
    ) -> None:
        """Remove a document from a collection."""
        self._conn.execute(
            "DELETE FROM collection_documents WHERE collection_id = ? AND doc_id = ?",
            (collection_id, doc_id),
        )
        self._conn.commit()

    def get_collection_doc_ids(self, collection_id: str) -> list[str]:
        """Get all document IDs in a collection."""
        rows = self._conn.execute(
            "SELECT doc_id FROM collection_documents WHERE collection_id = ? "
            "ORDER BY added_at",
            (collection_id,),
        ).fetchall()
        return [r["doc_id"] for r in rows]

    def get_document_collections(self, doc_id: str) -> list[str]:
        """Get all collection IDs a document belongs to."""
        rows = self._conn.execute(
            "SELECT collection_id FROM collection_documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
        return [r["collection_id"] for r in rows]

    def get_accessible_doc_ids(self, user_groups: list[str]) -> set[str]:
        """Get all document IDs accessible to a user based on their groups.

        Returns doc IDs from public collections + restricted collections
        the user has access to.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT cd.doc_id "
            "FROM collection_documents cd "
            "JOIN collections c ON cd.collection_id = c.collection_id",
        ).fetchall()

        accessible = set()
        # Get all collections this user can access
        allowed_collections = {
            c.collection_id for c in self.list_collections(user_groups=user_groups)
        }
        for row in rows:
            doc_id = row["doc_id"]
            doc_collections = set(self.get_document_collections(doc_id))
            if doc_collections & allowed_collections:
                accessible.add(doc_id)

        return accessible

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _row_to_collection(row: sqlite3.Row) -> Collection:
        d = dict(row)
        groups_str = d.pop("allowed_groups", "")
        groups = [g.strip() for g in groups_str.split(",") if g.strip()]
        return Collection(
            collection_id=d["collection_id"],
            name=d["name"],
            description=d["description"],
            access=d["access"],
            allowed_groups=groups,
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            doc_count=d.get("doc_count", 0),
        )
