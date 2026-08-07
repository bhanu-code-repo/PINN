"""SQLite file registry for deduplication and ingestion tracking.

Tracks which files have been processed, their content hashes, and
conversion metadata. Prevents re-indexing unchanged files and records
which pages used LLM vs pymupdf4llm for cost tracking.

Schema is intentionally simple — easy to migrate to PostgreSQL later
by swapping the connection string.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS files (
    file_hash   TEXT PRIMARY KEY,
    file_name   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    page_count  INTEGER,
    doc_id      TEXT,
    converter   TEXT NOT NULL DEFAULT 'text',
    llm_pages   TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_doc_id ON files(doc_id);
CREATE INDEX IF NOT EXISTS idx_files_path ON files(file_path);
"""


@dataclass
class FileRecord:
    """A registered file entry."""

    file_hash: str
    file_name: str
    file_path: str
    file_size: int
    page_count: int | None
    doc_id: str | None
    converter: str  # "text", "pymupdf4llm", "hybrid"
    llm_pages: str  # comma-separated page numbers that used LLM
    created_at: str
    updated_at: str


class FileRegistry:
    """SQLite-backed file registry for deduplication.

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
        self._conn.executescript(_SCHEMA)
        logger.debug("File registry opened at {}", self.db_path)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> FileRegistry:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Hashing -----------------------------------------------------------

    @staticmethod
    def compute_hash(path: str | Path) -> str:
        """Compute SHA-256 hash of a file's contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- Lookup ------------------------------------------------------------

    def lookup(self, file_hash: str) -> FileRecord | None:
        """Look up a file by its content hash."""
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if row is None:
            return None
        return FileRecord(**dict(row))

    def lookup_by_path(self, file_path: str | Path) -> FileRecord | None:
        """Look up a file by its path (most recent entry)."""
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_path = ? ORDER BY updated_at DESC LIMIT 1",
            (str(file_path),),
        ).fetchone()
        if row is None:
            return None
        return FileRecord(**dict(row))

    def is_indexed(self, path: str | Path) -> bool:
        """Check if a file has already been indexed (by content hash)."""
        file_hash = self.compute_hash(path)
        return self.lookup(file_hash) is not None

    # -- Registration ------------------------------------------------------

    def register(
        self,
        path: str | Path,
        *,
        doc_id: str | None = None,
        converter: str = "text",
        page_count: int | None = None,
        llm_pages: list[int] | None = None,
    ) -> FileRecord:
        """Register a file as processed.

        If the file hash already exists, updates the record.
        """
        path = Path(path)
        file_hash = self.compute_hash(path)
        now = datetime.now(tz=UTC).isoformat()
        llm_pages_str = ",".join(str(p) for p in (llm_pages or []))

        existing = self.lookup(file_hash)
        if existing is not None:
            self._conn.execute(
                "UPDATE files SET doc_id=?, converter=?, llm_pages=?, updated_at=? "
                "WHERE file_hash=?",
                (doc_id or existing.doc_id, converter, llm_pages_str, now, file_hash),
            )
            self._conn.commit()
            logger.info("Updated registry entry for '{}'", path.name)
        else:
            self._conn.execute(
                "INSERT INTO files (file_hash, file_name, file_path, file_size, "
                "page_count, doc_id, converter, llm_pages, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_hash,
                    path.name,
                    str(path),
                    path.stat().st_size,
                    page_count,
                    doc_id,
                    converter,
                    llm_pages_str,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            logger.info("Registered '{}' (hash={}…)", path.name, file_hash[:12])

        return self.lookup(file_hash)  # type: ignore[return-value]

    def unregister(self, file_hash: str) -> bool:
        """Remove a file from the registry. Returns True if it existed."""
        cursor = self._conn.execute(
            "DELETE FROM files WHERE file_hash = ?", (file_hash,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # -- Listing -----------------------------------------------------------

    def list_all(self) -> list[FileRecord]:
        """Return all registered files."""
        rows = self._conn.execute(
            "SELECT * FROM files ORDER BY updated_at DESC"
        ).fetchall()
        return [FileRecord(**dict(r)) for r in rows]

    def count(self) -> int:
        """Return the number of registered files."""
        row = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()
        return row[0]
