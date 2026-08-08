"""SQLite-backed user management with bcrypt password hashing.

Stores users with hashed passwords, group memberships, and admin flags.
Passwords are hashed with bcrypt — never stored in plaintext.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import bcrypt as _bcrypt
from loguru import logger

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS users (
    username    TEXT PRIMARY KEY,
    password    TEXT NOT NULL,
    groups      TEXT NOT NULL DEFAULT '',
    is_admin    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


@dataclass
class User:
    """A registered user."""

    username: str
    groups: list[str]
    is_admin: bool
    created_at: str
    updated_at: str


class UserManager:
    """Manages user accounts with bcrypt-hashed passwords.

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

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> UserManager:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Password hashing ----------------------------------------------------

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password with bcrypt."""
        return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its bcrypt hash."""
        return _bcrypt.checkpw(password.encode(), hashed.encode())

    # -- CRUD ----------------------------------------------------------------

    def create_user(
        self,
        username: str,
        password: str,
        *,
        groups: list[str] | None = None,
        is_admin: bool = False,
    ) -> User:
        """Create a new user with a hashed password."""
        now = datetime.now(tz=UTC).isoformat()
        hashed = self.hash_password(password)
        groups_str = ",".join(groups or [])

        try:
            self._conn.execute(
                "INSERT INTO users (username, password, groups, is_admin, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username, hashed, groups_str, int(is_admin), now, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as e:
            msg = f"User '{username}' already exists"
            raise ValueError(msg) from e

        logger.info("Created user '{}' (admin={})", username, is_admin)
        return self.get_user(username)

    def get_user(self, username: str) -> User:
        """Get a user by username (without password hash)."""
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            msg = f"User '{username}' not found"
            raise KeyError(msg)
        return self._row_to_user(row)

    def authenticate(self, username: str, password: str) -> User | None:
        """Verify credentials. Returns User on success, None on failure."""
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return None
        if not self.verify_password(password, row["password"]):
            return None
        return self._row_to_user(row)

    def list_users(self) -> list[User]:
        """Return all users (without password hashes)."""
        rows = self._conn.execute(
            "SELECT * FROM users ORDER BY username"
        ).fetchall()
        return [self._row_to_user(r) for r in rows]

    def update_password(self, username: str, new_password: str) -> None:
        """Reset a user's password."""
        self.get_user(username)  # raises KeyError if not found
        now = datetime.now(tz=UTC).isoformat()
        hashed = self.hash_password(new_password)
        self._conn.execute(
            "UPDATE users SET password=?, updated_at=? WHERE username=?",
            (hashed, now, username),
        )
        self._conn.commit()
        logger.info("Password updated for '{}'", username)

    def update_user(
        self,
        username: str,
        *,
        groups: list[str] | None = None,
        is_admin: bool | None = None,
    ) -> User:
        """Update user groups and/or admin status."""
        existing = self.get_user(username)
        now = datetime.now(tz=UTC).isoformat()

        new_groups = ",".join(groups) if groups is not None else ",".join(existing.groups)
        new_admin = int(is_admin) if is_admin is not None else int(existing.is_admin)

        self._conn.execute(
            "UPDATE users SET groups=?, is_admin=?, updated_at=? WHERE username=?",
            (new_groups, new_admin, now, username),
        )
        self._conn.commit()
        logger.info("Updated user '{}'", username)
        return self.get_user(username)

    def delete_user(self, username: str) -> None:
        """Delete a user."""
        self.get_user(username)  # raises KeyError if not found
        self._conn.execute("DELETE FROM users WHERE username = ?", (username,))
        self._conn.commit()
        logger.info("Deleted user '{}'", username)

    def count(self) -> int:
        """Return the number of registered users."""
        row = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()
        return row[0]

    def ensure_admin(self, username: str = "admin", password: str = "admin") -> None:
        """Ensure at least one admin user exists. Creates default if none found."""
        admins = [u for u in self.list_users() if u.is_admin]
        if not admins:
            try:
                self.create_user(
                    username, password,
                    groups=["admin", "research", "engineering"],
                    is_admin=True,
                )
                logger.info("Created default admin user '{}'", username)
            except ValueError:
                # User exists but not admin — promote
                self.update_user(username, is_admin=True)
                logger.info("Promoted existing user '{}' to admin", username)

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        d = dict(row)
        groups_str = d.get("groups", "")
        groups = [g.strip() for g in groups_str.split(",") if g.strip()]
        return User(
            username=d["username"],
            groups=groups,
            is_admin=bool(d["is_admin"]),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )
