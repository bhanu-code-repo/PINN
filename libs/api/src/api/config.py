"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root: PINN/
# __file__ = libs/api/src/api/config.py → parents[4] = PINN/
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class Settings:
    """Server configuration with sensible defaults."""

    # Knowledge base paths
    knowledge_store_dir: Path = _PROJECT_ROOT / "data" / "pinn-knowledge" / "store"
    knowledge_sources_dir: Path = _PROJECT_ROOT / "data" / "pinn-knowledge" / "sources"
    registry_db: Path = _PROJECT_ROOT / "data" / "pinn-knowledge" / "registry.db"
    collections_db: Path = _PROJECT_ROOT / "data" / "pinn-knowledge" / "collections.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Session secret (override in production)
    secret_key: str = "pinn-admin-dev-key-change-in-production"

    # User database
    users_db: Path = _PROJECT_ROOT / "data" / "pinn-knowledge" / "users.db"

    # Default admin credentials (used for initial admin bootstrap)
    admin_username: str = "admin"
    admin_password: str = "admin"

    # Default user groups
    default_groups: list[str] = field(
        default_factory=lambda: ["admin", "research", "engineering"]
    )
