# PINN Admin Server

FastAPI-based administration server for managing the PINN knowledge base, with
collection-based access control, user management, and a Bootstrap 5 dashboard.

---

## Quick Start

```bash
# Install (from repo root)
uv sync --all-packages

# Start the server (bootstraps default admin user on first run)
uv run pinn-admin serve

# Server runs at http://localhost:8000
# Default login: admin / admin
```

---

## User Management

Users are stored in SQLite with **bcrypt-hashed passwords**. Passwords are never
stored in plaintext. Each user has a username, group memberships, and an admin
flag.

### CLI Commands

All user management is done via the `pinn-admin` CLI:

```bash
# Create a user (prompts for password interactively)
uv run pinn-admin create-user alice --admin --groups "research,engineering"

# List all users
uv run pinn-admin list-users

# Reset a user's password
uv run pinn-admin reset-password alice

# Delete a user
uv run pinn-admin delete-user alice

# Delete without confirmation
uv run pinn-admin delete-user alice --force
```

### CLI Reference

| Command | Description |
|---------|-------------|
| `serve` | Start the admin server (`--host`, `--port`, `--reload`) |
| `create-user <username>` | Create user with bcrypt password (`--admin`, `--groups`) |
| `list-users` | Show all users in a Rich table |
| `reset-password <username>` | Reset password (interactive prompt) |
| `delete-user <username>` | Delete user (`--force` to skip confirmation) |

### Default Admin Bootstrap

On first startup, the server checks for admin users. If none exist, it creates
a default admin account:

- **Username**: `admin` (configurable via `Settings.admin_username`)
- **Password**: `admin` (configurable via `Settings.admin_password`)
- **Groups**: `admin`, `research`, `engineering`

**Change the default password immediately in production.**

### Password Security

- Passwords are hashed with **bcrypt** (work factor 12 by default)
- Each hash includes a unique random salt
- The `users` table stores only the bcrypt hash — never the plaintext
- Password verification uses constant-time comparison

---

## Architecture

```
libs/api/src/api/
├── __init__.py          Package init, exports create_app
├── app.py               FastAPI application factory
├── config.py            Settings dataclass (paths, server config)
├── cli.py               Typer CLI (serve, create-user, list-users, etc.)
├── deps.py              FastAPI dependency injection
├── users.py             UserManager — SQLite + bcrypt user accounts
├── routes/
│   ├── auth.py          Login/logout with session auth
│   ├── dashboard.py     Dashboard + settings page
│   ├── collections.py   Collection CRUD + document membership
│   └── documents.py     Document list/view/search/upload/delete
├── templates/           Jinja2 templates (Bootstrap 5 / Portal)
│   ├── layouts/         base.html, base-auth.html
│   ├── includes/        header, footer, flash, scripts, tree_node
│   └── pages/           dashboard, login, collections, documents, etc.
└── static/              CSS, JS, images (Bootstrap 5, FontAwesome)
```

---

## Authentication Flow

1. User visits any page → middleware checks session cookie
2. No session → redirect to `/login`
3. POST `/login` → `UserManager.authenticate(username, password)`
4. bcrypt verifies password → session created with username, groups, admin flag
5. Session cookie sent to browser (signed with `secret_key`)
6. Logout clears the session

---

## Access Control

Access control operates at the **collection** level:

| Access Level | Behavior |
|-------------|----------|
| **Public** | Visible to all authenticated users |
| **Restricted** | Visible only to users whose groups overlap with the collection's `allowed_groups` |

Users are assigned groups at creation time. Groups can be updated later:

```bash
uv run pinn-admin create-user bob --groups "research,data-science"
```

---

## Configuration

All settings are in `api/config.py` as a `Settings` dataclass:

| Setting | Default | Description |
|---------|---------|-------------|
| `knowledge_store_dir` | `data/pinn-knowledge/store` | Knowledge store path |
| `knowledge_sources_dir` | `data/pinn-knowledge/sources` | Source documents path |
| `registry_db` | `data/pinn-knowledge/registry.db` | File registry SQLite |
| `collections_db` | `data/pinn-knowledge/collections.db` | Collections SQLite |
| `users_db` | `data/pinn-knowledge/users.db` | User accounts SQLite |
| `host` | `0.0.0.0` | Server bind address |
| `port` | `8000` | Server bind port |
| `secret_key` | `pinn-admin-dev-key-...` | Session signing key |
| `admin_username` | `admin` | Default admin username |
| `admin_password` | `admin` | Default admin password |

**Production checklist:**
- Change `secret_key` to a random value
- Change default admin password after first login
- Consider restricting `host` to `127.0.0.1` if not behind a reverse proxy

---

## Database Schema

### Users (`users.db`)

```sql
CREATE TABLE users (
    username    TEXT PRIMARY KEY,
    password    TEXT NOT NULL,      -- bcrypt hash
    groups      TEXT NOT NULL DEFAULT '',  -- comma-separated
    is_admin    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,      -- ISO 8601
    updated_at  TEXT NOT NULL
);
```

### Collections (`collections.db`)

```sql
CREATE TABLE collections (
    id              TEXT PRIMARY KEY,   -- slug
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    access          TEXT DEFAULT 'public',
    allowed_groups  TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE collection_documents (
    collection_id   TEXT REFERENCES collections(id) ON DELETE CASCADE,
    doc_id          TEXT NOT NULL,
    added_at        TEXT NOT NULL,
    PRIMARY KEY (collection_id, doc_id)
);
```

---

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Login | `/login` | Username/password form |
| Dashboard | `/` | Stats cards, recent collections and documents |
| Collections | `/collections` | List, create, edit, delete collections |
| Collection Detail | `/collections/{id}` | View collection, manage document membership |
| Documents | `/documents` | Paginated list with search |
| Document Detail | `/documents/{id}` | Full document with tree structure |
| Upload | `/upload` | File upload (md/txt/pdf) with collection selector |
| Search | `/search?q=...` | BM25 keyword search |
| Settings | `/settings` | Server configuration overview |
