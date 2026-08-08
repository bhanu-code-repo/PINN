"""CLI entry point for the PINN Admin server."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn
from pyfiglet import figlet_format
from rich.console import Console
from rich.table import Table

from .config import Settings
from .users import UserManager

console = Console()
app = typer.Typer(help="PINN Knowledge Admin — server & user management")


def _get_user_manager(db: Path | None = None) -> UserManager:
    """Create a UserManager from an explicit path or the default settings."""
    path = db or Settings().users_db
    return UserManager(path)


def _print_banner(settings: Settings) -> None:
    """Print startup banner with server info."""
    banner = figlet_format("PINN Admin", font="slant")
    console.print(f"[bold cyan]{banner}[/bold cyan]", highlight=False)
    console.print("[bold]Knowledge Base Administration Server[/bold]\n")
    console.print(f"  [dim]Store:[/dim]       {settings.knowledge_store_dir}")
    console.print(f"  [dim]Collections:[/dim] {settings.collections_db}")
    console.print(f"  [dim]Registry:[/dim]    {settings.registry_db}")
    console.print(f"  [dim]Users DB:[/dim]    {settings.users_db}")
    console.print(f"  [dim]Server:[/dim]      http://{settings.host}:{settings.port}")
    console.print()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the PINN Knowledge Admin server."""
    settings = Settings(host=host, port=port, debug=reload)
    _print_banner(settings)
    uvicorn.run(
        "api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


@app.command("create-user")
def create_user(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Option(..., prompt=True, hide_input=True, confirmation_prompt=True, help="Password"),
    admin: bool = typer.Option(False, "--admin", help="Grant admin privileges"),
    groups: str = typer.Option("", help="Comma-separated group list"),
) -> None:
    """Create a new user with bcrypt-hashed password."""
    group_list = [g.strip() for g in groups.split(",") if g.strip()] if groups else []
    with _get_user_manager() as mgr:
        user = mgr.create_user(username, password, groups=group_list, is_admin=admin)
    console.print(f"[green]✓[/green] Created user [bold]{user.username}[/bold] (admin={user.is_admin}, groups={user.groups})")


@app.command("list-users")
def list_users() -> None:
    """List all registered users."""
    with _get_user_manager() as mgr:
        users = mgr.list_users()

    if not users:
        console.print("[dim]No users found.[/dim]")
        return

    table = Table(title="Registered Users")
    table.add_column("Username", style="bold")
    table.add_column("Admin")
    table.add_column("Groups")
    table.add_column("Created")

    for u in users:
        table.add_row(
            u.username,
            "✓" if u.is_admin else "",
            ", ".join(u.groups) or "—",
            u.created_at[:19],
        )
    console.print(table)


@app.command("reset-password")
def reset_password(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Option(..., prompt=True, hide_input=True, confirmation_prompt=True, help="New password"),
) -> None:
    """Reset a user's password."""
    with _get_user_manager() as mgr:
        mgr.update_password(username, password)
    console.print(f"[green]✓[/green] Password updated for [bold]{username}[/bold]")


@app.command("delete-user")
def delete_user(
    username: str = typer.Argument(..., help="Username to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a user."""
    if not force:
        typer.confirm(f"Delete user '{username}'?", abort=True)
    with _get_user_manager() as mgr:
        mgr.delete_user(username)
    console.print(f"[green]✓[/green] Deleted user [bold]{username}[/bold]")


def main():
    """Entry point for the pinn-admin CLI."""
    app()


if __name__ == "__main__":
    main()
