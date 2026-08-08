"""CLI entry point for the PINN Admin server."""

from __future__ import annotations

import uvicorn
from pyfiglet import figlet_format
from rich.console import Console

from .config import Settings

console = Console()


def _print_banner(settings: Settings) -> None:
    """Print startup banner with server info."""
    banner = figlet_format("PINN Admin", font="slant")
    console.print(f"[bold cyan]{banner}[/bold cyan]", highlight=False)
    console.print("[bold]Knowledge Base Administration Server[/bold]\n")
    console.print(f"  [dim]Store:[/dim]       {settings.knowledge_store_dir}")
    console.print(f"  [dim]Collections:[/dim] {settings.collections_db}")
    console.print(f"  [dim]Registry:[/dim]    {settings.registry_db}")
    console.print(f"  [dim]Server:[/dim]      http://{settings.host}:{settings.port}")
    console.print(f"  [dim]Login:[/dim]       {settings.admin_username} / ****")
    console.print()


def main():
    """Start the PINN Knowledge Admin server."""
    settings = Settings()
    _print_banner(settings)
    uvicorn.run(
        "api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
