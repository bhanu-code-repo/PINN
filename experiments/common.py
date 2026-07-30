"""Shared infrastructure for all experiment CLIs.

Centralises what every experiment needs and previously duplicated:
banner, device selection, run initialisation (seed + logging + output
directory), metrics persistence, and the summary table.
"""

import json
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from pinn import set_seed, setup_logging
from pyfiglet import Figlet
from rich.console import Console
from rich.table import Table

console = Console()


def show_banner(text: str, subtitle: str) -> None:
    """Display a startup banner using pyfiglet and rich."""
    banner = Figlet(font="slant").renderText(text)
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print(f"[bold yellow]{subtitle}[/bold yellow]")
    console.print("=" * 50)


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_run(experiment: str, output_dir: str | None, seed: int) -> tuple[Path, torch.device]:
    """Initialise an experiment run: output dir, file logging, seed, device.

    Args:
        experiment: Experiment name, used in the default output path.
        output_dir: Explicit output directory, or ``None`` for the default
            ``outputs/<experiment>/<timestamp>``.
        seed: Seed forwarded to :func:`pinn.set_seed`.

    Returns:
        ``(run_dir, device)`` — the created run directory and target device.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("outputs") / experiment / timestamp
    else:
        run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=run_dir / "logs")
    set_seed(seed)
    device = get_device()

    logger.info("Run directory: {}", run_dir)
    logger.info("Seed: {} | Device: {}", seed, device)
    return run_dir, device


def save_metrics(metrics: dict, run_dir: Path) -> Path:
    """Write a metrics dict as pretty-printed JSON to ``run_dir/metrics.json``."""
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to {}", path)
    return path


def print_summary(title: str, rows: dict[str, str]) -> None:
    """Print a rich summary table of metric name -> value."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for name, value in rows.items():
        table.add_row(name, value)
    console.print(table)
