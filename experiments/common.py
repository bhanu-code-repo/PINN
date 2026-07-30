"""Shared infrastructure for all experiment CLIs.

Centralises what every experiment needs and previously duplicated:
banner, device selection, run initialisation (seed + logging + output
directory), metrics persistence, and the summary table.
"""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from pinn import set_seed, setup_logging
from pyfiglet import Figlet
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUTS_ROOT = Path("outputs")


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


def find_latest_run(experiment: str) -> Path:
    """Return the newest run directory for an experiment that has a checkpoint.

    Run directories are timestamped (``outputs/<experiment>/<YYYYmmdd-HHMMSS>``),
    so lexicographic order equals chronological order.

    Raises:
        FileNotFoundError: If no completed run exists for this experiment.
    """
    root = OUTPUTS_ROOT / experiment
    runs = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "checkpoint.pt").exists()
    ) if root.exists() else []
    if not runs:
        raise FileNotFoundError(
            f"No completed runs found under {root}/ — train a model first."
        )
    return runs[-1]


def load_model(
    run_dir: str | Path,
    build_model: Callable[[dict], nn.Module],
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """Rebuild and load a trained model from a run directory's checkpoint.

    The checkpoint's ``metadata`` (the run config saved at training time) is
    passed to ``build_model`` so the architecture is reconstructed exactly —
    checkpoints are self-describing; no hyperparameters need to be remembered.

    Args:
        run_dir: A run directory containing ``checkpoint.pt``.
        build_model: Experiment factory ``config -> nn.Module``.
        device: Target device (default: CUDA if available, else CPU).

    Returns:
        ``(model, config)`` — the model in eval mode on ``device``, and the
        training-time config dict.
    """
    if device is None:
        device = get_device()
    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["metadata"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logger.info("Loaded model from {} (config: {})", checkpoint_path, config)
    return model, config


def compare_runs(experiment: str, sort_key: str = "final_total_loss") -> None:
    """Print a ranked table of all runs of an experiment from their metrics.json.

    Args:
        experiment: Experiment name (subdirectory of ``outputs/``).
        sort_key: Metric to rank by, ascending (must exist in ``metrics.json``).
    """
    root = OUTPUTS_ROOT / experiment
    rows = []
    if root.exists():
        for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            data = json.loads(metrics_path.read_text())
            rows.append((run_dir.name, data.get("config", {}), data.get("metrics", {})))

    if not rows:
        console.print(f"[yellow]No runs with metrics found under {root}/[/yellow]")
        return

    rows.sort(key=lambda r: r[2].get(sort_key, float("inf")))

    # Union of metric names across runs, sort key first
    metric_names = sorted({name for _, _, m in rows for name in m} - {sort_key})
    metric_names = [sort_key, *metric_names]

    table = Table(title=f"Runs: {experiment} (best {sort_key} first)")
    table.add_column("Run", style="cyan")
    table.add_column("Seed", style="white")
    table.add_column("Epochs", style="white")
    for name in metric_names:
        table.add_column(name, style="magenta")

    for run_name, config, metrics in rows:
        cells = [run_name, str(config.get("seed", "?")), str(config.get("epochs", "?"))]
        for name in metric_names:
            value = metrics.get(name)
            cells.append(f"{value:.4e}" if isinstance(value, float) else str(value))
        table.add_row(*cells)
    console.print(table)
