# Adding a New Experiment

Step-by-step guide for adding a new ODE/PDE experiment to the PINN monorepo.
Every experiment follows the same conventions: a Typer CLI with `train`,
`predict`, and `compare` commands, self-describing checkpoints, structured
artifacts, and a lifecycle test.

---

## 1. Create the Experiment Directory

```
experiments/<name>/
├── __init__.py    # empty
├── train.py       # CLI + solver + model + losses
└── README.md      # methodology, usage, CLI reference, output, caveats
```

Choose a short, descriptive name: `harmonic_oscillator`, `burgers`, `taylor_green`,
`cylinder_wake`, etc.

## 2. Write `train.py`

Every experiment `train.py` follows this structure:

```python
#!/usr/bin/env python3
"""One-line description of the experiment."""

from pathlib import Path
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import typer
from loguru import logger
from pinn import PINN, PINNTrainer

from experiments.common import (
    compare_runs,
    find_latest_run,
    get_device,
    init_run,
    load_model,
    print_summary,
    save_metrics,
    show_banner,
)

app = typer.Typer(help="Short description for --help.")

EXPERIMENT = "<name>"  # must match the directory name


# ── Model ──────────────────────────────────────────────────────

class MyModel(nn.Module):
    """Wrapper around PINN backbone with problem-specific structure."""

    def __init__(self, hidden_layers: int, hidden_neurons: int):
        super().__init__()
        self.network = PINN(
            input_dim=...,          # number of independent variables
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            output_dim=...,         # number of solution components
        )

    def forward(self, x, t, ...):
        out = self.network(torch.cat([x, t, ...], dim=1))
        return out[:, 0:1], out[:, 1:2], ...


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model from a run config (self-describing checkpoints)."""
    return MyModel(
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
    )


# ── Losses ─────────────────────────────────────────────────────

def build_losses(n_physics: int, device: torch.device) -> dict:
    """Create named loss functions (closures own their collocation points).

    Returns a dict like {"ic": ..., "bc": ..., "physics": ...}.
    Each value is a callable: fn(model) -> scalar tensor.
    """
    # 1. Create collocation points ONCE (closed over by the loss functions)
    x_physics = torch.rand(n_physics, 1, device=device, requires_grad=True)
    # ... more points for IC, BC ...

    def physics_loss(model):
        # Compute PDE residual using torch.autograd.grad
        u = model(x_physics, ...)
        u_x = autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]
        # ... build the residual ...
        residual = ...
        return torch.mean(residual ** 2)

    def ic_loss(model):
        # Match initial condition
        ...

    return {"ic": ic_loss, "physics": physics_loss}


# ── Evaluation ─────────────────────────────────────────────────

def evaluate(model, device) -> tuple[dict, dict]:
    """Evaluate on a test grid. Return (metrics, arrays)."""
    # Compare against exact solution or benchmark data
    ...
    metrics = {"rel_l2_error": ..., "some_other_metric": ...}
    arrays = {"x": ..., "u_pred": ..., "u_exact": ...}
    return metrics, arrays


# ── Solver ─────────────────────────────────────────────────────

def solve_my_problem(
    epochs=30000, lr=1e-3, hidden_neurons=64, hidden_layers=4,
    n_physics=10000, seed=42, output_dir=None, show=True,
) -> dict:
    """Train, evaluate, and persist a run."""
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_physics": n_physics, "seed": seed,
    }
    logger.info("Config: {}", config)

    model = build_model(config)
    loss_functions = build_losses(n_physics, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)

    # Train with best-model saving
    trainer.train(
        n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions,
        save_best=run_dir / "best_model.pt",
    )
    # Save final checkpoint (self-describing: config stored in metadata)
    trainer.save_checkpoint(run_dir / "checkpoint.pt", optimizer=optimizer, metadata=config)
    trainer.plot_loss_history(show_total=True, save_path=run_dir / "loss_history.png", show=show)

    # Evaluate and persist metrics
    metrics, arrays = evaluate(model, device)
    final = trainer.loss_history[-1]
    metrics.update({
        "final_total_loss": final["total"],
        "epochs_run": len(trainer.loss_history),
    })
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    # Plots + summary table
    # make_plots(arrays, ..., show)
    print_summary("Training Summary", {
        "Final Loss": f"{metrics['final_total_loss']:.4e}",
        "Artifacts": str(run_dir),
    })
    return metrics


# ── CLI ────────────────────────────────────────────────────────

@app.command()
def train(
    epochs: int = typer.Option(30000, "--epochs", "-e"),
    lr: float = typer.Option(1e-3, "--lr"),
    neurons: int = typer.Option(64, "--neurons", "-n"),
    layers: int = typer.Option(4, "--layers", "-l"),
    n_physics: int = typer.Option(10000, "--n-physics"),
    seed: int = typer.Option(42, "--seed"),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o"),
    show: bool = typer.Option(True, "--show/--no-show"),
):
    """Train the PINN."""
    show_banner("MY PROBLEM", "Description")
    solve_my_problem(
        epochs=epochs, lr=lr, hidden_neurons=neurons, hidden_layers=layers,
        n_physics=n_physics, seed=seed, output_dir=output_dir, show=show,
    )


@app.command()
def predict(
    run: str | None = typer.Option(None, "--run", "-r"),
    show: bool = typer.Option(True, "--show/--no-show"),
):
    """Load a trained model and evaluate."""
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()
    model, config = load_model(run_dir, build_model, device)

    metrics, arrays = evaluate(model, device)
    np.savez(run_dir / "predictions.npz", ...)
    # make_plots(...)
    print_summary("Prediction Summary", {"Run": str(run_dir), ...})


@app.command()
def compare():
    """Rank all runs by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
```

### Key conventions

- **`build_model(config)`** — a factory that reconstructs the model from a config dict.
  Shared by `train` and `predict` via `load_model(run_dir, build_model)`. This makes
  checkpoints self-describing: no hyperparameters need to be remembered.

- **`build_losses(...)`** — returns a `dict[str, Callable[[nn.Module], Tensor]]`. Each
  callable receives the model and returns a scalar loss. Collocation points are created
  once and closed over — the trainer never sees the physics.

- **`save_best=run_dir / "best_model.pt"`** — saves the best model weights during
  training and restores them at the end. The final `checkpoint.pt` (saved via
  `trainer.save_checkpoint`) then contains the best weights.

- **`init_run(EXPERIMENT, output_dir, seed)`** — creates the run directory, sets up
  logging (console + file), seeds everything, returns `(run_dir, device)`.

- **`save_metrics({"config": ..., "metrics": ...}, run_dir)`** — writes `metrics.json`
  used by `compare_runs()` to rank runs.

## 3. Register the Console Script

In the root `pyproject.toml`, add an entry under `[project.scripts]`:

```toml
[project.scripts]
train-<name> = "experiments.<name>.train:app"
```

Then re-sync: `uv sync --all-packages`.

## 4. Add a CLI Lifecycle Test

In `tests/test_experiments_cli.py`:

```python
from experiments.<name>.train import app as my_app

def test_my_experiment_lifecycle(tmp_path, monkeypatch):
    """My experiment: train -> predict -> compare."""
    run_dir = tmp_path / "<name>" / "run1"

    invoke(my_app, [
        "train", "-e", "3", "--n-physics", "200",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    for artifact in TRAIN_ARTIFACTS:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    invoke(my_app, ["predict", "--run", str(run_dir), "--no-show"])
    assert (run_dir / "predictions.npz").exists()

    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(my_app, ["compare"])
    assert "No runs" not in result.output
```

Run: `uv run pytest tests/test_experiments_cli.py::test_my_experiment_lifecycle -v`

## 5. Write `README.md`

Follow the structure used by every experiment README:

1. **Title + one-line summary**
2. **Problem statement** — the equation, domain, BCs
3. **Method** — model architecture, losses, any special techniques (ansatz, hard BCs,
   normalisation, learnable parameters)
4. **Usage** — commands + CLI reference tables for `train` and `predict`
5. **Output** — artifact listing + what to look for
6. **Caveats** — honest limitations
7. **References**

## 6. Update the Root README

Add the experiment to:

- The directory tree in "Repository Structure"
- The experiments table (Experiment | Equation | Entry point | Docs)

## 7. Verify Everything

```bash
uv run ruff check .                    # lint passes
uv run pytest -v                       # all tests pass
uv run train-<name> train -e 100 --no-show   # smoke run
uv run train-<name> predict --no-show  # loads checkpoint, evaluates
uv run train-<name> compare            # ranks the run
```

---

## Shared Infrastructure Reference

All of these live in `experiments/common.py`:

| Function                                                    | Purpose                                                    |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| `show_banner(text, subtitle)`                               | Startup banner (pyfiglet + rich)                           |
| `get_device()`                                              | CUDA if available, else CPU                                |
| `init_run(experiment, output_dir, seed)`                    | Create run dir, logging, seed; returns `(run_dir, device)` |
| `save_metrics(data, run_dir)`                               | Write `metrics.json`                                       |
| `print_summary(title, rows)`                                | Rich table of metric name → value                          |
| `find_latest_run(experiment)`                               | Newest timestamped run dir with a checkpoint               |
| `load_model(run_dir, build_model, device, checkpoint_name)` | Rebuild + load model from checkpoint                       |
| `compare_runs(experiment, sort_key)`                        | Ranked table of all runs' metrics                          |

## Core Library Reference

The `pinn` package (`libs/pinn`) provides:

| Class/Function                                               | Purpose                                                              |
| ------------------------------------------------------------ | -------------------------------------------------------------------- |
| `PINN(input_dim, hidden_layers, hidden_neurons, output_dim)` | `tanh` MLP backbone                                                  |
| `PINNTrainer(model, device)`                                 | Multi-loss trainer with best-model saving, early stopping, callbacks |
| `set_seed(seed)`                                             | Seed random/numpy/torch/CUDA                                         |
| `setup_logging(log_dir, level)`                              | loguru console + file sinks                                          |
| `plot_contour(X, Y, Z, ...)`                                 | 2D contour plot                                                      |
| `plot_comparison_1d(x, y_true, y_pred, ...)`                 | 1D comparison plot                                                   |
| `plot_loss_comparison(histories, ...)`                       | Multi-run loss comparison                                            |

### `PINNTrainer.train()` key parameters

```python
trainer.train(
    n_epochs=30000,
    optimizer=optimizer,
    loss_functions={"ic": ic_loss, "physics": physics_loss},
    weights={"ic": 1.0, "physics": 1.0},       # per-term weights (default 1.0)
    save_best=run_dir / "best_model.pt",        # save best model during training
    restore_best=True,                           # restore best weights at end
    early_stop_patience=5000,                    # stop if no improvement for N epochs
    grad_clip=1.0,                               # clip gradient norm
    callbacks=[my_callback],                     # fn(epoch, losses) per epoch
)
```

Both standard (Adam, SGD) and closure-based (L-BFGS) optimizers are supported.
L-BFGS is detected automatically and uses the `optimizer.step(closure)` pattern.

### Two-stage training (Adam → L-BFGS)

For inverse problems and hard-to-converge forward problems, a two-stage approach
works well: Adam for initial convergence (robust, handles noisy gradients), then
L-BFGS for refinement (quasi-Newton, better on smooth landscapes):

```python
# Stage 1: Adam
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer.train(n_epochs=20000, optimizer=optimizer_adam, ...)

# Stage 2: L-BFGS refinement
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0,
    max_iter=50, max_eval=50,
    history_size=50, line_search_fn="strong_wolfe",
)
trainer.train(n_epochs=1000, optimizer=optimizer_lbfgs, ...)
```

The trainer's loss history accumulates across both stages. See
`experiments/cylinder_wake/` and `experiments/navier_stokes_inverse/` for
production examples with `--lbfgs-epochs` CLI flags.

## Design Patterns Worth Knowing

### Spectral bias defeat (Ansatz)

If your solution oscillates at known frequencies, embed them analytically:

```python
class Ansatz(nn.Module):
    def forward(self, t, w0):
        envelopes = self.backbone(t)
        return envelopes[:, 0:1] * cos(w0 * t) + envelopes[:, 1:2] * sin(w0 * t)
```

The backbone only learns slowly-varying envelopes. See `experiments/harmonic_oscillator`.

### Hard boundary conditions

Encode BCs in the model so they're satisfied by construction:

```python
mask = x * (1 - x) * y   # vanishes at x=0, x=1, y=0
u = mask * nn_output + boundary_value
```

See `experiments/lid_driven_cavity`.

### Streamfunction for incompressible NS

Output `(ψ, p)`, derive `u = ψ_y, v = -ψ_x` via autograd. Continuity is exact.
See `experiments/cylinder_wake`.

### Parametric PINNs + deep ensembles

Parameters become network inputs; ensemble members train independently from different
seeds. See `experiments/parametric_harmonic` and [docs/parametric_pinns.md](parametric_pinns.md).

### Learnable PDE parameters (inverse problems)

Add `nn.Parameter` scalars to the model, use them in the physics loss, add a data loss
to anchor the solution. See `experiments/navier_stokes_inverse` and `experiments/cylinder_wake`.

### Residual normalisation

If a parameter multiplies the dominant term (e.g. `w0²` in the harmonic ODE), divide the
residual by it so the loss is O(1) across the parameter range. See Rule 2 in
[docs/parametric_pinns.md](parametric_pinns.md).
