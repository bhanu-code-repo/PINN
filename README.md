# PINN — Physics-Informed Neural Networks

A [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo for solving differential equations with **Physics-Informed Neural Networks** (PINNs) in PyTorch. A shared core library provides the network backbone and training loop; each experiment defines a specific ODE/PDE problem on top of it.

## What is a PINN?

A PINN approximates the solution `u` of a differential equation with a neural network. Instead of training on labelled data, the loss penalises:

1. **Physics residual** — the equation itself, evaluated at collocation points using exact derivatives from automatic differentiation.
2. **Initial / boundary conditions** — mismatch at the domain boundary.

Minimising the weighted sum drives the network toward a function that satisfies the equation everywhere.

## Repository Structure

```
PINN/
├── libs/
│   └── pinn/                     # Core library (workspace member) — see libs/pinn/README.md
│       └── src/pinn/
│           ├── core/             # PINN MLP backbone
│           ├── trainer/          # Generic multi-loss trainer
│           └── utils/            # Plotting helpers
├── experiments/
│   ├── harmonic_oscillator/      # Damped harmonic oscillator ODE — see its README.md
│   ├── burgers/                  # Burgers' equation
│   └── schrodinger/              # Schrödinger equation
├── notebooks/                    # Guided walkthrough notebooks (theory + analysis)
│   ├── 01_harmonic_analysis.ipynb    # Full deep-dive: PINN cost function & solving loop
│   ├── 02_burgers_analysis.ipynb     # Nonlinear PDE, shock formation
│   ├── 03_schrodinger_analysis.ipynb # Complex fields, periodic BCs
│   └── 04_model_as_solution.ipynb    # Prediction: derivatives, residual check, extrapolation
├── docs/
│   └── prediction.md             # Concept: how prediction works in a PINN
├── pyproject.toml                # Workspace root + console scripts
└── uv.lock
```

## Getting Started

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)

### Install

```bash
git clone <repo-url> && cd PINN
uv sync --all-packages
```

This installs the `pinn` library (editable, from `libs/pinn`), all experiment dependencies (`torch`, `numpy`, `matplotlib`, `tqdm`, `typer`, `rich`), and the console scripts.

### Run an experiment

```bash
uv run train-harmonic --help             # CLI reference (train / predict / compare)
uv run train-harmonic train              # train with defaults
uv run train-harmonic predict            # evaluate the latest trained model (no retraining)
uv run train-harmonic compare            # rank all runs by their recorded metrics
```

## Experiments

| Experiment | Equation | Entry point | Docs |
|------------|----------|-------------|------|
| Harmonic oscillator | `u'' + μu' + ku = 0` | `uv run train-harmonic train` | [README](experiments/harmonic_oscillator/README.md) |
| Burgers | `u_t + u·u_x = ν·u_xx` | `uv run train-burgers train` | [README](experiments/burgers/README.md) |
| Schrödinger | `i·h_t + ½·h_xx + |h|²·h = 0` | `uv run train-schrodinger train` | [README](experiments/schrodinger/README.md) |

Each CLI also provides `predict` (re-evaluate a saved model — defaults to the latest run;
checkpoints are self-describing, so the architecture is rebuilt automatically from the stored
config) and `compare` (rank all runs of an experiment by their `metrics.json`).

For what "prediction" actually means for a PINN — the model *is* the solution function — see
[docs/prediction.md](docs/prediction.md) and the hands-on demonstration in
[notebooks/04_model_as_solution.ipynb](notebooks/04_model_as_solution.ipynb) (mesh-free
evaluation, autograd derivatives, residual self-check, extrapolation failure mode).

## Adding a New Experiment

1. Create `experiments/<name>/` with `__init__.py` and a `train.py` exposing a Typer `app`.
2. Define the problem: collocation points, residual via `torch.autograd.grad`, IC/BC losses.
3. Build on the shared library and experiment infrastructure:
   ```python
   from pinn import PINN, PINNTrainer
   from experiments.common import init_run, print_summary, save_metrics, show_banner
   ```
4. Register a console script in the root `pyproject.toml`:
   ```toml
   [project.scripts]
   train-<name> = "experiments.<name>.train:app"
   ```
5. Re-sync: `uv sync --all-packages`.

Every experiment run writes a self-contained artifact directory under `outputs/`
(checkpoint, `metrics.json`, plots, loguru logs) — see any experiment README for details.

## Core Library

The `pinn` package (in `libs/pinn`) is documented in [libs/pinn/README.md](libs/pinn/README.md) — including a "solve your own equation in 5 steps" quickstart and a scaling guide. Highlights:

- `PINN` — configurable `tanh` MLP for smooth higher-order derivatives
- `PINNTrainer` — named multi-term losses with per-term weights, early stopping, gradient clipping, per-epoch callbacks, checkpoint save/load, full loss history
- `set_seed` / `setup_logging` — reproducibility and loguru console+file logging
- `utils.plotting` — contour, 1D-comparison, and loss-comparison plots (headless-safe)

## Testing

```bash
uv run pytest              # fast suite: unit + CLI smoke tests (~5s)
uv run pytest -m slow      # convergence regression tests (train real PINNs)
uv run ruff check .        # lint
```

Layout:

- `libs/pinn/tests/` — library unit tests: network shapes/gradients, trainer mechanics
  (weighted losses, early stopping, grad clipping, callbacks), checkpoint round-trip,
  seeding, headless plotting
- `tests/test_experiments_cli.py` — full `train → predict → compare` lifecycle per experiment
  via Typer's in-process `CliRunner`, asserting every artifact is written
- `tests/test_convergence.py` — marked `slow`: solves `u' = -u` against the exact solution
  (rel-L2 < 5%) and checks the harmonic Ansatz pipeline drops its loss by 100×

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
