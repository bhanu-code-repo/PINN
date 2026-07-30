# 1D Burgers' Equation PINN

A Physics-Informed Neural Network (PINN) that solves the **viscous Burgers' equation** — the canonical benchmark from Raissi et al. (2019) — capturing the steep shock that forms at `x = 0` without any labelled solution data.

---

## 1. Problem Statement

We solve the 1D viscous Burgers' equation on `x ∈ [−1, 1]`, `t ∈ [0, 1]`:

```
u_t + u·u_x − ν·u_xx = 0
```

with initial and boundary conditions:

```
u(0, x)  = −sin(π·x)          (initial condition)
u(t, −1) = u(t, 1) = 0        (homogeneous Dirichlet BCs)
```

| Symbol | Meaning | Default |
|--------|---------|---------|
| `ν` (`--nu`) | Viscosity coefficient | `0.01/π ≈ 0.00318` |

### Why this problem is hard

Burgers' equation is a nonlinear conservation law. The convective term `u·u_x` steepens the initial sine wave over time, and at the low default viscosity a **near-discontinuous shock** forms at `x = 0` by `t ≈ 0.4`. Classical numerical schemes need shock-capturing machinery (upwinding, limiters); mesh-free PINNs handle it via the smoothing `ν·u_xx` term and dense random collocation.

---

## 2. Method

### Architecture

```
PINN: Linear(2 → 50) → Tanh
      → [Linear(50 → 50) → Tanh] × 4
      → Linear(50 → 1)
```

Input is `(x, t)`; output is `u(t, x)`. Uses `pinn.core.network.PINN` directly (no ansatz needed).

### Loss formulation

| Term | Definition | Collocation points | Weight |
|------|-----------|--------------------|--------|
| `ic` | `mean[(u(0, x) + sin(πx))²]` | 100 uniform points on `x ∈ [−1, 1]` | `1.0` |
| `bc` | `mean[u(t, −1)² + u(t, 1)²]` | 50 uniform points on `t ∈ [0, 1]` | `1.0` |
| `physics` | `mean[(u_t + u·u_x − ν·u_xx)²]` | 5000 uniform-random points in the interior | `1.0` |

Derivatives (`u_t`, `u_x`, `u_xx`) are computed via `torch.autograd.grad` with `create_graph=True`.

### Optimisation

- Optimiser: Adam, `lr = 1e-3`
- Epochs: `30000` (full-batch)
- Trainer: `pinn.trainer.PINNTrainer` (live loss plot, tqdm, loss history)
- Device: CUDA if available, else CPU

---

## 3. Usage

### Quick start

From the repository root:

```bash
uv sync --all-packages
uv run train-burgers train           # train with defaults
```

### CLI reference

```
uv run train-burgers train [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `30000` | Number of training epochs |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `50` | Neurons per hidden layer |
| `--layers` | `-l` | int | `5` | Number of hidden layers |
| `--nu` | | float | `0.01/π` | Viscosity coefficient |
| `--seed` | | int | `42` | Random seed for reproducibility |
| `--output-dir` | `-o` | str | auto | Artifact directory (default: `outputs/burgers/<timestamp>`) |
| `--show/--no-show` | | flag | `--show` | Display plots interactively (`--no-show` for headless runs) |

### Examples

```bash
# Quick smoke test with a smoother (more viscous) solution
uv run train-burgers train -e 5000 --nu 0.1

# Full run, custom artifact directory, headless
uv run train-burgers train -o results/burgers --no-show
```

### Working with trained models

The CLI is multi-command — `train`, `predict`, and `compare`:

```bash
# Re-evaluate a trained model without retraining (defaults to the LATEST run)
uv run train-burgers predict
uv run train-burgers predict --run outputs/burgers/<timestamp> --no-show
# -> writes predictions.npz + prediction_contour.png + prediction_snapshots.png into the run directory

# Rank all runs by final loss (reads each run's metrics.json)
uv run train-burgers compare
```

Checkpoints are **self-describing**: the run config is stored inside
`checkpoint.pt`, so `predict` rebuilds the exact architecture automatically —
no hyperparameters to remember. Programmatic use:

```python
from experiments.common import find_latest_run, load_model
from experiments.burgers.train import build_model

model, config = load_model(find_latest_run("burgers"), build_model)
```

> **What does "prediction" mean for a PINN?** The trained model *is* the solution function —
> see [docs/prediction.md](../../docs/prediction.md) for the concept and
> [notebooks/04_model_as_solution.ipynb](../../notebooks/04_model_as_solution.ipynb) for a
> hands-on demonstration (derivatives via autograd, residual self-check, extrapolation limits).

---

## 4. Output

Every run writes a **self-contained artifact directory** (default
`outputs/burgers/<timestamp>`, override with `-o`):

```
<run-dir>/
├── checkpoint.pt           # model + optimizer state, loss history, run config
├── metrics.json            # config + final losses + rel-L2 error at t=0
├── loss_history.png        # ic / bc / physics curves, log scale
├── solution_contour.png    # u(t,x) over the full space-time domain
├── snapshots.png           # t=0 (vs exact IC) and t=1 (shock) profiles
└── logs/run_*.log          # full DEBUG-level training log (loguru)
```

What to look for:

1. **Contour plot** — the shock appears as a sharp colour transition along `x = 0` for `t ≳ 0.4`.
2. **Snapshots** — `t = 0` must match `−sin(πx)`; `t = 1` shows the fully-formed steep shock.
3. **Summary table** — final total/ic/bc/physics losses, rel-L2 error at `t=0`, epochs run.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Shock is smeared/rounded | Under-trained, or too few collocation points near the shock | More epochs; increase interior points in `train.py` (currently 5000), or sample more densely near `x = 0` |
| `t = 0` snapshot doesn't match `−sin(πx)` | Physics loss dominating the IC early | Raise the `ic` weight in `solve_burgers_equation` |
| Loss oscillates late in training | Adam lr too high for the sharpening solution | Lower `--lr` (e.g. `5e-4`), or add `grad_clip` in the `trainer.train` call |
| Very low `--nu` diverges | Shock too sharp for the network capacity | Increase `-n`/`-l`, or keep `ν ≥ 0.01/π` |
| No plot window appears | Headless environment | Run with `--no-show` — all plots are saved to the run directory anyway |

---

## 6. File Layout

```
experiments/burgers/
├── README.md      # this file
├── __init__.py
└── train.py       # Typer CLI, residual, losses, training, plots
```

Key entry points in `train.py`:

- `train(...)` — Typer command, exposed as the `train-burgers` console script
- `solve_burgers_equation(...)` — programmatic API (importable from other code)

---

## 7. References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378 — Burgers' setup (Sec. 3.1) with the same `ν = 0.01/π` benchmark.
- Basdevant et al. (1986). *Spectral and finite difference solutions of the Burgers equation.* Computers & Fluids 14.
