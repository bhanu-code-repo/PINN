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
uv run train-burgers           # train with defaults
```

### CLI reference

```
uv run train-burgers [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `30000` | Number of training epochs |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `50` | Neurons per hidden layer |
| `--layers` | `-l` | int | `5` | Number of hidden layers |
| `--nu` | | float | `0.01/π` | Viscosity coefficient |
| `--save-plot` | | flag | off | Save the final plots |
| `--plot-path` | | str | `None` | Output path for plots (used with `--save-plot`) |

### Examples

```bash
# Quick smoke test with a smoother (more viscous) solution
uv run train-burgers -e 5000 --nu 0.1

# Full run, saving figures
uv run train-burgers --save-plot --plot-path results/burgers.png
# -> writes results/burgers.png and results/burgers_validation.png
```

---

## 4. Output

1. **Live loss plot** — `ic`, `bc`, `physics` curves on a log scale during training.
2. **Contour plot** — `u(t, x)` over the full space-time domain (200×200 grid). The shock appears as a sharp colour transition along `x = 0` for `t ≳ 0.4`.
3. **Validation snapshots**:
   - `t = 0`: PINN vs. exact `−sin(πx)` (checks the IC was learned)
   - `t = 1`: PINN profile showing the fully-formed steep shock
4. **Summary table** — final total/ic/bc/physics losses and epochs run.

With `--save-plot --plot-path <p>.png`, the validation figure is saved alongside as `<p>_validation.png`.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Shock is smeared/rounded | Under-trained, or too few collocation points near the shock | More epochs; increase interior points in `train.py` (currently 5000), or sample more densely near `x = 0` |
| `t = 0` snapshot doesn't match `−sin(πx)` | Physics loss dominating the IC early | Raise the `ic` weight in `solve_burgers_equation` |
| Loss oscillates late in training | Adam lr too high for the sharpening solution | Lower `--lr` (e.g. `5e-4`), or add `grad_clip` in the `trainer.train` call |
| Very low `--nu` diverges | Shock too sharp for the network capacity | Increase `-n`/`-l`, or keep `ν ≥ 0.01/π` |
| No plot window appears | Headless environment | `--save-plot --plot-path out.png`, or `MPLBACKEND=Agg` |

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
