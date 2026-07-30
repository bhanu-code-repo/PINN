# 1D Nonlinear Schrödinger Equation PINN

A Physics-Informed Neural Network (PINN) that solves the **focusing nonlinear Schrödinger (NLS) equation** — a complex-valued PDE with periodic boundary conditions — by predicting real and imaginary parts with a two-output network.

---

## 1. Problem Statement

We solve the 1D nonlinear Schrödinger equation on `x ∈ [−5, 5]`, `t ∈ [0, π/2]`:

```
i·h_t + ½·h_xx + |h|²·h = 0
```

where `h(t, x)` is **complex-valued**, with:

```
h(0, x)      = 2·sech(x)                    (initial condition — bright soliton)
h(t, −5)     = h(t, 5)                      (periodic BC on the value)
h_x(t, −5)   = h_x(t, 5)                    (periodic BC on the derivative)
```

This equation models wave propagation in nonlinear media — optical fibres, deep-water waves, Bose–Einstein condensates. The `2·sech(x)` initial condition evolves as a breathing higher-order soliton whose magnitude peaks near `t = π/4`.

### Why this problem is hard

- **Complex-valued solution** — standard real networks can't output `h` directly.
- **Nonlinear term `|h|²·h`** — couples real and imaginary parts.
- **Periodic BCs on both value and derivative** — requires differentiating the network at both boundaries.

---

## 2. Method

### Handling complex values

The network outputs two channels interpreted as real and imaginary parts:

```
ComplexPINN(x, t) → (u, v),   h = u + i·v
```

```
PINN: Linear(2 → 100) → Tanh
      → [Linear(100 → 100) → Tanh] × 3
      → Linear(100 → 2)
```

Built on `pinn.core.network.PINN` with `output_dim=2`; a thin `ComplexPINN` wrapper (defined in `train.py`) splits the outputs.

The complex residual is assembled from real-part derivatives:

```
h_t  = u_t + i·v_t
h_xx = u_xx + i·v_xx
f    = i·h_t + ½·h_xx + (h·h̄)·h
loss = mean(|f|²)
```

### Loss formulation

| Term | Definition | Collocation points | Weight |
|------|-----------|--------------------|--------|
| `ic` | `mean[(u(0,x) − 2·sech(x))² + v(0,x)²]` | 100 uniform points on `x ∈ [−5, 5]` | `1.0` |
| `bc` | value + derivative mismatch between `x = −5` and `x = 5` | 50 uniform points on `t ∈ [0, π/2]` | `1.0` |
| `physics` | `mean[|i·h_t + ½·h_xx + |h|²·h|²]` | 5000 uniform-random interior points | `1.0` |

All derivatives via `torch.autograd.grad` with `create_graph=True` (including boundary derivatives `h_x(±5)` for the periodic BC).

### Optimisation

- Optimiser: Adam, `lr = 5e-4`
- Epochs: `25000` (full-batch)
- Trainer: `pinn.trainer.PINNTrainer` (live loss plot, tqdm, loss history)
- Device: CUDA if available, else CPU

---

## 3. Usage

### Quick start

From the repository root:

```bash
uv sync --all-packages
uv run train-schrodinger       # train with defaults
```

### CLI reference

```
uv run train-schrodinger [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `25000` | Number of training epochs |
| `--lr` | | float | `5e-4` | Adam learning rate |
| `--neurons` | `-n` | int | `100` | Neurons per hidden layer |
| `--layers` | `-l` | int | `4` | Number of hidden layers |
| `--save-plot` | | flag | off | Save the final plots |
| `--plot-path` | | str | `None` | Output path for plots (used with `--save-plot`) |

### Examples

```bash
# Quick smoke test
uv run train-schrodinger -e 5000 -n 50

# Full run, saving figures
uv run train-schrodinger --save-plot --plot-path results/nls.png
# -> writes results/nls.png and results/nls_validation.png
```

---

## 4. Output

1. **Live loss plot** — `ic`, `bc`, `physics` curves on a log scale during training.
2. **Contour plot** — solution magnitude `|h(t, x)|` on a 200×100 space-time grid. Expect the soliton to breathe: amplitude focusing near `t ≈ π/4`.
3. **Validation snapshot** — `|h(0, x)|` vs. the exact `2·sech(x)` initial condition.
4. **Summary table** — final total/ic/bc/physics losses and epochs run.

With `--save-plot --plot-path <p>.png`, the validation figure is saved alongside as `<p>_validation.png`.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Magnitude decays instead of breathing | Physics residual under-fit; network stuck in a smooth local minimum | More epochs; larger network (`-n 128`); or raise the `physics` weight in `solve_schrodinger_equation` |
| IC snapshot off from `2·sech(x)` | IC loss under-weighted relative to 5000-point physics loss | Raise the `ic` weight |
| Boundary seams in the contour plot | Periodic BC not converged | Raise the `bc` weight, or add more `t_bc` points in `train.py` |
| Loss spikes / NaN | lr too high for the nonlinear term | Lower `--lr` (e.g. `1e-4`), or pass `grad_clip` to `trainer.train` |
| No plot window appears | Headless environment | `--save-plot --plot-path out.png`, or `MPLBACKEND=Agg` |

---

## 6. File Layout

```
experiments/schrodinger/
├── README.md      # this file
├── __init__.py
└── train.py       # Typer CLI, ComplexPINN wrapper, residual, losses, training, plots
```

Key entry points in `train.py`:

- `train(...)` — Typer command, exposed as the `train-schrodinger` console script
- `solve_schrodinger_equation(...)` — programmatic API (importable from other code)
- `ComplexPINN` — two-output wrapper returning `(Re h, Im h)`

---

## 7. References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378 — NLS setup (Sec. 3.1.1) with the same `2·sech(x)` IC and domain.
- Sulem & Sulem (1999). *The Nonlinear Schrödinger Equation: Self-Focusing and Wave Collapse.* Springer.
