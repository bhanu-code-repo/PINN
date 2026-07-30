# Damped Harmonic Oscillator PINN

A Physics-Informed Neural Network (PINN) that solves the **under-damped harmonic oscillator** ODE — including the high-frequency regime (`w0 = 80`) where vanilla PINNs typically fail — using a learnable **sinusoidal Ansatz**.

---

## 1. Problem Statement

We solve the second-order linear ODE on the domain `t ∈ [0, 1]`:

```
u''(t) + μ·u'(t) + k·u(t) = 0
```

with initial conditions:

```
u(0)  = 1
u'(0) = 0
```

The physical parameters are derived from the CLI inputs:

| Symbol | Definition | Meaning | Default |
|--------|------------|---------|---------|
| `d`    | CLI `--damping` | Damping coefficient (δ) | `2.0` |
| `w0`   | CLI `--w0` | Natural (undamped) frequency | `80.0` |
| `μ`    | `2·d` | ODE friction term | `4.0` |
| `k`    | `w0²` | ODE stiffness term | `6400.0` |

### Exact solution (under-damped case, `d < w0`)

```
w   = sqrt(w0² − d²)
φ   = arctan(−d / w)
A   = 1 / (2·cos φ)
u(t) = e^(−d·t) · 2A · cos(φ + w·t)
```

This closed-form solution is used for validation in the final comparison plot.

---

## 2. Why an Ansatz?

At `w0 = 80` the solution oscillates ~13 times in `[0, 1]`. A plain MLP with `tanh` activations suffers from **spectral bias** — it learns low-frequency content first and often never resolves high-frequency oscillations, collapsing to the trivial solution `u ≡ 0`.

The fix used here wraps the network output in a trainable sinusoid:

```
u(t) = NN(t) · sin(a·t + b)
```

where `a` (frequency, initialised at `70.0`) and `b` (phase, initialised at `1.0`) are `nn.Parameter`s optimised jointly with the network weights. The network then only needs to learn the slowly-varying **envelope** `e^(−d·t)`-like amplitude, which is a low-frequency function well within an MLP's easy reach. The optimiser tunes `a` toward the true damped frequency `w ≈ sqrt(w0² − d²) ≈ 79.97`.

The learned `a` and `b` are reported in the final summary table, so you can verify the frequency was recovered.

---

## 3. Method

### Architecture

```
Ansatz(t) = PINN(t) · sin(a·t + b)

PINN: Linear(1 → 32) → Tanh
      → [Linear(32 → 32) → Tanh] × 2
      → Linear(32 → 1)
```

- Backbone: `pinn.core.network.PINN` (fully-connected, `tanh` activations)
- Wrapper: local `Ansatz` module defined in `train.py`

### Loss formulation

The total loss is a weighted sum of two terms:

| Term | Definition | Collocation points | Weight |
|------|-----------|--------------------|--------|
| `ic` | `(u(0) − 1)² + (u'(0) − 0)²` | 1 point at `t = 0` | `0.1` |
| `physics` | `mean[(u'' + μ·u' + k·u)²]` | 100 uniform points on `[0, 1]` | `1e-4` |

> **Note on weights:** the physics residual scales with `k = w0² = 6400`, so its raw magnitude is huge. The small `1e-4` weight rebalances it against the O(1) initial-condition loss. If you change `--w0` significantly, these weights may need retuning.

Derivatives `u'` and `u''` are computed exactly via `torch.autograd.grad` with `create_graph=True` (second-order autodiff).

### Optimisation

- Optimiser: Adam, `lr = 1e-3`
- Epochs: `15000` (full-batch — all collocation points every step)
- Device: CUDA if available, else CPU
- Training loop: `pinn.trainer.PINNTrainer` (live loss plot, tqdm progress bar, loss history)

---

## 4. Usage

### Quick start

From the repository root:

```bash
uv sync --all-packages
uv run train-harmonic          # train with defaults
```

Or run the module directly:

```bash
uv run python experiments/harmonic_oscillator/train.py
```

### CLI reference

```
uv run train-harmonic [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `15000` | Number of training epochs |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `32` | Neurons per hidden layer |
| `--layers` | `-l` | int | `3` | Number of hidden layers |
| `--w0` | | float | `80.0` | Natural frequency |
| `--damping` | `-d` | float | `2.0` | Damping coefficient |
| `--save-plot` | | flag | off | Save the final comparison plot |
| `--plot-path` | | str | `None` | Output path for the plot (used with `--save-plot`) |

### Examples

```bash
# Faster, lower-frequency sanity check
uv run train-harmonic -e 5000 --w0 20

# Bigger network, save the result figure
uv run train-harmonic -n 64 -l 4 --save-plot --plot-path results/harmonic.png

# Heavier damping
uv run train-harmonic -d 10 --w0 40
```

---

## 5. Output

During and after training you get:

1. **Live loss plot** — per-term (`ic`, `physics`) loss curves on a log scale, refreshed every 1000 epochs.
2. **Comparison plot** — PINN prediction (red) vs. analytical solution (black dashed) on 300 test points.
3. **Summary table**:

   | Metric | Meaning |
   |--------|---------|
   | Final Loss | Total weighted loss at the last epoch |
   | Learned 'a' (Frequency) | Should converge near `sqrt(w0² − d²)` |
   | Learned 'b' (Phase) | Learned phase offset |
   | Epochs Run | Actual epochs completed |

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Prediction collapses to `u ≈ 0` | Physics loss dominates; frequency `a` far from truth | Lower the physics weight, or initialise `a` closer to `sqrt(w0² − d²)` |
| Loss plateaus early | Learning rate too high/low for the regime | Try `--lr 3e-4` or `--lr 3e-3`; increase `--epochs` |
| Good fit near `t=0`, drifts later | Too few collocation points for the frequency | Increase collocation density in `train.py` (currently 100) |
| Changed `--w0`, training diverges | Loss weights tuned for `w0 = 80` | Retune `weights={'ic': ..., 'physics': ...}` in `solve_harmonic_oscillator` |
| No plot window appears | Headless environment / non-interactive matplotlib backend | Use `--save-plot --plot-path out.png`, or set `MPLBACKEND=Agg` |

---

## 7. File Layout

```
experiments/harmonic_oscillator/
├── README.md      # this file
├── __init__.py
└── train.py       # Typer CLI, problem definition, Ansatz, training, plotting
```

Key entry points in `train.py`:

- `train(...)` — Typer command, exposed as the `train-harmonic` console script
- `solve_harmonic_oscillator(...)` — programmatic API (importable from other code)
- `Ansatz` — the `NN(t)·sin(a·t + b)` wrapper module

---

## 8. References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs.* J. Comput. Phys. 378.
- Rahaman et al. (2019). *On the Spectral Bias of Neural Networks.* ICML.
- Moseley, Markham, Nissen-Meyer (2021). *Finite Basis Physics-Informed Neural Networks.* (motivating high-frequency PINN failure modes)
