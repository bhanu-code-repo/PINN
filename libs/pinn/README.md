# pinn

Core library for building and training **Physics-Informed Neural Networks (PINNs)** in PyTorch.
It provides the building blocks shared by every experiment in this monorepo: a configurable MLP
backbone, a generic multi-loss trainer with checkpointing, reproducibility and logging helpers,
and plotting utilities.

```python
from pinn import PINN, PINNTrainer, set_seed, setup_logging
```

## Installation

This package is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) member. From the
repository root:

```bash
uv sync --all-packages
```

Requires Python ≥ 3.11. Dependencies: `torch`, `numpy`, `matplotlib`, `tqdm`, `loguru`.

## Package Layout

```
pinn/
├── core/
│   └── network.py     # PINN — fully-connected MLP backbone
├── trainer/
│   └── trainer.py     # PINNTrainer — training loop + checkpointing
└── utils/
    ├── logging.py     # setup_logging — loguru console + file sinks
    ├── seed.py        # set_seed — reproducibility
    └── plotting.py    # contour / comparison / loss plots
```

---

## Quickstart: Solve Your Own Equation in 5 Steps

The library never needs to know your physics — you express it entirely through loss functions.
Example: the trivial ODE `u' = -u`, `u(0) = 1`.

```python
import torch
from pinn import PINN, PINNTrainer, set_seed, setup_logging

# 1. Reproducibility + logging (console; pass log_dir=... for a log file)
setup_logging()
set_seed(42)

# 2. Collocation points — where the physics will be enforced
t = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)

# 3. Loss functions: each takes the model, returns a scalar tensor
def physics_loss(m):
    u = m(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    return torch.mean((u_t + u) ** 2)               # residual of u' + u = 0

def ic_loss(m):
    return (m(torch.zeros(1, 1)) - 1.0).pow(2).squeeze()   # u(0) = 1

# 4. Model + trainer
model = PINN(input_dim=1, hidden_layers=3, hidden_neurons=32)
trainer = PINNTrainer(model)
trainer.train(
    n_epochs=5000,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    loss_functions={"physics": physics_loss, "ic": ic_loss},
    weights={"physics": 1.0, "ic": 10.0},
)

# 5. Persist and inspect
trainer.save_checkpoint("run/checkpoint.pt", metadata={"seed": 42})
trainer.plot_loss_history(save_path="run/loss.png", show=False)
```

Adapting to a new PDE means changing only steps 2–3: new collocation points, new residual, new
IC/BC losses. See `experiments/` for three complete worked examples (ODE, shock-forming PDE,
complex-valued PDE with periodic BCs).

---

## API Reference

### `PINN` — network backbone

```python
PINN(input_dim: int, hidden_layers: int, hidden_neurons: int, output_dim: int = 1)
```

| Parameter | Description |
|-----------|-------------|
| `input_dim` | Number of input coordinates (`1` for `t`, `2` for `(x, t)`, ...) |
| `hidden_layers` | Number of hidden layers |
| `hidden_neurons` | Width of each hidden layer |
| `output_dim` | Output channels — `2` for e.g. complex fields (`Re`, `Im`) |

Architecture: `Linear(in → h) → Tanh → [Linear(h → h) → Tanh] × (L−1) → Linear(h → out)`.
`tanh` is deliberate: PINN losses differentiate the output w.r.t. the *inputs*, often twice, so
the activation must be smooth (ReLU's second derivative is zero).

### `PINNTrainer` — training loop

```python
PINNTrainer(model: nn.Module, device: torch.device | None = None)   # device: auto CUDA/CPU
```

#### `train(...)`

```python
trainer.train(
    n_epochs,                  # int: full-batch epochs
    optimizer,                 # torch.optim.Optimizer
    loss_functions,            # dict[str, Callable]: name -> fn(model) -> scalar tensor
    weights=None,              # dict[str, float]: per-term weights (default 1.0)
    verbose=True,              # tqdm progress bar
    log_every=1000,            # loguru DEBUG summary (losses + grad norm) every N epochs
    early_stop_patience=None,  # stop after N epochs without improvement
    early_stop_threshold=1e-8, # minimum decrease that counts as improvement
    grad_clip=None,            # clip global grad norm (helps stiff PDEs)
    callbacks=None,            # list of fn(epoch, epoch_losses) run each epoch
) -> list[dict[str, float]]    # loss history (also on trainer.loss_history)
```

**Loss function contract:** `total = Σ weights[name] · loss_fn(model)` — the trainer backprops
and steps. Collocation points are closed over by the loss callables, keeping the trainer
agnostic to ODEs vs. PDEs, boundary types, or data-fitting terms.

**No plotting inside the loop** — the trainer is headless-safe by construction. Monitor via the
progress bar, log files, or a callback; plot afterwards with `plot_loss_history`.

#### `save_checkpoint(path, optimizer=None, metadata=None)` / `load_checkpoint(path, optimizer=None)`

Persist/restore model weights, optimizer state, loss history, and an arbitrary metadata dict
(hyperparameters, seed, problem config). Passing the optimizer enables exact training resume.

```python
trainer.save_checkpoint("ckpt.pt", optimizer=opt, metadata={"seed": 42, "w0": 80.0})
meta = trainer.load_checkpoint("ckpt.pt", optimizer=opt)   # returns the metadata dict
```

#### `plot_loss_history(show_total=False, save_path=None, show=True)`

Post-training log-scale plot of all loss terms. `show=False` for headless runs.

### `set_seed(seed)`

Seeds `random`, NumPy, and PyTorch (CPU + CUDA) in one call. Call it before creating collocation
points and models.

### `setup_logging(log_dir=None, level="INFO", file_level="DEBUG")`

Configures loguru: a tqdm-safe console sink (log lines don't mangle progress bars), plus an
optional rotating file sink under `log_dir`. Call once at application startup; library code just
uses `from loguru import logger`. Returns the log-file path (or `None`).

### Plotting utilities

| Function | Purpose |
|----------|---------|
| `plot_contour(X, Y, Z, ...)` | Filled contour of a 2D field; `X` is the horizontal axis, `Y` vertical |
| `plot_comparison_1d(x, y_exact, y_pred, ...)` | Exact (black solid) vs. prediction (red dashed) |
| `plot_loss_comparison(loss_history, ...)` | Overlay loss curves from multiple experiments |

All accept `save_path` (300 dpi PNG) and `show=False` for headless use.

---

## Scaling This Codebase

Guidance for growing beyond the current small-problem regime:

**More collocation points → mini-batching.** The trainer is full-batch (standard for small
PINNs). At ~10⁵⁺ points, resample or shard inside your loss callables — e.g. draw a fresh random
subset of points each call. Because losses own their data, this requires *no trainer changes*:

```python
def physics_loss(m):
    idx = torch.randint(0, x_all.shape[0], (4096,))
    return residual(m, x_all[idx], t_all[idx]).pow(2).mean()
```

**Better convergence → Adam then L-BFGS.** The standard PINN recipe is Adam for exploration
followed by L-BFGS for refinement. L-BFGS needs a closure-based step, so run it as a short
manual loop after `trainer.train(...)` using the same loss callables.

**Harder problems → adaptive loss weighting.** Hand-tuned weights (see the harmonic oscillator's
`1e-4` physics weight) stop scaling as terms multiply. Techniques such as gradient-norm balancing
can be implemented as a `callback` that adjusts a shared `weights` dict during training.

**High-frequency solutions → input embeddings or ansatz.** Spectral bias makes plain MLPs fail on
oscillatory solutions. Options: a problem-specific ansatz wrapper (see
`experiments/harmonic_oscillator`), Fourier feature embeddings before the first layer, or SIREN
activations — all composable with `PINNTrainer` since it accepts any `nn.Module`.

**GPU.** Everything follows `trainer.device` (auto-CUDA). Keep collocation tensors on the same
device you pass to `PINNTrainer` — create them with `.to(device)` as the experiments do.

**Reproducible experiments.** Combine `set_seed`, checkpoint `metadata`, and a per-run output
directory. `experiments/common.py` shows the pattern: every run directory is self-contained
(checkpoint + metrics JSON + plots + logs).

---

## Design Notes

- **Losses own their data.** The trainer never sees collocation points, only callables. This is
  what keeps a single trainer valid for ODEs, PDEs, complex fields, and inverse problems.
- **Logging over printing.** All library output goes through loguru; applications control sinks.
- **Headless by default.** Nothing in the training path opens a matplotlib window.

## Used By

- `experiments/harmonic_oscillator` — damped oscillator ODE, learnable sinusoidal ansatz
- `experiments/burgers` — Burgers' equation, shock formation
- `experiments/schrodinger` — nonlinear Schrödinger, complex field + periodic BCs
