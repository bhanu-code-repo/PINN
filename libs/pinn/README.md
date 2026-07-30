# pinn

Reusable core library for building and training **Physics-Informed Neural Networks (PINNs)** in PyTorch. It provides the building blocks shared by all experiments in this monorepo: a configurable MLP backbone, a generic multi-loss trainer, and plotting utilities.

## Installation

This package is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) member. From the repository root:

```bash
uv sync --all-packages
```

Requires Python ≥ 3.11. Dependencies: `torch`, `numpy`, `matplotlib`, `tqdm`.

## Package Layout

```
pinn/
├── core/
│   └── network.py     # PINN — fully-connected MLP backbone
├── trainer/
│   └── trainer.py     # PINNTrainer — generic multi-loss training loop
└── utils/
    └── plotting.py    # Contour / comparison / loss plots
```

---

## API Reference

### `pinn.core.network.PINN`

A fully-connected MLP with `tanh` activations.

```python
PINN(input_dim: int, hidden_layers: int, hidden_neurons: int, output_dim: int = 1)
```

| Parameter | Description |
|-----------|-------------|
| `input_dim` | Number of input coordinates (e.g. `1` for `t`, `2` for `(x, t)`) |
| `hidden_layers` | Number of hidden layers |
| `hidden_neurons` | Width of each hidden layer |
| `output_dim` | Number of outputs (default `1`; use `2` for e.g. complex-valued fields) |

Architecture: `Linear(in → h) → Tanh → [Linear(h → h) → Tanh] × (L−1) → Linear(h → out)`.

`tanh` is used because PINNs need smooth, infinitely-differentiable activations for higher-order autograd derivatives (ReLU has zero second derivative).

```python
from pinn.core.network import PINN

model = PINN(input_dim=2, hidden_layers=4, hidden_neurons=64)
u = model(xt)   # xt: (N, 2) tensor
```

### `pinn.trainer.trainer.PINNTrainer`

A problem-agnostic training loop for weighted multi-term PINN losses.

```python
PINNTrainer(model: nn.Module, device: Optional[torch.device] = None)
```

Device defaults to CUDA when available. The model is moved to the device on construction.

#### `PINNTrainer.train(...)`

```python
trainer.train(
    n_epochs,                  # int: number of epochs (full-batch)
    optimizer,                 # torch.optim.Optimizer
    loss_functions,            # Dict[str, Callable]: name -> fn(model) -> scalar tensor
    weights=None,              # Dict[str, float]: per-term weights (default 1.0 each)
    verbose=True,              # tqdm progress bar with live total loss
    plot_every=1000,           # live log-scale loss plot every N epochs (0 disables)
    debug_every=5000,          # print gradient norm every N epochs (0 disables)
    early_stop_patience=None,  # stop if no improvement for N epochs
    early_stop_threshold=1e-8, # minimum improvement to reset patience
    grad_clip=None,            # clip gradient norm (helps stiff PDEs)
)
```

**Loss function contract:** each entry in `loss_functions` is a callable that takes the model and returns a scalar tensor. The trainer computes
`total = Σ weights[name] · loss_fn(model)`, backpropagates, and steps the optimizer. Collocation points are typically closed over by the loss functions themselves.

**Loss history:** every epoch appends `{name: value, ..., 'total': value}` to `trainer.loss_history`, so you can inspect or re-plot after training.

#### `PINNTrainer.plot_loss_history(show_total=False, save_path=None)`

Static post-training log-scale plot of all loss terms; optionally saved to disk at 300 dpi.

#### Example

```python
import torch
from pinn.core.network import PINN
from pinn.trainer.trainer import PINNTrainer

model = PINN(input_dim=1, hidden_layers=3, hidden_neurons=32)

t = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)

def physics_loss(m):
    u = m(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    return torch.mean((u_t + u) ** 2)          # u' + u = 0

def ic_loss(m):
    return (m(torch.zeros(1, 1)) - 1.0).pow(2).squeeze()  # u(0) = 1

trainer = PINNTrainer(model)
trainer.train(
    n_epochs=5000,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    loss_functions={"physics": physics_loss, "ic": ic_loss},
    weights={"physics": 1.0, "ic": 10.0},
)
trainer.plot_loss_history(save_path="loss.png")
```

### `pinn.utils.plotting`

| Function | Purpose |
|----------|---------|
| `plot_contour(X, Y, Z, ...)` | Filled contour plot (viridis, 20 levels) for 2D fields, e.g. `u(x, t)` |
| `plot_comparison_1d(x, y_exact, y_pred, ...)` | Exact (black solid) vs. prediction (red dashed) line plot |
| `plot_loss_comparison(loss_history, ...)` | Overlay loss curves from multiple experiments on a log scale |

All plotting functions accept `save_path` to write a 300 dpi PNG and call `plt.show()`.

---

## Design Notes

- **Full-batch training.** PINN losses are computed over all collocation points each epoch — standard practice for PINNs, where the "dataset" is a small set of collocation points rather than mini-batched data.
- **Losses own their data.** The trainer never sees collocation points; loss callables close over them. This keeps the trainer generic across ODEs/PDEs, boundary conditions, and data-fitting terms.
- **Live plotting uses interactive matplotlib** (`plt.ion()`). In headless environments set `plot_every=0` or `MPLBACKEND=Agg`.

## Used By

- `experiments/harmonic_oscillator` — damped harmonic oscillator ODE (see its README)
- `experiments/burgers` — Burgers' equation
- `experiments/schrodinger` — Schrödinger equation
