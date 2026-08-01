# Navier-Stokes Inverse Problem PINN — Infer Re from Data (Kovasznay Flow)

**The inverse problem: given scattered velocity observations, infer the unknown Reynolds
number.** This demonstrates the same methodology as Raissi et al. (2019) cylinder wake, but
with the **Kovasznay flow** — an exact steady NS solution — so the experiment is fully
self-contained and rigorously validatable.

---

## 1. Problem Statement

The Kovasznay flow is an exact solution to the 2D steady incompressible Navier-Stokes
equations for any Reynolds number:

```
lambda = Re/2 - sqrt(Re^2/4 + 4*pi^2)
u  = 1 - exp(lambda*x) * cos(2*pi*y)
v  = (lambda / 2*pi) * exp(lambda*x) * sin(2*pi*y)
p  = (1 - exp(2*lambda*x)) / 2
```

over `[-0.5, 1.0] x [-0.5, 1.5]`.

**The inverse setup:** we observe noisy `(u, v)` at scattered points and ask the PINN to:

1. Learn the flow field `(u, v, p)` everywhere (forward problem)
2. **Simultaneously infer `Re`** (inverse problem)

The ground-truth `Re = 20` is unknown to the network; it starts from a deliberately wrong
initial guess (`Re_init = 10` by default).

## 2. Method

### Learnable Reynolds number

The model holds a **learnable scalar** `log_Re` (optimised in log-space for positivity):

```python
class InverseNavierStokesPINN(nn.Module):
    def __init__(self, ...):
        self.network = PINN(input_dim=2, ..., output_dim=3)
        self.log_re = nn.Parameter(torch.tensor(log_re_init))

    @property
    def re(self):
        return torch.exp(self.log_re)
```

The physics loss computes `nu = 1 / model.re` and uses it inside the NS residual. Since
`log_re` is a parameter, **gradients flow through** the residual back to `log_re`, and Adam
optimises both the network weights and the Reynolds number jointly.

### Losses

| Term | Enforces | Points |
|------|----------|--------|
| `data` | `(u, v)` match noisy observations | 200 scattered (default) |
| `physics` | Steady NS with *learnable* `nu = 1/Re` | 5,000 random interior |

No boundary conditions are needed — the data term anchors the solution, and the physics
term ensures NS consistency while providing the signal for Re inference.

### Synthetic data generation

Observations are generated from the exact Kovasznay solution with additive Gaussian noise:

```
u_obs = u_exact + noise * std(u_exact) * N(0, 1)
v_obs = v_exact + noise * std(v_exact) * N(0, 1)
```

Default noise level: 1% of the signal amplitude. The observations are saved in
`observations.npz` for reproducibility.

## 3. Usage

```bash
uv run train-ns-inverse train                              # default: Re_true=20, noise=1%
uv run train-ns-inverse train --re-true 40 --noise 0.05    # harder: higher Re, 5% noise
uv run train-ns-inverse train --re-init 1.0                # start from a very wrong guess
uv run train-ns-inverse train -e 10000 --lbfgs-epochs 500  # Adam → L-BFGS two-stage
uv run train-ns-inverse predict                             # report inferred Re and accuracy
uv run train-ns-inverse compare                             # rank all runs
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `30000` | Training epochs |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `5` | Hidden layers |
| `--n-physics` | | int | `5000` | Interior collocation points |
| `--n-obs` | | int | `200` | Number of observation points |
| `--re-true` | | float | `20` | Ground-truth Reynolds number |
| `--noise` | | float | `0.01` | Observation noise (fraction of amplitude) |
| `--re-init` | | float | `10` | Initial Re guess (deliberately wrong) |
| `--seed` | | int | `42` | Random seed |
| `--output-dir` | `-o` | str | auto | Artifact directory |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |
| `--lbfgs-epochs` | | int | `0` | L-BFGS refinement epochs after Adam (0 = skip) |
| `--lbfgs-lr` | | float | `1.0` | L-BFGS learning rate |

### Two-stage training (Adam → L-BFGS)

For sharper Re inference, use L-BFGS refinement after Adam converges. L-BFGS uses
quasi-Newton curvature information that is particularly effective for the smooth
landscape around the inverse parameter:

```bash
uv run train-ns-inverse train -e 10000 --lbfgs-epochs 500 --no-show
```

The loss history accumulates across both stages. L-BFGS is configured with
`history_size=50`, strong Wolfe line search.

### CLI reference — `predict`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--run` | `-r` | str | latest | Run directory to load |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

## 4. Output

```
<run-dir>/
├── checkpoint.pt                # self-describing (includes log_re as model state)
├── metrics.json                 # losses + Re_true, Re_inferred, Re error, velocity error
├── loss_history.png
├── inverse_results.png          # velocity comparison + Re convergence plot + pressure
├── observations.npz             # the noisy data used for training
├── predictions.npz              # full fields + inferred Re
└── logs/run_*.log
```

**What to look for:** the Re convergence subplot should show `log_re` climbing from the
initial guess (10) toward the true value (20). The velocity field should match the
Kovasznay pattern: a baseline flow `u ~ 1` with an exponentially decaying perturbation.

## 5. The Inverse Problem Methodology

This experiment demonstrates the general recipe for PINN inverse problems:

1. **Add learnable parameters** to the model (here: `log_Re`)
2. **Add a data loss** that anchors the solution to observations
3. **Use the learnable parameters inside the physics loss** (here: `nu = 1/Re`)
4. **Optimise everything jointly** — the optimizer adjusts weights *and* parameters
5. **Refine with L-BFGS** (optional) — quasi-Newton refinement after Adam plateaus

The physics loss provides the signal: if the current Re guess makes the NS residual large,
the gradient of that residual w.r.t. `log_Re` pushes Re toward the value that satisfies the
equations. The data loss prevents the trivial solution (Re -> infinity, zero viscosity, any
smooth field).

**Connection to Raissi's cylinder wake:** identical methodology, but Raissi uses DNS data
from a Re = 100 cylinder flow and infers two parameters (lambda_1, lambda_2 in the NS
parameterisation). The Kovasznay flow lets us verify the methodology against a known answer
without external data dependencies.

## 6. What Makes It Hard

- **Noise sensitivity:** higher noise levels make Re inference less precise (try `--noise 0.1`)
- **Data sparsity:** fewer observations -> weaker anchoring -> Re may not converge
  (try `--n-obs 20` to see this)
- **Wrong initial guess:** starting far from the true Re requires more epochs to converge
  (try `--re-init 1.0`)
- **Higher Re:** stronger nonlinearity makes both the forward solve and the inverse
  inference harder

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
- Kovasznay, L.I.G. (1948). *Laminar flow behind a two-dimensional grid.* Proc. Cambridge
  Phil. Soc. 44.
