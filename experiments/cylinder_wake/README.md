# Cylinder Wake Inverse PINN — Raissi et al. (2019) Benchmark

**The headline result from the original PINNs paper.** Given scattered noisy velocity
observations from a DNS of 2D flow past a cylinder at Re = 100, a PINN simultaneously
reconstructs the full velocity + pressure field and infers the unknown PDE parameters λ₁
and λ₂ — including the pressure, which was *never observed* during training.

---

## 1. Problem Statement

The incompressible Navier-Stokes equations, written in parameterised form:

```
u_t + λ₁(u·u_x + v·u_y) = -p_x + λ₂(u_xx + u_yy)
v_t + λ₁(u·v_x + v·v_y) = -p_y + λ₂(v_xx + v_yy)
u_x + v_y = 0   (incompressibility)
```

**True values:** λ₁ = 1, λ₂ = 1/Re = 0.01 (i.e. Re = 100).

The data comes from a spectral-element DNS (Nektar) of 2D flow past a circular cylinder:

| Property | Value |
|----------|-------|
| Domain | x ∈ [1, 8], y ∈ [−2, 2] (wake region behind cylinder at x = 0) |
| Time | t ∈ [0, 19.9], 200 snapshots at Δt = 0.1 |
| Spatial points | 5,000 per snapshot |
| Total data | 1,000,000 points of (x, y, t, u, v, p) |
| Flow regime | Periodic vortex shedding (Von Kármán street) |

**What the PINN sees:** a random subset of (u, v) observations (default: 5,000 out of 1M).
**What it never sees:** the pressure field p.
**What it must infer:** λ₁, λ₂, the full (u, v, p) field everywhere.

## 2. Method

### Streamfunction formulation (Rule 3: embed known structure)

The network outputs `(ψ, p)` from inputs `(x, y, t)`, and velocities are derived via
autograd:

```
u = ∂ψ/∂y,    v = −∂ψ/∂x
```

Incompressibility (`u_x + v_y = 0`) is satisfied **by construction** — not approximately,
not as a loss term, but identically. This is critical for the cylinder wake: enforcing
continuity as a soft constraint fights the momentum equations, and the resulting three-way
competition makes convergence fragile.

The cost: momentum residuals require second-order derivatives of u and v, which are
third-order derivatives of ψ. Autograd handles this via `create_graph=True` chains, but
per-epoch cost is higher than direct (u, v, p) formulations.

### Learnable PDE parameters

Two learnable scalars `lambda_1` and `lambda_2` are stored as `nn.Parameter` and used inside
the physics loss:

```python
self.lambda_1 = nn.Parameter(torch.tensor(lambda1_init))
self.lambda_2 = nn.Parameter(torch.tensor(lambda2_init))

# In physics_loss:
f_u = u_t + lam1 * (u * u_x + v * u_y) + p_x - lam2 * (u_xx + u_yy)
```

Gradients flow through the residual back to both parameters, so Adam jointly optimises the
network weights, λ₁, and λ₂.

### Losses

| Term | Enforces | Points |
|------|----------|--------|
| `data` | (u, v) match DNS observations | 5,000 random from the 1M pool |
| `physics` | Parameterised NS momentum residuals with learnable λ₁, λ₂ | 10,000 random from the full domain |

No boundary conditions are imposed — the data term anchors the solution, and the physics
term drives λ convergence.

### What makes this different from the Kovasznay inverse experiment

| | Kovasznay (`navier_stokes_inverse`) | Cylinder wake |
|---|---|---|
| Flow | Steady, laminar, exact solution | Unsteady, periodic vortex shedding |
| Data | Synthetic (generated from formula) | Real DNS (spectral-element simulation) |
| Parameters inferred | Re (1 scalar, log-space) | λ₁, λ₂ (2 scalars, direct) |
| Pressure validation | Against exact solution | Against DNS (never seen by PINN) |
| Formulation | Direct (u, v, p) output | Streamfunction ψ → u, v by construction |
| Difficulty | Entry-level inverse | Full benchmark |

## 3. Usage

### Data setup

Place the DNS data file in `.workspace/input/`:

```
.workspace/input/cylinder_nektar_wake.mat
```

This file is from the [Raissi PINNs repository](https://github.com/maziarraissi/PINNs).

### Commands

```bash
uv run train-cylinder train                        # default: 30k epochs, 5k observations
uv run train-cylinder train -e 50000 --n-train 10000  # more data + longer training
uv run train-cylinder predict                       # evaluate latest run, report λ errors
uv run train-cylinder predict --t-idx 150           # snapshot at a different time step
uv run train-cylinder compare                       # rank all runs
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `30000` | Training epochs |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `8` | Hidden layers (deeper than other experiments) |
| `--n-train` | | int | `5000` | Number of (u, v) observations from DNS |
| `--n-physics` | | int | `10000` | Collocation points for NS residual |
| `--lambda1-init` | | float | `1.0` | Initial guess for λ₁ |
| `--lambda2-init` | | float | `0.01` | Initial guess for λ₂ |
| `--seed` | | int | `42` | Random seed |
| `--output-dir` | `-o` | str | auto | Artifact directory |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |
| `--data-path` | | str | `.workspace/input/...` | Path to the .mat file |

### CLI reference — `predict`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--run` | `-r` | str | latest | Run directory to load |
| `--t-idx` | | int | `100` | Time snapshot index for plots (0–199) |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |
| `--data-path` | | str | `.workspace/input/...` | Path to the .mat file |

## 4. Output

```
<run-dir>/
├── checkpoint.pt                  # self-describing (includes λ₁, λ₂ as model state)
├── metrics.json                   # losses + λ errors + velocity/pressure rel-L2
├── loss_history.png
├── snapshot_comparison.png        # 3×3 grid: DNS vs PINN for u, v, p + pointwise error
├── lambda_convergence.png         # λ₁ and λ₂ over training epochs
├── predictions.npz                # snapshot fields + inferred λ values
└── logs/run_*.log
```

**What to look for:**

1. **λ convergence plot:** both parameters should converge toward the true values
   (λ₁ → 1.0, λ₂ → 0.01). λ₁ typically converges faster than λ₂.
2. **Snapshot comparison:** the u and v fields should match the DNS closely. The
   pressure field (bottom row) is the headline result — the PINN reconstructs it
   from velocity data alone, using the NS equations as the bridge.
3. **Error maps:** largest errors concentrate near the cylinder (strong gradients)
   and in the far wake (limited training data).

## 5. The Key Ideas

### Why this works: the physics constrains the inverse

The data loss alone would give you an interpolator — it could match (u, v) at the observed
points but would say nothing about pressure or the PDE parameters. The physics loss adds the
constraint that the *fields must satisfy the Navier-Stokes equations*. This:

- **Determines pressure from velocity:** the pressure gradient appears in the momentum
  equations, so the physics loss forces the network to learn a pressure field that is
  consistent with the observed velocities.
- **Determines λ₁, λ₂:** wrong parameter values produce large momentum residuals at the
  collocation points. The gradient of the physics loss w.r.t. λ₁, λ₂ pushes them toward
  values that make the equations consistent with the data.

### Why streamfunction: robustness

The standard formulation (output u, v, p directly) penalises continuity as a loss term.
In practice, the continuity loss can trade off against the momentum losses during
optimisation, leading to velocity fields that are not truly divergence-free. The
streamfunction eliminates this failure mode entirely.

### The two-diagnostic principle (from parametric_pinns.md)

Even with accurate λ inference, check both:

- **Data residual:** does the field match observations where you have them?
- **Physics residual:** does it satisfy NS everywhere, including where you have no data?

If the physics residual is large in the far wake but λ is accurate, the network has good
global parameter inference but poor local field reconstruction — likely needs more capacity
or collocation points in that region.

## 6. Honest Caveats

1. **This is Re = 100 (laminar periodic shedding).** Higher Re (turbulent wake) is a
   different beast — the DNS data would be three-dimensional and chaotic. PINNs struggle
   with turbulence; this experiment is the gentle end of the NS spectrum.

2. **External data dependency.** Unlike every other experiment in this repo, this one
   requires a `.mat` file that cannot be generated from first principles. The test skips
   gracefully if the file is missing.

3. **Computational cost.** The streamfunction formulation requires third-order autograd
   chains. Per-epoch cost is ~5–10× higher than direct (u, v, p). The deeper default
   network (8 layers vs 4–5) adds to this.

4. **λ₂ is harder than λ₁.** λ₂ = 0.01 is small and multiplies the viscous term, which
   is itself small relative to the convective term. The gradient signal for λ₂ is weak —
   it requires more epochs and more physics collocation points than λ₁.

5. **Pressure gauge.** Pressure in incompressible NS is defined up to a constant. The
   PINN may learn a different constant offset than the DNS. All pressure comparisons are
   mean-subtracted.

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks: A deep
  learning framework for solving forward and inverse problems involving nonlinear partial
  differential equations.* J. Comput. Phys. 378, 686–707.
- Raissi, Yazdani, Karniadakis (2019). *Hidden fluid mechanics: Learning velocity and
  pressure fields from flow visualizations.* Science 367(6481), 1026–1030.
