# Parametric Taylor-Green Vortex — Reynolds Sweep with a Single PINN

**One model learns the entire Taylor-Green solution family across two orders of
magnitude in viscosity.** Given `nu`, predict the full `(u, v, p)` field in
milliseconds — no retraining required.

---

## 1. Problem Statement

The Taylor-Green vortex is an exact solution to the 2D incompressible
Navier-Stokes equations, valid for any viscosity `nu` and all time `t`:

```
u(x,y,t; nu) = -cos(x) sin(y) exp(-2 nu t)
v(x,y,t; nu) =  sin(x) cos(y) exp(-2 nu t)
p(x,y,t; nu) = -1/4 (cos 2x + cos 2y) exp(-4 nu t)
```

Domain: `[0, 2pi]^2 x [0, 1]` with periodic boundary conditions.

**The parametric setup:** instead of training one model per `nu`, the network
takes `(x, y, t, nu)` as input and learns the full family over
`nu in [0.001, 0.1]` (Re = 10 to 1000). After training, `predict --nu 0.05`
evaluates a never-trained viscosity instantly.

## 2. Method

### Viscosity as network input

Viscosity enters through its **log**, normalised to `[-1, 1]`:

```python
nu_norm = 2 * (log10(nu) - log10(0.001)) / (log10(0.1) - log10(0.001)) - 1
```

This is the right coordinate because the decay rate `exp(-2*nu*t)` depends
multiplicatively on `nu` — a factor-of-100 range in linear space becomes
uniform in log-space.

### Network architecture

```
Input: (x, y, t, nu_norm) -> 4D
PINN backbone: 5 hidden layers x 64 neurons (configurable)
Output: (u, v, p) -> 3D
```

### Losses

| Term | Enforces | Points |
|------|----------|--------|
| `physics` | NS momentum + continuity with **varying nu** per point | 10,000 random in (x,y,t,nu) box |
| `ic` | u,v,p match exact IC at t=0 (same for all nu) | 500 random (x,y,nu) |
| `bc` | Periodic BCs at domain boundaries | 200 random (t,nu) per boundary pair |

All collocation points sample `nu` log-uniformly. Each point in the physics
loss uses its own `nu` value — the network sees the full viscosity range in
every training batch.

### Evaluation

Held-out viscosities (never sampled explicitly during training):

| nu | Re | Regime |
|----|-----|--------|
| 0.002 | 500 | Low viscosity, slow decay |
| 0.015 | 67 | Moderate |
| 0.07 | 14 | High viscosity, fast decay |

Since the Taylor-Green vortex has an exact solution for any `nu`, we compute
true rel-L2 errors at every held-out viscosity — unlike parametric Burgers
which can only check IC error and residual.

## 3. Usage

```bash
uv run train-parametric-tg train                              # single model, 40k epochs
uv run train-parametric-tg train --ensemble 3                 # 3-member deep ensemble
uv run train-parametric-tg train -e 20000 -n 128 -l 6        # bigger network, fewer epochs
uv run train-parametric-tg predict --nu 0.05                  # solve never-trained Re=20
uv run train-parametric-tg predict --nu 0.0005                # extrapolation (warns)
uv run train-parametric-tg compare                            # rank all runs
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `40000` | Epochs per ensemble member |
| `--lr` | | float | `1e-3` | Learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `5` | Hidden layers |
| `--n-physics` | | int | `10000` | Collocation points in the (x,y,t,nu) box |
| `--ensemble` | | int | `1` | Ensemble members (>1 enables uncertainty) |
| `--seed` | | int | `42` | Base random seed (member i uses seed+i) |
| `--output-dir` | `-o` | str | auto | Artifact directory |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

### CLI reference — `predict`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nu` | | float | `0.01` | Viscosity to solve |
| `--run` | `-r` | str | latest | Run directory to load |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

## 4. Output

```
<run-dir>/
├── checkpoint.pt                # self-describing (config in metadata)
├── checkpoint_1.pt              # (ensemble only) additional members
├── metrics.json                 # per-nu errors + mean held-out metrics
├── loss_history.png
├── comparison.png               # PINN vs exact at a held-out nu
├── predictions.npz              # full fields at the predicted nu
├── prediction_comparison.png    # predict-time comparison plot
└── logs/run_*.log
```

## 5. What Makes This Interesting

### Exact validation at any nu

Unlike parametric Burgers (no closed form), we can compute exact rel-L2 errors
at every held-out viscosity. This makes it the cleanest benchmark for parametric
PINN accuracy across flow regimes.

### Physical intuition across the range

- **Low nu (high Re):** the vortex decays slowly, maintaining structure to t=1.
  The network must capture near-steady oscillatory fields.
- **High nu (low Re):** the vortex decays rapidly, becoming nearly zero by t=1.
  The network must capture the fast exponential decay.

A single model handles both extremes — the log-normalised viscosity input gives
it the right coordinate to interpolate between them.

### Connection to the single-instance experiment

`experiments/taylor_green/` trains one model at `nu=0.01`. This experiment
generalises that to the full `[0.001, 0.1]` range. The physics loss is
identical except `nu` varies per collocation point.

## 6. Honest Caveats

1. **This is the easiest parametric NS problem.** The Taylor-Green solution is
   separable and exponentially decaying — there are no shocks, boundary layers,
   or turbulent cascades. Real parametric NS problems are much harder.

2. **4D input space.** With `(x, y, t, nu)`, the network needs more capacity
   than the 3D single-instance version. Default 5x64 may be undersized for
   high accuracy; consider 6x128 for production runs.

3. **No turbulence transition.** The solution is laminar for all Re in this
   range. True Reynolds-sweep experiments would hit instabilities around
   Re ~ 1000 and require fundamentally different approaches.

## References

- Taylor, G.I. and Green, A.E. (1937). *Mechanism of the production of small
  eddies from large ones.* Proc. R. Soc. Lond. A.
