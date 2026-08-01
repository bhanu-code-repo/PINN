# Taylor-Green Vortex PINN — 2D Unsteady Navier-Stokes

**The simplest Navier-Stokes benchmark with an exact solution.** One model learns the 2D
incompressible NS flow for the Taylor-Green vortex, validated rigorously against the
closed-form solution for all time.

---

## 1. Problem Statement

```
u_t + u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
v_t + u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
u_x + v_y = 0   (incompressibility)
```

over `[0, 2pi]^2 x [0, 1]` with periodic BCs and `nu = 0.01`.

**Exact solution (valid for all t, any nu):**

```
u = -cos(x) sin(y) exp(-2*nu*t)
v =  sin(x) cos(y) exp(-2*nu*t)
p = -1/4 (cos(2x) + cos(2y)) exp(-4*nu*t)
```

The nonlinear terms cancel identically for this initial condition, so the exact solution
reduces to pure diffusion — but the PINN still evaluates the full NS residual, testing the
complete machinery.

## 2. Method

The network outputs `(u, v, p)` from inputs `(x, y, t)` — three channels from a standard
`PINN` backbone. Three loss terms:

| Term | Enforces | Points |
|------|----------|--------|
| `physics` | NS momentum (x, y) + continuity residuals | 10,000 random interior |
| `ic` | Exact solution at t = 0 | 500 random (x, y) |
| `bc` | Periodic BCs: solution matches at x = 0/2pi, y = 0/2pi | 200 random per boundary pair |

**Pressure validation:** pressure is defined up to a constant in incompressible NS. The
evaluation mean-subtracts both PINN and exact pressure before computing the rel-L2 error.

## 3. Usage

```bash
uv run train-taylor-green train                  # train with defaults (30k epochs)
uv run train-taylor-green train -e 5000 --nu 0.1 # quick test, higher viscosity
uv run train-taylor-green predict                 # evaluate latest run vs exact
uv run train-taylor-green compare                 # rank all runs
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `30000` | Training epochs |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `5` | Hidden layers |
| `--n-physics` | | int | `10000` | Interior collocation points |
| `--nu` | | float | `0.01` | Kinematic viscosity |
| `--seed` | | int | `42` | Random seed |
| `--output-dir` | `-o` | str | auto | Artifact directory |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

### CLI reference — `predict`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--run` | `-r` | str | latest | Run directory to load |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

## 4. Output

```
<run-dir>/
├── checkpoint.pt              # self-describing (config in metadata)
├── metrics.json               # losses + rel-L2 velocity/pressure vs exact
├── loss_history.png
├── comparison.png             # 2x3 grid: exact vs PINN velocity/pressure + error
├── predictions.npz            # full 3D arrays of (u,v,p) predicted and exact
└── logs/run_*.log
```

**What to look for:** the velocity field should show the characteristic four-vortex pattern
decaying smoothly. With `nu = 0.01`, the flow barely decays over `t in [0, 1]` (exp(-0.02)
~ 0.98), so the spatial structure is the dominant feature. Pressure should match after
mean-subtraction.

## 5. Why This Experiment

This is the entry point for Navier-Stokes PINNs in this repo:

1. **Exact solution** — rigorous validation (unlike cavity or wake problems)
2. **Full NS residual** — tests all derivative computations (u_xx, u_yy, p_x, etc.)
3. **Periodic BCs** — same BC type as the Schrodinger experiment
4. **Foundation** — the same `NavierStokesPINN` pattern (inputs -> u,v,p) is reused by the
   cavity and inverse experiments

## References

- Taylor, G.I. (1923). *On the decay of vortices in a viscous fluid.* Phil. Mag. 46.
- Green, G. (1838). *On the motion of waves in a variable canal.* Trans. Cambridge Phil. Soc.
- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
