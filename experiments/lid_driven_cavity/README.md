# Lid-Driven Cavity PINN — 2D Steady Navier-Stokes

**The standard 2D benchmark for incompressible NS solvers.** Steady-state flow in a
square cavity driven by a sliding lid, validated against the Ghia, Ghia & Shin (1982)
tabulated centreline velocities at Re = 100.

---

## 1. Problem Statement

```
u*u_x + v*u_y = -p_x + (1/Re)*(u_xx + u_yy)
u*v_x + v*v_y = -p_y + (1/Re)*(v_xx + v_yy)
u_x + v_y = 0
```

over `[0, 1]^2` (steady — no time dimension), Re = 100.

**Boundary conditions:**

| Wall | u | v |
|------|---|---|
| Top (y = 1) | 1 | 0 |
| Bottom (y = 0) | 0 | 0 |
| Left (x = 0) | 0 | 0 |
| Right (x = 1) | 0 | 0 |

## 2. Method

### Hard-encoded boundary conditions

The model embeds three of the four walls analytically:

```python
mask = x * (1 - x) * y          # vanishes at x=0, x=1, y=0
u = mask * NN_u + y^10           # satisfies u=0 on three walls, ramps to ~1 at y=1
v = mask * (1 - y) * NN_v       # vanishes on all four walls
```

The `y^10` term provides a smooth ramp that is effectively zero except very near y = 1,
where it approaches the lid velocity u = 1. The soft `bc_lid` loss reinforces the lid
condition during early training when the network is far from the solution.

### Losses

| Term | Enforces | Points |
|------|----------|--------|
| `physics` | Steady NS momentum + continuity | 10,000 random interior |
| `bc_lid` | u = 1, v = 0 at y = 1 (soft reinforcement) | 200 random on lid |

Wall BCs on x = 0, x = 1, y = 0 are satisfied by construction (hard BC).

### No exact solution — Ghia benchmark

The cavity has no closed-form solution. Validation uses the Ghia et al. (1982) tabulated
velocities at 17 points along each centreline:

- **u** along the vertical centreline (x = 0.5)
- **v** along the horizontal centreline (y = 0.5)

The rel-L2 error against these benchmark values is the primary accuracy metric.

## 3. Usage

```bash
uv run train-cavity train                   # train with defaults (30k epochs, Re=100)
uv run train-cavity train -e 50000 --re 400 # higher Re (harder — sharper gradients)
uv run train-cavity predict                  # evaluate latest run vs Ghia
uv run train-cavity compare                  # rank all runs
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `30000` | Training epochs |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `5` | Hidden layers |
| `--n-physics` | | int | `10000` | Interior collocation points |
| `--re` | | float | `100` | Reynolds number |
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
├── checkpoint.pt               # self-describing
├── metrics.json                # losses + rel-L2 vs Ghia for u and v
├── loss_history.png
├── cavity_results.png          # centreline profiles vs Ghia + 2D contours
├── predictions.npz             # 2D fields + centreline data + Ghia reference
└── logs/run_*.log
```

**What to look for:** the centreline profiles should pass through the Ghia benchmark
points. At Re = 100, the primary vortex is centred slightly above the geometric centre
with a clear recirculation pattern. Higher Re pushes the vortex centre downward and
creates secondary corner vortices (harder for the PINN).

## 5. Why This Experiment

1. **Industry-standard benchmark** — every NS solver is validated against Ghia
2. **Steady state** — no time marching, simpler than Taylor-Green but with richer physics
3. **Hard BCs** — demonstrates the mask technique for embedding Dirichlet conditions
4. **No exact solution** — forces validation against external benchmark data (a realistic
   scenario for most practical PINN applications)
5. **Reynolds number sensitivity** — Re = 100 is tractable; Re = 400+ shows where PINNs
   struggle (thin boundary layers near the driven lid)

## References

- Ghia, Ghia & Shin (1982). *High-Re solutions for incompressible flow using the
  Navier-Stokes equations and a multigrid method.* J. Comput. Phys. 48.
- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
