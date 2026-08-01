# Parametric Burgers' Equation PINN (+ Deep Ensembles)

One trained model that solves Burgers' equation for the **whole viscosity family** — the PDE
counterpart of `experiments/parametric_harmonic`. Full method and tradeoff discussion:
[docs/parametric_pinns.md](../../docs/parametric_pinns.md).

---

## 1. Problem Statement

```
u_t + u·u_x − ν·u_xx = 0,   u(0,x) = −sin(πx),   u(t,±1) = 0
```

with the viscosity as a **network input** rather than a fixed constant:

| Input | Range | Sampling |
|-------|-------|----------|
| `x` | `[−1, 1]` | uniform |
| `t` | `[0, 1]` | uniform |
| `ν` | `[0.01/π, 0.1]` | **log-uniform** |

One checkpoint then covers the full physical spectrum — from the near-inviscid **sharp-shock**
regime (`ν = 0.01/π`) to the smooth **diffusive** regime (`ν = 0.1`):

```bash
uv run train-parametric-burgers predict --nu 0.05    # never-trained viscosity, milliseconds
```

## 2. Method

- **Model:** plain MLP `(x, t̃, ν̃) → u` (no ansatz needed). The viscosity enters as
  **normalised `log10(ν)`** — its physical effect (shock width ~ `1/ν`) is multiplicative, so
  log-space is the right coordinate; `t` is normalised to `[−1, 1]` alongside.
- **Losses** (all weight `1.0` — the Burgers residual is naturally O(1)):

  | Term | Points |
  |------|--------|
  | `ic` — `u(0,x;ν) = −sin(πx)` | 200 random `(x, ν)` |
  | `bc` — `u(t,±1;ν) = 0` | 200 random `(t, ν)` per side |
  | `physics` — full residual, per-sample `ν` | 10,000 random `(x, t, ν)` |

- **Validation without a closed form** — the two honest tools from
  [docs/prediction.md](../../docs/prediction.md), at held-out `ν ∈ {0.005, 0.02, 0.08}`:
  IC rel-L2 error and mean `|residual|`.
- **Deep ensembles:** `--ensemble N` trains N members (seeds `seed+i`); `predict` plots the
  member mean with a ±2σ band. Expect σ to concentrate near the shock at low `ν` — exactly
  where the family is hardest to pin down.

## 3. Usage

```bash
uv run train-parametric-burgers train                  # single model
uv run train-parametric-burgers train --ensemble 5     # with uncertainty bands
uv run train-parametric-burgers predict --nu 0.05
uv run train-parametric-burgers compare
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `40000` | Epochs per ensemble member |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `5` | Number of hidden layers |
| `--n-physics` | | int | `10000` | Collocation points in the `(x, t, ν)` box |
| `--ensemble` | | int | `1` | Ensemble members (`>1` enables σ bands) |
| `--seed` | | int | `42` | Base seed (member *i* uses `seed + i`) |
| `--output-dir` | `-o` | str | auto | Artifact directory |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

### CLI reference — `predict`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--nu` | float | `0.05` | Viscosity of the instance to solve |
| `--run` / `-r` | str | latest | Run directory to load |
| `--show/--no-show` | flag | `--show` | Display plots interactively |

> `ν` outside `[0.01/π, 0.1]` triggers a parameter-space extrapolation warning — the result
> is unreliable (limitation #1 of [docs/prediction.md](../../docs/prediction.md)).

## 4. Output

```
<run-dir>/
├── checkpoint.pt / checkpoint_1.pt ...   # self-describing ensemble members
├── metrics.json                          # per-member losses + held-out IC error + residuals
├── loss_history.png
├── prediction_contour.png                # written by predict: u(t,x) at the requested nu
├── prediction_snapshots.png              # t=0 (vs exact IC) and t=1 profiles, ±2σ bands
├── predictions.npz                       # grid mean/std + profiles
└── logs/run_*.log
```

## 5. Honest Caveats

1. **The low-ν corner is the hard part.** Shock sharpness ~ `1/ν`; the family's difficulty is
   concentrated at `ν → 0.01/π`. If the shock is smeared there, raise `--epochs`,
   `--n-physics`, or capacity — or narrow `NU_RANGE`.
2. **No closed form** → validation is IC error + residual only. A low residual *is* the
   correctness certificate here.
3. All general parametric caveats apply — see the tradeoff table in
   [docs/parametric_pinns.md](../../docs/parametric_pinns.md).

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
- Lakshminarayanan et al. (2017). *Deep Ensembles.* NeurIPS.
