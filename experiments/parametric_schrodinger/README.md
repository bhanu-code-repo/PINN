# Parametric Schrödinger PINN — The Fundamental Soliton Family (+ Deep Ensembles)

**How to train a complex-valued parametric PINN.** One model learns the nonlinear Schrödinger
soliton family over the amplitude `A` — with **closed-form validation for every member**,
thanks to a deliberate choice of the parametric family. Full method context:
[docs/parametric_pinns.md](../../docs/parametric_pinns.md).

---

## 1. Problem Statement — and Why This Family

```
i·h_t + ½·h_xx + |h|²·h = 0,     h(0, x) = A·sech(A·x)
```

over `x ∈ [−5, 5]`, `t ∈ [0, π/2]`, `A ∈ [0.75, 2]`, with periodic BCs on value and derivative.

The IC `A·sech(A·x)` (not `A·sech(x)`!) is chosen deliberately: it is the **fundamental
soliton** initial condition, and every member of the family has the exact solution

```
h(t, x) = A·sech(A·x) · e^(i·A²·t/2)
```

— a shape-invariant envelope with a pure phase rotation. Two consequences:

1. **No qualitative transitions across the family** (unlike `A·sech(x)`, where `A=2` becomes a
   breather — see "deferred" below).
2. **Rigorous validation everywhere**: full-complex rel-L2 against the exact solution at
   held-out amplitudes, just like the parametric harmonic oscillator.

## 2. Method

### Phase-embedded ansatz (Rule 3 of the parametric doc)

```
h(x, t; A) = W(x, t, A) · e^(i·A²·t/2)
```

The known phase rotation is computed analytically from the inputs; the backbone
(`PINN`, 3 inputs → 2 channels = Re/Im of `W`) only has to learn `W`. For the exact solution,
`W = A·sech(A·x)` — **real and time-independent**. The learning target is a nearly-static
envelope instead of a rotating complex field, which is what makes this tractable.

Inputs are normalised to `[−1, 1]` (Rule 1) before the backbone.

### Losses (all weight `1.0`)

| Term | Enforces | Points |
|------|----------|--------|
| `ic` | `h(0,x;A) = A·sech(A·x)` (real) | 300 random `(x, A)` |
| `bc` | `h` and `h_x` periodic at `x = ±5` | 200 random `(t, A)` |
| `physics` | `mean|i·h_t + ½·h_xx + \|h\|²·h|²`, complex residual | 10,000 random `(x, t, A)` |

The complex residual is assembled from real/imaginary derivatives exactly as in the
single-instance `experiments/schrodinger` — that file is the reference for the technique; this
one adds the parametric input and the analytic phase factor.

### Deep ensembles

`--ensemble N` trains N members (seeds `seed+i`); `predict` plots the `|h|` mean with ±2σ
bands and reports the full-complex rel-L2 against the exact soliton.

## 3. Usage

```bash
uv run train-parametric-schrodinger train                  # single model
uv run train-parametric-schrodinger train --ensemble 5     # with uncertainty bands
uv run train-parametric-schrodinger predict -a 1.3         # never-trained amplitude
uv run train-parametric-schrodinger compare
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `40000` | Epochs per ensemble member |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `4` | Number of hidden layers |
| `--n-physics` | | int | `10000` | Collocation points in the `(x, t, A)` box |
| `--ensemble` | | int | `1` | Ensemble members (`>1` enables σ bands) |
| `--seed` | | int | `42` | Base seed (member *i* uses `seed + i`) |
| `--output-dir` | `-o` | str | auto | Artifact directory |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

### CLI reference — `predict`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--amplitude` | `-a` | float | `1.3` | Soliton amplitude of the instance |
| `--run` | `-r` | str | latest | Run directory to load |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

> `A` outside `[0.75, 2]` triggers a parameter-space extrapolation warning. The lower bound
> also keeps the soliton tails small at `x = ±5` so the periodic BC is physically consistent.

## 4. Output

```
<run-dir>/
├── checkpoint.pt / checkpoint_1.pt ...   # self-describing ensemble members
├── metrics.json                          # per-member losses + held-out complex rel-L2
├── loss_history.png
├── prediction_contour.png                # |h(t,x)| at the requested A — should be STATIC in t
├── prediction_snapshots.png              # |h| at t=0 and t=π/4 vs exact envelope, ±2σ bands
├── predictions.npz                       # grid mean/std of Re h, Im h, |h|
└── logs/run_*.log
```

**What to look for:** the exact `|h| = A·sech(A·x)` is *time-independent* — any breathing or
drift in the contour plot is pure model error, which makes this family unusually easy to
inspect visually.

## 5. Honest Caveats — and What Remains Deferred

1. **This is the soliton family, not the general amplitude family.** The original
   single-instance experiment (`experiments/schrodinger`) uses `2·sech(x)` — a *breather* with
   periodic shape oscillations. The parametric **breather family** `A·sech(x)` remains
   deferred: qualitative solution changes across `A` make it a genuinely hard research
   problem (likely needing curriculum over `A` and Adam→L-BFGS refinement).
2. **Phase errors grow with `A²·t`** — the embedded rotation removes most of this, but
   residual phase drift is the dominant error mode at high `A`; the full-complex rel-L2
   (not just `|h|`) is the honest metric and is what we report.
3. All general parametric caveats apply — see the tradeoff table in
   [docs/parametric_pinns.md](../../docs/parametric_pinns.md).

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
- Sulem & Sulem (1999). *The Nonlinear Schrödinger Equation.* Springer — soliton solutions.
- Lakshminarayanan et al. (2017). *Deep Ensembles.* NeurIPS.
