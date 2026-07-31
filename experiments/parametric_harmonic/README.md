# Parametric Harmonic Oscillator PINN (+ Deep Ensembles)

One trained model that solves the **entire family** of damped harmonic oscillators — lifting the
classic PINN limitation of "one model = one problem instance" — with optional **deep-ensemble
uncertainty bands**. This experiment directly addresses limitations #2 and #3 from
[docs/prediction.md](../../docs/prediction.md).

Full method, design rules, and the measured single-instance-vs-parametric tradeoff table:
[docs/parametric_pinns.md](../../docs/parametric_pinns.md). The PDE counterpart is
[`experiments/parametric_burgers`](../parametric_burgers/README.md).

---

## 1. Problem Statement

The same ODE as `experiments/harmonic_oscillator`:

```
u'' + 2d·u' + w0²·u = 0,   u(0) = 1,  u'(0) = 0
```

but instead of fixing `(w0, d)`, the network learns the solution **as a function of the
parameters** over the box:

| Input | Range |
|-------|-------|
| `t`  | `[0, 1]` |
| `w0` | `[20, 100]` |
| `d`  | `[0.1, 4]` |

After one training run, **any** `(w0, d)` inside the box is solved by a forward pass —
`predict --w0 40 -d 1.5` answers a never-trained instance in milliseconds.

---

## 2. Method

### Parametric Ansatz

```
u(t; w0, d) = A(t, w0, d)·cos(ω·t) + B(t, w0, d)·sin(ω·t),   ω = √(w0² − d²)
```

- The **known damped frequency ω is embedded analytically** (computed from the inputs) — the
  backbone never has to fight spectral bias across a 20–100 rad/s frequency range.
- The backbone (`PINN`, 3 inputs → 2 outputs) learns only the two slowly-varying envelopes
  `A` and `B` — low-frequency functions across the whole box.
- Parameter inputs are **normalised to `[−1, 1]`** before the backbone (`w0` spans two orders
  of magnitude in `k = w0²`; unnormalised inputs saturate `tanh`).

### k-normalised residual

The raw residual scales with `k = w0² ∈ [400, 10⁴]`, so high-`w0` samples would dominate the
loss. The residual is therefore divided through by `k`:

```
r = u''/k + 2d·u'/k + u        →  O(1) across the whole parameter box
```

This removes the need for hand-tuned loss weights entirely (both terms use weight `1.0`).

### Losses

| Term | Definition | Collocation points |
|------|-----------|--------------------|
| `ic` | `mean[(u(0;w0,d) − 1)² + u'(0;w0,d)²]` | 200 random `(w0, d)` at `t = 0` |
| `physics` | `mean[r²]` (k-normalised) | 10,000 uniform-random points in the 3D `(t, w0, d)` box |

### Deep ensembles (`--ensemble N`)

`N > 1` trains N independent members from seeds `seed, seed+1, …`, saved as `checkpoint.pt`,
`checkpoint_1.pt`, …. At prediction time the member **mean** is the answer and **±2σ** of the
member spread is an epistemic-uncertainty band — where members disagree, trust less.

### Validation

Held-out `(w0, d)` combos never sampled explicitly during training — `(40, 1.5)`, `(90, 3)`,
`(25, 0.5)`, `(60, 2.5)` — each compared against the closed-form solution (rel-L2).

---

## 3. Usage

```bash
# Train the solution family (single model)
uv run train-parametric train

# Train a 5-member deep ensemble (uncertainty bands at predict time)
uv run train-parametric train --ensemble 5

# Solve NEVER-TRAINED instances in milliseconds — no retraining
uv run train-parametric predict --w0 40 -d 1.5
uv run train-parametric predict --w0 77.7 -d 0.33
uv run train-parametric compare
```

### CLI reference — `train`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | `40000` | Epochs **per ensemble member** |
| `--lr` | | float | `1e-3` | Adam learning rate |
| `--neurons` | `-n` | int | `64` | Neurons per hidden layer |
| `--layers` | `-l` | int | `4` | Number of hidden layers |
| `--n-physics` | | int | `10000` | Collocation points in the `(t, w0, d)` box |
| `--ensemble` | | int | `1` | Ensemble members (`>1` enables σ bands) |
| `--seed` | | int | `42` | Base seed (member *i* uses `seed + i`) |
| `--output-dir` | `-o` | str | auto | Artifact directory |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

### CLI reference — `predict`

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--w0` | | float | `55.0` | Natural frequency of the instance to solve |
| `--damping` | `-d` | float | `2.0` | Damping coefficient of the instance |
| `--run` | `-r` | str | latest | Run directory to load |
| `--n-points` | | int | `300` | Evaluation points |
| `--show/--no-show` | | flag | `--show` | Display plots interactively |

> Requesting `(w0, d)` **outside the trained box** triggers a warning: that is parameter-space
> extrapolation — limitation #1 of [docs/prediction.md](../../docs/prediction.md) in a new
> guise — and the result is unreliable.

---

## 4. Output

```
<run-dir>/
├── checkpoint.pt           # member 0 (self-describing: config in metadata)
├── checkpoint_1.pt ...     # further ensemble members (if --ensemble > 1)
├── metrics.json            # per-member losses + held-out rel-L2 per combo
├── loss_history.png        # member 0 loss curves
├── prediction.png          # written by `predict`: exact vs mean ± 2σ band
├── predictions.npz         # t, u_mean, u_std, u_exact, w0, d
└── logs/run_*.log
```

---

## 5. Honest Caveats

1. **Harder training than single-instance.** The network learns a 3D function family; expect
   longer runs and a larger network than `experiments/harmonic_oscillator`. If held-out rel-L2
   plateaus, raise `--epochs`, `--n-physics`, or capacity.
2. **Interpolation in parameter space only.** The box boundary is a hard validity edge — the
   same "no extrapolation" rule as the time domain.
3. **Ensemble σ is epistemic, not a guarantee.** Small σ means members *agree*, not that they
   are *right* — combine with the residual check (notebook 04) for the full picture.

---

## 6. References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
- Lakshminarayanan et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using
  Deep Ensembles.* NeurIPS.
- Lu et al. (2021). *Learning nonlinear operators via DeepONet.* Nature Machine Intelligence —
  the operator-learning generalisation of this idea.
