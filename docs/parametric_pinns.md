# Parametric PINNs and Deep Ensembles

How this repo lifts two of the classic PINN limitations described in
[prediction.md](prediction.md): **one model = one problem instance** (limitation #2, fixed by
parametric PINNs) and **no uncertainty awareness** (limitation #3, mitigated by deep
ensembles).

Implementations:

- [`experiments/parametric_harmonic`](../experiments/parametric_harmonic/README.md) — ODE
  family over `(w0, d)`, validated against the closed-form solution
- [`experiments/parametric_burgers`](../experiments/parametric_burgers/README.md) — PDE family
  over viscosity `ν`, validated via IC error + residual (no closed form exists)

---

## 1. The Idea: Parameters Become Inputs

A standard PINN bakes the physics parameters into the **loss**:

```
single-instance:   NN(t) -> u        residual uses fixed w0=80, d=2
```

A parametric PINN moves them into the **input**:

```
parametric:        NN(t, w0, d) -> u     residual uses each sample's own (w0, d)
```

Collocation points are sampled over the full box `t × w0 × d`, so the network is forced to
satisfy the ODE **for every parameter combination simultaneously**. The result is one
checkpoint that encodes the whole solution *family* — any `(w0, d)` inside the box is solved
by a forward pass:

```bash
uv run train-parametric predict --w0 40 -d 1.5     # never trained on this — milliseconds
uv run train-parametric-burgers predict --nu 0.05  # same idea for a PDE
```

## 2. Design Rules That Make It Work

These came out of building the two experiments; they generalise.

**Rule 1 — Normalise parameter inputs.** `w0 ∈ [20, 100]` saturates `tanh` units; the
harmonic experiment maps parameters to `[−1, 1]`. When a parameter acts *multiplicatively*
(viscosity: shock width ~ `1/ν`), feed the network its **log** — the Burgers experiment uses
normalised `log10(ν)`, and samples `ν` log-uniformly for the same reason.

**Rule 2 — Normalise the residual.** The raw harmonic residual scales with `k = w0² ∈ [400,
10⁴]`: high-`w0` samples would dominate the loss 25:1. Dividing the ODE through by `k` makes
the residual O(1) across the entire box — and removes hand-tuned loss weights entirely.
Whenever your parameter multiplies a dominant term, divide it out.

**Rule 3 — Embed known structure analytically.** The harmonic ansatz is
`A(t,w0,d)·cos(ωt) + B(t,w0,d)·sin(ωt)` with `ω = √(w0²−d²)` **computed from the inputs**, not
learned. The backbone only learns two slowly-varying envelopes — otherwise it would have to
defeat spectral bias across a 20–100 rad/s frequency range simultaneously, which in practice
does not converge. Use every closed-form fact you have.

**Rule 4 — Validate on held-out parameter combos.** Both experiments evaluate at parameter
values never explicitly sampled during training — that is the honest test of family learning
(`(40,1.5), (90,3), …` for harmonic; `ν ∈ {0.005, 0.02, 0.08}` for Burgers). Without a closed
form (Burgers), use the IC error plus the **residual check** from
[prediction.md](prediction.md).

## 3. The Tradeoff Table — Parametric vs Single-Instance

Measured on this repo's experiments (CPU, Apple Silicon):

| | Single-instance | Parametric |
|---|---|---|
| **Training cost** | ~20 s (harmonic, 15k epochs) | 10–50× more: 3D collocation box, larger net, more epochs (×N again for ensembles) |
| **Accuracy per instance** | excellent — rel-L2 **1.3e-4** (harmonic, converged) | good, not great — capacity is *shared* across the family. Measured at 8k epochs (defaults are 40k): harmonic held-out rel-L2 **0.08–0.40**; Burgers held-out IC rel-L2 **0.009–0.067** with residuals 0.013–0.067, worst in the sharp-shock low-ν corner |
| **New problem instance** | full retrain (~minutes) | **milliseconds** (one forward pass) |
| **Validity region** | one point in parameter space | a box — with hard edges (see below) |
| **Loss weighting effort** | hand-tuned per problem (`1e-4` physics weight) | eliminated by residual normalisation (Rule 2) |
| **Uncertainty** | residual check only | residual check **+** ensemble σ |
| **Break-even point** | you need 1–2 instances | worth it from ~3+ instances, or *any* parameter sweep / design-loop / inverse-search workload |

**The core trade:** you pay a large one-time training cost and accept lower per-instance
accuracy, in exchange for **amortised, instant evaluation across the whole family**. Choose
parametric when the *family* is the object of study (design optimisation, sensitivity
analysis, real-time applications); choose single-instance when you need maximum accuracy on
one configuration.

**The boundary caveat:** the parameter box has hard validity edges — `w0 = 150` or `ν = 0.5`
outside the trained range is limitation #1 of [prediction.md](prediction.md) (no
extrapolation) in a new guise. Both `predict` CLIs warn when you cross it.

## 4. Deep Ensembles: Honest (Epistemic) Uncertainty

A single PINN gives one answer with no error bars. A **deep ensemble** trains N independent
members from different seeds (`--ensemble N` → seeds `seed, seed+1, …`, saved as
`checkpoint.pt`, `checkpoint_1.pt`, …). At prediction time:

- **mean** of the members = the prediction
- **±2σ** of the member spread = the epistemic-uncertainty band

Where the loss landscape constrains the solution tightly, independently-trained members agree
(small σ); where the data/physics under-constrain it — near shock fronts, box edges, sparse
collocation regions — they diverge (large σ).

**What σ means — and does not mean:**

| σ tells you | σ does NOT tell you |
|---|---|
| Where independently trained models *disagree* (epistemic) | That the mean is *correct* — all members can share a bias |
| Which regions are under-constrained by the loss | Aleatoric noise (there is none in a deterministic ODE/PDE) |
| Where to add collocation points or capacity | A calibrated probability — 2σ is a heuristic band, not a guarantee |

**The two-diagnostic principle.** Combine ensemble σ with the pointwise residual for two
*independent* trust signals:

| Ensemble σ | Residual | Verdict |
|---|---|---|
| small | small | trustworthy — constrained *and* physics-consistent |
| small | large | shared systematic failure (e.g. under-resolved shock) — σ alone would mislead you |
| large | small | members found different near-solutions — under-constrained region |
| large | large | invalid region (e.g. outside the domain/box) |

Cost note: ensembles multiply training time by N (members are independent — trivially
parallelisable across machines) but predictions stay effectively instant.

## 5. When to Reach for What

```
one configuration, maximum accuracy      -> single-instance PINN
3+ configurations, sweeps, design loops  -> parametric PINN
need error bars on top                   -> + --ensemble 5
parameter far outside any trained box    -> retrain (no free lunch)
```

## 6. Deferred: Parametric Schrödinger

Deliberately not implemented yet — for honest reasons, not oversight:

1. The NLS as posed (`i·h_t + ½·h_xx + |h|²·h = 0`) has **no free physical parameter**; the
   natural family is the IC amplitude `h(0,x) = A·sech(x)`.
2. That family is *qualitatively* diverse: `A=1` is a fundamental soliton (shape-invariant),
   `A=2` a breather (periodic shape oscillation) — the hardest kind of family for one network
   to represent, compounded by complex output and periodic BCs.
3. The right sequence is: validate the parametric pattern on Burgers (PDE, done), then attempt
   NLS with a dedicated effort — likely needing larger capacity, curriculum over `A`, and the
   Adam→L-BFGS two-stage recipe.

## 7. Beyond: Operator Learning

Parametric PINNs interpolate over a low-dimensional parameter box. The general version —
learning the map from *functions* (arbitrary ICs, coefficient fields) to solutions — is
**operator learning**: DeepONet (Lu et al., 2021) and Fourier Neural Operators (Li et al.,
2021). Same trade, larger scale: expensive training, instant amortised inference.

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
- Lakshminarayanan, Pritzel, Blundell (2017). *Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles.* NeurIPS.
- Lu, Jin, Pang, Zhang, Karniadakis (2021). *Learning nonlinear operators via DeepONet.*
  Nature Machine Intelligence.
- Li et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR.
