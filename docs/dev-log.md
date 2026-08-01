# Development Log

Timestamped progress tracker for the PINN monorepo. Each entry records what
was built, key decisions made, and the commit that landed it.

---

## 2026-07-30

### Phase 0 — Initial scaffolding
**Commit:** `06c6b18` Initial commit

- Empty repo with `.gitignore`, basic structure.

---

## 2026-07-31

### Phase 1 — Core library + first three experiments
**Commit:** `622bce9` Add PINN monorepo: core library, three experiments, notebooks, and docs

- Built `libs/pinn` core library: `PINN` MLP backbone (`tanh` activation for smooth
  higher-order autograd), `PINNTrainer` (weighted multi-term losses, tqdm progress bar).
- Three single-instance experiments:
  - `harmonic_oscillator` — damped ODE, learnable sinusoidal Ansatz to defeat spectral bias.
  - `burgers` — nonlinear PDE, shock formation at low viscosity.
  - `schrodinger` — complex-valued PDE (`h = u + iv`), periodic BCs, `2*sech(x)` breather IC.
- Four walkthrough notebooks (01–04): harmonic deep-dive, Burgers, Schrodinger, prediction.
- `pyproject.toml` uv workspace with `libs/pinn` as a member.
- Production READMEs for each experiment + root README.

### Phase 2 — Production hardening
**Commit:** `dd0edaa` Production-harden library and experiments: logging, checkpointing, artifacts

- Replaced print-based logging with **loguru** (tqdm-safe console sink + rotating file sinks).
- Added `set_seed()` for reproducibility (random/numpy/torch/CUDA).
- Added `setup_logging()` with configurable log levels and file output.
- Structured artifact directories: each run writes to `outputs/<experiment>/<timestamp>/`.
- Shared infrastructure in `experiments/common.py`: `init_run`, `save_metrics`, `show_banner`,
  `print_summary`, `get_device`.
- Ruff linting with `select = [E, W, F, I, UP, B, SIM]`.

### Phase 3 — Predict/compare commands + self-describing checkpoints
**Commit:** `f9a1b75` Add predict/compare commands and self-describing checkpoints

- Multi-command Typer CLIs: `train`, `predict`, `compare` for every experiment.
- Self-describing checkpoints: training config stored in `metadata` field, `build_model(config)`
  factory reconstructs architecture from checkpoint — no hyperparameters need remembering.
- `predict` loads latest run by default, evaluates, writes `predictions.npz` + plots.
- `compare` ranks all runs of an experiment by `metrics.json`.
- `load_model(run_dir, build_model)` shared utility in `common.py`.

### Phase 4 — Test suite
**Commit:** `f391306` Add test suite: unit, convergence regression, and CLI lifecycle tests

- `libs/pinn/tests/` — 20 unit tests: network shapes/gradients, trainer mechanics (weighted
  losses, early stopping, grad clipping, callbacks), checkpoint round-trip, seeding, plotting.
- `tests/test_experiments_cli.py` — full `train -> predict -> compare` lifecycle per experiment
  via Typer `CliRunner` (in-process, tiny epoch counts, temp dirs).
- `tests/test_convergence.py` — `@pytest.mark.slow`: exponential decay rel-L2 < 5%, harmonic
  Ansatz loss drops 100x.
- `conftest.py` with Agg backend, seeded fixtures, `tiny_model` fixture.

### Phase 5 — Prediction concept doc + notebook
**Commit:** `a5d1ed9` Add prediction concept doc and model-as-solution notebook

- `docs/prediction.md` — "How Prediction Works in a PINN": model IS the solution function.
  Three legitimate expectations (mesh-free eval, free derivatives, residual self-check).
  Three limitations (no extrapolation, one model per instance, no uncertainty) — limitations
  #2 and #3 annotated as "addressed in this repo" pointing to upcoming parametric experiments.
- `notebooks/04_model_as_solution.ipynb` — hands-on demo: 10k-point evaluation, autograd
  derivatives (u', u''), pointwise residual self-check, extrapolation failure (26x blow-up).

### Phase 6 — Parametric PINNs + deep ensembles
**Commit:** `7311585` Add parametric PINN experiment with deep-ensemble uncertainty

- `experiments/parametric_harmonic/` — parameters `(w0, d)` become network inputs. One model
  learns the whole solution family `u(t; w0, d)` over `w0 in [20, 100]`, `d in [0.1, 4]`.
- Learnable sinusoidal Ansatz generalised across the parameter family:
  `u = A(t,w0,d)*cos(wt) + B(t,w0,d)*sin(wt)` with `w = sqrt(w0^2 - d^2)`.
- Four design rules established:
  1. Normalise parameter inputs to [-1, 1].
  2. Normalise residual by dominant scaling (divide ODE by k = w0^2).
  3. Embed known structure analytically (frequency, phase).
  4. Validate on held-out parameter combos.
- Deep ensembles: `--ensemble N` trains N independent members from different seeds.
  Prediction: mean +/- 2sigma epistemic uncertainty band.
- Out-of-box extrapolation warnings in `predict`.

**Commit:** `d6bd897` Add parametric Burgers experiment and parametric PINN documentation

- `experiments/parametric_burgers/` — PDE family over viscosity nu (log-uniform sampling,
  normalised log10(nu) input). No closed form; validated via IC error + residual.
- `docs/parametric_pinns.md` — full method documentation: 4 design rules, measured tradeoff
  table (single-instance vs parametric), deep ensemble semantics (sigma-vs-residual 4-quadrant
  decision table), operator learning forward-look.

**Commit:** `ea7f727` Add parametric Schrodinger: complex parametric PINN via the soliton family

- `experiments/parametric_schrodinger/` — complex-valued parametric PINN over soliton amplitude
  `A`. IC `A*sech(A*x)` chosen for exact solution `h = A*sech(A*x)*exp(i*A^2*t/2)`.
- Phase-embedded ansatz: `h = W(x,t,A) * exp(i*A^2*t/2)`. Backbone target is real and
  time-independent — the best-converging parametric experiment (~1% rel-L2 at 8k epochs).
- Breather family `A*sech(x)` documented as deferred (qualitative transitions across A).
- Convergence validated at 8k epochs: harmonic 0.08-0.40, Burgers 0.009-0.067,
  Schrodinger 0.010-0.014 rel-L2.

### Phase 7 — Navier-Stokes experiments
**Commit:** `b4c0231` Add Navier-Stokes experiments: Taylor-Green, cavity, Kovasznay inverse, Raissi cylinder wake

Four NS experiments covering the full spectrum:

- `experiments/taylor_green/` — 2D unsteady NS, exact closed-form solution.
  Direct `(u, v, p)` output from `(x, y, t)`. Full NS residual (momentum + continuity),
  periodic BCs. Pressure validated mean-subtracted (gauge invariance). nu = 0.01.

- `experiments/lid_driven_cavity/` — 2D steady NS, Ghia et al. (1982) benchmark.
  Hard-encoded wall BCs via mask `x(1-x)y` — three walls satisfied by construction.
  Ghia tabulated centreline velocities (17 points each) hardcoded for validation.

- `experiments/navier_stokes_inverse/` — inverse NS, infer Re from noisy synthetic data.
  Kovasznay flow (exact steady NS solution). Learnable `log_Re` parameter optimised jointly
  with network weights. Data loss + physics loss with `nu = 1/model.re`.

- `experiments/cylinder_wake/` — Raissi et al. (2019) benchmark. Real DNS data from
  `cylinder_nektar_wake.mat` (5000 spatial x 200 time = 1M points). Streamfunction
  formulation: `u = psi_y, v = -psi_x` for exact incompressibility. Learnable lambda_1,
  lambda_2. Hidden pressure recovery (pressure never observed during training). 8-layer
  network. scipy added for .mat loading.

- Test for cylinder wake skips gracefully if data file is missing.

### Phase 8 — Best-model saving + experiment quickstart guide
**Commit:** `e99a7ac` Add best-model checkpointing, experiment quickstart guide, and dev log

- `PINNTrainer.train()` — new `save_best` and `restore_best` parameters. Saves best model
  weights (lowest total loss) during training; restores them at end so `checkpoint.pt`
  contains the best model, not the last. All 10 experiments updated.
- `docs/adding_experiments.md` — full step-by-step guide for adding new experiments: complete
  `train.py` template, test pattern, README structure, shared infrastructure reference,
  core library reference, design patterns (ansatz, hard BCs, streamfunction, parametric,
  learnable parameters, residual normalisation).
- 2 new trainer unit tests for best-model feature.
- 34 tests pass total (up from 28 at Phase 4).

### Phase 9 — Learn PINN curriculum
**Commit:** `86cd477` Add learn/ curriculum: 8 progressive notebooks for learning PINNs from scratch

- `learn/` directory: 8 progressive Jupyter notebooks (~4 hours total) teaching PINNs from
  first principles. No prior PINN experience required.
- Notebooks: (01) what PINNs are, (02) autograd deep dive, (03) first PINN from scratch
  (u'=-u), (04) data vs physics vs hybrid (the aha moment), (05) PDEs and BCs (Burgers),
  (06) training tricks (weighting, spectral bias, Ansatz), (07) parametric and inverse PINNs,
  (08) honest assessment with decision framework.
- `learn/README.md` — course overview, learning path table, prerequisites, relationship to
  production code.
- Root README updated with `learn/` in directory tree and "Learn PINNs" section.
- `learn/` added to ruff's extend-exclude.

## 2026-08-01

### Phase 10 — L-BFGS two-stage training
**Commits:** `455ab2d` Add L-BFGS two-stage training support for inverse problems
           `f09b4f2` Update READMEs with L-BFGS two-stage training documentation

- `PINNTrainer.train()` — automatic L-BFGS support. Detects closure-based optimizers
  (via `_is_closure_optimizer`) and uses `optimizer.step(closure)` pattern instead of
  the standard `zero_grad → backward → step` flow. Both optimizer types support all
  existing features (best-model saving, early stopping, callbacks, grad clipping).
- `experiments/cylinder_wake/train.py` — two-stage training: `--lbfgs-epochs` and
  `--lbfgs-lr` CLI flags. Adam for initial convergence, L-BFGS (strong Wolfe line search,
  history_size=50) for refinement. Following Raissi et al. (2019) methodology.
- `experiments/navier_stokes_inverse/train.py` — same two-stage pattern for Re inference.
- 3 new trainer unit tests: L-BFGS basic, L-BFGS with save_best, two-stage Adam→L-BFGS.
- 37 tests pass total (up from 34 at Phase 9).
- Updated READMEs: `libs/pinn/README.md`, `experiments/cylinder_wake/README.md`,
  `experiments/navier_stokes_inverse/README.md`, `docs/adding_experiments.md` — all
  reflect L-BFGS support, two-stage training examples, and updated CLI reference tables.

### Phase 11 — Production hardening of `libs/pinn`
**Pending commit**

- **Input validation** — `PINN.__init__` validates all dimension args >= 1 and activation
  name. `PINNTrainer.__init__` validates model is `nn.Module`. `train()` validates
  `n_epochs >= 1`, non-empty `loss_functions`, `grad_clip > 0`, `early_stop_patience >= 1`.
- **NaN/Inf detection** — training loop checks `math.isfinite(total_loss)` each epoch;
  logs error and stops immediately on divergence.
- **LR scheduler support** — new `scheduler` parameter on `train()`. Auto-detects
  `ReduceLROnPlateau` vs standard schedulers (StepLR, CosineAnnealingLR, etc.).
- **Custom activations** — `PINN` now accepts `activation` parameter (`tanh`, `silu`,
  `gelu`). Registry-based with `_ACTIVATIONS` dict.
- **Xavier initialisation** — `_init_weights()` applies Xavier uniform to all Linear layers.
- **`__version__`** — `pinn.__version__ = "0.1.0"` exported in `__all__`.
- **Checkpoint error handling** — `save_checkpoint` catches `OSError` with logging,
  `load_checkpoint` raises `FileNotFoundError` for missing files, wraps corrupt files.
- **Module docstrings + copyright headers** — all 6 implementation files and 2 test files.
- **`-> None` return annotations** — plotting functions now fully typed.
- **Utility methods** — `PINN.count_parameters()`, improved `__repr__` showing dims +
  activation + param count.
- **Enhanced logging** — periodic debug log now includes current learning rate.
- 57 tests pass total (up from 37 at Phase 10): 8 new tests for validation, NaN detection,
  LR scheduling, checkpoint errors, activations.

---

## Roadmap / Deferred

- **GitHub Actions CI** — run `pytest` + `ruff check` on push.
- **Inverse-problem notebook** — walkthrough of the data-loss term methodology.
- **Breather family `A*sech(x)`** — qualitative transitions across A; needs curriculum +
  Adam->L-BFGS. Documented as deferred research problem.
- **Convergence validation runs** — medium background runs for NS experiments (Taylor-Green,
  cavity, cylinder wake) to establish baseline numbers before committing to docs.
- **Parametric NS** — Taylor-Green parametric in nu (Reynolds sweep), reusing the
  parametric PINN pattern.
