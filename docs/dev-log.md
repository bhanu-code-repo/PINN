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
**Commits:** `3f8baba` Production-harden pinn library: validation, NaN detection, LR scheduling, activations
           `bc8a7a8` Update pinn README to reflect production-hardening changes

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

### Phase 12 — GitHub Actions CI
**Commit:** `d419445` Add GitHub Actions CI: ruff lint + pytest on push and PR

- `.github/workflows/ci.yml` — two-job CI: `lint` (ruff check) and `test` (pytest).
- Triggers on push to `main`/`v*` branches and PRs to `main`.
- Uses `astral-sh/setup-uv@v6` for fast uv-based dependency resolution.
- Slow convergence tests excluded from CI (via `-m 'not slow'` in pytest config).

### Phase 13 — Convergence validation tests for NS experiments
**Commit:** `c754407` Add parametric Taylor-Green experiment and NS convergence tests

- 3 new `@pytest.mark.slow` convergence tests in `tests/test_convergence.py`:
  - `test_taylor_green_loss_drops` — 2000 epochs, asserts 10x loss reduction. Exercises
    full NS pipeline (momentum + continuity residual, IC, periodic BCs).
  - `test_lid_driven_cavity_loss_drops` — 2000 epochs, asserts 10x loss reduction.
    Exercises steady NS with hard-encoded wall BCs (mask-based).
  - `test_navier_stokes_inverse_re_moves_toward_truth` — 5000 epochs, asserts inferred
    Re moves closer to truth (Re=20) from initial guess (Re=10). Uses 3-layer 32-neuron
    network with data weight=10 to anchor the solution.
- 62 total tests (57 fast + 5 slow), up from 59 at Phase 12.

### Phase 14 — Parametric Taylor-Green (Reynolds sweep)
**Commit:** `c754407` Add parametric Taylor-Green experiment and NS convergence tests

- `experiments/parametric_taylor_green/` — parametric PINN for the Taylor-Green vortex
  over `nu in [0.001, 0.1]` (Re = 10 to 1000). Network takes `(x, y, t, log10(nu)_norm)`
  as input, outputs `(u, v, p)`. Log-uniform viscosity sampling during training.
- Exact closed-form validation at held-out viscosities (nu = 0.002, 0.015, 0.07) —
  the cleanest parametric NS benchmark (unlike Burgers, no residual-only validation).
- Full CLI: `train-parametric-tg train/predict/compare`. Ensemble support via `--ensemble`.
- Lifecycle test in `test_experiments_cli.py` including out-of-range warning check.
- 63 total tests (58 fast + 5 slow), up from 62 at Phase 13.

### Phase 15 — Streamlit dashboard
**Commits:** `1b9efe1` Add Streamlit dashboard for browsing runs and interactive parametric prediction
           `5882a99` Upgrade dashboard to premium UI: Plotly charts, custom CSS, icons
           `37f1d31` Light-only theme with brand #00205B, plot expand dialog
           `357e3eb` Fix sidebar heading color and increase nav font size

- `dashboard.py` — single-file Streamlit app for browsing and interacting with
  PINN training runs. Four pages:
  - **Overview** — summary stats bar (experiment count, total runs, best loss),
    experiment cards with Material Design icons, color-coded loss indicators.
  - **Run Detail** — config and metrics cards, interactive Plotly loss curves
    (zoom, hover, pan), artifacts listing with file-type icons, plot gallery
    with "Expand" button (opens full-width `st.dialog`).
  - **Compare** — metrics dataframe, overlaid Plotly loss curves across runs.
    Component losses togglable via legend clicks.
  - **Parametric Predictor** — interactive sliders for all 4 parametric experiments
    (harmonic, Burgers, Schrodinger, Taylor-Green). Loads trained checkpoints,
    runs inference on the fly, plots PINN prediction vs exact solution.
    `@st.cache_resource` for model caching. Loading spinners.
- Light theme with brand navy `#00205B`: gradient sidebar, tinted metric cards,
  branded headers. Custom CSS for card styling and typography.
- `plotly>=6.1.0` and `streamlit>=1.45.0` added to `pyproject.toml`.
- `docs/dashboard.md` — full usage guide: quick start, page descriptions,
  how it works, customisation.
- Launch: `uv run streamlit run dashboard.py`.

### Phase 16 — Inverse Navier-Stokes learning notebook

- `learn/09_inverse_navier_stokes.ipynb` — advanced deep-dive notebook (~45 min):
  - **Kovasznay flow** — exact steady Navier-Stokes solution for any Re, used as
    ground truth for inverse inference.
  - **Inverse PINN model** — `InverseNSPINN` with `log_re` as `nn.Parameter`,
    joint optimization of network weights + physical parameter.
  - **NS residual** — momentum-x, momentum-y, continuity terms computed via autograd.
  - **Training with two losses** — data loss (match sparse observations) + physics
    loss (NS residual), weighted combination.
  - **Re convergence visualization** — watch learned Re converge from wrong initial
    guess to true value over training.
  - **Field evaluation** — full u, v, p field comparison against exact solution.
  - **Pressure gauge invariance** — mean-subtract before comparison (pressure
    defined up to additive constant in incompressible NS).
  - **Streamfunction trick** — `u = psi_y`, `v = -psi_x` eliminates continuity
    equation by construction. `StreamfunctionPINN` class provided.
  - **6 exercises** including noise robustness, fewer observations, two-parameter
    inference, streamfunction training.
- `learn/README.md` — updated learning path table (9 notebooks, ~5 hours).
- `README.md` — updated directory tree and notebook count.

### Phase 17 — Residual-based Adaptive Refinement (RAR)
**Commit:** `02f2a59` Add Residual-based Adaptive Refinement (RAR) to pinn library
**PR:** #2 (feature/adaptive-collocation → main)

- `libs/pinn/src/pinn/rar.py` — new RAR module with two public functions:
  - `select_rar_points(model, candidates, residual_fn, n_select)` — scores candidate
    points by PDE residual magnitude, returns top-K. Handles multi-component residuals
    via L2-norm.
  - `adaptive_train(trainer, build_losses, residual_fn, ...)` — multi-phase training
    orchestrator: train for E epochs → evaluate residuals on dense candidate set →
    append top-K points to collocation set → rebuild loss closures → repeat for P phases.
- `experiments/burgers/train.py` — RAR integration as proof-of-concept:
  - New CLI flags: `--rar`, `--rar-phases` (default 5), `--rar-points` (default 500).
  - Refactored `build_losses` into `build_losses_with_points()` to support point set
    rebuilds between RAR phases.
  - Extracted `_pde_residual()` for shared use between training and RAR point selection.
- `libs/pinn/tests/test_rar.py` — 11 unit tests covering point selection (correct count,
  highest residuals selected, detached outputs, multi-dim input, multi-component residuals)
  and adaptive training (point growth per phase, loss history continuity, single-phase
  no-op, loss decrease, weight forwarding).
- `tests/test_experiments_cli.py` — 1 new CLI smoke test for `--rar` flag.
- 70 total fast tests (up from 58 pre-RAR), all passing.
- Exported `select_rar_points` and `adaptive_train` from `pinn` package `__init__.py`.

### Phase 18 — Weights & Biases integration
**Commits:** `7c3e83f` Add Weights & Biases integration with lazy imports
            `3286410` Add project overview for technical and non-technical audiences
**PR:** #3 (feature/wandb-integration → main)

- `libs/pinn/src/pinn/wandb_integration.py` — optional W&B logging module:
  - `wandb_init(project, config, name, tags, group)` — initialise a W&B run.
  - `wandb_callback(log_every, prefix)` — epoch callback that logs per-loss metrics.
  - `wandb_finish(run_dir, artifact_name)` — saves checkpoint, metrics, and plots
    as a W&B artifact and closes the run.
  - Lazy import via `_import_wandb()` — core library works without wandb installed;
    clear `ImportError` message if missing.
- `libs/pinn/src/pinn/__init__.py` — lazy `__getattr__` for `wandb_init`,
  `wandb_callback`, `wandb_finish` to avoid import-time dependency on wandb.
- `experiments/burgers/train.py` — new `--wandb` and `--wandb-project` CLI flags
  as proof-of-concept. Callback wired into both standard and RAR training paths.
- `pyproject.toml` — `wandb>=0.18.0` added as optional dependency
  (`[project.optional-dependencies] wandb`).
- `libs/pinn/tests/test_wandb.py` — 9 mock-based tests covering init, callback
  (every-epoch, log_every, prefix), finish (with/without artifacts, custom name),
  and ImportError messaging.
- `docs/project-overview.md` — balanced technical/non-technical project showcase
  covering all 11 experiments, 84 tests, learning curriculum, dashboard, and roadmap.
- 79 total fast tests (up from 70 pre-W&B), all passing.

### Phase 19 — Training Feedback Agent
**Commit:** `d363120` Add training feedback: health monitor, adaptive loss weighting, quality eval
**PR:** #4 (feature/feedback-agent → main)

Inspired by the Feedback Agent concept from Lang-PINN (He et al. 2025, ICLR 2026
Workshop Spotlight). Adapted as pure callbacks within the existing PINNTrainer.

- `libs/pinn/src/pinn/feedback.py` — three components:
  - `TrainingHealthMonitor` — epoch callback tracking loss smoothness
    (`1 - Std(ΔL)/Mean(L)`), gradient health (norm in `[ε, κ]` range),
    and convergence detection. `monitor.report()` returns full health summary.
  - `AdaptiveLossWeighter` — epoch callback that dynamically rebalances loss
    weights when one term dominates (max/min ratio > threshold). Rebalances
    inversely proportional to mean loss, with clamping to `[min_weight, max_weight]`.
    Mutates the `weights` dict in-place — trainer sees changes on the next epoch.
  - `evaluate_quality(loss_history)` — post-training scoring across three
    dimensions: effectiveness (final MSE via log-scale normalization), efficiency
    (convergence speed as fraction of total epochs), robustness (loss smoothness).
    Returns overall `quality_score` (weighted 0.4/0.3/0.3).
- `experiments/burgers/train.py` — new `--adaptive-weights` CLI flag. Health
  monitor always active, quality and health reports saved in `metrics.json`.
- `libs/pinn/tests/test_feedback.py` — 20 unit tests covering health monitor
  (loss tracking, smoothness stable/unstable, gradient health, convergence
  detection, report fields, trainer integration), adaptive weighter (balanced
  no-op, imbalanced rebalance, clamping, rebalance count, trainer integration),
  and quality evaluator (empty history, converged/non-converged, smooth/oscillating,
  field presence, score bounds).
- `tests/test_experiments_cli.py` — 1 new CLI smoke test for `--adaptive-weights`
  verifying quality and health reports in `metrics.json`.
- 100 total fast tests (up from 79 pre-feedback), all passing.

### Phase 20 — LLM Provider Library
**Commit:** `beca8e2` Add llm-provider library: LiteLLM-based LLM abstraction layer
**PR:** #5 (feature/llm-provider → main)

Foundation for Lang-PINN multi-agent framework. New workspace member
`libs/llm-provider/` providing a thin, LLM-agnostic client layer.

- `libs/llm-provider/src/llm_provider/config.py` — `LLMConfig` dataclass with
  resolution order: explicit kwargs > env vars > `.env` file > defaults.
  `LLMConfig.cloud()` for Ollama Cloud (auto-injects bearer auth header),
  `LLMConfig.local()` for localhost:11434. Any LiteLLM-supported provider
  (Anthropic, OpenAI, Vertex) works by changing `model` + `api_key` — no
  code changes required.
- `libs/llm-provider/src/llm_provider/client.py` — `LLMClient` with sync
  (`ask`, `ask_batch`) and async (`ask_async`, `ask_batch_async`) APIs.
  Supports system messages, streaming, and multi-turn conversations.
  `LLMClient()` / `.local()` / `.cloud()` convenience constructors.
- `libs/llm-provider/tests/` — 28 tests (14 config + 14 client), all mocked
  against LiteLLM so no API key needed to run.
- `pyproject.toml` — registered as workspace member, test path added.
- Default: Ollama Cloud `gpt-oss:120b`, reads `OLLAMA_API_KEY` from `.env`.
- 128 total fast tests (up from 100 pre-llm-provider), all passing.

### Phase 21 — Lang-PINN Multi-Agent Framework
**PR:** #6 (feature/lang-pinn → main)

LLM-guided PINN construction inspired by Lang-PINN (He et al. 2025).
Three operating modes: library (deterministic), code-agent (full LLM),
hybrid (LLM generates code targeting `pinn` library API with feedback loop).

- `libs/lang-pinn/` — new workspace member depending on `pinn` + `llm-provider`.
- `libs/lang-pinn/src/lang_pinn/schemas.py` — `PDESpec` (structured PDE
  representation: equation, variables, domain, conditions, parameters, feature
  flags) and `ArchitectureRec` (architecture recommendation with reasoning).
- `libs/lang-pinn/src/lang_pinn/agents/pde_agent.py` — PDE Agent: chain-of-thought
  LLM prompt → JSON extraction → validated `PDESpec`. Handles markdown fences,
  validates domain bounds, resolves defaults.
- `libs/lang-pinn/src/lang_pinn/agents/pinn_agent.py` — PINN Agent: rule-based
  architecture recommendation (ODE→compact, 1D PDE→medium, 2D→large), with
  feature-driven adjustments (high frequency → sinusoidal Ansatz, sharp gradients
  → more collocation). Optional LLM mode for unusual problems.
- `libs/lang-pinn/src/lang_pinn/agents/code_agent.py` — Code Agent: template-based
  (deterministic, always valid) or LLM-generated experiment code. Both modes
  target the `pinn` library API (`PINN`, `PINNTrainer`, `TrainingHealthMonitor`,
  `evaluate_quality`).
- `libs/lang-pinn/src/lang_pinn/orchestrator.py` — 3-mode orchestrator:
  - **Library mode**: PDE Agent (LLM) → PINN Agent (rules) → Code Agent (template).
  - **Code-Agent mode**: all agents use LLM.
  - **Hybrid mode**: LLM generates + executes + feedback loop (quality threshold
    with iterative refinement up to `max_iterations`).
- `libs/lang-pinn/tests/` — 47 tests: schemas (5), PDE Agent (9), PINN Agent (13),
  Code Agent (12), Orchestrator (8). All LLM calls mocked.
- 175 total fast tests (up from 128 pre-lang-pinn), all passing.

---

## Roadmap / Deferred

- **Lang-PINN enhancements** — SymPy-based PDE verification, more PDE templates,
  CLI entry point (`uv run lang-pinn solve "..."`), real-world integration tests.
- **Breather family `A*sech(x)`** — qualitative transitions across A; needs curriculum +
  Adam->L-BFGS. Documented as deferred research problem.
- **ONNX/FastAPI export** — serve trained models for real-time inference.
