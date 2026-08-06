# PINN — Physics-Informed Neural Networks

A [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo for solving differential equations with **Physics-Informed Neural Networks** (PINNs) in PyTorch. A shared core library provides the network backbone and training loop; each experiment defines a specific ODE/PDE problem on top of it.

## What is a PINN?

A PINN approximates the solution `u` of a differential equation with a neural network. Instead of training on labelled data, the loss penalises:

1. **Physics residual** — the equation itself, evaluated at collocation points using exact derivatives from automatic differentiation.
2. **Initial / boundary conditions** — mismatch at the domain boundary.

Minimising the weighted sum drives the network toward a function that satisfies the equation everywhere.

## Repository Structure

```
PINN/
├── libs/
│   ├── pinn/                     # Core library (workspace member) — see libs/pinn/README.md
│   │   └── src/pinn/
│   │       ├── core/             # PINN MLP backbone
│   │       ├── trainer/          # Generic multi-loss trainer
│   │       ├── rar.py            # Residual-based Adaptive Refinement
│   │       ├── feedback.py       # Training health monitor, adaptive weights, quality eval
│   │       └── utils/            # Plotting, logging, seeding
│   ├── llm-provider/             # LLM abstraction layer — see libs/llm-provider/README.md
│   │   └── src/llm_provider/
│   │       ├── config.py         # LLMConfig: env/.env/kwargs resolution, Ollama Cloud auth
│   │       └── client.py         # LLMClient: sync + async, streaming, system messages
│   └── lang-pinn/                # Lang-PINN multi-agent framework — see libs/lang-pinn/README.md
│       └── src/lang_pinn/
│           ├── agents/           # PDE Agent, PINN Agent, Code Agent
│           ├── orchestrator.py   # 3-mode runner (library / code-agent / hybrid)
│           └── schemas.py        # PDESpec, ArchitectureRec dataclasses
├── experiments/
│   ├── harmonic_oscillator/      # Damped harmonic oscillator ODE — see its README.md
│   ├── burgers/                  # Burgers' equation
│   ├── schrodinger/              # Schrödinger equation
│   ├── parametric_harmonic/      # Parametric family (t,w0,d) + deep ensembles
│   ├── parametric_burgers/       # Parametric viscosity family (x,t,nu) + ensembles
│   ├── parametric_schrodinger/   # Complex parametric soliton family (x,t,A) + ensembles
│   ├── taylor_green/             # Taylor-Green vortex — 2D unsteady NS, exact solution
│   ├── lid_driven_cavity/        # Lid-driven cavity — 2D steady NS, Ghia benchmark
│   ├── navier_stokes_inverse/    # Inverse NS — infer Re from data (Kovasznay flow)
│   └── cylinder_wake/            # Raissi cylinder wake — inverse NS with real DNS data
├── learn/                        # Hands-on PINN course (9 notebooks, ~5 hours)
│   ├── 01_what_are_pinns.ipynb       # Motivation, landscape, when to use
│   ├── 02_autodiff_the_key_idea.ipynb # torch.autograd.grad deep dive
│   ├── 03_first_pinn_from_scratch.ipynb # Solve u'=-u with raw PyTorch
│   ├── 04_data_vs_physics_vs_hybrid.ipynb # Same ODE 3 ways — the aha moment
│   ├── 05_pdes_and_boundary_conditions.ipynb # Burgers' equation from scratch
│   ├── 06_training_tricks.ipynb      # Loss weighting, spectral bias, Ansatz
│   ├── 07_parametric_and_inverse.ipynb # Parameters as inputs + learnable params
│   ├── 08_honest_assessment.ipynb    # Failure modes, alternatives, decision framework
│   └── 09_inverse_navier_stokes.ipynb # Infer Re from flow data — Kovasznay, streamfunction
├── notebooks/                    # Experiment-specific analysis notebooks
│   ├── 01_harmonic_analysis.ipynb    # Full deep-dive: PINN cost function & solving loop
│   ├── 02_burgers_analysis.ipynb     # Nonlinear PDE, shock formation
│   ├── 03_schrodinger_analysis.ipynb # Complex fields, periodic BCs
│   └── 04_model_as_solution.ipynb    # Prediction: derivatives, residual check, extrapolation
├── dashboard.py                  # Streamlit dashboard — see docs/dashboard.md
├── docs/
│   ├── prediction.md             # Concept: how prediction works in a PINN
│   ├── parametric_pinns.md       # Parametric PINNs + deep ensembles: method & tradeoffs
│   ├── adding_experiments.md     # Step-by-step guide for adding a new experiment
│   └── dashboard.md              # Dashboard usage guide
├── pyproject.toml                # Workspace root + console scripts
└── uv.lock
```

## Learn PINNs

New to PINNs? The **[learn/](learn/)** directory is a hands-on course (9 notebooks, ~5 hours) that takes you from zero to solving your own equations. No prior PINN experience required — just Python, PyTorch basics, and some calculus. See [learn/README.md](learn/README.md) for the full learning path.

```bash
uv sync --all-packages
jupyter lab learn/
```

## Dashboard

A Streamlit dashboard for browsing runs, comparing experiments, and running
interactive parametric predictions — no code needed.

```bash
uv run streamlit run dashboard.py
```

Four pages: **Run Browser** (all experiments and runs at a glance), **Run Detail**
(config, loss curves, artifacts), **Compare** (side-by-side metrics and overlaid
loss curves), and **Parametric Predictor** (interactive sliders for all four
parametric experiments — loads a trained checkpoint and runs inference live).

Full guide: [docs/dashboard.md](docs/dashboard.md).

## Getting Started

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)

### Install

```bash
git clone <repo-url> && cd PINN
uv sync --all-packages
```

This installs the `pinn` library (editable, from `libs/pinn`), all experiment dependencies (`torch`, `numpy`, `matplotlib`, `tqdm`, `typer`, `rich`), and the console scripts.

### Run an experiment

```bash
uv run train-harmonic --help             # CLI reference (train / predict / compare)
uv run train-harmonic train              # train with defaults
uv run train-harmonic predict            # evaluate the latest trained model (no retraining)
uv run train-harmonic compare            # rank all runs by their recorded metrics
```

## Experiments

| Experiment | Equation | Entry point | Docs |
|------------|----------|-------------|------|
| Harmonic oscillator | `u'' + μu' + ku = 0` | `uv run train-harmonic train` | [README](experiments/harmonic_oscillator/README.md) |
| Burgers | `u_t + u·u_x = ν·u_xx` | `uv run train-burgers train` | [README](experiments/burgers/README.md) |
| Schrödinger | `i·h_t + ½·h_xx + |h|²·h = 0` | `uv run train-schrodinger train` | [README](experiments/schrodinger/README.md) |
| Parametric harmonic | whole `(w0, d)` family, one model | `uv run train-parametric train` | [README](experiments/parametric_harmonic/README.md) |
| Parametric Burgers | whole viscosity family, one model | `uv run train-parametric-burgers train` | [README](experiments/parametric_burgers/README.md) |
| Parametric Schrödinger | complex soliton family, one model | `uv run train-parametric-schrodinger train` | [README](experiments/parametric_schrodinger/README.md) |
| **Taylor-Green vortex** | 2D unsteady NS, exact solution | `uv run train-taylor-green train` | [README](experiments/taylor_green/README.md) |
| **Lid-driven cavity** | 2D steady NS, Ghia benchmark | `uv run train-cavity train` | [README](experiments/lid_driven_cavity/README.md) |
| **NS inverse (Kovasznay)** | infer Re from scattered data | `uv run train-ns-inverse train` | [README](experiments/navier_stokes_inverse/README.md) |
| **Cylinder wake (Raissi)** | inverse NS with real DNS data | `uv run train-cylinder train` | [README](experiments/cylinder_wake/README.md) |

The Navier-Stokes experiments cover the full spectrum: **forward unsteady** (Taylor-Green,
exact solution), **forward steady** (cavity, Ghia benchmark), **inverse self-contained**
(Kovasznay, infer Re from synthetic data), and **inverse with real data** (Raissi cylinder
wake — infer λ₁, λ₂ from DNS, reconstruct hidden pressure). The cylinder wake reproduces
the headline result from the original PINNs paper using a streamfunction formulation for
exact incompressibility.

The parametric experiments lift "one model = one problem instance": parameters become network
*inputs*, so `predict --w0 40 -d 1.5` (or `--nu 0.05`, or `-a 1.3`) solves a **never-trained**
instance in milliseconds. All support deep ensembles (`train --ensemble 5`) for ±2σ
epistemic-uncertainty bands. Method, design rules, and the measured cost/accuracy tradeoff vs
single-instance PINNs: [docs/parametric_pinns.md](docs/parametric_pinns.md).

Each CLI also provides `predict` (re-evaluate a saved model — defaults to the latest run;
checkpoints are self-describing, so the architecture is rebuilt automatically from the stored
config) and `compare` (rank all runs of an experiment by their `metrics.json`).

For what "prediction" actually means for a PINN — the model *is* the solution function — see
[docs/prediction.md](docs/prediction.md) and the hands-on demonstration in
[notebooks/04_model_as_solution.ipynb](notebooks/04_model_as_solution.ipynb) (mesh-free
evaluation, autograd derivatives, residual self-check, extrapolation failure mode).

## Adding a New Experiment

See **[docs/adding_experiments.md](docs/adding_experiments.md)** for the full step-by-step
guide — including the complete `train.py` template, test pattern, README structure, and a
reference of all shared infrastructure and design patterns (ansatz, hard BCs, streamfunction,
parametric inputs, learnable parameters, residual normalisation).

Quick summary:

1. Create `experiments/<name>/` with `__init__.py` and a `train.py` exposing a Typer `app`.
2. Implement `build_model(config)`, `build_losses(...)`, `solve_*(...)`, and the `train`/`predict`/`compare` CLI commands.
3. Register a console script in `pyproject.toml` and re-sync: `uv sync --all-packages`.
4. Add a lifecycle test in `tests/test_experiments_cli.py`.
5. Write a `README.md` with problem statement, method, CLI reference, and caveats.

Every experiment run writes a self-contained artifact directory under `outputs/`
(checkpoint, `metrics.json`, plots, loguru logs) — see any experiment README for details.

## Core Libraries

The `pinn` package (in `libs/pinn`) is documented in [libs/pinn/README.md](libs/pinn/README.md) — including a "solve your own equation in 5 steps" quickstart and a scaling guide. Highlights:

- `PINN` — configurable `tanh` MLP for smooth higher-order derivatives
- `PINNTrainer` — named multi-term losses with per-term weights, early stopping, gradient clipping, per-epoch callbacks, checkpoint save/load, full loss history
- `set_seed` / `setup_logging` — reproducibility and loguru console+file logging
- `utils.plotting` — contour, 1D-comparison, and loss-comparison plots (headless-safe)

The `llm-provider` package (in `libs/llm-provider`) is documented in [libs/llm-provider/README.md](libs/llm-provider/README.md). LLM abstraction layer:

- `LLMClient` — sync + async LLM queries with streaming, system messages, batch support
- `LLMConfig` — settings resolution (kwargs > env > `.env` > defaults), Ollama Cloud auth
- Ollama Cloud default (`gpt-oss:120b`); any LiteLLM provider works by changing `model` + `api_key`

The `lang-pinn` package (in `libs/lang-pinn`) is documented in [libs/lang-pinn/README.md](libs/lang-pinn/README.md). LLM-guided PINN construction:

- **PDE Agent** — parse natural language into structured `PDESpec` via chain-of-thought LLM prompting
- **PINN Agent** — recommend architecture from PDE features (rule-based or LLM-assisted)
- **Code Agent** — generate experiment code targeting the `pinn` library API (template or LLM)
- **Orchestrator** — 3-mode runner: library (deterministic), code-agent (full LLM), hybrid (LLM + feedback loop)

## Testing

```bash
uv run pytest              # fast suite: unit + CLI smoke tests (~5s)
uv run pytest -m slow      # convergence regression tests (train real PINNs)
uv run ruff check .        # lint
```

Layout:

- `libs/pinn/tests/` — library unit tests: network shapes/gradients, trainer mechanics
  (weighted losses, early stopping, grad clipping, callbacks), checkpoint round-trip,
  seeding, headless plotting, feedback agent
- `libs/llm-provider/tests/` — config resolution, client sync/async, all mocked (no API key)
- `libs/lang-pinn/tests/` — schemas, PDE/PINN/Code agents, orchestrator, all LLM calls mocked
- `tests/test_experiments_cli.py` — full `train → predict → compare` lifecycle per experiment
  via Typer's in-process `CliRunner`, asserting every artifact is written
- `tests/test_convergence.py` — marked `slow`: solves `u' = -u` against the exact solution
  (rel-L2 < 5%) and checks the harmonic Ansatz pipeline drops its loss by 100×

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378.
