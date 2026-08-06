# PINN — Project Overview

**Physics-Informed Neural Networks: A Production-Grade Research Framework**

---

## The Big Picture

Solving differential equations is at the heart of science and engineering — from predicting how fluids flow, to modelling quantum particles, to simulating structural mechanics. Traditional methods (finite elements, finite differences) require painstaking mesh generation, struggle with irregular geometries, and must be re-run from scratch when parameters change.

**Physics-Informed Neural Networks (PINNs)** offer a fundamentally different approach: train a neural network to satisfy the governing equations directly. No mesh. No grid. The trained model *is* the solution — a smooth, differentiable function you can evaluate anywhere, instantly.

This project is a complete framework for building, training, and deploying PINNs — from a beginner's first ODE to production-scale Navier-Stokes fluid simulations.

---

## What We Built

### A Modular Core Library

The `pinn` library provides the building blocks that every experiment shares:

| Component | What It Does |
|-----------|-------------|
| **PINN Network** | Configurable neural network backbone optimised for physics problems |
| **Trainer** | Multi-loss training loop with early stopping, gradient clipping, NaN detection, learning rate scheduling, and checkpointing |
| **Adaptive Refinement (RAR)** | Automatically concentrates collocation points where the equation is hardest to satisfy — like adaptive mesh refinement, but for neural networks |
| **Training Feedback** | Health monitoring (loss smoothness, gradient health), adaptive loss weighting, post-training quality scoring — inspired by Lang-PINN |
| **W&B Integration** | Optional Weights & Biases logging for experiment tracking and hyperparameter sweeps |
| **Utilities** | Reproducible seeding, structured logging, publication-quality plotting |

### LLM Abstraction Layer

The `llm-provider` library provides a unified interface for LLM access — the foundation for our Lang-PINN multi-agent framework:

| Feature | Details |
|---------|---------|
| **Provider-agnostic** | Ollama Cloud (default), local Ollama, Anthropic, OpenAI, Vertex — any LiteLLM-supported provider works by changing `model` + `api_key` |
| **Zero-friction default** | Ollama Cloud `gpt-oss:120b` with automatic bearer auth from `.env` |
| **Sync + Async** | `ask()`, `ask_async()`, batch queries, streaming, system messages |
| **Config resolution** | Explicit kwargs > environment variables > `.env` file > sensible defaults |

### Lang-PINN Multi-Agent Framework

An LLM-guided system for constructing and training PINNs from natural language, inspired by [Lang-PINN (He et al. 2025)](https://arxiv.org/abs/2510.05158):

| Agent | What It Does |
|-------|-------------|
| **PDE Agent** | Parses natural language problem descriptions into structured specifications (equation, domain, conditions, parameters) |
| **PINN Agent** | Recommends network architecture based on PDE features — rule-based heuristics from 11 experiments, or LLM-assisted |
| **Code Agent** | Generates runnable experiment code targeting the `pinn` library API — deterministic templates or LLM-generated |
| **Orchestrator** | 3-mode pipeline: library (deterministic), code-agent (full LLM), hybrid (LLM + iterative feedback refinement) |

The hybrid mode is where it gets interesting: the LLM generates code, the `pinn` library executes it, the Feedback Agent scores it, and the LLM refines — a quality-gated loop that combines LLM flexibility with library reliability.

### 11 Experiments Spanning 4 Domains

Each experiment is a self-contained CLI application with train, predict, and compare commands.

#### Ordinary Differential Equations
- **Damped Harmonic Oscillator** — the "hello world" of PINNs, with a learnable sinusoidal Ansatz that overcomes spectral bias at high frequencies

#### Partial Differential Equations
- **Burgers' Equation** — the classic shock-formation benchmark (Raissi et al. 2019), now with adaptive collocation for improved shock resolution
- **Schrodinger Equation** — complex-valued quantum soliton with periodic boundaries

#### Fluid Dynamics (Navier-Stokes)
- **Taylor-Green Vortex** — 2D unsteady flow with exact analytical solution for rigorous validation
- **Lid-Driven Cavity** — benchmark steady-state flow validated against Ghia et al. reference data at Re=100
- **Cylinder Wake** — the headline result from Raissi et al.: reconstruct pressure and infer physical parameters from DNS velocity data (1M data points)

#### Parametric & Inverse Problems
- **Parametric Harmonic / Burgers / Schrodinger / Taylor-Green** — a single model learns the *entire family* of solutions across a parameter range. Slide a parameter, get instant predictions — no retraining
- **Inverse Navier-Stokes** — infer the Reynolds number from sparse, noisy flow observations. The network simultaneously learns the solution *and* discovers the physics

### Deep Ensemble Uncertainty Quantification

For parametric experiments, we train multiple independent models (deep ensembles) and use their disagreement to estimate prediction uncertainty. This gives honest confidence bands — critical for any real-world deployment where you need to know *how much to trust the prediction*.

### Interactive Dashboard

A Streamlit web application for exploring results without writing code:

- **Overview** — browse all experiments and runs at a glance
- **Run Detail** — interactive loss curves (zoom, hover, pan), configuration, artifacts
- **Compare** — overlay loss histories across runs to compare convergence
- **Parametric Predictor** — drag sliders to explore how solutions change with physical parameters in real time

### Learning Curriculum

A 9-notebook progressive course (~5 hours) that takes someone from zero PINN knowledge to solving their own equations:

| Notebook | Topic |
|----------|-------|
| 01 | What are PINNs? Motivation and landscape |
| 02 | Automatic differentiation — the key enabling technology |
| 03 | Build your first PINN from scratch (raw PyTorch) |
| 04 | Data vs Physics vs Hybrid — the "aha moment" |
| 05 | Step up to PDEs and boundary conditions |
| 06 | Training tricks that actually matter |
| 07 | Parametric PINNs and inverse problems |
| 08 | Honest assessment: when to use PINNs (and when not to) |
| 09 | Advanced: Inverse Navier-Stokes with Reynolds number inference |

Each notebook is self-contained, runnable, and produces its own visualisations. The curriculum is designed to build intuition, not just demonstrate code.

---

## Why This Matters

### For Researchers
- **Rapid prototyping**: define your PDE, write a loss function, train. The library handles everything else.
- **Reproducibility**: every run saves a self-describing checkpoint with full configuration. Re-run or re-analyse months later with no guesswork.
- **Adaptive refinement**: RAR automatically concentrates computational effort where the physics is stiffest — no manual mesh tuning.
- **Uncertainty quantification**: deep ensembles give calibrated uncertainty bands out of the box.

### For Engineers
- **Mesh-free**: no mesh generation, no grid convergence studies. The solution is a smooth function you evaluate at any point.
- **Parametric models**: train once, predict instantly at any parameter value. Ideal for design exploration and real-time digital twins.
- **Inverse capability**: infer unknown material properties or operating conditions directly from sensor data.
- **Production-ready CLI**: every experiment is a command-line tool with configurable hyperparameters, artifact management, and structured logging.

### For Educators & Students
- **Progressive curriculum**: from first principles to Navier-Stokes in 9 notebooks, with exercises at every step.
- **Honest assessment**: Notebook 08 doesn't oversell — it covers failure modes, alternatives, and gives a decision framework for when PINNs are the right tool.
- **Reference implementations**: 11 experiments serve as templates for new problems.

---

## By the Numbers

| Metric | Count |
|--------|-------|
| Python source files | 60+ |
| Lines of code | ~10,500 |
| Experiments | 11 |
| Learning notebooks | 9 |
| Analysis notebooks | 4 |
| Automated tests | 180 (175 fast + 5 convergence) |
| Development phases | 21 |
| Documentation files | 6 |
| CI/CD | GitHub Actions (lint + test) |

---

## Technical Highlights

- **Two-stage optimisation**: Adam for fast initial convergence, then L-BFGS for precision refinement — combining the best of both worlds
- **Spectral bias mitigation**: learnable sinusoidal Ansatz for high-frequency problems where standard networks fail
- **Streamfunction formulation**: for fluid problems, `u = psi_y, v = -psi_x` enforces incompressibility by construction — eliminating an entire equation from the loss
- **Pressure gauge invariance**: proper handling of the pressure-up-to-a-constant ambiguity in incompressible flow
- **NaN/Inf detection**: training halts immediately with diagnostic output rather than silently producing garbage
- **Self-describing checkpoints**: model architecture is reconstructed from saved metadata — no need to remember hyperparameters
- **Adaptive loss weighting**: automatic rebalancing when one loss term dominates — prevents gradient starvation without manual tuning
- **Training quality scoring**: post-training effectiveness/efficiency/robustness evaluation inspired by the Lang-PINN feedback agent
- **Lazy optional dependencies**: W&B integration loads only when used; the core library has no heavy optional deps

---

## Architecture

```
PINN/
├── libs/pinn/          Core library (network, trainer, RAR, feedback, W&B, utils)
├── libs/llm-provider/  LLM abstraction layer (Ollama, Anthropic, OpenAI via LiteLLM)
├── libs/lang-pinn/     Lang-PINN multi-agent framework (PDE/PINN/Code agents, orchestrator)
├── experiments/        11 self-contained experiment CLIs
├── learn/              9-notebook progressive curriculum
├── notebooks/          4 experiment analysis notebooks
├── dashboard.py        Streamlit interactive dashboard
├── docs/               Technical documentation
├── tests/              Integration & convergence tests
└── .github/workflows/  CI pipeline
```

The **workspace architecture** (powered by uv) keeps the core library and experiments cleanly separated. The library knows nothing about specific PDEs; experiments know nothing about training mechanics. This separation makes it trivial to add new problems — define your equation, write a loss function, and the entire infrastructure (training, checkpointing, logging, CLI, dashboard) works automatically.

---

## Getting Started

```bash
# Clone and install
git clone https://github.com/bhanu-code-repo/PINN.git
cd PINN && uv sync --all-packages

# Train your first PINN (takes ~30 seconds)
uv run train-harmonic train -e 500 --no-show

# Launch the dashboard
uv run streamlit run dashboard.py

# Start the learning curriculum
jupyter lab learn/
```

---

## Roadmap

| Item | Status |
|------|--------|
| Core library (network, trainer, utils) | Done |
| 11 experiments (ODE → PDE → NS → parametric → inverse) | Done |
| Deep ensemble uncertainty quantification | Done |
| 9-notebook learning curriculum | Done |
| Streamlit interactive dashboard | Done |
| GitHub Actions CI | Done |
| Residual-based Adaptive Refinement (RAR) | Done |
| Weights & Biases integration | Done |
| Training Feedback Agent (health, adaptive weights, quality) | Done |
| LLM Provider library (Ollama Cloud + local, any provider) | Done |
| Lang-PINN multi-agent framework (LLM-guided PINNs) | Done |
| Breather soliton family (research problem) | Planned |
| ONNX/FastAPI model serving | Planned |

---

*Built with PyTorch, uv, and a commitment to making physics-informed machine learning accessible, rigorous, and production-ready.*
