# Learn PINNs — From Zero to Solving Your Own Equations

A hands-on, progressive course for learning **Physics-Informed Neural Networks**
from first principles. No prior PINN experience required — just basic Python, PyTorch
familiarity, and some calculus/differential equations background.

## Who This Is For

- **Researchers** exploring PINNs for their domain (fluids, materials, biology, finance)
- **Engineers** evaluating whether PINNs fit their simulation workflow
- **Students** building intuition for physics-informed machine learning
- **Practitioners** who've read papers but haven't built one from scratch

## Prerequisites

- Python basics (functions, classes, NumPy)
- PyTorch basics (tensors, `nn.Module`, optimisers, backprop concept)
- Calculus (derivatives, chain rule)
- Some exposure to differential equations (what an ODE/PDE is, what "solving" means)

You do **not** need experience with numerical methods (FEM, FDM), scientific computing
libraries, or prior PINN knowledge.

## The Learning Path

| # | Notebook | What You'll Learn | Time |
|---|----------|-------------------|------|
| 01 | [What Are PINNs?](01_what_are_pinns.ipynb) | Motivation, how they work, the landscape, when to use (and when NOT to) | 20 min |
| 02 | [Autodiff: The Key Idea](02_autodiff_the_key_idea.ipynb) | `torch.autograd.grad` — computing exact derivatives of neural networks | 25 min |
| 03 | [Your First PINN from Scratch](03_first_pinn_from_scratch.ipynb) | Solve `u' = -u` with raw PyTorch — no library, every line explained | 40 min |
| 04 | [Data vs Physics vs Hybrid](04_data_vs_physics_vs_hybrid.ipynb) | Same ODE solved 3 ways — see exactly when physics helps | 30 min |
| 05 | [PDEs and Boundary Conditions](05_pdes_and_boundary_conditions.ipynb) | Step up to Burgers' equation — spatial derivatives, BCs, collocation | 40 min |
| 06 | [Training Tricks That Matter](06_training_tricks.ipynb) | Loss weighting, spectral bias, ansatz, learning rate, best-model saving | 35 min |
| 07 | [Parametric PINNs and Inverse Problems](07_parametric_and_inverse.ipynb) | One model for a whole family + infer unknown parameters from data | 40 min |
| 08 | [Honest Assessment: When to Use PINNs](08_honest_assessment.ipynb) | Failure modes, alternatives, the decision framework | 25 min |
| 09 | [Inverse Navier-Stokes](09_inverse_navier_stokes.ipynb) | Infer Reynolds number from flow data — Kovasznay flow, pressure gauge invariance, streamfunction trick | 45 min |
| **10** | **[Lang-PINN Intro](10_lang_pinn_intro.ipynb)** | **LLM-guided PINNs: 3 agents, 3 modes, PDESpec, SymPy verification** | **30 min** |
| **11** | **[Hybrid Mode Deep Dive](11_hybrid_mode_deep_dive.ipynb)** | **Feedback loop, quality scoring, adaptive weights, health monitoring** | **30 min** |
| **12** | **[Bring Your Own PDE](12_bring_your_own_pde.ipynb)** | **End-to-end workflow: describe → verify → generate → customize → train** | **40 min** |

**Total: ~7 hours of hands-on work.**

## How to Use

```bash
cd PINN/
uv sync --all-packages    # install everything
jupyter lab learn/         # open the notebooks
```

Work through the notebooks in order. Each one builds on the previous:
- **01–02** build conceptual foundation
- **03** is the core hands-on experience (don't skip this)
- **04** is the "aha moment" — understanding the data/physics tradeoff
- **05–07** progressively tackle harder problems
- **08** gives you the honest picture for real-world decisions
- **09** is an advanced deep-dive into inverse Navier-Stokes
- **10–12** introduce Lang-PINN: LLM-guided PINN construction and the full workflow

Every notebook is self-contained (runs independently) but references earlier
concepts. Code cells are meant to be executed — the outputs tell the story.

## Relationship to the Rest of This Repo

This `learn/` directory is the **teaching track**. The rest of the repo is the
**production track**:

- `libs/pinn/` — the core library these notebooks eventually use
- `experiments/` — production experiments that demonstrate everything at scale
- `notebooks/` — experiment-specific analysis (assumes PINN knowledge)
- `docs/` — reference documentation

After completing this course, you'll be ready to:
1. Use the `pinn` library to solve your own equations
2. Read and modify any experiment in `experiments/`
3. Follow the [adding experiments guide](../docs/adding_experiments.md) to add your own
