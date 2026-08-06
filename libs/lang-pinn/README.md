# lang-pinn

**LLM-guided Physics-Informed Neural Network construction and training.**

Inspired by [Lang-PINN (He et al. 2025)](https://arxiv.org/abs/2510.05158) — an LLM-orchestrated multi-agent framework for solving differential equations with PINNs.

## Three Operating Modes

| Mode | LLM Usage | Best For |
|------|-----------|----------|
| **Library** | PDE parsing only | Production, reproducibility |
| **Code Agent** | All agents use LLM | Exploration, prototyping |
| **Hybrid** | LLM + feedback loop | Best of both — LLM targets `pinn` library API |

## CLI

```bash
# Full pipeline: parse → recommend → generate (hybrid mode)
uv run lang-pinn solve "u'' + 2u' + 6400u = 0, u(0)=1, u'(0)=0 on [0,1]"

# Library mode (deterministic, no LLM for arch/code)
uv run lang-pinn solve "Burgers equation" --mode library

# Save generated code + specs
uv run lang-pinn solve "heat equation" --save-code --output-dir ./my_experiment

# Execute the generated code
uv run lang-pinn solve "exponential decay" --execute

# Parse only (PDE Agent)
uv run lang-pinn parse "damped harmonic oscillator"

# Parse + recommend architecture (PDE + PINN Agents)
uv run lang-pinn recommend "Schrodinger equation with periodic BCs"
```

## Python API

```python
from lang_pinn import Orchestrator

# Hybrid mode (default): LLM generates, library validates
orch = Orchestrator()
result = orch.solve("Solve u'' + 2u' + 6400u = 0, u(0)=1, u'(0)=0 on [0,1]")

print(result.spec)          # Parsed PDE specification
print(result.architecture)  # Recommended architecture
print(result.code)          # Generated Python code
```

## Architecture

```
lang-pinn/
├── agents/
│   ├── pde_agent.py      # NL → PDESpec (LLM + JSON extraction)
│   ├── pinn_agent.py     # PDESpec → ArchitectureRec (rules or LLM)
│   └── code_agent.py     # PDESpec + Arch → Python code (template or LLM)
├── orchestrator.py       # 3-mode pipeline runner
├── cli.py                # Typer CLI (solve, parse, recommend)
├── sympy_verify.py       # SymPy-based PDE verification
└── schemas.py            # PDESpec, ArchitectureRec dataclasses
```

### Agents

**PDE Agent** — Parses natural language into a structured `PDESpec`:
- Chain-of-thought prompting for equation extraction
- JSON output with validation
- Extracts: equation, variables, domain, conditions, parameters, feature flags

**PINN Agent** — Recommends architecture from PDE features:
- Rule-based mode: deterministic heuristics from 11 experiments
- LLM mode: consults LLM for architecture advice
- Output: layer count, width, activation, Ansatz, loss weights, collocation density

**Code Agent** — Generates runnable experiment code:
- Template mode: deterministic, always valid, uses `pinn` library API
- LLM mode: full code generation with `pinn` imports
- Output: complete Python script ready to execute

### Feedback Agent

Lives in `libs/pinn/feedback.py` (pure training logic, no LLM dependency):
- `TrainingHealthMonitor` — loss smoothness, gradient health, convergence detection
- `AdaptiveLossWeighter` — dynamic weight rebalancing
- `evaluate_quality()` — post-training scoring

### Orchestrator

Ties everything together with a feedback loop (hybrid mode):

```
User describes problem
  → PDE Agent parses it
  → PINN Agent recommends architecture
  → Code Agent generates code
  → Execute + Feedback Agent evaluates
  → If quality < threshold: LLM refines code
  → Repeat until quality target met
```

## Dependencies

- `pinn` — core PINN library (execution engine)
- `llm-provider` — LLM abstraction layer (Ollama default)
- `sympy` — symbolic math for future PDE verification

## Adding New PDE Types

The rule-based PINN Agent already handles:
- ODEs (any order)
- 1D PDEs (with sharp gradient detection)
- 2D+ PDEs (wider networks)
- High-frequency problems (sinusoidal Ansatz)
- Complex-valued outputs

For unusual problems, use `mode="hybrid"` — the LLM adapts.
