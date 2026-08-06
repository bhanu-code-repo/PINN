"""lang-pinn — LLM-guided Physics-Informed Neural Network construction.

Three operating modes:

- **Library mode**: user writes Python using the ``pinn`` library directly
- **Code Agent mode**: LLM generates experiment code from natural language
- **Hybrid mode**: LLM generates code targeting the ``pinn`` library API,
  with training feedback driving iterative refinement

Quick start::

    from lang_pinn import Orchestrator

    orch = Orchestrator()
    result = orch.solve("Solve u'' + 2u' + 6400u = 0, u(0)=1, u'(0)=0 on [0,1]")
"""

from .agents.code_agent import CodeAgent
from .agents.pde_agent import PDEAgent
from .agents.pinn_agent import PINNAgent
from .orchestrator import Orchestrator
from .schemas import PDESpec

__all__ = [
    "CodeAgent",
    "Orchestrator",
    "PDEAgent",
    "PDESpec",
    "PINNAgent",
]
