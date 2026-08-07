"""PINN Agent — recommend network architecture based on PDE features.

Two modes:
1. **Rule-based** (default): deterministic feature→architecture mapping.
   Fast, reproducible, no LLM call needed.
2. **LLM-assisted**: asks the LLM for architecture advice given PDE features.
   Useful for unusual problems where rules don't cover well.

Both modes optionally augment recommendations with retrieved knowledge
from the PINN knowledge base (BM25 search, no LLM cost for retrieval).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_provider import LLMClient
from loguru import logger

from ..schemas import ArchitectureRec, PDESpec
from .knowledge import build_search_query, load_knowledge, search_knowledge

# ---------------------------------------------------------------------------
# Rule-based architecture recommendation
# ---------------------------------------------------------------------------


def recommend_architecture(spec: PDESpec) -> ArchitectureRec:
    """Deterministic architecture recommendation from PDE features.

    This captures the design heuristics we've developed across 11 experiments:
    - High frequency → sinusoidal Ansatz, more neurons
    - Higher order → deeper network
    - Sharp gradients → more collocation points, adaptive refinement
    - Complex output → output_dim=2
    - 2D spatial → wider network
    """
    input_dim = len(spec.independent_vars)
    output_dim = spec.output_dim

    # Base architecture scales with problem complexity
    if spec.spatial_dim == 0:
        # ODE: compact network
        hidden_layers = 3
        hidden_neurons = 32
        n_collocation = 200
        epochs = 10000
    elif spec.spatial_dim == 1:
        # 1D PDE: medium network
        hidden_layers = 4
        hidden_neurons = 64
        n_collocation = 2000
        epochs = 15000
    else:
        # 2D+ PDE: larger network
        hidden_layers = 5
        hidden_neurons = 128
        n_collocation = 5000
        epochs = 20000

    # Higher-order PDEs benefit from deeper networks
    if spec.order >= 3:
        hidden_layers += 1

    # Activation: tanh is the safe default for PINNs
    activation = "tanh"

    # Ansatz for high-frequency problems
    use_ansatz = spec.has_high_frequency
    ansatz_type = "sinusoidal" if use_ansatz else None
    if use_ansatz:
        hidden_neurons = max(hidden_neurons, 64)

    # Sharp gradients: more collocation points
    if spec.has_sharp_gradients:
        n_collocation = int(n_collocation * 2.5)
        epochs = int(epochs * 1.5)

    # Loss weights: IC/BC weighted lower than physics for stiff problems
    loss_weights: dict[str, float] = {}
    if spec.initial_conditions:
        loss_weights["ic"] = 0.1 if spec.has_high_frequency else 1.0
    if spec.boundary_conditions:
        loss_weights["bc"] = 1.0
    loss_weights["physics"] = 1e-4 if spec.has_high_frequency else 1.0

    # Learning rate
    lr = 1e-3

    reasoning = _build_reasoning(spec, hidden_layers, hidden_neurons, activation,
                                 use_ansatz, n_collocation)

    return ArchitectureRec(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        hidden_neurons=hidden_neurons,
        activation=activation,
        learning_rate=lr,
        epochs=epochs,
        use_ansatz=use_ansatz,
        ansatz_type=ansatz_type,
        loss_weights=loss_weights,
        n_collocation=n_collocation,
        reasoning=reasoning,
    )


def _build_reasoning(spec: PDESpec, layers: int, neurons: int,
                     activation: str, ansatz: bool, n_coll: int) -> str:
    parts = [f"Order-{spec.order} {'ODE' if spec.spatial_dim == 0 else 'PDE'} "
             f"in {len(spec.independent_vars)} variable(s)."]
    if ansatz:
        parts.append("High-frequency content detected → sinusoidal Ansatz recommended.")
    if spec.has_sharp_gradients:
        parts.append("Sharp gradients → increased collocation density.")
    parts.append(f"Architecture: {layers}×{neurons} {activation}, {n_coll} collocation points.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM-assisted architecture recommendation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a PINN architecture expert. Given a PDE specification, recommend the \
optimal neural network architecture.

Return ONLY a JSON object with these fields:
{
  "hidden_layers": 4,
  "hidden_neurons": 64,
  "activation": "tanh",
  "learning_rate": 0.001,
  "epochs": 15000,
  "use_ansatz": false,
  "ansatz_type": null,
  "loss_weights": {"ic": 1.0, "physics": 1.0},
  "n_collocation": 2000,
  "reasoning": "Brief explanation of your choices"
}

Design principles:
- tanh is the standard PINN activation (smooth, infinitely differentiable)
- silu/gelu for problems where tanh causes vanishing gradients
- High-frequency oscillations need sinusoidal Ansatz (learnable freq + phase)
- Sharp gradients (shocks, fronts) need dense collocation or adaptive refinement
- IC/BC weights < 1.0 for stiff problems; physics weight can be very small (1e-4)
- Deeper networks (4-6 layers) for higher-order PDEs
- Wider networks (128+ neurons) for 2D spatial problems
"""


class PINNAgent:
    """Recommend PINN architecture from a PDESpec.

    Args:
        client: LLM client. Only needed if ``use_llm=True`` in :meth:`recommend`.
        knowledge_store_dir: Path to pre-built knowledge store.
            Uses the default store if not specified.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        knowledge_store_dir: str | Path | None = None,
    ):
        self.client = client
        self._knowledge = load_knowledge(knowledge_store_dir)

    def recommend(self, spec: PDESpec, *, use_llm: bool = False) -> ArchitectureRec:
        """Produce an architecture recommendation.

        Args:
            spec: The parsed PDE specification.
            use_llm: If True, consult the LLM. Otherwise use deterministic rules.

        Returns:
            ArchitectureRec with recommended hyperparameters.
        """
        if use_llm:
            return self._recommend_llm(spec)
        return self._recommend_rules(spec)

    def _get_knowledge_context(self, spec: PDESpec) -> str:
        """Retrieve relevant knowledge base context for this PDE."""
        if self._knowledge is None:
            return ""
        store, engine = self._knowledge
        query = build_search_query(spec)
        return search_knowledge(query, store, engine)

    def _recommend_rules(self, spec: PDESpec) -> ArchitectureRec:
        """Rule-based recommendation augmented with knowledge context."""
        rec = recommend_architecture(spec)

        context = self._get_knowledge_context(spec)
        if context:
            rec.knowledge_context = context
            logger.info("PINN Agent: augmented recommendation with knowledge base context")

        return rec

    def _recommend_llm(self, spec: PDESpec) -> ArchitectureRec:
        if self.client is None:
            self.client = LLMClient()

        prompt = (
            f"Problem: {spec.name}\n"
            f"Equation: {spec.equation}\n"
            f"Variables: {spec.independent_vars} → {spec.dependent_var}\n"
            f"Order: {spec.order}, Spatial dim: {spec.spatial_dim}\n"
            f"Domain: {spec.domain}\n"
            f"ICs: {spec.initial_conditions}\n"
            f"BCs: {spec.boundary_conditions}\n"
            f"Parameters: {spec.parameters}\n"
            f"Linear: {spec.is_linear}, Time-dependent: {spec.is_time_dependent}\n"
            f"High frequency: {spec.has_high_frequency}\n"
            f"Sharp gradients: {spec.has_sharp_gradients}\n"
            f"Periodic BC: {spec.has_periodic_bc}\n"
            f"Output dim: {spec.output_dim}"
        )

        # Augment with knowledge base context
        context = self._get_knowledge_context(spec)
        if context:
            prompt += (
                "\n\n--- Relevant Literature Context ---\n"
                "Use this retrieved knowledge to inform your recommendation:\n\n"
                f"{context}"
            )
            logger.info("PINN Agent: injected knowledge context into LLM prompt")

        logger.info("PINN Agent: consulting LLM for architecture advice")
        raw = self.client.ask(prompt, system=_SYSTEM_PROMPT)
        logger.debug("LLM response: {}", raw)

        data = _extract_json(raw)

        return ArchitectureRec(
            input_dim=len(spec.independent_vars),
            output_dim=spec.output_dim,
            hidden_layers=int(data.get("hidden_layers", 4)),
            hidden_neurons=int(data.get("hidden_neurons", 64)),
            activation=data.get("activation", "tanh"),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            epochs=int(data.get("epochs", 10000)),
            use_ansatz=bool(data.get("use_ansatz", False)),
            ansatz_type=data.get("ansatz_type"),
            loss_weights=data.get("loss_weights", {"ic": 1.0, "physics": 1.0}),
            n_collocation=int(data.get("n_collocation", 2000)),
            reasoning=data.get("reasoning", "LLM-recommended architecture"),
        )


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)
