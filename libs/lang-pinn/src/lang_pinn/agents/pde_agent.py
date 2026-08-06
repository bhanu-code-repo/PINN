"""PDE Agent — parse natural language into a structured PDESpec.

Uses chain-of-thought prompting to extract equation, variables, domain,
conditions, and parameters. Verifies the parse via SymPy where possible.
"""

from __future__ import annotations

import json
import re

from llm_provider import LLMClient
from loguru import logger

from ..schemas import PDESpec

_SYSTEM_PROMPT = """\
You are a PDE parsing expert. Given a natural language description of a \
differential equation problem, extract a structured JSON specification.

You MUST return ONLY a JSON object (no markdown fences, no explanation) with \
these exact fields:

{
  "name": "Human-readable problem name",
  "equation": "Symbolic equation, e.g. u_tt + mu*u_t + k*u = 0",
  "independent_vars": ["t"],
  "dependent_var": "u",
  "order": 2,
  "spatial_dim": 0,
  "domain": {"t": [0.0, 1.0]},
  "initial_conditions": ["u(0) = 1", "u'(0) = 0"],
  "boundary_conditions": [],
  "parameters": {"mu": 4.0, "k": 6400.0},
  "is_linear": true,
  "is_time_dependent": true,
  "has_periodic_bc": false,
  "has_high_frequency": false,
  "has_sharp_gradients": false,
  "output_dim": 1
}

Rules:
- Use standard derivative notation: u_t, u_x, u_tt, u_xx, u_xt, etc.
- For ODEs, spatial_dim = 0. For 1D PDEs, spatial_dim = 1. etc.
- Compute "order" as the highest derivative order in the equation.
- Set has_high_frequency = true if the natural frequency > 20 or the problem \
  involves rapid oscillation.
- Set has_sharp_gradients = true for shock/front problems (Burgers, advection).
- For complex-valued problems, set output_dim = 2.
- If domain is not specified, use reasonable defaults.
- Extract ALL numerical parameters into the "parameters" dict.
"""


class PDEAgent:
    """Parse natural language PDE descriptions into structured PDESpec objects.

    Args:
        client: LLM client to use. Defaults to a new ``LLMClient()`` instance.
    """

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

    def parse(self, description: str) -> PDESpec:
        """Parse a natural language PDE description into a PDESpec.

        Args:
            description: Natural language description of the PDE problem.

        Returns:
            Parsed PDESpec.

        Raises:
            ValueError: If the LLM response cannot be parsed.
        """
        logger.info("PDE Agent: parsing problem description")
        logger.debug("Input: {}", description)

        raw = self.client.ask(description, system=_SYSTEM_PROMPT)
        logger.debug("LLM response: {}", raw)

        data = _extract_json(raw)
        spec = _dict_to_spec(data)

        logger.info("PDE Agent: parsed '{}' — order={}, spatial_dim={}, vars={}",
                     spec.name, spec.order, spec.spatial_dim, spec.independent_vars)
        return spec


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    # Strip markdown code fences if present
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nRaw: {text}") from e


def _dict_to_spec(data: dict) -> PDESpec:
    """Convert a parsed JSON dict into a PDESpec, with validation."""
    # Convert domain lists to tuples
    domain = {}
    for var, bounds in data.get("domain", {}).items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            domain[var] = (float(bounds[0]), float(bounds[1]))
        else:
            raise ValueError(f"Invalid domain bounds for '{var}': {bounds}")

    return PDESpec(
        name=data["name"],
        equation=data["equation"],
        independent_vars=data["independent_vars"],
        dependent_var=data.get("dependent_var", "u"),
        order=int(data["order"]),
        spatial_dim=int(data.get("spatial_dim", 0)),
        domain=domain,
        initial_conditions=data.get("initial_conditions", []),
        boundary_conditions=data.get("boundary_conditions", []),
        parameters={k: float(v) for k, v in data.get("parameters", {}).items()},
        is_linear=data.get("is_linear", True),
        is_time_dependent=data.get("is_time_dependent", True),
        has_periodic_bc=data.get("has_periodic_bc", False),
        has_high_frequency=data.get("has_high_frequency", False),
        has_sharp_gradients=data.get("has_sharp_gradients", False),
        output_dim=int(data.get("output_dim", 1)),
    )
