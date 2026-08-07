"""Data structures for PDE specification and architecture recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PDESpec:
    """Structured representation of a PDE problem parsed by the PDE Agent.

    This is the contract between agents: PDE Agent produces it, PINN Agent
    and Code Agent consume it.
    """

    # Human-readable name (e.g. "Damped Harmonic Oscillator")
    name: str

    # Symbolic equation string (e.g. "u_tt + mu*u_t + k*u = 0")
    equation: str

    # Independent variables (e.g. ["t"] or ["x", "t"])
    independent_vars: list[str]

    # Dependent variable name (e.g. "u")
    dependent_var: str

    # PDE order (highest derivative order)
    order: int

    # Spatial dimension (0 for ODE, 1 for 1D PDE, 2 for 2D, ...)
    spatial_dim: int

    # Domain bounds: {var_name: (low, high)}
    domain: dict[str, tuple[float, float]]

    # Initial conditions: list of strings (e.g. ["u(0) = 1", "u'(0) = 0"])
    initial_conditions: list[str] = field(default_factory=list)

    # Boundary conditions: list of strings
    boundary_conditions: list[str] = field(default_factory=list)

    # Physical parameters: {name: value} (e.g. {"mu": 4.0, "k": 6400.0})
    parameters: dict[str, float] = field(default_factory=dict)

    # PDE feature flags (used by PINN Agent for architecture matching)
    is_linear: bool = True
    is_time_dependent: bool = True
    has_periodic_bc: bool = False
    has_high_frequency: bool = False
    has_sharp_gradients: bool = False
    output_dim: int = 1  # 2 for complex-valued (real + imag)


@dataclass
class ArchitectureRec:
    """Architecture recommendation produced by the PINN Agent."""

    input_dim: int
    output_dim: int
    hidden_layers: int
    hidden_neurons: int
    activation: str = "tanh"
    learning_rate: float = 1e-3
    epochs: int = 10000
    use_ansatz: bool = False
    ansatz_type: str | None = None  # "sinusoidal", "exponential", etc.
    loss_weights: dict[str, float] = field(default_factory=lambda: {"ic": 1.0, "physics": 1.0})
    n_collocation: int = 200
    reasoning: str = ""  # Why this architecture was chosen
    knowledge_context: str = ""  # Retrieved literature context from RAG
