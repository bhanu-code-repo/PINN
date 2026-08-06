"""Tests for PINN Agent — rule-based recommendations (no LLM needed)."""

from lang_pinn.agents.pinn_agent import PINNAgent, recommend_architecture
from lang_pinn.schemas import PDESpec


def _make_spec(**kwargs) -> PDESpec:
    """Helper to build a PDESpec with sensible defaults."""
    defaults = {
        "name": "Test",
        "equation": "u_t = 0",
        "independent_vars": ["t"],
        "dependent_var": "u",
        "order": 1,
        "spatial_dim": 0,
        "domain": {"t": (0.0, 1.0)},
    }
    defaults.update(kwargs)
    return PDESpec(**defaults)


class TestRuleBasedRecommendation:
    def test_ode_compact_architecture(self):
        spec = _make_spec(spatial_dim=0)
        rec = recommend_architecture(spec)
        assert rec.hidden_layers == 3
        assert rec.hidden_neurons == 32
        assert rec.input_dim == 1

    def test_1d_pde_medium_architecture(self):
        spec = _make_spec(
            independent_vars=["x", "t"],
            spatial_dim=1,
            domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        )
        rec = recommend_architecture(spec)
        assert rec.hidden_layers == 4
        assert rec.hidden_neurons == 64
        assert rec.input_dim == 2

    def test_2d_pde_larger_architecture(self):
        spec = _make_spec(
            independent_vars=["x", "y", "t"],
            spatial_dim=2,
            domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
        )
        rec = recommend_architecture(spec)
        assert rec.hidden_layers == 5
        assert rec.hidden_neurons == 128
        assert rec.input_dim == 3

    def test_high_frequency_enables_ansatz(self):
        spec = _make_spec(has_high_frequency=True)
        rec = recommend_architecture(spec)
        assert rec.use_ansatz is True
        assert rec.ansatz_type == "sinusoidal"
        assert rec.hidden_neurons >= 64
        assert rec.loss_weights["physics"] == 1e-4

    def test_no_high_frequency_no_ansatz(self):
        spec = _make_spec(has_high_frequency=False)
        rec = recommend_architecture(spec)
        assert rec.use_ansatz is False
        assert rec.ansatz_type is None

    def test_sharp_gradients_more_collocation(self):
        spec_normal = _make_spec(has_sharp_gradients=False, spatial_dim=1,
                                 independent_vars=["x", "t"],
                                 domain={"x": (-1, 1), "t": (0, 1)})
        spec_sharp = _make_spec(has_sharp_gradients=True, spatial_dim=1,
                                independent_vars=["x", "t"],
                                domain={"x": (-1, 1), "t": (0, 1)})
        rec_normal = recommend_architecture(spec_normal)
        rec_sharp = recommend_architecture(spec_sharp)
        assert rec_sharp.n_collocation > rec_normal.n_collocation
        assert rec_sharp.epochs > rec_normal.epochs

    def test_higher_order_deeper_network(self):
        spec2 = _make_spec(order=2)
        spec3 = _make_spec(order=3)
        rec2 = recommend_architecture(spec2)
        rec3 = recommend_architecture(spec3)
        assert rec3.hidden_layers > rec2.hidden_layers

    def test_complex_output_dim(self):
        spec = _make_spec(output_dim=2)
        rec = recommend_architecture(spec)
        assert rec.output_dim == 2

    def test_ic_generates_ic_weight(self):
        spec = _make_spec(initial_conditions=["u(0) = 1"])
        rec = recommend_architecture(spec)
        assert "ic" in rec.loss_weights

    def test_bc_generates_bc_weight(self):
        spec = _make_spec(boundary_conditions=["u(0,t) = 0"])
        rec = recommend_architecture(spec)
        assert "bc" in rec.loss_weights

    def test_activation_is_tanh(self):
        spec = _make_spec()
        rec = recommend_architecture(spec)
        assert rec.activation == "tanh"

    def test_reasoning_is_populated(self):
        spec = _make_spec()
        rec = recommend_architecture(spec)
        assert len(rec.reasoning) > 0


class TestPINNAgentInterface:
    def test_default_uses_rules(self):
        agent = PINNAgent()
        spec = _make_spec()
        rec = agent.recommend(spec)
        assert rec.hidden_layers == 3  # rule-based ODE default
