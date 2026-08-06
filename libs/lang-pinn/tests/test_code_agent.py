"""Tests for Code Agent — template and LLM modes."""

from unittest.mock import MagicMock

from lang_pinn.agents.code_agent import CodeAgent, _extract_code, generate_code_template
from lang_pinn.schemas import ArchitectureRec, PDESpec


def _harmonic_spec() -> PDESpec:
    return PDESpec(
        name="Damped Harmonic Oscillator",
        equation="u_tt + mu*u_t + k*u = 0",
        independent_vars=["t"],
        dependent_var="u",
        order=2,
        spatial_dim=0,
        domain={"t": (0.0, 1.0)},
        initial_conditions=["u(0) = 1", "u'(0) = 0"],
        parameters={"mu": 4.0, "k": 6400.0},
        has_high_frequency=True,
    )


def _harmonic_arch() -> ArchitectureRec:
    return ArchitectureRec(
        input_dim=1, output_dim=1,
        hidden_layers=3, hidden_neurons=64,
        activation="tanh",
        learning_rate=1e-3,
        epochs=10000,
        use_ansatz=True, ansatz_type="sinusoidal",
        loss_weights={"ic": 0.1, "physics": 1e-4},
        n_collocation=200,
    )


class TestExtractCode:
    def test_plain_code(self):
        assert _extract_code("print('hi')") == "print('hi')"

    def test_python_fence(self):
        raw = "```python\nprint('hi')\n```"
        assert _extract_code(raw) == "print('hi')"

    def test_bare_fence(self):
        raw = "```\nprint('hi')\n```"
        assert _extract_code(raw) == "print('hi')"


class TestTemplateGeneration:
    def test_generates_valid_python(self):
        code = generate_code_template(_harmonic_spec(), _harmonic_arch())
        # Should be parseable Python
        compile(code, "<test>", "exec")

    def test_contains_pinn_imports(self):
        code = generate_code_template(_harmonic_spec(), _harmonic_arch())
        assert "from pinn import" in code
        assert "PINN" in code
        assert "PINNTrainer" in code

    def test_contains_model_config(self):
        code = generate_code_template(_harmonic_spec(), _harmonic_arch())
        assert "hidden_layers=3" in code
        assert "hidden_neurons=64" in code
        assert 'activation="tanh"' in code

    def test_contains_loss_functions(self):
        code = generate_code_template(_harmonic_spec(), _harmonic_arch())
        assert "def physics_loss" in code
        assert "def ic_loss" in code
        assert "loss_functions" in code

    def test_contains_training_loop(self):
        code = generate_code_template(_harmonic_spec(), _harmonic_arch())
        assert "trainer.train(" in code
        assert "optimizer" in code

    def test_contains_quality_eval(self):
        code = generate_code_template(_harmonic_spec(), _harmonic_arch())
        assert "evaluate_quality" in code
        assert "TrainingHealthMonitor" in code

    def test_pde_2var_uses_meshgrid(self):
        spec = PDESpec(
            name="Burgers",
            equation="u_t + u*u_x = nu*u_xx",
            independent_vars=["x", "t"],
            dependent_var="u",
            order=2,
            spatial_dim=1,
            domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        )
        arch = ArchitectureRec(input_dim=2, output_dim=1,
                               hidden_layers=4, hidden_neurons=64)
        code = generate_code_template(spec, arch)
        assert "meshgrid" in code


class TestCodeAgentInterface:
    def test_template_mode_no_llm(self):
        agent = CodeAgent()
        code = agent.generate(_harmonic_spec(), _harmonic_arch(), use_llm=False)
        assert "from pinn import" in code

    def test_llm_mode_calls_client(self):
        mock_client = MagicMock()
        mock_client.ask.return_value = "```python\nprint('generated')\n```"
        agent = CodeAgent(client=mock_client)
        code = agent.generate(_harmonic_spec(), _harmonic_arch(), use_llm=True)
        assert code == "print('generated')"
        mock_client.ask.assert_called_once()
