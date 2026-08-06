"""Tests for the Orchestrator — 3-mode runner."""

import json
from unittest.mock import MagicMock

import pytest
from lang_pinn.orchestrator import Orchestrator, SolveResult
from lang_pinn.schemas import PDESpec


def _mock_client_with_pde_response():
    """Create a mock LLM client that returns a valid PDE parse."""
    response_data = {
        "name": "Exponential Decay",
        "equation": "u_t + u = 0",
        "independent_vars": ["t"],
        "dependent_var": "u",
        "order": 1,
        "spatial_dim": 0,
        "domain": {"t": [0.0, 1.0]},
        "initial_conditions": ["u(0) = 1"],
        "parameters": {},
    }
    mock = MagicMock()
    mock.ask.return_value = json.dumps(response_data)
    return mock


def _simple_spec() -> PDESpec:
    return PDESpec(
        name="Decay",
        equation="u_t + u = 0",
        independent_vars=["t"],
        dependent_var="u",
        order=1,
        spatial_dim=0,
        domain={"t": (0.0, 1.0)},
        initial_conditions=["u(0) = 1"],
    )


class TestOrchestratorInit:
    def test_valid_modes(self):
        for mode in ("library", "code-agent", "hybrid"):
            orch = Orchestrator(client=MagicMock(), mode=mode)
            assert orch.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            Orchestrator(client=MagicMock(), mode="invalid")

    def test_default_mode_is_hybrid(self):
        orch = Orchestrator(client=MagicMock())
        assert orch.mode == "hybrid"


class TestSolveWithSpec:
    def test_library_mode_no_execute(self):
        mock = MagicMock()
        orch = Orchestrator(client=mock, mode="library")
        result = orch.solve_from_spec(_simple_spec(), execute=False)

        assert isinstance(result, SolveResult)
        assert result.spec.name == "Decay"
        assert result.mode == "library"
        assert result.executed is False
        assert "from pinn import" in result.code
        # Library mode doesn't call LLM for architecture or code
        mock.ask.assert_not_called()

    def test_code_agent_mode_calls_llm(self):
        mock = MagicMock()
        mock.ask.return_value = '{"hidden_layers": 3, "hidden_neurons": 32}'
        orch = Orchestrator(client=mock, mode="code-agent")
        result = orch.solve_from_spec(_simple_spec(), execute=False)

        assert result.mode == "code-agent"
        # Should have called LLM for architecture + code
        assert mock.ask.call_count == 2

    def test_hybrid_mode_calls_llm(self):
        mock = MagicMock()
        mock.ask.return_value = '{"hidden_layers": 3, "hidden_neurons": 32}'
        orch = Orchestrator(client=mock, mode="hybrid")
        result = orch.solve_from_spec(_simple_spec(), execute=False)

        assert result.mode == "hybrid"
        assert mock.ask.call_count == 2


class TestSolveWithDescription:
    def test_parses_description_first(self):
        mock = _mock_client_with_pde_response()
        orch = Orchestrator(client=mock, mode="library")
        result = orch.solve("Solve exponential decay", execute=False)

        assert result.spec.name == "Exponential Decay"
        # PDE Agent called LLM once (library mode doesn't call for arch/code)
        assert mock.ask.call_count == 1


class TestSolveResult:
    def test_default_values(self):
        result = SolveResult(
            spec=_simple_spec(),
            architecture=MagicMock(),
            code="print('hi')",
            mode="library",
        )
        assert result.executed is False
        assert result.quality_score is None
        assert result.error is None
        assert result.iterations == 1
        assert result.history == []
