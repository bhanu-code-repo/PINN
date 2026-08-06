"""Tests for Lang-PINN CLI — LLM calls are mocked."""

import json
from unittest.mock import MagicMock, patch

from lang_pinn.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def _mock_client_response():
    """Mock that returns valid PDE JSON for any ask() call."""
    response_data = {
        "name": "Simple Decay",
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


class TestSolveCommand:
    @patch("lang_pinn.orchestrator.LLMClient")
    def test_solve_library_mode(self, mock_cls):
        mock_cls.return_value = _mock_client_response()
        result = runner.invoke(app, [
            "solve", "exponential decay", "--mode", "library", "--no-code",
        ])
        assert result.exit_code == 0, result.output
        assert "Simple Decay" in result.output

    @patch("lang_pinn.orchestrator.LLMClient")
    def test_solve_shows_tables(self, mock_cls):
        mock_cls.return_value = _mock_client_response()
        result = runner.invoke(app, [
            "solve", "exponential decay", "--mode", "library",
        ])
        assert result.exit_code == 0, result.output
        assert "Parsed PDE" in result.output
        assert "Architecture" in result.output

    @patch("lang_pinn.orchestrator.LLMClient")
    def test_solve_save_code(self, mock_cls, tmp_path):
        mock_cls.return_value = _mock_client_response()
        result = runner.invoke(app, [
            "solve", "exponential decay",
            "--mode", "library",
            "--save-code",
            "--output-dir", str(tmp_path),
            "--no-code",
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "generated_experiment.py").exists()
        assert (tmp_path / "pde_spec.json").exists()
        assert (tmp_path / "architecture.json").exists()


class TestParseCommand:
    @patch("lang_pinn.agents.pde_agent.LLMClient")
    def test_parse_outputs_table(self, mock_cls):
        mock_cls.return_value = _mock_client_response()
        result = runner.invoke(app, ["parse", "exponential decay", "--no-verify"])
        assert result.exit_code == 0, result.output
        assert "Simple Decay" in result.output


class TestRecommendCommand:
    @patch("lang_pinn.agents.pde_agent.LLMClient")
    def test_recommend_outputs_tables(self, mock_cls):
        mock_cls.return_value = _mock_client_response()
        result = runner.invoke(app, ["recommend", "exponential decay"])
        assert result.exit_code == 0, result.output
        assert "Architecture" in result.output
