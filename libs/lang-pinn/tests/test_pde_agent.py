"""Tests for PDE Agent — LLM calls are mocked."""

import json
from unittest.mock import MagicMock

import pytest
from lang_pinn.agents.pde_agent import PDEAgent, _dict_to_spec, _extract_json


class TestExtractJson:
    def test_plain_json(self):
        data = _extract_json('{"name": "test"}')
        assert data["name"] == "test"

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"name": "test"}\n```'
        data = _extract_json(raw)
        assert data["name"] == "test"

    def test_json_in_bare_fence(self):
        raw = '```\n{"order": 2}\n```'
        data = _extract_json(raw)
        assert data["order"] == 2

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            _extract_json("not json at all")


class TestDictToSpec:
    def test_converts_domain_lists_to_tuples(self):
        data = {
            "name": "ODE",
            "equation": "u_t = -u",
            "independent_vars": ["t"],
            "dependent_var": "u",
            "order": 1,
            "spatial_dim": 0,
            "domain": {"t": [0.0, 1.0]},
        }
        spec = _dict_to_spec(data)
        assert spec.domain["t"] == (0.0, 1.0)

    def test_defaults_for_optional_fields(self):
        data = {
            "name": "ODE",
            "equation": "u_t = -u",
            "independent_vars": ["t"],
            "order": 1,
            "domain": {"t": [0, 1]},
        }
        spec = _dict_to_spec(data)
        assert spec.dependent_var == "u"
        assert spec.is_linear is True
        assert spec.output_dim == 1

    def test_invalid_domain_raises(self):
        data = {
            "name": "Bad",
            "equation": "u = 0",
            "independent_vars": ["t"],
            "order": 1,
            "domain": {"t": [1, 2, 3]},
        }
        with pytest.raises(ValueError, match="Invalid domain bounds"):
            _dict_to_spec(data)


class TestPDEAgentParse:
    def test_parse_calls_llm_and_returns_spec(self):
        response_data = {
            "name": "Damped Harmonic Oscillator",
            "equation": "u_tt + mu*u_t + k*u = 0",
            "independent_vars": ["t"],
            "dependent_var": "u",
            "order": 2,
            "spatial_dim": 0,
            "domain": {"t": [0.0, 1.0]},
            "initial_conditions": ["u(0) = 1", "u'(0) = 0"],
            "boundary_conditions": [],
            "parameters": {"mu": 4.0, "k": 6400.0},
            "is_linear": True,
            "is_time_dependent": True,
            "has_high_frequency": True,
            "output_dim": 1,
        }

        mock_client = MagicMock()
        mock_client.ask.return_value = json.dumps(response_data)

        agent = PDEAgent(client=mock_client)
        spec = agent.parse("Solve the damped harmonic oscillator")

        assert spec.name == "Damped Harmonic Oscillator"
        assert spec.order == 2
        assert spec.has_high_frequency is True
        assert spec.parameters["k"] == 6400.0
        mock_client.ask.assert_called_once()

    def test_parse_handles_markdown_fence(self):
        response_data = {
            "name": "Decay",
            "equation": "u_t + u = 0",
            "independent_vars": ["t"],
            "order": 1,
            "spatial_dim": 0,
            "domain": {"t": [0, 1]},
        }
        mock_client = MagicMock()
        mock_client.ask.return_value = f"```json\n{json.dumps(response_data)}\n```"

        agent = PDEAgent(client=mock_client)
        spec = agent.parse("Exponential decay")
        assert spec.name == "Decay"
