"""Tests for LLMConfig resolution and constructors."""

import os
from unittest.mock import patch

from llm_provider.config import LLMConfig


class TestDefaults:
    def test_default_model(self):
        cfg = LLMConfig()
        assert cfg.model == "ollama/gpt-oss:120b"

    def test_default_api_base_is_cloud(self):
        cfg = LLMConfig()
        assert cfg.api_base == "https://ollama.com"

    def test_default_temperature(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.7


class TestEnvResolution:
    def test_reads_ollama_api_key(self):
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "test-key-123"}, clear=False):
            cfg = LLMConfig()
        assert cfg.api_key == "test-key-123"

    def test_reads_llm_api_key_fallback(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "fallback-key"}, clear=False):
            # Remove OLLAMA_API_KEY if present
            env = {k: v for k, v in os.environ.items() if k != "OLLAMA_API_KEY"}
            env["LLM_API_KEY"] = "fallback-key"
            with patch.dict(os.environ, env, clear=True):
                cfg = LLMConfig()
        assert cfg.api_key == "fallback-key"

    def test_explicit_key_overrides_env(self):
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "env-key"}, clear=False):
            cfg = LLMConfig(api_key="explicit-key")
        assert cfg.api_key == "explicit-key"


class TestOllamaCloudAuth:
    def test_cloud_injects_bearer_header(self):
        cfg = LLMConfig(api_key="my-key")
        assert cfg.extra_headers["Authorization"] == "Bearer my-key"

    def test_non_ollama_base_no_auth_header(self):
        cfg = LLMConfig(api_base="https://api.openai.com", api_key="sk-xxx")
        assert "Authorization" not in cfg.extra_headers


class TestConstructors:
    def test_local_points_at_localhost(self):
        cfg = LLMConfig.local()
        assert cfg.api_base == "http://localhost:11434"
        assert cfg.api_key == ""

    def test_cloud_points_at_ollama_com(self):
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "cloud-key"}, clear=False):
            cfg = LLMConfig.cloud()
        assert cfg.api_base == "https://ollama.com"
        assert cfg.api_key == "cloud-key"

    def test_local_custom_model(self):
        cfg = LLMConfig.local(model="ollama/llama3:8b")
        assert cfg.model == "ollama/llama3:8b"


class TestToLitellmParams:
    def test_includes_model_and_stream(self):
        cfg = LLMConfig(api_key="k")
        params = cfg.to_litellm_params(stream=True)
        assert params["model"] == "ollama/gpt-oss:120b"
        assert params["stream"] is True

    def test_cloud_uses_extra_headers(self):
        cfg = LLMConfig(api_key="k")
        params = cfg.to_litellm_params()
        assert "extra_headers" in params
        assert "api_key" not in params  # key is in header, not as param

    def test_non_ollama_uses_api_key_param(self):
        cfg = LLMConfig(model="openai/gpt-4", api_base=None, api_key="sk-xxx")
        params = cfg.to_litellm_params()
        assert params["api_key"] == "sk-xxx"
        assert "extra_headers" not in params
