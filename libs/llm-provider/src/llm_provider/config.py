"""LLM configuration with env / .env / kwargs resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env from project root (or wherever the process was launched)
load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "ollama/gpt-oss:120b"
DEFAULT_CLOUD_BASE = "https://ollama.com"
DEFAULT_LOCAL_BASE = "http://localhost:11434"


@dataclass
class LLMConfig:
    """Resolved LLM configuration.

    Resolution order: explicit kwarg > environment variable > default.

    For Ollama Cloud (default):
        model = "ollama/gpt-oss:120b"
        api_base = "https://ollama.com"
        api_key = $OLLAMA_API_KEY

    For local Ollama:
        LLMConfig.local()  — points at localhost:11434, no key needed.

    For any other provider (future):
        LLMConfig(model="anthropic/claude-sonnet-4-20250514", api_key="sk-...")
        LiteLLM routes it automatically.
    """

    model: str = DEFAULT_MODEL
    api_base: str | None = DEFAULT_CLOUD_BASE
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Resolve API key from env if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("LLM_API_KEY")

        # Auto-detect Ollama Cloud and inject auth header
        if self.api_base and "ollama.com" in self.api_base.lower() and self.api_key:
            self.extra_headers.setdefault("Authorization", f"Bearer {self.api_key}")

    # ----- Convenience constructors -----

    @classmethod
    def local(
        cls,
        model: str = "ollama/gpt-oss:120b",
        api_base: str = DEFAULT_LOCAL_BASE,
        **kwargs,
    ) -> LLMConfig:
        """Config for a local Ollama instance (no API key needed)."""
        return cls(model=model, api_base=api_base, api_key="", **kwargs)

    @classmethod
    def cloud(
        cls,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        **kwargs,
    ) -> LLMConfig:
        """Config for Ollama Cloud (reads OLLAMA_API_KEY from env)."""
        return cls(model=model, api_base=DEFAULT_CLOUD_BASE, api_key=api_key, **kwargs)

    def to_litellm_params(self, *, stream: bool = False) -> dict:
        """Build the kwargs dict for litellm.completion / acompletion."""
        params: dict = {
            "model": self.model,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            params["api_base"] = self.api_base
        if self.extra_headers:
            params["extra_headers"] = dict(self.extra_headers)
        elif self.api_key:
            # Non-Ollama providers: pass key directly, LiteLLM routes it
            params["api_key"] = self.api_key
        return params
