"""llm-provider — LLM abstraction layer for the PINN workspace.

Ollama (local + cloud) out of the box. Any LiteLLM-supported provider
(Anthropic, OpenAI, Vertex, etc.) works by changing ``model`` and ``api_key``.

Quick start::

    from llm_provider import LLMClient

    # Ollama Cloud (default)
    client = LLMClient()
    print(client.ask("What is a Physics-Informed Neural Network?"))

    # Local Ollama
    client = LLMClient.local()

    # Any provider — just set model + key
    client = LLMClient(model="anthropic/claude-sonnet-4-20250514", api_key="sk-...")
"""

from .client import LLMClient
from .config import LLMConfig

__all__ = ["LLMClient", "LLMConfig"]
