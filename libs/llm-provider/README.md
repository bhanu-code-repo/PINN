# llm-provider

LLM abstraction layer for the PINN workspace. Uses [LiteLLM](https://docs.litellm.ai/) under the hood — Ollama (local + cloud) out of the box, any LiteLLM-supported provider (Anthropic, OpenAI, Vertex, etc.) via config.

## Quick Start

```python
from llm_provider import LLMClient

# Ollama Cloud (default — reads OLLAMA_API_KEY from .env)
client = LLMClient()
answer = client.ask("What is a Physics-Informed Neural Network?")

# Local Ollama (no API key needed)
client = LLMClient.local()

# Any provider — just set model + key
client = LLMClient(model="anthropic/claude-sonnet-4-20250514", api_key="sk-...")
```

## Configuration

Settings resolve in order: **explicit kwarg > env var > `.env` file > default**.

| Setting | Default | Env Var |
|---------|---------|---------|
| `model` | `ollama/gpt-oss:120b` | — |
| `api_base` | `https://ollama.com` | — |
| `api_key` | — | `OLLAMA_API_KEY` or `LLM_API_KEY` |
| `temperature` | 0.7 | — |
| `max_tokens` | 4096 | — |

Create a `.env` file in the project root:

```bash
OLLAMA_API_KEY=your-key-here
```

## API

### Sync

```python
client.ask("prompt")                        # single query
client.ask("prompt", system="You are...")    # with system message
client.ask("prompt", stream=True)           # streaming (prints tokens)
client.ask_batch(["q1", "q2", "q3"])        # multiple queries
```

### Async

```python
await client.ask_async("prompt")
await client.ask_batch_async(["q1", "q2"])  # concurrent
```

### Config Object

```python
from llm_provider import LLMConfig

cfg = LLMConfig.cloud()                     # Ollama Cloud
cfg = LLMConfig.local()                     # localhost:11434
cfg = LLMConfig(model="openai/gpt-4o")     # any provider

client = LLMClient(config=cfg)
```

## Adding a New Provider

No code changes needed — LiteLLM routes by model prefix:

```python
# Anthropic
client = LLMClient(model="anthropic/claude-sonnet-4-20250514", api_key="sk-ant-...")

# OpenAI
client = LLMClient(model="openai/gpt-4o", api_key="sk-...")

# Google Vertex
client = LLMClient(model="vertex_ai/gemini-pro")

# Azure
client = LLMClient(model="azure/gpt-4", api_base="https://my-resource.openai.azure.com")
```

See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for the full list.
