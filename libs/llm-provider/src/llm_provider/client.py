"""LLM client — thin wrapper over LiteLLM with sync + async support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from litellm import acompletion, completion

from .config import LLMConfig

if TYPE_CHECKING:
    pass

Message = dict[str, str]  # {"role": "user", "content": "..."}


class LLMClient:
    """Sync + async LLM client backed by LiteLLM.

    Usage::

        from llm_provider import LLMClient

        # Ollama Cloud (default — reads OLLAMA_API_KEY from .env)
        client = LLMClient()
        answer = client.ask("What is a PINN?")

        # Local Ollama
        client = LLMClient.local()
        answer = client.ask("Explain Burgers' equation")

        # Any provider (future)
        client = LLMClient(model="anthropic/claude-sonnet-4-20250514", api_key="sk-...")
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if config is not None:
            self.config = config
        else:
            kwargs: dict = {}
            if model is not None:
                kwargs["model"] = model
            if api_key is not None:
                kwargs["api_key"] = api_key
            if api_base is not None:
                kwargs["api_base"] = api_base
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            self.config = LLMConfig(**kwargs)

    # ----- Convenience constructors -----

    @classmethod
    def local(cls, **kwargs) -> LLMClient:
        """Client pointing at a local Ollama instance."""
        return cls(config=LLMConfig.local(**kwargs))

    @classmethod
    def cloud(cls, **kwargs) -> LLMClient:
        """Client pointing at Ollama Cloud."""
        return cls(config=LLMConfig.cloud(**kwargs))

    # ----- Message preparation -----

    @staticmethod
    def _to_messages(prompt: str | list[Message]) -> list[Message]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    # ----- Sync API -----

    def ask(
        self,
        prompt: str | list[Message],
        *,
        stream: bool = False,
        system: str | None = None,
    ) -> str:
        """Send a prompt and return the response text.

        Args:
            prompt: A string or list of message dicts.
            stream: If True, print tokens as they arrive and return full text.
            system: Optional system message prepended to the conversation.
        """
        messages = self._to_messages(prompt)
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        params = self.config.to_litellm_params(stream=stream)

        resp = completion(messages=messages, **params)

        if stream:
            text = ""
            for chunk in resp:
                content = chunk["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
                text += content
            print()
            return text

        return resp["choices"][0]["message"]["content"]

    def ask_batch(
        self,
        prompts: list[str],
        *,
        system: str | None = None,
    ) -> list[str]:
        """Send multiple independent prompts sequentially."""
        return [self.ask(p, system=system) for p in prompts]

    # ----- Async API -----

    async def ask_async(
        self,
        prompt: str | list[Message],
        *,
        stream: bool = False,
        system: str | None = None,
    ) -> str:
        """Async version of ask()."""
        messages = self._to_messages(prompt)
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        params = self.config.to_litellm_params(stream=stream)

        resp = await acompletion(messages=messages, **params)

        if stream:
            text = ""
            async for chunk in resp:
                content = chunk["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
                text += content
            print()
            return text

        return resp["choices"][0]["message"]["content"]

    async def ask_batch_async(
        self,
        prompts: list[str],
        *,
        system: str | None = None,
    ) -> list[str]:
        """Send multiple prompts concurrently."""
        tasks = [self.ask_async(p, system=system) for p in prompts]
        return await asyncio.gather(*tasks)

    # ----- Repr -----

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.config.model!r}, "
            f"api_base={self.config.api_base!r})"
        )
