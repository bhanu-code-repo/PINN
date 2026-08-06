"""Tests for LLMClient — all LiteLLM calls are mocked."""

import asyncio
from unittest.mock import MagicMock, patch

from llm_provider.client import LLMClient
from llm_provider.config import LLMConfig


def _mock_response(content: str = "mock answer"):
    """Build a fake LiteLLM response object."""
    msg = MagicMock()
    msg.__getitem__ = lambda self, key: {"content": content}[key]
    msg.get = lambda key, default=None: {"content": content}.get(key, default)

    choice = MagicMock()
    choice.__getitem__ = lambda self, key: {"message": msg, "delta": msg}[key]

    resp = MagicMock()
    resp.__getitem__ = lambda self, key: {"choices": [choice]}[key]
    return resp


class TestClientConstruction:
    def test_default_creates_cloud_config(self):
        client = LLMClient()
        assert client.config.api_base == "https://ollama.com"

    def test_local_constructor(self):
        client = LLMClient.local()
        assert client.config.api_base == "http://localhost:11434"

    def test_cloud_constructor(self):
        client = LLMClient.cloud()
        assert client.config.api_base == "https://ollama.com"

    def test_explicit_config(self):
        cfg = LLMConfig(model="test/model", api_base=None, api_key="k")
        client = LLMClient(config=cfg)
        assert client.config.model == "test/model"

    def test_kwarg_overrides(self):
        client = LLMClient(model="openai/gpt-4", temperature=0.2)
        assert client.config.model == "openai/gpt-4"
        assert client.config.temperature == 0.2

    def test_repr(self):
        client = LLMClient()
        assert "LLMClient" in repr(client)
        assert "ollama" in repr(client)


class TestSyncAsk:
    @patch("llm_provider.client.completion")
    def test_ask_string(self, mock_completion):
        mock_completion.return_value = _mock_response("hello")
        client = LLMClient(api_key="k")
        result = client.ask("hi")
        assert result == "hello"
        mock_completion.assert_called_once()

    @patch("llm_provider.client.completion")
    def test_ask_with_system(self, mock_completion):
        mock_completion.return_value = _mock_response("ok")
        client = LLMClient(api_key="k")
        client.ask("hi", system="You are helpful")
        args = mock_completion.call_args
        messages = args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("llm_provider.client.completion")
    def test_ask_message_list(self, mock_completion):
        mock_completion.return_value = _mock_response("ok")
        client = LLMClient(api_key="k")
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "sure"},
            {"role": "user", "content": "second"},
        ]
        client.ask(msgs)
        args = mock_completion.call_args
        assert len(args.kwargs["messages"]) == 3

    @patch("llm_provider.client.completion")
    def test_ask_batch(self, mock_completion):
        mock_completion.return_value = _mock_response("ans")
        client = LLMClient(api_key="k")
        results = client.ask_batch(["q1", "q2", "q3"])
        assert len(results) == 3
        assert mock_completion.call_count == 3


class TestAsyncAsk:
    @patch("llm_provider.client.acompletion")
    def test_ask_async(self, mock_acompletion):
        mock_acompletion.return_value = _mock_response("async hello")
        client = LLMClient(api_key="k")
        result = asyncio.run(client.ask_async("hi"))
        assert result == "async hello"

    @patch("llm_provider.client.acompletion")
    def test_ask_batch_async(self, mock_acompletion):
        mock_acompletion.return_value = _mock_response("ans")
        client = LLMClient(api_key="k")
        results = asyncio.run(client.ask_batch_async(["q1", "q2"]))
        assert len(results) == 2


class TestMessagePrep:
    def test_string_becomes_user_message(self):
        msgs = LLMClient._to_messages("hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_list_passes_through(self):
        original = [{"role": "user", "content": "hi"}]
        assert LLMClient._to_messages(original) is original
