"""Provider-agnostic LLM client wrapper.

Normalizes OpenAI and Anthropic chat APIs into a single interface that returns
either a final text message or a list of tool calls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """A normalized tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """Normalized chat completion response."""

    text: str | None  # final assistant text (None if only tool calls)
    tool_calls: list[ToolCall]  # tool calls requested (empty if final text)
    raw: Any  # provider-native response, for debugging
    finish_reason: str | None = None


class LLMClient:
    """Abstract base class."""

    DEFAULT_MODEL: str = "<unset>"

    def __init__(self, model: str | None = None):
        self.model = model or self.DEFAULT_MODEL

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        calls: list[ToolCall] = []
        if msg.tool_calls:
            import json

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ChatResponse(
            text=msg.content,
            tool_calls=calls,
            raw=resp,
            finish_reason=choice.finish_reason,
        )


class AnthropicClient(LLMClient):
    DEFAULT_MODEL = "claude-3-5-sonnet-latest"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=api_key)

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        # Anthropic uses a separate `system` field, not a system message in the list.
        system = None
        anthropic_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] if isinstance(m["content"], str) else m["content"]
            else:
                anthropic_messages.append(m)

        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if block.input else {},
                    )
                )

        return ChatResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=calls,
            raw=resp,
            finish_reason=resp.stop_reason,
        )


def make_client(provider: str, model: str | None = None) -> LLMClient:
    """Factory for the supported providers."""
    if provider == "openai":
        return OpenAIClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic.")
