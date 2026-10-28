"""Groq-hosted chat LLM, streamed."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import groq

from vgw.providers.base import LLMDelta, LLMProvider, Message
from vgw.providers.groq_client import translate_error

DEFAULT_MODEL = "openai/gpt-oss-120b"


def to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Our Message objects -> OpenAI-style dicts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            calls = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in m.tool_calls
            ]
            out.append({"role": "assistant", "content": m.content or None, "tool_calls": calls})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class GroqLLM(LLMProvider):
    name = "groq-llm"

    def __init__(
        self,
        client: groq.AsyncGroq,
        model: str = DEFAULT_MODEL,
        *,
        temperature: float = 0.6,
        max_tokens: int | None = None,
        reasoning_effort: str = "low",
    ):
        self._client = client
        self._model = model
        self._temperature = temperature
        # gpt-oss models "think" before answering, and the thinking tokens count against the
        # limit, so they need a much bigger budget than a plain chat model.
        self._reasoning = "gpt-oss" in model
        self._max_tokens = max_tokens or (1024 if self._reasoning else 300)
        self._reasoning_effort = reasoning_effort

    async def stream(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMDelta]:
        # `tools` is accepted for interface compatibility; the tool loop arrives with the agent step.
        extra: dict[str, Any] = {}
        if self._reasoning:
            # Low effort keeps the thinking short: it sits directly on the latency path.
            extra["extra_body"] = {"reasoning_effort": self._reasoning_effort}

        stream = None
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=to_wire(messages),
                stream=True,
                temperature=self._temperature,
                max_completion_tokens=self._max_tokens,
                **extra,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                # Only the answer text. Reasoning arrives in a separate field and is never spoken.
                text = chunk.choices[0].delta.content
                if text:
                    yield LLMDelta(text=text)
        except groq.APIError as exc:
            raise translate_error(exc, self.name) from exc
        finally:
            if stream is not None:
                await stream.close()  # on barge-in this drops the HTTP stream immediately