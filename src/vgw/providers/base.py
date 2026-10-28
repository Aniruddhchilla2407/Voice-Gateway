"""Provider interfaces: the only thing the pipeline knows about STT / LLM / TTS.

Cancellation contract: the gateway cancels by cancelling the asyncio task that is
consuming a provider call. Providers must let asyncio.CancelledError propagate
(never swallow it) and release network resources in `finally` / `async with`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal


class ProviderError(Exception):
    """A provider call failed. transient=True means a retry or fallback may help."""

    def __init__(self, message: str, *, provider: str = "", transient: bool = True):
        super().__init__(message)
        self.provider = provider
        self.transient = transient


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class LLMDelta:
    """One streamed piece of LLM output: a text fragment or a tool call."""

    text: str = ""
    tool_call: ToolCall | None = None


class STTProvider(ABC):
    name: str = "stt"

    @abstractmethod
    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        """Transcribe one finished utterance (PCM16 mono 16 kHz).

        The gateway's own VAD decides where utterances end, so STT adapters never need
        to be streaming. That keeps request/response APIs (like Whisper) usable.
        """


class LLMProvider(ABC):
    name: str = "llm"

    @abstractmethod
    def stream(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMDelta]:
        """Stream the reply. Implement as an `async def` generator that yields LLMDelta."""


class TTSProvider(ABC):
    name: str = "tts"

    @abstractmethod
    def synthesize(self, text: str, *, language: str | None = None) -> AsyncIterator[bytes]:
        """Stream PCM16 mono 16 kHz audio for `text`. Adapters convert from their native format."""


@dataclass(frozen=True, slots=True)
class ProviderSet:
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider