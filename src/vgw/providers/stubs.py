"""Fake providers with controllable latency and failures.

Why these exist: they let us load-test the *gateway itself* with hundreds of concurrent
sessions, for free, with repeatable results, and inject faults on demand. Numbers from
stub runs measure gateway overhead, not model speed.
"""
from __future__ import annotations

import asyncio
import math
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from itertools import cycle
from typing import Any

from vgw.audio import bytes_to_ms, silence, tone
from vgw.providers.base import (
    LLMDelta,
    LLMProvider,
    Message,
    ProviderError,
    STTProvider,
    Transcript,
    TTSProvider,
)


@dataclass(frozen=True)
class LatencyProfile:
    """Lognormal latency: median_ms * exp(sigma * N(0,1)), plus an optional rare slow tail."""

    median_ms: float
    sigma: float = 0.25
    tail_prob: float = 0.0
    tail_multiplier: float = 6.0

    def sample_s(self, rng: random.Random) -> float:
        ms = self.median_ms * math.exp(rng.gauss(0.0, self.sigma))
        if self.tail_prob and rng.random() < self.tail_prob:
            ms *= self.tail_multiplier
        return ms / 1000.0

    def p95_ms(self) -> float:
        """Approximate p95 of the lognormal body (ignores the tail)."""
        return self.median_ms * math.exp(1.645 * self.sigma)


@dataclass(frozen=True)
class FaultSpec:
    error_prob: float = 0.0  # fail before producing anything
    stall_prob: float = 0.0  # hang for stall_s before continuing
    stall_s: float = 10.0
    mid_stream_error_prob: float = 0.0  # streams only: fail halfway through


NO_FAULTS = FaultSpec()

# Default latency profiles (rough, plausible numbers; override per test or benchmark).
STT_LATENCY = LatencyProfile(120)
LLM_FIRST_TOKEN = LatencyProfile(250)
LLM_INTER_TOKEN = LatencyProfile(25, sigma=0.3)
TTS_FIRST_AUDIO = LatencyProfile(150)


async def _apply_faults(rng: random.Random, faults: FaultSpec, provider: str) -> None:
    if faults.error_prob and rng.random() < faults.error_prob:
        raise ProviderError("injected failure", provider=provider, transient=True)
    if faults.stall_prob and rng.random() < faults.stall_prob:
        await asyncio.sleep(faults.stall_s)


class StubSTT(STTProvider):
    name = "stub-stt"

    def __init__(
        self,
        transcripts: Sequence[str] = ("hello, i would like to book an appointment",),
        latency: LatencyProfile = STT_LATENCY,
        faults: FaultSpec = NO_FAULTS,
        seed: int | None = None,
    ):
        self._transcripts = cycle(transcripts)
        self._latency = latency
        self._faults = faults
        self._rng = random.Random(seed)

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        await _apply_faults(self._rng, self._faults, self.name)
        await asyncio.sleep(self._latency.sample_s(self._rng))
        return Transcript(text=next(self._transcripts), language=language)


class StubLLM(LLMProvider):
    name = "stub-llm"

    def __init__(
        self,
        responses: Sequence[str] = ("Sure, I can help with that. What day works best for you?",),
        first_token: LatencyProfile = LLM_FIRST_TOKEN,
        inter_token: LatencyProfile = LLM_INTER_TOKEN,
        faults: FaultSpec = NO_FAULTS,
        seed: int | None = None,
    ):
        self._responses = cycle(responses)
        self._first_token = first_token
        self._inter_token = inter_token
        self._faults = faults
        self._rng = random.Random(seed)

    async def stream(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMDelta]:
        await _apply_faults(self._rng, self._faults, self.name)
        words = next(self._responses).split(" ")
        drop_at = (
            len(words) // 2
            if self._faults.mid_stream_error_prob
            and self._rng.random() < self._faults.mid_stream_error_prob
            else None
        )
        await asyncio.sleep(self._first_token.sample_s(self._rng))
        for i, word in enumerate(words):
            if i:
                await asyncio.sleep(self._inter_token.sample_s(self._rng))
            if drop_at is not None and i == drop_at:
                raise ProviderError("injected mid-stream failure", provider=self.name)
            yield LLMDelta(text=word + (" " if i < len(words) - 1 else ""))


class StubTTS(TTSProvider):
    name = "stub-tts"

    def __init__(
        self,
        ms_per_char: float = 65.0,  # roughly conversational speech
        chunk_ms: int = 100,
        first_audio: LatencyProfile = TTS_FIRST_AUDIO,
        realtime_factor: float = 8.0,  # generation speed vs playback speed
        audible: bool = False,  # True = 440 Hz beep instead of silence
        faults: FaultSpec = NO_FAULTS,
        seed: int | None = None,
    ):
        self._ms_per_char = ms_per_char
        self._chunk_ms = chunk_ms
        self._first_audio = first_audio
        self._realtime_factor = realtime_factor
        self._faults = faults
        self._rng = random.Random(seed)
        # Built once and reused: generating audio per call would burn event-loop CPU and
        # distort load tests. 440 Hz * 100 ms = 44 whole cycles, so joins are click-free.
        self._chunk = tone(chunk_ms, 440.0) if audible else silence(chunk_ms)

    async def synthesize(self, text: str, *, language: str | None = None) -> AsyncIterator[bytes]:
        await _apply_faults(self._rng, self._faults, self.name)
        total_ms = max(self._ms_per_char * len(text), self._chunk_ms)
        n_chunks = math.ceil(total_ms / self._chunk_ms)
        drop_at = (
            n_chunks // 2
            if self._faults.mid_stream_error_prob
            and self._rng.random() < self._faults.mid_stream_error_prob
            else None
        )
        await asyncio.sleep(self._first_audio.sample_s(self._rng))
        pace_s = (self._chunk_ms / 1000.0) / self._realtime_factor
        for i in range(n_chunks):
            if drop_at is not None and i == drop_at:
                raise ProviderError("injected mid-stream failure", provider=self.name)
            yield self._chunk
            await asyncio.sleep(pace_s)


def audio_ms(pcm: bytes) -> float:
    """Convenience for tests and benchmarks."""
    return bytes_to_ms(len(pcm))