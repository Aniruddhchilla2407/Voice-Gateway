"""Builds a set of providers for each session, chosen by name (VGW_PROVIDERS)."""
from __future__ import annotations

import os

from vgw.providers.base import ProviderSet
from vgw.providers.stubs import LatencyProfile, StubLLM, StubSTT, StubTTS

DEMO_REPLY = (
    "Sure, I can help with that. We have openings on Monday, Tuesday and Wednesday. "
    "Which of those days works best for you?"
)
BENCH_REPLY = "Sure, I can help with that. Which day works best for you?"


def build_providers(name: str) -> ProviderSet:
    if name == "stub":
        # The stub STT ignores what you say and the TTS "speaks" as a 440 Hz beep.
        # That is enough to test turn-taking and barge-in by ear.
        return ProviderSet(
            stt=StubSTT(),
            llm=StubLLM(responses=[DEMO_REPLY]),
            tts=StubTTS(audible=True),
        )
    if name == "bench":
        # Silent audio, near-zero provider latency: what you measure is the gateway itself.
        fast = LatencyProfile(0, sigma=0.0)
        return ProviderSet(
            stt=StubSTT(latency=fast),
            llm=StubLLM(responses=[BENCH_REPLY], first_token=fast, inter_token=fast),
            tts=StubTTS(ms_per_char=40, first_audio=fast, realtime_factor=100.0),
        )
    if name == "bench-real":
        # Silent audio, plausible provider latencies: for comparing gateway modes under load.
        return ProviderSet(
            stt=StubSTT(),
            llm=StubLLM(responses=[DEMO_REPLY]),
            tts=StubTTS(ms_per_char=40),
        )
    if name == "groq":
        # Imported here so the stub setups work without the optional "providers" extra.
        from vgw.providers.groq_client import get_client
        from vgw.providers.llm_groq import DEFAULT_MODEL as LLM_MODEL
        from vgw.providers.llm_groq import GroqLLM
        from vgw.providers.stt_groq import DEFAULT_MODEL as STT_MODEL
        from vgw.providers.stt_groq import GroqSTT
        from vgw.providers.tts_edge import EdgeTTS

        client = get_client()  # raises a clear error if GROQ_API_KEY is missing
        return ProviderSet(
            stt=GroqSTT(client, os.getenv("GROQ_STT_MODEL") or STT_MODEL),
            llm=GroqLLM(
                client,
                os.getenv("GROQ_LLM_MODEL") or LLM_MODEL,
                reasoning_effort=os.getenv("GROQ_REASONING_EFFORT") or "low",
            ),
            tts=EdgeTTS(),
        )
    raise ValueError(f"unknown provider set {name!r} (available: stub, bench, bench-real, groq)")