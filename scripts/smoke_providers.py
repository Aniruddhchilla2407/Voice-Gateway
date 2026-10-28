"""Smoke test for the real providers: edge-tts -> Groq STT -> Groq LLM.

Needs GROQ_API_KEY in .env and ffmpeg on PATH. Run from the repo root:
    python scripts/smoke_providers.py
    python scripts/smoke_providers.py --lang hi --text "नमस्ते, मुझे मंगलवार के लिए अपॉइंटमेंट चाहिए"
"""
from __future__ import annotations

import argparse
import asyncio
import time

from dotenv import load_dotenv

from vgw.agent.prompts import SYSTEM_PROMPT
from vgw.audio import SAMPLE_RATE, SAMPLE_WIDTH
from vgw.providers.base import Message
from vgw.providers.groq_client import get_client
from vgw.providers.registry import build_providers


def ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


async def main(text: str, lang: str | None) -> None:
    load_dotenv()
    p = build_providers("groq")
    stt_model = getattr(p.stt, "_model", "?")
    llm_model = getattr(p.llm, "_model", "?")
    effort = getattr(p.llm, "_reasoning_effort", "-")
    print(f"models: stt={stt_model} llm={llm_model} effort={effort}")

    # Baseline: a trivial API call, so mostly network round trip + request overhead.
    client = get_client()
    for label in ("cold", "warm"):
        t0 = time.perf_counter()
        await client.models.list()
        print(f"NET  Groq round trip, {label}: {ms_since(t0):.0f} ms")

    t0 = time.perf_counter()
    pcm = b"".join([c async for c in p.tts.synthesize(text, language=lang)])
    seconds = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
    print(f"TTS  {seconds:.1f} s of audio in {ms_since(t0):.0f} ms")

    t0 = time.perf_counter()
    heard = await p.stt.transcribe(pcm, language=lang)
    print(f"STT  {ms_since(t0):.0f} ms -> {heard.text!r}")

    t0 = time.perf_counter()
    first_ms = float("nan")
    reply = ""
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=heard.text),
    ]
    async for delta in p.llm.stream(messages):
        if not reply and delta.text:
            first_ms = ms_since(t0)
        reply += delta.text
    print(f"LLM  first token {first_ms:.0f} ms, complete in {ms_since(t0):.0f} ms")
    print(f"     {reply!r}")

    await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello, I would like to book an appointment for Tuesday.")
    parser.add_argument("--lang", default=None, help="en, hi or te (default: auto-detect)")
    args = parser.parse_args()
    asyncio.run(main(args.text, args.lang))