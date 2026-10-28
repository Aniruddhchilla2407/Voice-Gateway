"""Microsoft Edge neural voices (edge-tts package), decoded from MP3 to PCM with ffmpeg.

First version: each chunk (sentence) is fetched completely, then decoded in one ffmpeg call.
If the trace shows TTS time-to-first-audio dominating, streaming decode is the optimization.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator
from contextlib import aclosing

import edge_tts

from vgw.audio import SAMPLE_RATE, ms_to_bytes
from vgw.providers.base import ProviderError, TTSProvider

logger = logging.getLogger(__name__)

VOICES = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
}
_OUT_CHUNK = ms_to_bytes(100)  # hand PCM to the pipeline in 100 ms pieces


def voice_for(text: str, language: str | None) -> str:
    """An explicit language wins; otherwise pick the voice from the script of the text."""
    if language:
        voice = VOICES.get(language.split("-")[0].lower())
        if voice:
            return voice
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:  # Devanagari
            return VOICES["hi"]
        if 0x0C00 <= code <= 0x0C7F:  # Telugu
            return VOICES["te"]
    return VOICES["en"]


def _ffmpeg_decode(mp3: bytes) -> bytes:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "mp3", "-i", "pipe:0",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, input=mp3, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ProviderError(
            "ffmpeg was not found on PATH", provider="edge-tts", transient=False
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace")[:200]
        raise ProviderError(f"ffmpeg could not decode: {detail}", provider="edge-tts", transient=False)
    return proc.stdout


async def decode_mp3(mp3: bytes) -> bytes:
    """MP3 bytes -> PCM16 mono 16 kHz. Runs in a thread so it works on any event loop."""
    return await asyncio.to_thread(_ffmpeg_decode, mp3)


class EdgeTTS(TTSProvider):
    name = "edge-tts"

    async def synthesize(self, text: str, *, language: str | None = None) -> AsyncIterator[bytes]:
        mp3 = bytearray()
        try:
            communicate = edge_tts.Communicate(text, voice_for(text, language))
            async with aclosing(communicate.stream()) as stream:
                async for chunk in stream:
                    if chunk["type"] == "audio":
                        mp3 += chunk["data"]
        except Exception as exc:  # noqa: BLE001 - network, websocket and library errors alike
            logger.warning("edge-tts failed: %r", exc)
            raise ProviderError(
                f"edge-tts: {type(exc).__name__}", provider=self.name, transient=True
            ) from exc
        if not mp3:
            raise ProviderError("edge-tts returned no audio", provider=self.name, transient=True)
        pcm = await decode_mp3(bytes(mp3))
        for i in range(0, len(pcm), _OUT_CHUNK):
            yield pcm[i : i + _OUT_CHUNK]