"""Groq-hosted Whisper speech-to-text."""
from __future__ import annotations

import io
import wave

import groq

from vgw.audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from vgw.providers.base import STTProvider, Transcript
from vgw.providers.groq_client import translate_error

DEFAULT_MODEL = "whisper-large-v3-turbo"


def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class GroqSTT(STTProvider):
    name = "groq-stt"

    def __init__(self, client: groq.AsyncGroq, model: str = DEFAULT_MODEL):
        self._client = client
        self._model = model

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        extra = {"language": language} if language else {}  # omit = Whisper auto-detects
        try:
            result = await self._client.audio.transcriptions.create(
                file=("utterance.wav", pcm_to_wav(pcm)),
                model=self._model,
                response_format="json",
                temperature=0.0,
                **extra,
            )
        except groq.APIError as exc:
            raise translate_error(exc, self.name) from exc
        return Transcript(text=result.text.strip(), language=language)