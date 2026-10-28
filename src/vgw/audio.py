"""Audio constants and tiny PCM helpers.

Internal format everywhere in the gateway: PCM16, little-endian, mono, 16 kHz.
Provider adapters convert to/from whatever their API needs.
"""
from __future__ import annotations

import math
import sys
from array import array
from collections.abc import Iterator

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (int16)
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH  # 640 bytes


def ms_to_bytes(ms: int) -> int:
    return (SAMPLE_RATE * ms // 1000) * SAMPLE_WIDTH


def bytes_to_ms(n_bytes: int) -> float:
    return n_bytes * 1000 / (SAMPLE_RATE * SAMPLE_WIDTH)


def silence(ms: int) -> bytes:
    return bytes(ms_to_bytes(ms))


def tone(ms: int, freq: float = 440.0, amplitude: float = 0.2) -> bytes:
    """Sine tone as PCM16. Pick freq * ms / 1000 = whole number for click-free joins."""
    n = SAMPLE_RATE * ms // 1000
    peak = int(amplitude * 32767)
    buf = array(
        "h", (int(peak * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n))
    )
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def iter_frames(pcm: bytes, frame_bytes: int = FRAME_BYTES) -> Iterator[bytes]:
    """Split PCM into fixed-size frames; the last one is zero-padded."""
    for i in range(0, len(pcm), frame_bytes):
        frame = pcm[i : i + frame_bytes]
        if len(frame) < frame_bytes:
            frame += bytes(frame_bytes - len(frame))
        yield frame


class FrameSplitter:
    """Re-cuts arbitrarily sized byte chunks into fixed-size frames, carrying the remainder over.

    Needed on both sides: clients send whatever chunk size their audio API produces, and TTS
    engines yield chunks that are not multiples of 20 ms.
    """

    def __init__(self, frame_bytes: int = FRAME_BYTES):
        self._n = frame_bytes
        self._rest = b""

    def feed(self, data: bytes) -> list[bytes]:
        buf = self._rest + data
        cut = len(buf) - len(buf) % self._n
        self._rest = buf[cut:]
        return [buf[i : i + self._n] for i in range(0, cut, self._n)]

    def flush(self) -> bytes | None:
        """The leftover bytes as one zero-padded frame, or None if nothing is pending."""
        if not self._rest:
            return None
        frame = self._rest + bytes(self._n - len(self._rest))
        self._rest = b""
        return frame