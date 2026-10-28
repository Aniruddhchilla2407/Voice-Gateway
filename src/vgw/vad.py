"""Voice activity detection and endpointing.

Two layers:
  * a VoiceActivityDetector answers one question per 20 ms frame: is this speech?
  * the Endpointer turns that noisy per-frame stream into clean events:
    SpeechStarted, and SpeechEnded carrying the finished utterance's audio.

The Endpointer is deterministic and clock-free (it counts frames), so it is easy to test.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from vgw.audio import FRAME_BYTES, FRAME_MS, SAMPLE_RATE


class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: bytes) -> bool: ...


class SyntheticVad:
    """Any non-zero byte counts as speech.

    Only valid for synthetic audio where silence is exactly zero. It costs ~1 microsecond per
    frame, which matters for load tests: a real VAD inside the load generator's hot path would
    distort the numbers we are trying to measure.
    """

    def is_speech(self, frame: bytes) -> bool:
        return bool(frame.strip(b"\x00"))


class WebRtcVad:
    """Adapter for Google's WebRTC VAD (pip install -e ".[vad]"). 20 ms frames at 16 kHz."""

    def __init__(self, aggressiveness: int = 2):
        """aggressiveness 0..3: higher filters more noise but may clip quiet speech."""
        try:
            import webrtcvad
        except ImportError as e:
            raise RuntimeError('webrtcvad is not installed: pip install -e ".[vad]"') from e
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        return self._vad.is_speech(frame, SAMPLE_RATE)


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """Speech onset confirmed (it lasted long enough to rule out a click or cough)."""


@dataclass(frozen=True, slots=True)
class SpeechEnded:
    """The user stopped talking. `pcm` is the whole utterance, ready for STT."""

    pcm: bytes
    forced: bool = False  # True if cut at max_utterance_ms rather than by silence


VadEvent = SpeechStarted | SpeechEnded


class Endpointer:
    """Frame-by-frame turn detection.

    Knobs (the latency / cut-off trade-off lives here):
      onset_ms      speech needed before we believe the user started. Public and mutable: the
                    turn manager raises it while the agent is talking, so noise and short
                    "mm-hmm"s do not interrupt it.
      onset_ratio   share of voiced frames required inside that window. Real VADs flicker
                    inside genuine speech, so demanding 100% would delay every onset.
      hangover_ms   silence needed to end the utterance. Lower = snappier replies but more
                    users cut off mid-sentence. Public and mutable.
      preroll_ms    audio kept from before the confirmed onset, so the first syllable is not
                    clipped (the onset window itself is always kept too).
      tail_ms       trailing silence kept after the last voiced frame.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        *,
        onset_ms: int = 60,
        onset_ratio: float = 0.8,
        hangover_ms: int = 600,
        preroll_ms: int = 200,
        tail_ms: int = 100,
        max_utterance_ms: int = 15_000,
        max_onset_ms: int = 1_000,
    ):
        self._vad = vad
        self.onset_ms = onset_ms
        self.hangover_ms = hangover_ms
        self._ratio = onset_ratio
        self._preroll_frames = preroll_ms // FRAME_MS
        self._tail_frames = tail_ms // FRAME_MS
        self._max_utterance_ms = max_utterance_ms
        self._pre: deque[bytes] = deque(maxlen=(preroll_ms + max_onset_ms) // FRAME_MS)
        self._flags: deque[bool] = deque(maxlen=max(1, max_onset_ms // FRAME_MS))
        self._utt: list[bytes] = []
        self._in_speech = False
        self._silence_run = 0
        self._last_voiced_idx = -1
        self.last_frame_voiced = False  # lets the caller timestamp the last voiced frame

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._pre.clear()
        self._flags.clear()
        self._utt = []
        self._in_speech = False
        self._silence_run = 0
        self._last_voiced_idx = -1

    def process(self, frame: bytes) -> list[VadEvent]:
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"expected a {FRAME_BYTES}-byte frame, got {len(frame)}")
        voiced = self._vad.is_speech(frame)
        self.last_frame_voiced = voiced
        return self._in_utterance(frame, voiced) if self._in_speech else self._idle(frame, voiced)

    # ---- internals
    def _idle(self, frame: bytes, voiced: bool) -> list[VadEvent]:
        self._pre.append(frame)
        self._flags.append(voiced)
        if not voiced:
            return []
        n = min(max(1, math.ceil(self.onset_ms / FRAME_MS)), self._flags.maxlen or 1)
        if len(self._flags) < n:
            return []
        if sum(list(self._flags)[-n:]) < math.ceil(n * self._ratio):
            return []
        self._utt = list(self._pre)[-(self._preroll_frames + n) :]
        self._in_speech = True
        self._silence_run = 0
        self._last_voiced_idx = len(self._utt) - 1
        self._pre.clear()
        self._flags.clear()
        return [SpeechStarted()]

    def _in_utterance(self, frame: bytes, voiced: bool) -> list[VadEvent]:
        self._utt.append(frame)
        if voiced:
            self._silence_run = 0
            self._last_voiced_idx = len(self._utt) - 1
        else:
            self._silence_run += 1
        ended_by_silence = self._silence_run * FRAME_MS >= self.hangover_ms
        too_long = len(self._utt) * FRAME_MS >= self._max_utterance_ms
        if ended_by_silence or too_long:
            return [self._finish(forced=too_long and not ended_by_silence)]
        return []

    def _finish(self, *, forced: bool) -> SpeechEnded:
        end = min(len(self._utt), self._last_voiced_idx + 1 + self._tail_frames)
        pcm = b"".join(self._utt[:end])
        self.reset()
        return SpeechEnded(pcm=pcm, forced=forced)