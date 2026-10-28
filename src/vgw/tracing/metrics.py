"""Turns raw trace events into numbers: per-turn metrics, percentiles, a rolling store."""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from vgw.tracing.events import EventKind as K
from vgw.tracing.recorder import TraceRecorder

LATENCY_FIELDS = (
    "ttfa_ms",
    "endpointing_ms",
    "stt_ms",
    "llm_first_token_ms",
    "llm_first_sentence_ms",
    "tts_first_audio_ms",
    "out_ms",
    "barge_in_cutoff_ms",
    "cancel_ms",
)


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    session_id: str
    turn_id: int
    outcome: str
    finished_at: float  # wall-clock seconds, for the dashboard
    # --- latency in ms; None when that stage never happened in this turn
    ttfa_ms: float | None  # last voiced frame -> first audio frame sent
    endpointing_ms: float | None  # last voiced frame -> "the user is done" decision
    stt_ms: float | None
    llm_first_token_ms: float | None
    llm_first_sentence_ms: float | None  # includes the first token
    tts_first_audio_ms: float | None
    out_ms: float | None  # TTS audio ready -> first frame sent
    barge_in_cutoff_ms: float | None  # interruption detected -> client told to stop
    cancel_ms: float | None  # cancel requested -> all work for the turn stopped
    # --- efficiency (characters: a rough stand-in for tokens, about 4 per token in English)
    input_chars: int | None
    history_messages: int | None
    generated_chars: int
    tts_chars: int
    spoken_chars: int


def turn_metrics(rec: TraceRecorder, turn_id: int) -> TurnMetrics | None:
    """Metrics for a finished turn, or None if it has no TURN_END yet."""
    end = rec.first(turn_id, K.TURN_END)
    if end is None:
        return None
    ms = rec.interval_ms
    llm_start = rec.first(turn_id, K.LLM_START)
    a = end.attrs
    return TurnMetrics(
        session_id=rec.session_id,
        turn_id=turn_id,
        outcome=str(a.get("outcome", "")),
        finished_at=time.time(),
        ttfa_ms=ms(turn_id, K.SPEECH_END, K.AUDIO_OUT_FIRST),
        endpointing_ms=ms(turn_id, K.SPEECH_END, K.ENDPOINT),
        stt_ms=ms(turn_id, K.STT_START, K.STT_END),
        llm_first_token_ms=ms(turn_id, K.LLM_START, K.LLM_FIRST_TOKEN),
        llm_first_sentence_ms=ms(turn_id, K.LLM_START, K.LLM_FIRST_SENTENCE),
        tts_first_audio_ms=ms(turn_id, K.TTS_START, K.TTS_FIRST_AUDIO),
        out_ms=ms(turn_id, K.TTS_FIRST_AUDIO, K.AUDIO_OUT_FIRST),
        barge_in_cutoff_ms=ms(turn_id, K.BARGE_IN, K.PLAYBACK_STOP_SENT),
        cancel_ms=ms(turn_id, K.CANCEL_START, K.CANCEL_DONE),
        input_chars=llm_start.attrs.get("input_chars") if llm_start else None,
        history_messages=llm_start.attrs.get("messages") if llm_start else None,
        generated_chars=int(a.get("generated_chars", 0)),
        tts_chars=int(a.get("tts_chars", 0)),
        spoken_chars=int(a.get("spoken_chars", 0)),
    )


def percentile(values: list[float], p: float) -> float:
    """p in 0..100, linear interpolation between the two nearest ranks."""
    if not values:
        raise ValueError("percentile of an empty list")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _share_unheard(heard: int, total: int) -> float | None:
    return round(1 - heard / total, 3) if total else None


def summarize(turns: list[TurnMetrics]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    for t in turns:
        outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1

    latency: dict[str, dict[str, float]] = {}
    for name in LATENCY_FIELDS:
        values = [v for t in turns if (v := getattr(t, name)) is not None]
        if values:
            latency[name] = {
                "n": len(values),
                "p50": round(percentile(values, 50), 1),
                "p95": round(percentile(values, 95), 1),
                "p99": round(percentile(values, 99), 1),
                "max": round(max(values), 1),
            }

    generated = sum(t.generated_chars for t in turns)
    tts = sum(t.tts_chars for t in turns)
    spoken = sum(t.spoken_chars for t in turns)
    inputs = [t.input_chars for t in turns if t.input_chars is not None]
    return {
        "turns": len(turns),
        "outcomes": outcomes,
        "latency": latency,
        "efficiency": {
            "generated_chars": generated,
            "tts_chars": tts,
            "spoken_chars": spoken,
            "unheard_generated_share": _share_unheard(spoken, generated),
            "unheard_tts_share": _share_unheard(spoken, tts),
            "avg_input_chars": round(sum(inputs) / len(inputs), 1) if inputs else None,
        },
    }


class MetricsStore:
    """Rolling window of recent turns across all sessions (in memory, bounded)."""

    def __init__(self, maxlen: int = 500):
        self._turns: deque[TurnMetrics] = deque(maxlen=maxlen)

    def add(self, metrics: TurnMetrics) -> None:
        self._turns.append(metrics)

    def recent(self, n: int = 30) -> list[TurnMetrics]:
        return list(self._turns)[-n:]

    def summary(self) -> dict[str, Any]:
        return summarize(list(self._turns))