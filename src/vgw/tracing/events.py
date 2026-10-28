"""Trace event schema.

Every interesting moment in a turn becomes one TraceEvent. The dashboard and the
benchmarks both compute their numbers from these events, so this file is the contract.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def now_ns() -> int:
    """High-resolution clock for all timing.

    perf_counter_ns rather than monotonic_ns: on older Python versions on Windows,
    monotonic() can tick as coarsely as ~15 ms, which is useless for latency work.
    """
    return time.perf_counter_ns()


class EventKind(str, Enum):
    # --- turn lifecycle
    TURN_START = "turn_start"
    TURN_END = "turn_end"  # attrs: outcome (TurnOutcome value)

    # --- user speech. SPEECH_END is the last voiced frame (may be recorded retroactively),
    # ENDPOINT is when the gateway decided the user is finished. The gap is the endpointing delay.
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    ENDPOINT = "endpoint"

    # --- STT
    STT_START = "stt_start"
    STT_END = "stt_end"  # attrs: chars

    # --- LLM
    LLM_START = "llm_start"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_FIRST_SENTENCE = "llm_first_sentence"
    LLM_END = "llm_end"

    # --- TTS
    TTS_START = "tts_start"
    TTS_FIRST_AUDIO = "tts_first_audio"
    TTS_END = "tts_end"

    # --- outbound audio to the client
    AUDIO_OUT_FIRST = "audio_out_first"  # time-to-first-audio is measured to here
    AUDIO_OUT_LAST = "audio_out_last"

    # --- interruption
    BARGE_IN = "barge_in"  # VAD heard the user while the agent was speaking
    CANCEL_START = "cancel_start"
    CANCEL_DONE = "cancel_done"  # all in-flight work for the old turn has stopped
    PLAYBACK_STOP_SENT = "playback_stop_sent"  # client told to stop playing

    # --- resilience
    FALLBACK_FILLER = "fallback_filler"
    PROVIDER_ERROR = "provider_error"  # attrs: provider, stage, transient
    PROVIDER_TIMEOUT = "provider_timeout"  # attrs: provider, stage


class TurnOutcome(str, Enum):
    COMPLETED = "completed"  # agent finished speaking
    INTERRUPTED = "interrupted"  # user barged in
    ERROR = "error"  # provider failure the gateway could not recover from
    ABANDONED = "abandoned"  # client disconnected mid-turn


@dataclass(frozen=True, slots=True)
class TraceEvent:
    session_id: str
    turn_id: int
    kind: EventKind
    ts_ns: int
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "kind": self.kind.value,
            "ts_ns": self.ts_ns,
            "attrs": self.attrs,
        }