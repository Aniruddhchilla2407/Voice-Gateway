"""All tunables in one place. Frozen: derive variants with dataclasses.replace(cfg, ...)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    # --- endpointing (see vad.Endpointer)
    onset_ms: int = 60  # speech needed to start a turn while the agent is silent
    barge_in_onset_ms: int = 200  # speech needed to interrupt the agent while it is talking
    onset_ratio: float = 0.8
    hangover_ms: int = 600  # silence that ends the user's turn: the main latency/cut-off knob
    preroll_ms: int = 200
    tail_ms: int = 100
    max_utterance_ms: int = 15_000

    # --- conversation
    language: str | None = None  # passed to STT and TTS; None = let providers decide
    first_chunk_min_chars: int = 20  # see chunker: earliest clause-boundary flush
    max_history_messages: int = 40

    # --- pipeline plumbing
    chunk_queue_size: int = 4  # LLM -> TTS. Small on purpose: backpressure should reach the LLM
    playout_queue_frames: int = 100  # TTS -> client; 20 ms each, so 1 s of audio
    client_buffer_ms: float = 40.0  # audio the client holds before playing; subtracted when
    # working out how much the user actually heard

    # --- timeouts (seconds)
    stt_timeout_s: float = 8.0
    llm_first_token_timeout_s: float = 8.0
    tts_first_audio_timeout_s: float = 8.0  # applied to the start of every TTS chunk
    cancel_timeout_s: float = 1.0  # how long barge-in waits for cancelled work to stop
    stream_sentences: bool = True
    barge_in_enabled: bool = True