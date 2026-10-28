"""One WebSocket connection = one Session: its own conversation, providers, trace, turn manager.

Wire protocol (browser <-> gateway):
  client -> server  binary: PCM16 mono 16 kHz audio, any chunk size
  server -> client  binary: 4-byte big-endian turn id + one 20 ms PCM16 frame
                    text:   {"type": "stop", "turn_id": N}
                            = stop playing and drop buffered audio for turns up to N
"""
from __future__ import annotations

import asyncio
import logging
import struct
import uuid

from fastapi import WebSocket

from vgw.agent.prompts import SYSTEM_PROMPT
from vgw.audio import FrameSplitter
from vgw.config import GatewayConfig
from vgw.history import Conversation
from vgw.providers.base import ProviderSet
from vgw.tracing.events import EventKind, TraceEvent
from vgw.tracing.metrics import MetricsStore, TurnMetrics, turn_metrics
from vgw.tracing.recorder import TraceRecorder
from vgw.turn_manager import TurnManager
from vgw.vad import VoiceActivityDetector

logger = logging.getLogger("vgw.session")

_HEADER = struct.Struct(">I")


class WebSocketSink:
    def __init__(self, ws: WebSocket):
        self._ws = ws
        # Audio frames (playout task) and stop messages (receive loop) share one socket.
        self._lock = asyncio.Lock()

    async def send_audio(self, turn_id: int, pcm: bytes) -> None:
        async with self._lock:
            await self._ws.send_bytes(_HEADER.pack(turn_id) + pcm)

    async def send_stop(self, turn_id: int) -> None:
        async with self._lock:
            await self._ws.send_json({"type": "stop", "turn_id": turn_id})


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _log_turn(session_id: str, m: TurnMetrics) -> None:
    parts = [
        ("TTFA", m.ttfa_ms),
        ("endpointing", m.endpointing_ms),
        ("stt", m.stt_ms),
        ("llm first token", m.llm_first_token_ms),
        ("llm first sentence", m.llm_first_sentence_ms),
        ("tts first audio", m.tts_first_audio_ms),
        ("out", m.out_ms),
    ]
    detail = " | ".join(f"{name} {_ms(value)}" for name, value in parts)
    logger.info(
        "session %s turn %d %s | %s | chars generated/tts/heard %d/%d/%d",
        session_id,
        m.turn_id,
        m.outcome,
        detail,
        m.generated_chars,
        m.tts_chars,
        m.spoken_chars,
    )


class Session:
    def __init__(
        self,
        ws: WebSocket,
        *,
        providers: ProviderSet,
        vad: VoiceActivityDetector,
        cfg: GatewayConfig,
        metrics: MetricsStore,
    ):
        self.id = uuid.uuid4().hex[:8]
        self._ws = ws
        self._metrics = metrics
        self.recorder = TraceRecorder(self.id, on_event=self._on_event)
        self._tm = TurnManager(
            providers=providers,
            conversation=Conversation(SYSTEM_PROMPT, max_messages=cfg.max_history_messages),
            sink=WebSocketSink(ws),
            recorder=self.recorder,
            vad=vad,
            cfg=cfg,
        )

    async def run(self) -> None:
        await self._ws.accept()
        logger.info("session %s connected", self.id)
        splitter = FrameSplitter()
        try:
            while True:
                message = await self._ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data:
                    for frame in splitter.feed(data):
                        await self._tm.on_frame(frame)
        finally:
            await self._tm.close()
            turns = max((e.turn_id for e in self.recorder.events), default=0)
            logger.info("session %s closed after %d turns", self.id, turns)

    def _on_event(self, event: TraceEvent) -> None:
        if event.kind is not EventKind.TURN_END:
            return
        metrics = turn_metrics(self.recorder, event.turn_id)
        if metrics is not None:
            self._metrics.add(metrics)
            _log_turn(self.id, metrics)