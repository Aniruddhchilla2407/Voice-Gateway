"""FastAPI app: serves the browser client and the /ws audio endpoint.

Run:   uvicorn vgw.server:app --port 8000     then open http://localhost:8000
Metrics dashboard: http://localhost:8000/dashboard.html
(Do not use --reload with real providers on Windows: it switches the event loop type.)

Settings come from .env or environment variables, all optional:
  VGW_PROVIDERS=stub|bench|bench-real|groq   which provider set to use
  VGW_VAD=webrtc|synthetic        webrtc for real microphones, synthetic for load tests
  VGW_VAD_AGGRESSIVENESS=2        0..3, higher filters more noise
  VGW_MODE=gateway|naive          naive = sentence streaming and barge-in switched off
  VGW_LANGUAGE=en|hi|te           fixes the STT/TTS language; empty = auto-detect
  VGW_LOG_LEVEL=INFO              use WARNING during load tests
  VGW_HANGOVER_MS, VGW_BARGE_IN_MS   override the endpointing knobs
  GROQ_API_KEY, GROQ_LLM_MODEL, GROQ_STT_MODEL, GROQ_REASONING_EFFORT   for VGW_PROVIDERS=groq
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from vgw.config import GatewayConfig
from vgw.providers.base import ProviderSet
from vgw.providers.registry import build_providers
from vgw.session import Session
from vgw.tracing.metrics import MetricsStore
from vgw.vad import SyntheticVad, VoiceActivityDetector, WebRtcVad

CLIENT_DIR = Path(__file__).resolve().parents[2] / "client"  # <repo>/client


def _env_cfg() -> GatewayConfig:
    overrides: dict[str, Any] = {}
    if value := os.getenv("VGW_HANGOVER_MS"):
        overrides["hangover_ms"] = int(value)
    if value := os.getenv("VGW_BARGE_IN_MS"):
        overrides["barge_in_onset_ms"] = int(value)
    if value := os.getenv("VGW_LANGUAGE"):
        overrides["language"] = value
    if os.getenv("VGW_MODE", "gateway") == "naive":
        # Ablation baseline: the same code with sentence streaming and barge-in switched off.
        overrides.update(stream_sentences=False, barge_in_enabled=False)
    return replace(GatewayConfig(), **overrides)


def _default_vad() -> VoiceActivityDetector:
    if os.getenv("VGW_VAD", "webrtc") == "synthetic":
        return SyntheticVad()
    return WebRtcVad(int(os.getenv("VGW_VAD_AGGRESSIVENESS", "2")))


def create_app(
    *,
    providers_factory: Callable[[], ProviderSet] | None = None,
    vad_factory: Callable[[], VoiceActivityDetector] | None = None,
    cfg: GatewayConfig | None = None,
) -> FastAPI:
    load_dotenv()  # real environment variables win over .env
    logging.basicConfig(
        level=os.getenv("VGW_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(name)s %(message)s",
    )
    provider_set = os.getenv("VGW_PROVIDERS", "stub")
    providers_factory = providers_factory or (lambda: build_providers(provider_set))
    providers_factory()  # fail fast on bad configuration, e.g. a missing API key
    vad_factory = vad_factory or _default_vad
    cfg = cfg or _env_cfg()
    logging.getLogger("vgw.server").info(
        "settings: providers=%s vad=%s mode=%s hangover=%d ms barge_in_enabled=%s",
        provider_set,
        os.getenv("VGW_VAD", "webrtc"),
        os.getenv("VGW_MODE", "gateway"),
        cfg.hangover_ms,
        cfg.barge_in_enabled,
    )

    app = FastAPI(title="voice-gateway")
    app.state.metrics = MetricsStore()
    app.state.active_sessions = 0

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        store: MetricsStore = app.state.metrics
        return {"summary": store.summary(), "turns": [asdict(m) for m in store.recent(30)]}

    @app.get("/api/process")
    async def process_stats() -> dict[str, Any]:
        """CPU time and memory of this server process, for the load generator."""
        import psutil

        proc = psutil.Process()
        times = proc.cpu_times()
        return {
            "cpu_s": round(times.user + times.system, 3),
            "rss_mb": round(proc.memory_info().rss / 1e6, 1),
            "threads": proc.num_threads(),
            "sessions": app.state.active_sessions,
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        session = Session(
            ws,
            providers=providers_factory(),
            vad=vad_factory(),
            cfg=cfg,
            metrics=app.state.metrics,
        )
        app.state.active_sessions += 1
        try:
            await session.run()
        finally:
            app.state.active_sessions -= 1

    # Mounted last so it only handles paths the routes above did not.
    app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")
    return app


app = create_app()