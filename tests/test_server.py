import json
import struct

from fastapi.testclient import TestClient

from vgw.audio import FRAME_MS, silence, tone
from vgw.config import GatewayConfig
from vgw.providers.base import ProviderSet
from vgw.providers.stubs import LatencyProfile, StubLLM, StubSTT, StubTTS
from vgw.server import create_app
from vgw.vad import SyntheticVad

SPEECH = tone(FRAME_MS)
QUIET = silence(FRAME_MS)
FAST = LatencyProfile(median_ms=1, sigma=0.0)
LONG = (
    "Sure, I can help with that. We have openings on Monday, Tuesday and Wednesday. "
    "Which of those days works best for you?"
)
CFG = GatewayConfig(
    onset_ms=40, barge_in_onset_ms=100, hangover_ms=100, preroll_ms=40, tail_ms=40,
    client_buffer_ms=0.0,
)  # fmt: skip


def make_client() -> TestClient:
    def providers() -> ProviderSet:
        return ProviderSet(
            StubSTT(["hi"], latency=FAST),
            StubLLM([LONG], first_token=FAST, inter_token=FAST),
            StubTTS(ms_per_char=65, first_audio=FAST, realtime_factor=1000),
        )

    return TestClient(create_app(providers_factory=providers, vad_factory=SyntheticVad, cfg=CFG))


def user_says(ws):
    for _ in range(15):
        ws.send_bytes(SPEECH)
    for _ in range(7):
        ws.send_bytes(QUIET)


def test_health():
    assert make_client().get("/health").json() == {"status": "ok"}


def test_audio_comes_back_tagged_with_the_turn_id():
    with make_client().websocket_connect("/ws") as ws:
        user_says(ws)
        data = ws.receive_bytes()
        (turn_id,) = struct.unpack(">I", data[:4])
        assert turn_id == 1
        assert len(data) == 4 + len(SPEECH)  # header + one 20 ms frame


def test_barge_in_sends_a_stop_message():
    with make_client().websocket_connect("/ws") as ws:
        user_says(ws)
        ws.receive_bytes()  # the agent is now speaking
        for _ in range(8):
            ws.send_bytes(SPEECH)  # the user talks over it
        for _ in range(500):
            message = ws.receive()
            if message.get("text"):
                assert json.loads(message["text"]) == {"type": "stop", "turn_id": 1}
                return
        raise AssertionError("no stop message arrived")
def test_metrics_endpoint_starts_empty():
    body = make_client().get("/api/metrics").json()
    assert body["turns"] == [] and body["summary"]["turns"] == 0