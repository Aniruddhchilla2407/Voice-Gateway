import asyncio
import time
from dataclasses import replace

from vgw.audio import FRAME_MS, silence, tone
from vgw.config import GatewayConfig
from vgw.history import INTERRUPTED_MARKER, Conversation
from vgw.providers.base import ProviderSet
from vgw.providers.stubs import FaultSpec, LatencyProfile, StubLLM, StubSTT, StubTTS
from vgw.tracing.events import EventKind as K
from vgw.tracing.recorder import TraceRecorder
from vgw.turn_manager import State, TurnManager
from vgw.vad import SyntheticVad
from vgw.tracing.metrics import turn_metrics
SPEECH = tone(FRAME_MS)
QUIET = silence(FRAME_MS)
FAST = LatencyProfile(median_ms=1, sigma=0.0)

CFG = GatewayConfig(
    onset_ms=40,
    barge_in_onset_ms=100,
    hangover_ms=100,
    preroll_ms=40,
    tail_ms=40,
    client_buffer_ms=0.0,
    stt_timeout_s=2.0,
    llm_first_token_timeout_s=2.0,
    tts_first_audio_timeout_s=2.0,
)

SHORT = "Sure, I can help with that. What day works?"
LONG = (
    "Sure, I can help with that. We have openings on Monday, Tuesday and Wednesday. "
    "Which of those days works best for you?"
)


class RecordingSink:
    def __init__(self):
        self.audio: list[tuple[float, int]] = []  # (time, turn_id)
        self.stops: list[tuple[float, int]] = []

    async def send_audio(self, turn_id, pcm):
        self.audio.append((time.perf_counter(), turn_id))

    async def send_stop(self, turn_id):
        self.stops.append((time.perf_counter(), turn_id))


class RecordingSTT(StubSTT):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[int] = []  # length in bytes of the audio of each call

    async def transcribe(self, pcm, *, language=None):
        self.calls.append(len(pcm))
        return await super().transcribe(pcm, language=language)


class Harness:
    def __init__(self, *, stt=None, llm=None, tts=None, cfg=CFG):
        self.cfg = cfg
        self.sink = RecordingSink()
        self.rec = TraceRecorder("test")
        self.conv = Conversation("sys")
        self.stt = stt or StubSTT(["hello there"], latency=FAST)
        self.llm = llm or StubLLM([SHORT], first_token=FAST, inter_token=FAST)
        self.tts = tts or StubTTS(ms_per_char=20, first_audio=FAST, realtime_factor=1000)
        self.tm = TurnManager(
            providers=ProviderSet(self.stt, self.llm, self.tts),
            conversation=self.conv,
            sink=self.sink,
            recorder=self.rec,
            vad=SyntheticVad(),
            cfg=cfg,
        )

    async def feed(self, frame, n):
        for _ in range(n):
            await self.tm.on_frame(frame)

    async def say(self, ms):
        """User speaks for `ms`, then falls silent long enough to be endpointed."""
        await self.feed(SPEECH, ms // FRAME_MS)
        await self.feed(QUIET, self.cfg.hangover_ms // FRAME_MS + 2)

    async def wait_for(self, turn_id, kind, timeout=5.0):
        deadline = time.perf_counter() + timeout
        while (event := self.rec.first(turn_id, kind)) is None:
            if time.perf_counter() > deadline:
                raise AssertionError(f"timed out waiting for {kind.value} in turn {turn_id}")
            await asyncio.sleep(0.01)
        return event

    def kinds(self, turn_id):
        return [e.kind for e in self.rec.turn_events(turn_id)]

    def outcome(self, turn_id):
        return self.rec.first(turn_id, K.TURN_END).attrs["outcome"]


async def test_full_turn_streams_paced_audio_and_commits_history():
    h = Harness()
    await h.say(300)
    await h.wait_for(1, K.TURN_END)

    kinds = h.kinds(1)
    order = [
        K.SPEECH_START, K.SPEECH_END, K.ENDPOINT, K.TURN_START, K.STT_START, K.STT_END,
        K.LLM_START, K.LLM_FIRST_TOKEN, K.LLM_FIRST_SENTENCE, K.TTS_START,
        K.TTS_FIRST_AUDIO, K.AUDIO_OUT_FIRST, K.AUDIO_OUT_LAST, K.TURN_END,
    ]  # fmt: skip
    positions = [kinds.index(k) for k in order]
    assert positions == sorted(positions), kinds
    assert h.outcome(1) == "completed"
    assert h.tm.state is State.IDLE

    assert [m.role for m in h.conv.messages()] == ["system", "user", "assistant"]
    assert h.conv.messages()[-1].content == SHORT

    assert h.rec.interval_ms(1, K.ENDPOINT, K.AUDIO_OUT_FIRST) > 0
    assert all(tid == 1 for _, tid in h.sink.audio)
    n = len(h.sink.audio)
    elapsed = h.sink.audio[-1][0] - h.sink.audio[0][0]
    assert elapsed >= 0.9 * (n - 1) * FRAME_MS / 1000  # real-time pacing, not a burst


async def test_barge_in_stops_audio_truncates_history_and_starts_the_next_turn():
    baseline = len(asyncio.all_tasks())
    llm = StubLLM([LONG, "Okay."], first_token=FAST, inter_token=FAST)
    tts = StubTTS(ms_per_char=65, first_audio=FAST, realtime_factor=1000)
    h = Harness(llm=llm, tts=tts)

    await h.say(300)
    await h.wait_for(1, K.AUDIO_OUT_FIRST)
    await asyncio.sleep(0.5)  # let about half a second of the reply play
    assert h.tm.state is State.SPEAKING

    await h.feed(SPEECH, 8)  # 160 ms of speech > 100 ms barge-in onset

    assert len(h.sink.stops) == 1 and h.sink.stops[0][1] == 1
    stop_time = h.sink.stops[0][0]
    await asyncio.sleep(0.3)
    assert all(t <= stop_time for t, tid in h.sink.audio if tid == 1)  # nothing after the stop

    barge = h.rec.first(1, K.BARGE_IN)
    assert barge.attrs["speaking"] is True and barge.attrs["played_ms"] > 0
    assert h.rec.interval_ms(1, K.BARGE_IN, K.PLAYBACK_STOP_SENT) < 100
    order = [K.BARGE_IN, K.CANCEL_START, K.PLAYBACK_STOP_SENT, K.CANCEL_DONE, K.TURN_END]
    kinds = h.kinds(1)
    assert [kinds.index(k) for k in order] == sorted(kinds.index(k) for k in order)
    assert h.outcome(1) == "interrupted"
    assert h.rec.first(1, K.CANCEL_DONE).attrs["timed_out"] is False

    interrupted = h.conv.messages()[-1].content
    assert interrupted.endswith(INTERRUPTED_MARKER)
    spoken = interrupted.removesuffix(INTERRUPTED_MARKER)
    assert LONG.startswith(spoken) and 0 < len(spoken) < len(LONG)

    # The user keeps talking, then stops: that becomes turn 2 and completes normally.
    await h.feed(SPEECH, 6)
    await h.feed(QUIET, h.cfg.hangover_ms // FRAME_MS + 2)
    await h.wait_for(2, K.TURN_END)
    assert h.outcome(2) == "completed"
    roles = [m.role for m in h.conv.messages()]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert h.conv.messages()[-1].content == "Okay."

    await asyncio.sleep(0.05)
    assert len(asyncio.all_tasks()) == baseline  # no leaked tasks


async def test_a_short_blip_does_not_interrupt_the_agent():
    llm = StubLLM([LONG], first_token=FAST, inter_token=FAST)
    tts = StubTTS(ms_per_char=20, first_audio=FAST, realtime_factor=1000)
    h = Harness(llm=llm, tts=tts)
    await h.say(300)
    await h.wait_for(1, K.AUDIO_OUT_FIRST)
    await h.feed(SPEECH, 3)  # 60 ms < 100 ms barge-in onset
    await h.feed(QUIET, 10)
    await h.wait_for(1, K.TURN_END)
    assert h.rec.first(1, K.BARGE_IN) is None
    assert h.outcome(1) == "completed"
    assert h.conv.messages()[-1].content == LONG


async def test_speech_during_stt_carries_the_unprocessed_audio_over():
    stt = RecordingSTT(["hello there"], latency=LatencyProfile(300, sigma=0.0))
    h = Harness(stt=stt)
    await h.say(300)
    await asyncio.sleep(0.05)  # turn 1 is now waiting on STT
    assert h.tm.state is State.THINKING

    await h.feed(SPEECH, 10)  # user speaks again: turn 1 is cancelled before it was transcribed
    assert h.outcome(1) == "interrupted"
    await h.feed(QUIET, h.cfg.hangover_ms // FRAME_MS + 2)
    await h.wait_for(2, K.TURN_END)

    assert len(stt.calls) == 2
    assert stt.calls[1] > stt.calls[0]  # second call got the carried audio plus the new speech
    assert [m.role for m in h.conv.messages()] == ["system", "user", "assistant"]  # one user msg


async def test_llm_failure_ends_the_turn_with_an_error_and_keeps_the_session_alive():
    llm = StubLLM(first_token=FAST, faults=FaultSpec(error_prob=1.0))
    h = Harness(llm=llm)
    await h.say(300)
    await h.wait_for(1, K.TURN_END)
    assert h.outcome(1) == "error"
    err = h.rec.first(1, K.PROVIDER_ERROR)
    assert err.attrs["stage"] == "llm" and err.attrs["provider"] == "stub-llm"
    assert h.tm.state is State.IDLE
    assert [m.role for m in h.conv.messages()] == ["system", "user"]  # heard, but no reply


async def test_stt_timeout_is_reported_per_stage():
    stt = StubSTT(latency=FAST, faults=FaultSpec(stall_prob=1.0, stall_s=5.0))
    h = Harness(stt=stt, cfg=replace(CFG, stt_timeout_s=0.1))
    await h.say(300)
    await h.wait_for(1, K.TURN_END)
    assert h.outcome(1) == "error"
    assert h.rec.first(1, K.PROVIDER_TIMEOUT).attrs["stage"] == "stt"
    assert h.tm.state is State.IDLE


async def test_close_cancels_the_turn_in_flight_and_leaks_nothing():
    baseline = len(asyncio.all_tasks())
    llm = StubLLM([LONG], first_token=FAST, inter_token=FAST)
    tts = StubTTS(ms_per_char=65, first_audio=FAST, realtime_factor=1000)
    h = Harness(llm=llm, tts=tts)
    await h.say(300)
    await h.wait_for(1, K.AUDIO_OUT_FIRST)
    await h.tm.close()
    assert h.outcome(1) == "abandoned"
    assert h.sink.stops == []  # the client is gone: nobody to tell
    await asyncio.sleep(0.05)
    assert len(asyncio.all_tasks()) == baseline
    await h.tm.on_frame(SPEECH)  # frames after close are ignored
    assert h.rec.first(2, K.SPEECH_START) is None

async def test_llm_start_and_turn_end_carry_the_efficiency_counters():
    h = Harness()
    await h.say(300)
    await h.wait_for(1, K.TURN_END)
    assert h.rec.first(1, K.LLM_START).attrs == {
        "input_chars": len("sys") + len("hello there"),
        "messages": 2,
    }
    end = h.rec.first(1, K.TURN_END).attrs
    assert end["generated_chars"] == end["tts_chars"] == end["spoken_chars"] == len(SHORT)


async def test_an_interrupted_reply_shows_how_much_was_written_but_never_heard():
    llm = StubLLM([LONG], first_token=FAST, inter_token=FAST)
    tts = StubTTS(ms_per_char=65, first_audio=FAST, realtime_factor=1000)
    h = Harness(llm=llm, tts=tts)
    await h.say(300)
    await h.wait_for(1, K.AUDIO_OUT_FIRST)
    await asyncio.sleep(0.5)
    await h.feed(SPEECH, 8)

    end = h.rec.first(1, K.TURN_END).attrs
    assert end["generated_chars"] == len(LONG)  # the fast LLM had already written all of it
    assert 0 < end["spoken_chars"] < end["tts_chars"] < end["generated_chars"]
    m = turn_metrics(h.rec, 1)
    assert m.barge_in_cutoff_ms is not None and m.barge_in_cutoff_ms < 100
    
async def test_naive_mode_waits_for_the_whole_reply_before_speaking():
    h = Harness(cfg=replace(CFG, stream_sentences=False))
    await h.say(300)
    await h.wait_for(1, K.TURN_END)
    kinds = h.kinds(1)
    assert kinds.index(K.LLM_END) < kinds.index(K.TTS_START)
    assert h.conv.messages()[-1].content == SHORT


async def test_with_barge_in_disabled_the_agent_keeps_talking_over_the_user():
    llm = StubLLM([LONG], first_token=FAST, inter_token=FAST)
    tts = StubTTS(ms_per_char=65, first_audio=FAST, realtime_factor=1000)
    h = Harness(llm=llm, tts=tts, cfg=replace(CFG, barge_in_enabled=False))
    await h.say(300)
    await h.wait_for(1, K.AUDIO_OUT_FIRST)
    await asyncio.sleep(0.3)
    await h.feed(SPEECH, 10)
    assert h.rec.first(1, K.BARGE_IN) is None and h.sink.stops == []
    assert h.tm.state is State.SPEAKING
    await h.tm.close()  # don't wait for the whole reply to finish