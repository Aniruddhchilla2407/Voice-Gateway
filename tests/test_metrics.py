import pytest

from vgw.tracing.events import EventKind as K
from vgw.tracing.metrics import MetricsStore, percentile, summarize, turn_metrics
from vgw.tracing.recorder import TraceRecorder

MS = 1_000_000  # nanoseconds per millisecond


def recorded_turn(*, interrupted: bool = False) -> TraceRecorder:
    rec = TraceRecorder("s1")
    e = rec.emit
    e(1, K.SPEECH_END, ts_ns=0)
    e(1, K.ENDPOINT, ts_ns=600 * MS)
    e(1, K.STT_START, ts_ns=600 * MS)
    e(1, K.STT_END, ts_ns=1100 * MS)
    e(1, K.LLM_START, ts_ns=1100 * MS, input_chars=500, messages=3)
    e(1, K.LLM_FIRST_TOKEN, ts_ns=1600 * MS)
    e(1, K.LLM_FIRST_SENTENCE, ts_ns=1700 * MS)
    e(1, K.TTS_START, ts_ns=1700 * MS)
    e(1, K.TTS_FIRST_AUDIO, ts_ns=2400 * MS)
    e(1, K.AUDIO_OUT_FIRST, ts_ns=2410 * MS)
    if interrupted:
        e(1, K.BARGE_IN, ts_ns=4000 * MS)
        e(1, K.CANCEL_START, ts_ns=4000 * MS)
        e(1, K.PLAYBACK_STOP_SENT, ts_ns=4003 * MS)
        e(1, K.CANCEL_DONE, ts_ns=4010 * MS)
        e(1, K.TURN_END, ts_ns=4011 * MS, outcome="interrupted",
          generated_chars=120, tts_chars=80, spoken_chars=20)  # fmt: skip
    else:
        e(1, K.TURN_END, ts_ns=6000 * MS, outcome="completed",
          generated_chars=100, tts_chars=100, spoken_chars=100)  # fmt: skip
    return rec


def test_turn_metrics_stage_latencies():
    m = turn_metrics(recorded_turn(), 1)
    assert m.outcome == "completed"
    assert (m.ttfa_ms, m.endpointing_ms, m.stt_ms) == (2410.0, 600.0, 500.0)
    assert (m.llm_first_token_ms, m.llm_first_sentence_ms) == (500.0, 600.0)
    assert (m.tts_first_audio_ms, m.out_ms) == (700.0, 10.0)
    assert (m.input_chars, m.history_messages) == (500, 3)
    assert m.barge_in_cutoff_ms is None


def test_interrupted_turn_has_barge_in_numbers():
    m = turn_metrics(recorded_turn(interrupted=True), 1)
    assert m.outcome == "interrupted"
    assert (m.barge_in_cutoff_ms, m.cancel_ms) == (3.0, 10.0)
    assert (m.generated_chars, m.tts_chars, m.spoken_chars) == (120, 80, 20)


def test_a_turn_that_has_not_ended_has_no_metrics():
    rec = TraceRecorder("s1")
    rec.emit(1, K.ENDPOINT)
    assert turn_metrics(rec, 1) is None


def test_percentile_interpolates_between_ranks():
    xs = [10.0, 20.0, 30.0, 40.0]
    assert percentile(xs, 0) == 10.0
    assert percentile(xs, 50) == 25.0
    assert percentile(xs, 100) == 40.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summary_reports_percentiles_and_the_unheard_share():
    turns = [turn_metrics(recorded_turn(), 1), turn_metrics(recorded_turn(interrupted=True), 1)]
    s = summarize(turns)
    assert s["turns"] == 2 and s["outcomes"] == {"completed": 1, "interrupted": 1}
    assert s["latency"]["ttfa_ms"]["n"] == 2
    assert s["latency"]["barge_in_cutoff_ms"]["n"] == 1
    eff = s["efficiency"]
    assert (eff["generated_chars"], eff["spoken_chars"]) == (220, 120)
    assert eff["unheard_generated_share"] == pytest.approx(1 - 120 / 220, abs=1e-3)


def test_store_is_bounded_and_returns_the_most_recent_turns():
    store = MetricsStore(maxlen=3)
    for i in range(5):
        rec = TraceRecorder("s")
        rec.emit(i, K.TURN_END, outcome="completed")
        store.add(turn_metrics(rec, i))
    assert [m.turn_id for m in store.recent(10)] == [2, 3, 4]
    assert store.summary()["turns"] == 3