from vgw.tracing.events import EventKind
from vgw.tracing.recorder import TraceRecorder


def test_interval_between_two_events():
    rec = TraceRecorder("s1")
    rec.emit(1, EventKind.ENDPOINT, ts_ns=1_000_000)
    rec.emit(1, EventKind.AUDIO_OUT_FIRST, ts_ns=6_000_000)
    assert rec.interval_ms(1, EventKind.ENDPOINT, EventKind.AUDIO_OUT_FIRST) == 5.0


def test_interval_is_none_when_an_event_is_missing():
    rec = TraceRecorder("s1")
    rec.emit(1, EventKind.ENDPOINT)
    assert rec.interval_ms(1, EventKind.ENDPOINT, EventKind.AUDIO_OUT_FIRST) is None
    assert rec.first(1, EventKind.AUDIO_OUT_FIRST) is None


def test_events_are_filtered_by_turn_and_keep_attrs():
    rec = TraceRecorder("s1")
    rec.emit(1, EventKind.TURN_START)
    rec.emit(2, EventKind.TURN_START)
    rec.emit(2, EventKind.TURN_END, outcome="completed")
    assert [e.kind for e in rec.turn_events(2)] == [EventKind.TURN_START, EventKind.TURN_END]
    assert rec.first(2, EventKind.TURN_END).attrs == {"outcome": "completed"}


def test_on_event_callback_sees_every_event():
    seen = []
    rec = TraceRecorder("s1", on_event=seen.append)
    e = rec.emit(1, EventKind.SPEECH_START)
    assert seen == [e]
    assert e.to_dict()["session_id"] == "s1"


def test_recorder_is_bounded():
    rec = TraceRecorder("s1", max_events=10)
    for i in range(50):
        rec.emit(i, EventKind.TURN_START)
    assert len(rec.events) == 10