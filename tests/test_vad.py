import pytest

from vgw.audio import FRAME_BYTES, FRAME_MS, silence, tone
from vgw.vad import Endpointer, SpeechEnded, SpeechStarted, SyntheticVad, WebRtcVad

SPEECH = tone(FRAME_MS)  # one 20 ms frame of "speech" (non-zero)
QUIET = silence(FRAME_MS)  # one 20 ms frame of silence


def feed(ep, frame, n):
    events = []
    for _ in range(n):
        events += ep.process(frame)
    return events


def make(**kw):
    defaults = {"onset_ms": 60, "hangover_ms": 400, "preroll_ms": 100, "tail_ms": 100}
    return Endpointer(SyntheticVad(), **{**defaults, **kw})


def test_silence_produces_no_events():
    assert feed(make(), QUIET, 100) == []


def test_a_single_blip_is_not_speech():
    ep = make()
    assert feed(ep, SPEECH, 1) + feed(ep, QUIET, 50) == []


def test_utterance_start_end_and_audio_bounds():
    ep = make()
    feed(ep, QUIET, 20)
    assert feed(ep, SPEECH, 20) == [SpeechStarted()]
    assert feed(ep, QUIET, 19) == []  # 380 ms of silence < 400 ms hangover
    (ended,) = feed(ep, QUIET, 1)
    assert isinstance(ended, SpeechEnded) and not ended.forced
    # 100 ms preroll + 20 speech frames + 100 ms tail = 5 + 20 + 5 frames
    assert len(ended.pcm) == 30 * FRAME_BYTES
    assert ended.pcm[: 5 * FRAME_BYTES] == bytes(5 * FRAME_BYTES)  # preroll is silence


def test_short_pause_inside_speech_does_not_split_the_utterance():
    ep = make()
    events = feed(ep, SPEECH, 15) + feed(ep, QUIET, 10) + feed(ep, SPEECH, 15)
    events += feed(ep, QUIET, 20)
    assert [type(e) for e in events] == [SpeechStarted, SpeechEnded]
    # nothing before the onset to use as preroll here, so: 3 onset frames are all there is
    assert len(events[1].pcm) == (15 + 10 + 15 + 5) * FRAME_BYTES


def test_endpointer_resets_and_hears_the_next_utterance():
    ep = make()
    first = feed(ep, SPEECH, 10) + feed(ep, QUIET, 25)
    second = feed(ep, SPEECH, 10) + feed(ep, QUIET, 25)
    assert [type(e) for e in first + second] == [
        SpeechStarted,
        SpeechEnded,
        SpeechStarted,
        SpeechEnded,
    ]


def test_utterance_is_force_ended_at_max_length():
    ep = make(max_utterance_ms=400, hangover_ms=10_000)
    events = feed(ep, SPEECH, 40)
    ended = [e for e in events if isinstance(e, SpeechEnded)]
    assert ended and ended[0].forced


def test_onset_can_be_raised_at_runtime_to_resist_short_bursts():
    ep = make()
    ep.onset_ms = 300  # what the turn manager does while the agent is talking
    assert feed(ep, SPEECH, 8) + feed(ep, QUIET, 5) == []  # 160 ms burst is ignored
    assert feed(ep, SPEECH, 16) == [SpeechStarted()]


def test_onset_tolerates_vad_flicker_but_not_sparse_noise():
    flicker = make(onset_ms=200)  # window of 10 frames, needs 8 voiced
    events = feed(flicker, SPEECH, 4) + feed(flicker, QUIET, 1) + feed(flicker, SPEECH, 5)
    assert events == [SpeechStarted()]

    sparse = make(onset_ms=200)
    events = []
    for _ in range(5):
        events += feed(sparse, SPEECH, 1) + feed(sparse, QUIET, 1)
    assert events == []


def test_wrong_frame_size_is_rejected():
    with pytest.raises(ValueError):
        make().process(b"\x00" * 10)


def test_last_frame_voiced_flag_tracks_the_latest_frame():
    ep = make()
    ep.process(SPEECH)
    assert ep.last_frame_voiced is True
    ep.process(QUIET)
    assert ep.last_frame_voiced is False


def test_webrtc_vad_says_silence_is_not_speech():
    pytest.importorskip("webrtcvad")
    assert WebRtcVad(aggressiveness=2).is_speech(QUIET) is False