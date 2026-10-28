import asyncio
import random
import time

import pytest

from vgw.audio import FRAME_BYTES, bytes_to_ms, iter_frames
from vgw.providers.base import Message, ProviderError
from vgw.providers.stubs import FaultSpec, LatencyProfile, StubLLM, StubSTT, StubTTS

FAST = LatencyProfile(median_ms=1, sigma=0.0)


async def test_stt_cycles_through_canned_transcripts():
    stt = StubSTT(transcripts=["one", "two"], latency=FAST)
    assert [(await stt.transcribe(b"")).text for _ in range(3)] == ["one", "two", "one"]


async def test_llm_stream_rejoins_to_the_response():
    llm = StubLLM(responses=["Hello there. How are you?"], first_token=FAST, inter_token=FAST)
    deltas = [d async for d in llm.stream([Message(role="user", content="hi")])]
    assert len(deltas) == 5
    assert "".join(d.text for d in deltas) == "Hello there. How are you?"


async def test_tts_audio_length_scales_with_text():
    tts = StubTTS(ms_per_char=50, chunk_ms=100, first_audio=FAST, realtime_factor=1000)
    short = b"".join([c async for c in tts.synthesize("hi")])
    long = b"".join([c async for c in tts.synthesize("hello there, how are you today")])
    assert bytes_to_ms(len(long)) > bytes_to_ms(len(short))
    assert bytes_to_ms(len(long)) >= 50 * len("hello there, how are you today")


async def test_error_injection_raises_provider_error():
    stt = StubSTT(latency=FAST, faults=FaultSpec(error_prob=1.0))
    with pytest.raises(ProviderError):
        await stt.transcribe(b"")


async def test_mid_stream_error_happens_after_some_output():
    llm = StubLLM(
        responses=["a b c d e f"],
        first_token=FAST,
        inter_token=FAST,
        faults=FaultSpec(mid_stream_error_prob=1.0),
    )
    got = []
    with pytest.raises(ProviderError):
        async for d in llm.stream([]):
            got.append(d.text)
    assert 0 < len(got) < 6


async def test_stall_can_be_cut_off_by_a_timeout():
    stt = StubSTT(latency=FAST, faults=FaultSpec(stall_prob=1.0, stall_s=5.0))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stt.transcribe(b""), timeout=0.05)


async def test_cancelling_a_consumer_stops_the_stream_promptly():
    llm = StubLLM(
        responses=["word " * 200], first_token=FAST, inter_token=LatencyProfile(20, sigma=0.0)
    )

    async def consume():
        async for _ in llm.stream([]):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.perf_counter() - t0 < 0.05


def test_latency_profile_is_reproducible_with_a_seed():
    p = LatencyProfile(100, sigma=0.3, tail_prob=0.1)
    a, b = random.Random(7), random.Random(7)
    assert [p.sample_s(a) for _ in range(20)] == [p.sample_s(b) for _ in range(20)]


def test_iter_frames_pads_the_last_frame():
    frames = list(iter_frames(bytes(FRAME_BYTES + 10)))
    assert len(frames) == 2
    assert all(len(f) == FRAME_BYTES for f in frames)