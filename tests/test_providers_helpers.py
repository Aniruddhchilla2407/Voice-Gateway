import io
import json
import shutil
import subprocess
import wave

import pytest

from vgw.audio import SAMPLE_RATE, tone
from vgw.providers.base import Message, ToolCall
from vgw.providers.llm_groq import to_wire
from vgw.providers.stt_groq import pcm_to_wav
from vgw.providers.tts_edge import decode_mp3, voice_for


def test_pcm_to_wav_is_a_valid_16k_mono_wav():
    pcm = tone(500)
    with wave.open(io.BytesIO(pcm_to_wav(pcm))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE)
        assert w.getnframes() == len(pcm) // 2


def test_voice_follows_the_script_of_the_text():
    assert voice_for("Hello there", None) == "en-US-AriaNeural"
    assert voice_for("नमस्ते, आप कैसे हैं", None) == "hi-IN-SwaraNeural"
    assert voice_for("మీరు ఎలా ఉన్నారు", None) == "te-IN-ShrutiNeural"


def test_an_explicit_language_overrides_the_script():
    assert voice_for("Hello", "hi") == "hi-IN-SwaraNeural"
    assert voice_for("Hello", "te-IN") == "te-IN-ShrutiNeural"
    assert voice_for("नमस्ते", "en") == "en-US-AriaNeural"


def test_messages_are_converted_to_the_wire_format():
    call = ToolCall(id="t1", name="book", arguments={"day": "tue"})
    wire = to_wire(
        [
            Message(role="system", content="s"),
            Message(role="user", content="hi"),
            Message(role="assistant", tool_calls=(call,)),
            Message(role="tool", content="ok", tool_call_id="t1"),
        ]
    )
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool"]
    assert wire[2]["content"] is None
    fn = wire[2]["tool_calls"][0]["function"]
    assert fn["name"] == "book" and json.loads(fn["arguments"]) == {"day": "tue"}
    assert wire[3]["tool_call_id"] == "t1"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_ffmpeg_decodes_mp3_to_16k_mono_pcm():
    make_mp3 = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=24000",
        "-ac", "1", "-f", "mp3", "pipe:1",
    ]  # fmt: skip
    mp3 = subprocess.run(make_mp3, capture_output=True, check=True).stdout
    pcm = await decode_mp3(mp3)
    seconds = len(pcm) / (SAMPLE_RATE * 2)
    assert 0.9 <= seconds <= 1.2