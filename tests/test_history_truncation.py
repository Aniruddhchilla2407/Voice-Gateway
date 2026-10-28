from vgw.history import INTERRUPTED_MARKER, AssistantUtterance, Conversation
from vgw.providers.base import ToolCall


def make_utterance() -> AssistantUtterance:
    u = AssistantUtterance()
    i = u.add_segment("Hello there.")  # 2 words
    u.add_audio(i, 600)
    u.finish_segment(i)
    j = u.add_segment("How are you doing today?")  # 5 words
    u.add_audio(j, 1200)
    u.finish_segment(j)
    return u


def test_completed_reply_is_stored_in_full():
    conv = Conversation("sys")
    stored = conv.commit_utterance(make_utterance(), interrupted=False)
    assert stored == "Hello there. How are you doing today?"
    assert conv.messages()[-1].role == "assistant"


def test_interrupt_mid_second_chunk_keeps_only_heard_words():
    u = make_utterance()
    assert u.spoken_text(600 + 600) == "Hello there. How are"  # half of 5 words -> 2


def test_interrupt_mid_first_chunk():
    assert make_utterance().spoken_text(300) == "Hello"


def test_exact_chunk_boundary_counts_as_fully_spoken():
    assert make_utterance().spoken_text(600) == "Hello there."


def test_playing_past_the_end_returns_everything():
    assert make_utterance().spoken_text(99_999) == "Hello there. How are you doing today?"


def test_nothing_played_means_nothing_stored():
    conv = Conversation("sys")
    stored = conv.commit_utterance(make_utterance(), interrupted=True, played_ms=0)
    assert stored == ""
    assert [m.role for m in conv.messages()] == ["system"]


def test_interrupted_reply_gets_the_marker():
    conv = Conversation("sys")
    stored = conv.commit_utterance(make_utterance(), interrupted=True, played_ms=1200)
    assert stored == "Hello there. How are" + INTERRUPTED_MARKER
    assert conv.messages()[-1].content == stored


def test_unfinished_tts_chunk_is_not_assumed_to_be_fully_played():
    u = AssistantUtterance(est_ms_per_char=100)
    i = u.add_segment("one two three four")  # 18 chars -> estimated 1800 ms
    u.add_audio(i, 900)  # TTS has only produced half so far, and is not finished
    assert u.spoken_text(900) == "one two"
    u.finish_segment(i)  # once finished, 900 ms really is the whole chunk
    assert u.spoken_text(900) == "one two three four"


def test_tool_calls_and_results_survive_an_interruption():
    conv = Conversation("sys")
    conv.add_user("book me for tuesday")
    call = ToolCall(id="t1", name="book_slot", arguments={"day": "tue"})
    conv.add_assistant(tool_calls=(call,))
    conv.add_tool_result("t1", "booked")
    conv.commit_utterance(make_utterance(), interrupted=True, played_ms=0)
    roles = [m.role for m in conv.messages()]
    assert roles == ["system", "user", "assistant", "tool"]
    assert conv.messages()[2].tool_calls == (call,)


def test_messages_returns_a_copy():
    conv = Conversation("sys")
    conv.messages().append(None)  # type: ignore[arg-type]
    assert len(conv.messages()) == 1