from vgw.history import Conversation
from vgw.providers.base import ToolCall


def fill(conv: Conversation, exchanges: int) -> None:
    for i in range(exchanges):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")


def test_no_cap_means_everything_is_sent():
    conv = Conversation("sys")
    fill(conv, 10)
    assert len(conv.messages()) == 21


def test_cap_keeps_the_system_prompt_and_the_most_recent_exchanges():
    conv = Conversation("sys", max_messages=4)
    fill(conv, 10)
    msgs = conv.messages()
    assert msgs[0].role == "system"
    assert [m.content for m in msgs[1:]] == ["u8", "a8", "u9", "a9"]


def test_the_window_never_starts_in_the_middle_of_an_exchange():
    conv = Conversation("sys", max_messages=3)
    fill(conv, 10)
    assert [m.content for m in conv.messages()[1:]] == ["u8", "a8", "u9", "a9"]


def test_a_tool_result_is_never_separated_from_its_call():
    conv = Conversation("sys", max_messages=3)
    conv.add_user("book it")
    conv.add_assistant(tool_calls=(ToolCall("t1", "book", {}),))
    conv.add_tool_result("t1", "ok")
    conv.add_assistant("Booked.")
    assert [m.role for m in conv.messages()[1:]] == ["user", "assistant", "tool", "assistant"]