"""Conversation history, including truncate-to-spoken after a barge-in.

The problem: when the user interrupts, the LLM has *generated* a full reply but the user only
*heard* the first part. If we store the full reply, the model believes it said things the user
never heard. So we store only what was actually played.

How we know what was played: the gateway paces outbound audio in real time, so "milliseconds
of audio sent" ~ "milliseconds played". Each chunk's text is mapped to its audio duration, and a
partly played chunk is cut at a word boundary, assuming an even speaking rate inside the chunk.
(Word-level cut needs space-separated text, which covers English, Hindi and Telugu.)

Rule for tool use: only spoken TEXT is ever truncated. Tool calls and tool results are always
kept in full, because the side effects (e.g. a booking) really happened.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from vgw.providers.base import Message, ToolCall

INTERRUPTED_MARKER = " [interrupted by user]"
ERROR_MARKER = " [cut off by a technical error]"
_WORD = re.compile(r"\S+")


def _first_words(text: str, fraction: float) -> str:
    """The leading part of `text` covering `fraction` of its words (rounded down)."""
    spans = list(_WORD.finditer(text))
    k = int(fraction * len(spans))
    return text[: spans[k - 1].end()] if k > 0 else ""


@dataclass(slots=True)
class _Segment:
    text: str
    audio_ms: float = 0.0  # audio produced so far for this chunk
    complete: bool = False  # TTS has finished this chunk


class AssistantUtterance:
    """Tracks one agent reply as ordered (text chunk, audio duration) segments."""

    def __init__(self, est_ms_per_char: float = 65.0):
        self._segments: list[_Segment] = []
        self._est = est_ms_per_char

    def add_segment(self, text: str) -> int:
        self._segments.append(_Segment(text=text))
        return len(self._segments) - 1

    def add_audio(self, index: int, audio_ms: float) -> None:
        self._segments[index].audio_ms += audio_ms

    def finish_segment(self, index: int) -> None:
        self._segments[index].complete = True

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self._segments).strip()

    @property
    def text_chars(self) -> int:
        """Characters handed to TTS so far (what we paid to synthesize)."""
        return len(self.full_text)

    def _total_ms(self, seg: _Segment) -> float:
        if seg.complete:
            return seg.audio_ms
        # TTS still running for this chunk: don't treat "audio so far" as the whole chunk.
        return max(seg.audio_ms, len(seg.text) * self._est)

    def spoken_text(self, played_ms: float) -> str:
        """The text the user has heard after `played_ms` of playback."""
        remaining = max(0.0, played_ms)
        parts: list[str] = []
        for seg in self._segments:
            total = self._total_ms(seg)
            if remaining <= 0 or total <= 0:
                break
            if remaining + 1e-6 >= total:
                parts.append(seg.text)
                remaining -= total
            else:
                parts.append(_first_words(seg.text, remaining / total))
                break
        return " ".join(p for p in parts if p).strip()


class Conversation:
    def __init__(self, system_prompt: str, *, max_messages: int | None = None):
        self._system = Message(role="system", content=system_prompt)
        self._turns: list[Message] = []
        self._max_messages = max_messages

    def messages(self) -> list[Message]:
        """What to send to the LLM: system prompt first, then the most recent turns.

        With max_messages set, the oldest turns are dropped. The window may be slightly larger
        than the cap so that it always starts at a user message: a tool result can never be
        separated from the assistant message that requested it.
        """
        turns = self._turns
        if self._max_messages is not None and len(turns) > self._max_messages:
            cut = len(turns) - self._max_messages
            start = next((i for i in range(cut, -1, -1) if turns[i].role == "user"), cut)
            turns = turns[start:]
        return [self._system, *turns]

    def add_user(self, text: str) -> None:
        # If the user interrupted before any reply was heard, two user messages in a row is
        # fine and honest: they did say both things.
        self._turns.append(Message(role="user", content=text))

    def add_assistant(self, text: str = "", *, tool_calls: tuple[ToolCall, ...] = ()) -> None:
        self._turns.append(Message(role="assistant", content=text, tool_calls=tool_calls))

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._turns.append(Message(role="tool", content=content, tool_call_id=tool_call_id))

    def commit_utterance(
        self,
        utterance: AssistantUtterance,
        *,
        interrupted: bool,
        played_ms: float = 0.0,
        marker: str = INTERRUPTED_MARKER,
    ) -> str:
        """Store the agent's spoken reply; return exactly what was stored (may be empty).

        If interrupted, only the part played so far is stored, followed by `marker`.
        """
        if interrupted:
            text = utterance.spoken_text(played_ms)
            if text:
                text += marker
        else:
            text = utterance.full_text
        if text:
            self.add_assistant(text)
        return text