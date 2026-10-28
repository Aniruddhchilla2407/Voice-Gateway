"""Turn a stream of LLM text fragments into speakable chunks.

Why: TTS should start on the first sentence while the LLM is still writing the rest.

Trade-off: a small first chunk gives faster first audio but choppier prosody, so only the
FIRST chunk of a turn may be released early at a clause boundary (comma, semicolon...).
Every later chunk waits for a real sentence end. A hard size cap protects against LLM
output with no punctuation at all.

The splitter errs toward NOT splitting: a chunk that is a bit too long costs a little
latency, a split in the wrong place sounds wrong. It never splits "3.5", "e.g.", "Dr. Rao".

Sans-IO on purpose: `SentenceChunker` is plain synchronous logic, so it is trivial to test.
Create a new one per turn.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator

_TERMINATORS = ".!?…।॥"  # includes the Devanagari danda used in Hindi
# A terminator run, optional closing quotes/brackets, then whitespace. Requiring the trailing
# whitespace is what keeps "3.5" and "example.com" in one piece.
_SENTENCE_END = re.compile(r"[.!?…।॥]+[\"'”’)\]]*\s")
_CLAUSE_END = re.compile(r"[,;:—–]\s")
_RUN = re.compile(r"[.!?…।॥]+")

_ABBREVIATIONS = frozenset(
    {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "e.g.", "i.e."}
)


def _speakable(text: str) -> bool:
    """False for chunks like '.' or '...' that would make a TTS engine error out."""
    return any(ch.isalnum() for ch in text)


def _is_false_end(buf: str, m: re.Match[str]) -> bool:
    """True if this '.' is an abbreviation or an initial, not a sentence end."""
    run = _RUN.match(m.group(0))
    if run is None or run.group(0) != ".":
        return False
    words = buf[: m.start()].split()
    prev = words[-1] if words else ""
    if (prev + ".").lower() in _ABBREVIATIONS:
        return True
    return len(prev) == 1 and prev.isalpha() and prev.isupper()  # "A. Kumar"


class SentenceChunker:
    def __init__(self, *, first_chunk_min_chars: int = 20, max_chars: int = 200):
        self._first_min = first_chunk_min_chars
        self._max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        """Add a fragment; return any chunks that are now complete."""
        self._buf += text
        out: list[str] = []
        while (chunk := self._pop()) is not None:
            if _speakable(chunk):
                out.append(chunk)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        """End of stream: release whatever is left."""
        rest, self._buf = self._buf.strip(), ""
        if rest and _speakable(rest):
            self._emitted += 1
            return [rest]
        return []

    # ---- internals
    def _pop(self) -> str | None:
        buf = self._buf
        cut = self._sentence_cut(buf)
        if self._emitted == 0:
            clause = self._clause_cut(buf)
            if clause is not None and (cut is None or clause < cut):
                cut = clause
        if len(buf) > self._max_chars and (cut is None or cut > self._max_chars):
            cut = self._forced_cut(buf)
        if cut is None:
            return None
        chunk, self._buf = buf[:cut], buf[cut:].lstrip()
        return chunk.strip()

    @staticmethod
    def _sentence_cut(buf: str) -> int | None:
        for m in _SENTENCE_END.finditer(buf):
            if not _is_false_end(buf, m):
                return m.end()
        return None

    def _clause_cut(self, buf: str) -> int | None:
        for m in _CLAUSE_END.finditer(buf):
            if m.start() >= self._first_min:
                return m.end()
        return None

    def _forced_cut(self, buf: str) -> int:
        window = buf[: self._max_chars + 1]
        spaces = [m.start() for m in re.finditer(r"\s", window)]
        idx = spaces[-1] if spaces else -1
        return idx + 1 if idx > 0 else self._max_chars


async def chunk_stream(
    texts: AsyncIterator[str], chunker: SentenceChunker | None = None
) -> AsyncIterator[str]:
    """Async adapter: text fragments in, speakable chunks out.

    If the consuming task is cancelled (barge-in), the generator just stops: the unflushed
    remainder is deliberately dropped, because it will never be spoken.
    """
    chunker = chunker or SentenceChunker()
    async for text in texts:
        for chunk in chunker.feed(text):
            yield chunk
    for chunk in chunker.flush():
        yield chunk