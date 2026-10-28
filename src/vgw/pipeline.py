"""One conversational turn: STT -> LLM -> sentence chunker -> TTS -> paced playout.

Structure: the STT call runs first (we need the text), then three stages run concurrently
inside an asyncio.TaskGroup, connected by BOUNDED queues:

    LLM stage --chunk_q--> TTS stage --playout queue--> Playout (real-time pacing) --> client

  * Bounded queues are the backpressure: a slow client stalls Playout, which fills its queue,
    which blocks TTS, which fills chunk_q, which blocks the LLM stage. Nothing buffers
    without limit.
  * TaskGroup is the cancellation guarantee: cancelling the turn cancels all three stages and
    waits for them, so a barge-in cannot leave orphaned tasks behind.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import aclosing, contextmanager
from typing import Protocol

from vgw.audio import FRAME_MS, FrameSplitter, bytes_to_ms
from vgw.chunker import SentenceChunker
from vgw.config import GatewayConfig
from vgw.history import AssistantUtterance, Conversation
from vgw.providers.base import ProviderError, ProviderSet
from vgw.tracing.events import EventKind

FRAME_S = FRAME_MS / 1000.0

Emit = Callable[..., None]  # emit(kind, **attrs): already bound to this turn's id


class OutputSink(Protocol):
    """Where the gateway's output goes (a WebSocket in production, a recorder in tests)."""

    async def send_audio(self, turn_id: int, pcm: bytes) -> None: ...

    async def send_stop(self, turn_id: int) -> None:
        """Tell the client to stop playing and discard anything buffered for this turn."""
        ...


class StageError(Exception):
    """A provider stage failed or timed out; carries enough detail for a trace event."""

    def __init__(self, stage: str, provider: str, *, timeout: bool, transient: bool = True):
        super().__init__(f"{stage} {'timed out' if timeout else 'failed'} ({provider})")
        self.stage = stage
        self.provider = provider
        self.timeout = timeout
        self.transient = transient


@contextmanager
def _stage(stage: str, provider: str):
    try:
        yield
    except TimeoutError as e:
        raise StageError(stage, provider, timeout=True) from e
    except ProviderError as e:
        raise StageError(
            stage, e.provider or provider, timeout=False, transient=e.transient
        ) from e


class Playout:
    """Sends audio frames to the client at real-time speed and counts what has gone out.

    TTS usually produces audio faster than real time. If we forwarded it immediately, the
    client would buffer seconds of speech and a barge-in could not stop it. Pacing at 20 ms per
    frame keeps "audio sent" within a small buffer of "audio heard", which is also what makes
    truncate-to-spoken accurate.
    """

    def __init__(
        self,
        turn_id: int,
        sink: OutputSink,
        *,
        queue_frames: int,
        client_buffer_ms: float,
        is_cancelled: Callable[[], bool],
        emit: Emit,
        on_first_frame: Callable[[], None],
    ):
        self.turn_id = turn_id
        self.sent_frames = 0
        self._sink = sink
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_frames)
        self._client_buffer_ms = client_buffer_ms
        self._is_cancelled = is_cancelled
        self._emit = emit
        self._on_first_frame = on_first_frame

    @property
    def sent_ms(self) -> float:
        return self.sent_frames * FRAME_MS

    @property
    def played_ms(self) -> float:
        """Estimate of what the user has actually heard."""
        return max(0.0, self.sent_ms - self._client_buffer_ms)

    async def put(self, frame: bytes) -> None:
        await self._queue.put(frame)

    async def finish(self) -> None:
        await self._queue.put(None)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        t0 = 0.0
        while True:
            frame = await self._queue.get()
            if frame is None:
                if self.sent_frames:
                    self._emit(EventKind.AUDIO_OUT_LAST, frames=self.sent_frames)
                return
            if self._is_cancelled():
                return
            now = loop.time()
            if self.sent_frames == 0:
                t0 = now  # the first frame goes out immediately: it is on the latency path
            due = t0 + self.sent_frames * FRAME_S
            if now - due > 2 * FRAME_S:
                # We starved (TTS was slow). Re-anchor the schedule instead of bursting
                # frames out to "catch up", which would defeat the pacing.
                t0 = now - self.sent_frames * FRAME_S
                due = now
            if due > now:
                await asyncio.sleep(due - now)
                if self._is_cancelled():
                    return
            await self._sink.send_audio(self.turn_id, frame)
            self.sent_frames += 1
            if self.sent_frames == 1:
                self._emit(EventKind.AUDIO_OUT_FIRST)
                self._on_first_frame()


class TurnPipeline:
    def __init__(
        self,
        *,
        turn_id: int,
        providers: ProviderSet,
        conversation: Conversation,
        cfg: GatewayConfig,
        sink: OutputSink,
        emit: Emit,
        on_speaking: Callable[[], None],
    ):
        self.turn_id = turn_id
        self.utterance = AssistantUtterance()
        self.cancelled = False  # set synchronously by the turn manager the instant it interrupts
        self.user_committed = False  # True once the transcript is in the conversation history
        self.chars_generated = 0
        self._p = providers
        self._conv = conversation
        self._cfg = cfg
        self._emit = emit
        self.playout = Playout(
            turn_id,
            sink,
            queue_frames=cfg.playout_queue_frames,
            client_buffer_ms=cfg.client_buffer_ms,
            is_cancelled=lambda: self.cancelled,
            emit=emit,
            on_first_frame=on_speaking,
        )

    async def run(self, pcm: bytes) -> str:
        """Run the whole turn. Returns the user's transcript ('' if nothing was said)."""
        text = await self._stt(pcm)
        if not text:
            return ""
        self._conv.add_user(text)
        self.user_committed = True
        chunk_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self._cfg.chunk_queue_size)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._llm_stage(chunk_q))
            tg.create_task(self._tts_stage(chunk_q))
            tg.create_task(self.playout.run())
        return text

    async def _stt(self, pcm: bytes) -> str:
        stt = self._p.stt
        self._emit(EventKind.STT_START, audio_ms=bytes_to_ms(len(pcm)))
        with _stage("stt", stt.name):
            async with asyncio.timeout(self._cfg.stt_timeout_s):
                transcript = await stt.transcribe(pcm, language=self._cfg.language)
        text = transcript.text.strip()
        self._emit(EventKind.STT_END, chars=len(text))
        return text

    async def _llm_stage(self, chunk_q: asyncio.Queue[str | None]) -> None:
        llm = self._p.llm
        chunker = SentenceChunker(first_chunk_min_chars=self._cfg.first_chunk_min_chars)
        first_token = True
        first_sentence = True
        messages = self._conv.messages()
        self._emit(
            EventKind.LLM_START,
            input_chars=sum(len(m.content) for m in messages),
            messages=len(messages),
        )

        async def emit_chunks(chunks: list[str]) -> None:
            nonlocal first_sentence
            for chunk in chunks:
                if first_sentence:
                    first_sentence = False
                    self._emit(EventKind.LLM_FIRST_SENTENCE, chars=len(chunk))
                await chunk_q.put(chunk)  # blocks when TTS is behind: this is the backpressure

        naive_text: list[str] = []  # only used when sentence streaming is switched off
        with _stage("llm", llm.name):
            async with aclosing(llm.stream(messages)) as stream:
                it = aiter(stream)
                while True:
                    try:
                        if first_token:
                            async with asyncio.timeout(self._cfg.llm_first_token_timeout_s):
                                delta = await anext(it)
                        else:
                            delta = await anext(it)
                    except StopAsyncIteration:
                        break
                    if delta.text:
                        self.chars_generated += len(delta.text)
                        if first_token:
                            first_token = False
                            self._emit(EventKind.LLM_FIRST_TOKEN)
                        if self._cfg.stream_sentences:
                            await emit_chunks(chunker.feed(delta.text))
                        else:
                            naive_text.append(delta.text)
                    # delta.tool_call is ignored for now: the agent step adds the tool loop.
        if self._cfg.stream_sentences:
            await emit_chunks(chunker.flush())
        elif naive_text:
            await emit_chunks(["".join(naive_text).strip()])
        self._emit(EventKind.LLM_END)
        await chunk_q.put(None)

    async def _tts_stage(self, chunk_q: asyncio.Queue[str | None]) -> None:
        tts = self._p.tts
        started = False
        first_audio = True
        while (chunk := await chunk_q.get()) is not None:
            if not started:
                started = True
                self._emit(EventKind.TTS_START)
            idx = self.utterance.add_segment(chunk)
            splitter = FrameSplitter()
            first_in_chunk = True
            with _stage("tts", tts.name):
                async with aclosing(tts.synthesize(chunk, language=self._cfg.language)) as stream:
                    it = aiter(stream)
                    while True:
                        try:
                            if first_in_chunk:
                                async with asyncio.timeout(self._cfg.tts_first_audio_timeout_s):
                                    pcm = await anext(it)
                            else:
                                pcm = await anext(it)
                        except StopAsyncIteration:
                            break
                        first_in_chunk = False
                        if first_audio:
                            first_audio = False
                            self._emit(EventKind.TTS_FIRST_AUDIO)
                        for frame in splitter.feed(pcm):
                            await self._put_frame(idx, frame)
            tail = splitter.flush()
            if tail is not None:
                await self._put_frame(idx, tail)
            self.utterance.finish_segment(idx)
        if started:
            self._emit(EventKind.TTS_END)
        await self.playout.finish()

    async def _put_frame(self, segment: int, frame: bytes) -> None:
        await self.playout.put(frame)
        # Counted only after the frame is accepted, so utterance audio always matches playout.
        self.utterance.add_audio(segment, FRAME_MS)