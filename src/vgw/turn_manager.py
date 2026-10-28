"""The turn manager: owns the conversation's timing for one session.

States:  IDLE -> LISTENING (user talking) -> THINKING (turn running, no audio out yet)
         -> SPEAKING (first audio frame sent) -> IDLE

Every user utterance gets a turn id when speech starts. A turn is the utterance plus the
agent's reply. Frames are tagged with their turn id, and a turn that has been interrupted is
marked cancelled synchronously, so stale audio cannot leak out after a barge-in.

Barge-in order matters, because what the user perceives is the moment audio stops:
  1. mark the turn cancelled        (sync: Playout will not send another frame)
  2. snapshot how much was played   (sync: before anything else changes)
  3. cancel the turn's task         (sync: scheduled; the TaskGroup unwinds all stages)
  4. tell the client to stop        (the one await on the critical path)
  5. wait for the cancelled work to finish (bounded by cancel_timeout_s)
  6. commit only the spoken text to the conversation history

If the user speaks again while we are still THINKING and their first utterance was never
transcribed, we do not throw that audio away: it is carried over and joined to the next
utterance, so one thought split by a long pause is transcribed as one.

Robustness rule: however a turn ends, the manager must return to IDLE. A bug in the
bookkeeping may lose a trace event, but it must never leave a live call stuck.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass
from enum import Enum

from vgw.config import GatewayConfig
from vgw.history import ERROR_MARKER, Conversation
from vgw.pipeline import OutputSink, StageError, TurnPipeline
from vgw.providers.base import ProviderSet
from vgw.tracing.events import EventKind, TurnOutcome, now_ns
from vgw.tracing.recorder import TraceRecorder
from vgw.vad import Endpointer, SpeechEnded, SpeechStarted, VoiceActivityDetector

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(slots=True)
class _Turn:
    id: int
    pcm: bytes
    pipeline: TurnPipeline
    task: asyncio.Task[None] | None = None


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten (possibly nested) exception groups raised by a TaskGroup."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for e in exc.exceptions for leaf in _leaves(e)]
    return [exc]


def _char_counts(pipeline: TurnPipeline, heard_ms: float | None) -> dict[str, int]:
    """Characters (a rough stand-in for tokens): written by the LLM, sent to TTS, actually heard.

    Best effort: the counters must never stop a turn from ending.
    """
    try:
        u = pipeline.utterance
        heard = u.full_text if heard_ms is None else u.spoken_text(heard_ms)
        return {
            "generated_chars": pipeline.chars_generated,
            "tts_chars": u.text_chars,
            "spoken_chars": len(heard),
        }
    except Exception:  # noqa: BLE001
        logger.exception("could not compute the character counters")
        return {}


class TurnManager:
    def __init__(
        self,
        *,
        providers: ProviderSet,
        conversation: Conversation,
        sink: OutputSink,
        recorder: TraceRecorder,
        vad: VoiceActivityDetector,
        cfg: GatewayConfig | None = None,
    ):
        self._cfg = cfg or GatewayConfig()
        self._providers = providers
        self._conv = conversation
        self._sink = sink
        self._rec = recorder
        self._ep = Endpointer(
            vad,
            onset_ms=self._cfg.onset_ms,
            onset_ratio=self._cfg.onset_ratio,
            hangover_ms=self._cfg.hangover_ms,
            preroll_ms=self._cfg.preroll_ms,
            tail_ms=self._cfg.tail_ms,
            max_utterance_ms=self._cfg.max_utterance_ms,
        )
        self._state = State.IDLE
        self._turn: _Turn | None = None
        self._turn_counter = 0
        self._pending_id = 0  # id of the turn the current user speech will become
        self._carry_pcm = b""
        self._last_voiced_ns = 0
        self._closed = False

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, state: State) -> None:
        self._state = state
        # While the agent talks, demand longer speech before treating it as an interruption.
        self._ep.onset_ms = (
            self._cfg.barge_in_onset_ms if state is State.SPEAKING else self._cfg.onset_ms
        )

    # ------------------------------------------------------------------ input
    async def on_frame(self, frame: bytes) -> None:
        """Feed one 20 ms PCM frame. Called from the session's receive loop, in order."""
        if self._closed:
            return
        if self._turn is not None and not self._cfg.barge_in_enabled:
            return
        now = now_ns()
        for event in self._ep.process(frame):
            if isinstance(event, SpeechStarted):
                await self._on_speech_started(now)
            elif isinstance(event, SpeechEnded):
                self._on_speech_ended(event, now)
        if self._ep.last_frame_voiced:
            self._last_voiced_ns = now

    async def close(self) -> None:
        """Client disconnected: stop everything in flight."""
        self._closed = True
        await self._interrupt(now=now_ns(), outcome=TurnOutcome.ABANDONED, send_stop=False)
        self._ep.reset()

    # ------------------------------------------------------------- speech events
    async def _on_speech_started(self, now: int) -> None:
        self._turn_counter += 1
        self._pending_id = self._turn_counter
        if self._turn is not None:
            await self._interrupt(now=now, outcome=TurnOutcome.INTERRUPTED, send_stop=True)
        self._set_state(State.LISTENING)
        self._rec.emit(self._pending_id, EventKind.SPEECH_START, ts_ns=now)

    def _on_speech_ended(self, event: SpeechEnded, now: int) -> None:
        tid = self._pending_id
        # SPEECH_END is the last voiced frame, recorded retroactively; ENDPOINT is "now".
        self._rec.emit(tid, EventKind.SPEECH_END, ts_ns=self._last_voiced_ns, forced=event.forced)
        self._rec.emit(tid, EventKind.ENDPOINT, ts_ns=now)
        pcm, self._carry_pcm = self._carry_pcm + event.pcm, b""
        self._start_turn(tid, pcm)

    # --------------------------------------------------------------- turn lifecycle
    def _start_turn(self, tid: int, pcm: bytes) -> None:
        emit = functools.partial(self._rec.emit, tid)
        pipeline = TurnPipeline(
            turn_id=tid,
            providers=self._providers,
            conversation=self._conv,
            cfg=self._cfg,
            sink=self._sink,
            emit=emit,
            on_speaking=lambda: self._mark_speaking(tid),
        )
        turn = _Turn(id=tid, pcm=pcm, pipeline=pipeline)
        self._turn = turn
        self._set_state(State.THINKING)
        emit(EventKind.TURN_START)
        turn.task = asyncio.create_task(self._run_turn(turn), name=f"turn-{tid}")

    def _mark_speaking(self, tid: int) -> None:
        if self._turn is not None and self._turn.id == tid:
            self._set_state(State.SPEAKING)

    async def _run_turn(self, turn: _Turn) -> None:
        """Task body. Handles normal completion and provider failures.

        Cancellation (barge-in, disconnect) is finished by _interrupt, not here.
        """
        try:
            text = await turn.pipeline.run(turn.pcm)
        except Exception as exc:  # noqa: BLE001 - includes ExceptionGroup from the TaskGroup
            if not turn.pipeline.cancelled:
                self._finish_turn(turn, TurnOutcome.ERROR, errors=_leaves(exc))
            return
        if not turn.pipeline.cancelled:
            self._finish_turn(turn, TurnOutcome.COMPLETED, empty=not text)

    def _finish_turn(
        self,
        turn: _Turn,
        outcome: TurnOutcome,
        *,
        errors: list[BaseException] | None = None,
        empty: bool = False,
    ) -> None:
        """Record how a turn ended and return to IDLE. Never raises."""
        pipeline = turn.pipeline
        emit = functools.partial(self._rec.emit, turn.id)
        try:
            for err in errors or []:
                self._report(emit, turn.id, err)
            completed = outcome is TurnOutcome.COMPLETED
            if completed:
                spoken = self._conv.commit_utterance(pipeline.utterance, interrupted=False)
            else:  # the reply may have been cut off partway by the failure
                spoken = self._conv.commit_utterance(
                    pipeline.utterance,
                    interrupted=True,
                    played_ms=pipeline.playout.played_ms,
                    marker=ERROR_MARKER,
                )
            heard_ms = None if completed else pipeline.playout.played_ms
            emit(
                EventKind.TURN_END,
                outcome=outcome.value,
                spoken=spoken,
                empty_transcript=empty,
                **_char_counts(pipeline, heard_ms),
            )
        except Exception:  # noqa: BLE001
            logger.exception("turn %s: failed while recording how it ended", turn.id)
        finally:
            self._turn = None
            self._set_state(State.IDLE)

    @staticmethod
    def _report(emit, turn_id: int, err: BaseException) -> None:
        if isinstance(err, StageError):
            kind = EventKind.PROVIDER_TIMEOUT if err.timeout else EventKind.PROVIDER_ERROR
            emit(kind, stage=err.stage, provider=err.provider, transient=err.transient)
        else:
            logger.error("turn %s failed unexpectedly", turn_id, exc_info=err)
            emit(EventKind.PROVIDER_ERROR, stage="unknown", error=type(err).__name__)

    async def _interrupt(self, *, now: int, outcome: TurnOutcome, send_stop: bool) -> None:
        turn = self._turn
        if turn is None or turn.task is None:
            return
        pipeline = turn.pipeline
        emit = functools.partial(self._rec.emit, turn.id)

        pipeline.cancelled = True  # 1. no further frames leave Playout
        played_ms = pipeline.playout.played_ms  # 2. snapshot before anything else changes
        was_speaking = pipeline.playout.sent_frames > 0
        turn.task.cancel()  # 3. TaskGroup unwinds every stage

        try:
            if outcome is TurnOutcome.INTERRUPTED:
                emit(EventKind.BARGE_IN, ts_ns=now, played_ms=played_ms, speaking=was_speaking)
            emit(EventKind.CANCEL_START)
            if send_stop and was_speaking:
                await self._sink.send_stop(turn.id)  # 4. the moment the user hears it stop
                emit(EventKind.PLAYBACK_STOP_SENT)

            _, pending = await asyncio.wait({turn.task}, timeout=self._cfg.cancel_timeout_s)  # 5.
            if not pending and not turn.task.cancelled():
                turn.task.exception()  # mark retrieved; _run_turn swallows errors, so a no-op
            emit(EventKind.CANCEL_DONE, timed_out=bool(pending))

            spoken = self._conv.commit_utterance(  # 6.
                pipeline.utterance, interrupted=True, played_ms=played_ms
            )
            if outcome is TurnOutcome.INTERRUPTED and not pipeline.user_committed:
                self._carry_pcm = turn.pcm  # never transcribed: join it to the next utterance
            emit(
                EventKind.TURN_END,
                outcome=outcome.value,
                spoken=spoken,
                empty_transcript=False,
                **_char_counts(pipeline, played_ms),
            )
        finally:
            self._turn = None
            self._set_state(State.IDLE)