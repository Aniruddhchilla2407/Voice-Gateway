"""Per-session trace recorder: collects TraceEvents and answers simple timing questions."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from vgw.tracing.events import EventKind, TraceEvent, now_ns


class TraceRecorder:
    def __init__(
        self,
        session_id: str,
        *,
        on_event: Callable[[TraceEvent], None] | None = None,
        max_events: int = 20_000,
    ):
        self.session_id = session_id
        self.events: deque[TraceEvent] = deque(maxlen=max_events)  # bounded: sessions can be long
        self._on_event = on_event  # e.g. push to the dashboard or a metrics aggregator

    def emit(
        self, turn_id: int, kind: EventKind, *, ts_ns: int | None = None, **attrs: Any
    ) -> TraceEvent:
        """Record an event. Pass ts_ns to record something retroactively."""
        ts = now_ns() if ts_ns is None else ts_ns
        event = TraceEvent(self.session_id, turn_id, kind, ts, attrs)
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)
        return event

    def turn_events(self, turn_id: int) -> list[TraceEvent]:
        return [e for e in self.events if e.turn_id == turn_id]

    def first(self, turn_id: int, kind: EventKind) -> TraceEvent | None:
        return next((e for e in self.events if e.turn_id == turn_id and e.kind == kind), None)

    def interval_ms(self, turn_id: int, start: EventKind, end: EventKind) -> float | None:
        """Milliseconds from the first `start` event to the first `end` event of a turn."""
        a, b = self.first(turn_id, start), self.first(turn_id, end)
        if a is None or b is None:
            return None
        return (b.ts_ns - a.ts_ns) / 1e6