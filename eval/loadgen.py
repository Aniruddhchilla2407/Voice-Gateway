"""Load generator: simulated callers talking to a running gateway over real WebSockets.

Each caller streams 20 ms audio frames in real time (tone = "speech", zeros = silence), so the
server must use the synthetic VAD. Terminal 1 (PowerShell):

    $env:VGW_PROVIDERS = "bench"        # near-zero provider latency: measures the gateway itself
    $env:VGW_VAD = "synthetic"
    $env:VGW_LOG_LEVEL = "warning"
    uvicorn vgw.server:app --port 8000 --log-level warning

Terminal 2 (repo root):

    python eval/loadgen.py --sessions 1 --turns 3                  # sanity check first
    python eval/loadgen.py --sessions 50 --turns 5 --label n50
    python eval/loadgen.py --sessions 20 --scenario barge --label barge20

Numbers are only as trustworthy as the load generator: it reports how often its own frames
left late and warns when it was the bottleneck. Server and load generator share this machine,
so compare runs with each other rather than quoting absolute capacity.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import websockets

from vgw.audio import FRAME_MS, silence, tone
from vgw.tracing.metrics import percentile

FRAME_S = FRAME_MS / 1000.0
SPEECH = tone(FRAME_MS)
QUIET = silence(FRAME_MS)
HEADER = struct.Struct(">I")
LATE_MS = 40.0  # a frame that left this much later than scheduled counts as "late"


class Caller:
    def __init__(self, url: str):
        self.url = url
        self._ws: Any = None
        self._script: deque[tuple[bytes, str | None]] = deque()
        self.marks: dict[str, float] = {}  # label -> perf_counter when that frame was sent
        self.audio: list[tuple[float, int]] = []  # (receive time, turn id) of every audio frame
        self.stops: list[tuple[float, int]] = []  # (receive time, turn id) of stop messages
        self._changed = asyncio.Event()
        self.errors: list[str] = []
        self.frames_sent = 0
        self.late_frames = 0
        self.max_lag_ms = 0.0

    # ----------------------------------------------------------------- audio out
    def say(self, ms: int, tag: str) -> None:
        """Queue `ms` of speech. The first and last frames are labelled so we know when they left."""
        n = ms // FRAME_MS
        for i in range(n):
            label = f"{tag}:start" if i == 0 else (f"{tag}:end" if i == n - 1 else None)
            self._script.append((SPEECH, label))

    async def _sender(self) -> None:
        """Send one frame every 20 ms: queued speech first, silence otherwise."""
        loop = asyncio.get_running_loop()
        due = loop.time()
        while True:
            frame, label = self._script.popleft() if self._script else (QUIET, None)
            await self._ws.send(frame)
            self.frames_sent += 1
            if label:
                self.marks[label] = time.perf_counter()
                self._changed.set()
            due += FRAME_S
            now = loop.time()
            lag_ms = max(0.0, (now - due) * 1000)
            self.max_lag_ms = max(self.max_lag_ms, lag_ms)
            if lag_ms > LATE_MS:
                self.late_frames += 1
            if lag_ms > 100:
                due = now  # far behind: re-anchor instead of bursting to catch up
            elif due > now:
                await asyncio.sleep(due - now)

    # ---------------------------------------------------------------- audio in
    async def _receiver(self) -> None:
        async for message in self._ws:
            now = time.perf_counter()
            if isinstance(message, bytes):
                self.audio.append((now, HEADER.unpack_from(message)[0]))
            else:
                data = json.loads(message)
                if data.get("type") == "stop":
                    self.stops.append((now, int(data["turn_id"])))
            self._changed.set()

    # ------------------------------------------------------------------ waiting
    async def _wait_for(self, probe, timeout: float):
        """Wait until probe() returns something other than None (or the timeout passes)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            value = probe()
            if value is not None:
                return value
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except TimeoutError:
                return probe()

    async def _wait_idle(self, quiet_s: float = 0.4, timeout: float = 30.0) -> None:
        """Wait until no agent audio has arrived for `quiet_s`."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            last = self.audio[-1][0] if self.audio else 0.0
            if time.perf_counter() - last > quiet_s:
                return
            await asyncio.sleep(0.05)

    def _first_audio_after(self, t: float, start: int) -> tuple[float, int] | None:
        for i in range(start, len(self.audio)):
            if self.audio[i][0] > t:
                return self.audio[i]
        return None

    def _audio_after_turn(self, turn_id: int, start: int) -> tuple[float, int] | None:
        for i in range(start, len(self.audio)):
            if self.audio[i][1] > turn_id:
                return self.audio[i]
        return None

    # ---------------------------------------------------------------- scenarios
    async def turn_taking(self, turns: int, think_s: float) -> list[dict[str, Any]]:
        """Speak, wait for the full reply, think, repeat. Measures time to first audio."""
        out: list[dict[str, Any]] = []
        for i in range(turns):
            tag = f"u{i}"
            start = len(self.audio)
            self.say(600, tag)
            end = await self._wait_for(lambda tag=tag: self.marks.get(f"{tag}:end"), 5.0)
            first = None
            if end is not None:
                first = await self._wait_for(
                    lambda end=end, start=start: self._first_audio_after(end, start), 15.0
                )
            if first is None:
                out.append({"kind": "turn", "failed": "no reply"})
                continue
            out.append({"kind": "turn", "ttfa_ms": (first[0] - end) * 1000})
            await self._wait_idle()
            await asyncio.sleep(think_s * random.uniform(0.5, 1.5))
        return out

    async def barge_in(self, rounds: int, listen_s: float) -> list[dict[str, Any]]:
        """Speak, let the agent talk for `listen_s`, then talk over it."""
        out: list[dict[str, Any]] = []
        for i in range(rounds):
            utag, itag = f"u{i}", f"i{i}"
            start = len(self.audio)
            self.say(600, utag)
            end = await self._wait_for(lambda utag=utag: self.marks.get(f"{utag}:end"), 5.0)
            first = None
            if end is not None:
                first = await self._wait_for(
                    lambda end=end, start=start: self._first_audio_after(end, start), 15.0
                )
            if first is None:
                out.append({"kind": "barge", "failed": "no reply"})
                continue
            reply_turn = first[1]
            await asyncio.sleep(listen_s)  # the agent is mid-reply now

            stops_before = len(self.stops)
            self.say(400, itag)  # the caller talks over it
            onset = await self._wait_for(lambda itag=itag: self.marks.get(f"{itag}:start"), 5.0)
            stop = await self._wait_for(
                lambda n=stops_before: self.stops[n] if len(self.stops) > n else None, 3.0
            )
            # The gateway answers the interruption as a new turn; the naive mode never does.
            await self._wait_for(lambda: self._audio_after_turn(reply_turn, start), 2.5)
            await self._wait_idle()

            reply_times = [t for t, tid in self.audio[start:] if tid == reply_turn]
            stale = 0
            if stop is not None:
                stale = sum(1 for t, tid in self.audio[start:] if t > stop[0] and tid <= stop[1])
            out.append(
                {
                    "kind": "barge",
                    "stopped": stop is not None,
                    "cutoff_ms": (stop[0] - onset) * 1000 if stop and onset else None,
                    "talk_over_ms": (reply_times[-1] - onset) * 1000 if reply_times and onset else None,
                    "stale_frames": stale,
                }
            )
        return out

    async def run(self, scenario: str, turns: int, think_s: float, listen_s: float) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        try:
            async with websockets.connect(self.url, max_size=None, ping_interval=None) as ws:
                self._ws = ws
                sender = asyncio.create_task(self._sender())
                receiver = asyncio.create_task(self._receiver())
                try:
                    if scenario == "turns":
                        results = await self.turn_taking(turns, think_s)
                    else:
                        results = await self.barge_in(turns, listen_s)
                finally:
                    for task in (sender, receiver):
                        task.cancel()
                    await asyncio.gather(sender, receiver, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001 - a failed caller is a result, not a crash
            self.errors.append(f"{type(exc).__name__}: {exc}")
        return {
            "results": results,
            "errors": self.errors,
            "frames": self.frames_sent,
            "late": self.late_frames,
            "max_lag_ms": self.max_lag_ms,
        }


# ------------------------------------------------------------------ reporting
def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "p99": round(percentile(values, 99), 1),
        "max": round(max(values), 1),
    }


def line(name: str, s: dict[str, float] | None, unit: str = "ms") -> str:
    if s is None:
        return f"{name:<28} no data"
    return (
        f"{name:<28} n={s['n']:<4.0f} p50={s['p50']:<8} p95={s['p95']:<8} "
        f"p99={s['p99']:<8} max={s['max']} {unit}"
    )


async def amain(args: argparse.Namespace) -> None:
    parsed = urlparse(args.url)
    base = f"{'https' if parsed.scheme == 'wss' else 'http'}://{parsed.netloc}"
    async with httpx.AsyncClient(base_url=base, timeout=10) as http:
        try:
            before = (await http.get("/api/process")).json()
        except httpx.HTTPError as exc:
            sys.exit(f"Cannot reach the server at {base} ({exc}). Start it first; see the top of this file.")

        callers = [Caller(args.url) for _ in range(args.sessions)]

        async def start(i: int, caller: Caller) -> dict[str, Any]:
            await asyncio.sleep(i * args.ramp_s / max(1, args.sessions))
            return await caller.run(args.scenario, args.turns, args.think_s, args.listen_s)

        t0 = time.perf_counter()
        outcomes = await asyncio.gather(*(start(i, c) for i, c in enumerate(callers)))
        wall = time.perf_counter() - t0
        after = (await http.get("/api/process")).json()
        efficiency = (await http.get("/api/metrics")).json()["summary"]["efficiency"]

    results = [r for o in outcomes for r in o["results"]]
    errors = [e for o in outcomes for e in o["errors"]]
    ttfa = [r["ttfa_ms"] for r in results if "ttfa_ms" in r]
    cutoff = [r["cutoff_ms"] for r in results if r.get("cutoff_ms") is not None]
    talk_over = [r["talk_over_ms"] for r in results if r.get("talk_over_ms") is not None]
    failures = sum(1 for r in results if "failed" in r)
    stale = sum(r.get("stale_frames", 0) for r in results)
    frames = sum(o["frames"] for o in outcomes)
    late = sum(o["late"] for o in outcomes)
    max_lag = max((o["max_lag_ms"] for o in outcomes), default=0.0)
    cpu_s = after["cpu_s"] - before["cpu_s"]

    report = {
        "label": args.label,
        "args": vars(args),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
        },
        "wall_s": round(wall, 1),
        "server": {
            "cpu_pct_of_one_core": round(cpu_s / wall * 100, 1),
            "cpu_ms_per_session_second": round(cpu_s * 1000 / (args.sessions * wall), 2),
            "rss_mb": after["rss_mb"],
        },
        "loadgen": {
            "frames": frames,
            "late_frames": late,
            "late_pct": round(100 * late / frames, 2) if frames else 0.0,
            "max_lag_ms": round(max_lag, 1),
        },
        "ttfa_ms": stats(ttfa),
        "barge_in_cutoff_ms": stats(cutoff),
        "talk_over_ms": stats(talk_over),
        "stale_frames": stale,
        "failures": failures,
        "errors": errors[:5],
        "efficiency": efficiency,
    }

    print(f"\n== {args.label}: {args.sessions} sessions x {args.turns} {args.scenario} "
          f"({wall:.0f} s) ==")  # fmt: skip
    print(line("time to first audio", report["ttfa_ms"]))
    if report["ttfa_ms"]:
        print(f"{'  minus the hangover':<28} p50={report['ttfa_ms']['p50'] - args.hangover_ms:.0f} ms"
              f"   (what the gateway and providers add on top of the endpointing wait)")  # fmt: skip
    if args.scenario == "barge":
        print(line("barge-in: onset -> stop sent", report["barge_in_cutoff_ms"]))
        print(line("barge-in: agent talked over", report["talk_over_ms"]))
        print(f"{'stale frames after stop':<28} {stale}   (must be 0)")
    print(f"{'failed turns':<28} {failures}")
    print(f"{'server CPU':<28} {report['server']['cpu_pct_of_one_core']}% of one core, "
          f"{report['server']['cpu_ms_per_session_second']} ms per session-second, "
          f"{report['server']['rss_mb']} MB resident")  # fmt: skip
    print(f"{'load generator':<28} {late}/{frames} frames late ({report['loadgen']['late_pct']}%), "
          f"worst lag {report['loadgen']['max_lag_ms']} ms")  # fmt: skip
    if report["loadgen"]["late_pct"] > 1.0:
        print("WARNING: the load generator itself fell behind, so these numbers are unreliable. "
              "Use fewer sessions or run it on another machine.")  # fmt: skip
    if errors:
        print(f"errors (first {min(5, len(errors))}): {errors[:5]}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--turns", type=int, default=4, help="turns (or interruption rounds) per caller")
    p.add_argument("--scenario", choices=["turns", "barge"], default="turns")
    p.add_argument("--ramp-s", type=float, default=2.0, help="spread connection starts over this long")
    p.add_argument("--think-s", type=float, default=1.0, help="average pause between turns")
    p.add_argument("--listen-s", type=float, default=0.6, help="barge scenario: how long the agent talks before the interruption")
    p.add_argument("--hangover-ms", type=float, default=600, help="the server's endpointing hangover")
    p.add_argument("--label", default="run")
    p.add_argument("--out", default="eval/results")
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()