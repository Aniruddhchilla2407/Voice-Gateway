# Voice Gateway

**The timing layer for voice AI.** Most voice assistants feel robotic because of timing, not intelligence: long silences, talking over you, and rambling on after you interrupt. Voice Gateway sits between a caller and the AI services (speech-to-text, LLM, text-to-speech) and fixes the timing. It answers sooner, stops the moment you interrupt, remembers only what you actually heard, and measures every step, so each claim below comes with numbers.

## In plain words

Picture phoning a company and talking to an AI assistant. Behind the scenes three helpers pass notes: one listens (voice to text), one thinks (writes the reply), one speaks (text to voice). Voice Gateway is the conversation manager in the middle, like a good phone operator. It:

- notices when you have finished talking, without jumping in on a short pause
- starts speaking the first sentence while the rest is still being written
- stops within a fraction of a second when you interrupt, and keeps only what you heard in its memory
- contains failures: if a helper fails or times out, that turn ends with a labelled error and the call stays alive
- times every step, so we can show where each millisecond goes

It is not a new AI model: the three helpers are existing services. The work is the traffic control.

## Results

All numbers are measured by `python eval/run_benchmarks.py` and written to [docs/results.md](docs/results.md). They come from one Windows 11 machine (8 logical CPUs, Python 3.13) with the server and load generator on the same machine, so read them as comparisons between rows, not absolute capacity. A run is only counted as valid if the load generator's own frames left on time.

**Time to first audio** is measured from the caller's last speech frame to the first agent audio frame received. It includes a 600 ms endpointing wait, so the last column is what the gateway itself adds.

### Gateway cost (zero-latency stand-in providers)

| Concurrent callers | TTFA p50 / p99 | Added on top of the 600 ms wait (p50) | Server CPU | Memory |
|---|---|---|---|---|
| 1 | 595 / 614 ms | about 0 | 8% of one core | 53 MB |
| 10 | 603 / 613 ms | +3 ms | 23% | 55 MB |
| 50 | 607 / 635 ms | +7 ms | 57% | 65 MB |
| 100 | 608 / 654 ms | +8 ms | 61% | 76 MB |
| 200 | 735 / 958 ms | +135 ms | 85% | 96 MB |

One Python process holds about 100 concurrent callers with single-digit-millisecond overhead. Around 200 the single event loop saturates. Sessions are independent, so the scaling path is more processes (not measured here).

### Barge-in

20 callers, 80 interruptions: the agent stops **142 ms** after the caller starts talking (p95 146 ms, p99 154 ms), with **0** stale audio frames after the stop signal and 0 failed rounds. Most of that time is a deliberate guard: while the agent talks, the gateway needs about 160 ms of sustained speech before treating a sound as an interruption, so noise and "mm-hmm" do not cut it off.

### Against a naive pipeline

The "naive" mode is the same code with sentence streaming and barge-in switched off, so it waits for the whole reply before speaking and ignores the caller while talking. Stand-in providers with realistic latencies, three-sentence reply:

| Mode | TTFA p50 | Stops when interrupted | Agent keeps talking after interruption starts |
|---|---|---|---|
| Voice Gateway | 1,310 ms | 142 ms | 141 ms |
| Naive | 1,809 ms | never | 4,168 ms |

The stand-in TTS takes the same time for any text length, while real TTS takes longer for longer text, so the true time-to-first-audio gap is larger than shown.

### Efficiency, honestly

In the interrupt-every-reply benchmark, 39% of the text the LLM wrote and the TTS synthesized was never heard. Both figures are equal because with fast providers the whole reply is already written and synthesized before the caller interrupts. So cancellation prevents wrong memory and wasted playback, but we do not claim it saves compute here. Efficiency figures use characters (about 4 per token in English), not tokens.

## How it works

```
 browser mic ──20 ms audio frames──▶ WebSocket ──▶ turn manager
                                                      │ voice-activity detection + endpointing
                                                      ▼
            speech-to-text ─▶ LLM (streamed) ─▶ sentence chunker ─▶ text-to-speech ─▶ paced playout ─▶ speaker
                                    bounded queues give backpressure; one cancel stops every stage
```

- **Endpointing.** Voice-activity detection with a silence "hangover" decides when the caller is done. The speech needed to start a turn is raised while the agent is talking.
- **Streaming pipeline.** Stages run concurrently inside one `asyncio.TaskGroup`, connected by bounded queues. TTS starts on the first sentence while the LLM is still writing. Cancelling a turn cancels every stage, and a test checks that no tasks leak.
- **Barge-in, in a fixed order.** Mark the turn cancelled, snapshot how much audio was played, cancel the task, tell the client to stop, wait for the work to unwind, then commit only what was spoken.
- **Memory correctness.** After an interruption the conversation history keeps only the words the caller actually heard (plus a marker), never the unspoken rest of the reply. Tool calls are never truncated.
- **Tracing.** Every stage emits timestamped events. A dashboard at `/dashboard.html` shows the per-turn breakdown and p50/p95/p99.
- **Measured baselines.** The ablation switches (`VGW_MODE=naive`) make the comparison fair: same code, features off.

## Run it

Requires Python 3.11+ and ffmpeg on your PATH (used to decode TTS audio).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,vad,providers]"
cp .env.example .env                   # add GROQ_API_KEY to use the real providers
uvicorn vgw.server:app --port 8000     # no --reload on Windows with real providers
```

Open http://localhost:8000 (use headphones) and the dashboard at http://localhost:8000/dashboard.html. With `VGW_PROVIDERS=stub` (the default) the agent answers with a beep, which is enough to test turn-taking and interruption by ear. With `VGW_PROVIDERS=groq` it uses Groq Whisper for speech-to-text, a Groq-hosted `gpt-oss` model for replies, and edge-tts for speech.

```powershell
pytest -q                                   # unit and integration tests
python eval/run_benchmarks.py --quick       # about 2 minutes
python eval/run_benchmarks.py               # full matrix, writes docs/results.md
```

## Limitations and future work

- The demo agent cannot actually book anything; the tool-calling loop is not built yet.
- A failed provider call ends that turn with an error in the trace, but the caller hears silence; spoken recovery and provider failover are future work.
- Each TTS sentence is fetched and decoded in full before it plays. Streaming the decode would cut time to first audio.
- Whisper can hallucinate text on pure noise, which would produce a nonsense turn after a false trigger.
- edge-tts uses an unofficial Microsoft endpoint and may need upgrading if it breaks.
- Benchmarks were run on one machine; scaling beyond one process is untested.

## Project layout

```
src/vgw/          the gateway: turn manager, pipeline, endpointing, providers, tracing
client/           browser client and metrics dashboard
eval/             load generator and benchmark runner
docs/results.md   latest benchmark output
tests/            automated tests
```

Built with Python, FastAPI/asyncio, WebRTC VAD, Groq (Whisper large-v3-turbo and gpt-oss), edge-tts and ffmpeg.