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

The first four sections come from `python eval/run_benchmarks.py` (full output in [docs/results.md](docs/results.md)); the last comes from one live session. The benchmarks ran on one Windows 11 machine (8 logical CPUs, Python 3.13) with the server and load generator on the same machine, so read them as comparisons between rows, not absolute capacity. A run is only counted as valid if the load generator's own frames left on time and no caller failed.

**Time to first audio (TTFA)** is measured from the caller's last speech frame to the first agent audio frame received. It includes a 600 ms endpointing wait, so the "added" column below is what the gateway itself contributes.

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

In the interrupt-every-reply benchmark, 39% of the text the LLM wrote and the TTS synthesized was never heard. The two figures are equal because with fast stand-in providers the whole reply is already written and synthesized before the caller interrupts, so there cancellation prevents wrong memory and wasted playback but saves no compute. With real services the picture differs; see the live test below. Efficiency figures use characters (about 4 per token in English), not tokens.

### Live test with real services

One session on one machine and home connection, using Groq Whisper (`whisper-large-v3-turbo`), a Groq-hosted `openai/gpt-oss-120b` with low reasoning effort, and edge-tts. Most turns in this session were interrupted by new speech or noise before the agent finished, so only 14 turns reached audio. Treat these numbers as indicative, not a benchmark. Latency to real APIs also depends heavily on network distance.

| Stage | Median |
|---|---|
| Endpointing wait | 591 ms |
| Speech-to-text | 227 ms |
| LLM to first sentence | 508 ms |
| TTS to first audio | 1,007 ms |
| **Time to first audio** | **2,358 ms** (p95 2,917 ms) |

- Speech synthesis is the largest stage (about 43%), because each sentence is fetched and decoded in full before it plays. The deliberate 0.6 s endpointing wait is next. Streaming the decode and a shorter wait are the obvious next optimizations.
- In 10 interruptions during playback, the stop signal was sent in under 1 ms after detection (detection itself needs about 140 ms of sustained speech; see Barge-in). All work for the cancelled turn had stopped within 4 ms in every one of 39 cancellations.
- Efficiency: the model wrote 4,325 characters, of which 1,928 (45%) were sent to speech synthesis and 1,167 (27%) were heard. Cancelling early meant about 55% of the written text was never synthesized, which is real saved work because real TTS is slower than the stand-in used in the benchmarks above.

## How it works

```
 browser mic ──20 ms audio frames──▶ WebSocket ──▶ turn manager
                                                      │ voice-activity detection + endpointing
                                                      ▼
            speech-to-text ─▶ LLM (streamed) ─▶ sentence chunker ─▶ text-to-speech ─▶ paced playout ─▶ speaker
                                    bounded queues give backpressure; one cancel stops every stage
```

See [docs/architecture.md](docs/architecture.md) for the full design: the state machine, the barge-in sequence, failure handling and the design trade-offs.

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
cp .env.example .env                   # add GROQ_API_KEY and set VGW_PROVIDERS=groq for real providers
uvicorn vgw.server:app --port 8000     # no --reload on Windows with real providers
```

Open http://localhost:8000 (use headphones) and the dashboard at http://localhost:8000/dashboard.html. With `VGW_PROVIDERS=stub` (the default) the agent answers with a beep, which is enough to test turn-taking and interruption by ear. With `VGW_PROVIDERS=groq` it uses Groq Whisper for speech-to-text, a Groq-hosted `gpt-oss` model for replies, and edge-tts for speech. The first log line after startup shows the settings in effect. Note that variables set in your shell override `.env`.

```powershell
pytest -q                                   # unit and integration tests
python eval/run_benchmarks.py --quick       # about 2 minutes
python eval/run_benchmarks.py               # full matrix, writes docs/results.md
```

## Limitations and future work

- The demo agent cannot actually book anything; the tool-calling loop is not built yet.
- A failed provider call ends that turn with an error in the trace, but the caller hears silence; spoken recovery and provider failover are future work.
- If the caller speaks again before the agent has started talking, the pending reply is cancelled and the new speech is handled instead. With a reply latency of 2+ seconds this can chain.
- Each TTS sentence is fetched and decoded in full before it plays. Streaming the decode would cut time to first audio.
- Whisper can hallucinate text on pure noise, which would produce a nonsense turn after a false trigger.
- No speech-recognition accuracy (word error rate) evaluation yet; speech-to-text is a pluggable provider, so this is future work.
- No scripted multi-language task evaluation (for example clinic booking in English, Hindi and Telugu); this needs the tool-calling loop.
- No authentication on the WebSocket, metrics or dashboard endpoints: local use only.
- edge-tts uses an unofficial Microsoft endpoint and may need upgrading if it breaks.
- Benchmarks were run on one machine; scaling beyond one process is untested.

## Project layout

```
src/vgw/          the gateway: turn manager, pipeline, endpointing, providers, tracing
client/           browser client and metrics dashboard
eval/             load generator and benchmark runner
docs/             architecture.md and the latest benchmark output (results.md)
tests/            automated tests
LICENSE           MIT
```

Built with Python, FastAPI/asyncio, WebRTC VAD, Groq (Whisper large-v3-turbo and gpt-oss), edge-tts and ffmpeg.

## License

MIT. See [LICENSE](LICENSE).