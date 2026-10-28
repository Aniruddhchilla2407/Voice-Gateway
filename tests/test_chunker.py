from vgw.chunker import SentenceChunker, chunk_stream
from vgw.providers.stubs import LatencyProfile, StubLLM

FAST = LatencyProfile(median_ms=1, sigma=0.0)


def run(chunker, fragments):
    out = []
    for f in fragments:
        out += chunker.feed(f)
    return out + chunker.flush()


def test_splits_on_sentence_ends_across_token_boundaries():
    c = SentenceChunker()
    fragments = ["Hello ", "there. ", "How ", "are ", "you? ", "Fine"]
    assert run(c, fragments) == ["Hello there.", "How are you?", "Fine"]


def test_waits_for_whitespace_after_a_period():
    c = SentenceChunker()
    assert c.feed("Hello there.") == []  # could still be "there.com" or "3.5"
    assert c.feed(" Next") == ["Hello there."]


def test_does_not_split_decimals_or_abbreviations():
    c = SentenceChunker()
    text = "The fee is 3.5 rupees. Dr. Rao will see you."
    assert run(c, [text]) == ["The fee is 3.5 rupees.", "Dr. Rao will see you."]


def test_splits_hindi_on_danda():
    c = SentenceChunker()
    assert run(c, ["नमस्ते। आप कैसे हैं? "]) == ["नमस्ते।", "आप कैसे हैं?"]


def test_first_chunk_may_flush_early_at_a_clause_boundary():
    c = SentenceChunker(first_chunk_min_chars=15)
    chunks = c.feed("Sure, I can help you with that, and more. Next")
    assert chunks == ["Sure, I can help you with that,", "and more."]
    assert c.flush() == ["Next"]


def test_later_chunks_do_not_split_on_commas():
    c = SentenceChunker(first_chunk_min_chars=5)
    chunks = run(c, ["First thing done. Then this, and that, and more. "])
    assert chunks == ["First thing done.", "Then this, and that, and more."]


def test_hard_cap_splits_unpunctuated_text_on_word_boundaries():
    c = SentenceChunker(max_chars=50)
    text = "word " * 100
    chunks = run(c, [text])
    assert all(len(ch) <= 50 for ch in chunks)
    assert " ".join(chunks) == text.strip()


def test_unspeakable_chunks_are_dropped():
    c = SentenceChunker()
    chunks = run(c, ["Okay. . Fine. "])
    assert chunks == ["Okay.", "Fine."]


def test_flush_is_empty_when_nothing_is_left():
    c = SentenceChunker()
    c.feed("Done. ")
    assert c.flush() == []
    assert c.flush() == []


async def test_chunk_stream_over_a_stub_llm():
    llm = StubLLM(first_token=FAST, inter_token=FAST)

    async def texts():
        async for d in llm.stream([]):
            yield d.text

    chunks = [c async for c in chunk_stream(texts())]
    assert chunks == ["Sure, I can help with that.", "What day works best for you?"]