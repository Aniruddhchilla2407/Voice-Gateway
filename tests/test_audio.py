from vgw.audio import FRAME_BYTES, FrameSplitter


def test_splitter_emits_whole_frames_and_carries_the_remainder():
    s = FrameSplitter()
    data = bytes(range(256)) * 10  # 2560 bytes = 4 frames + 0 remainder
    assert len(s.feed(data[:700])) == 1  # 700 = 1 frame + 60 bytes pending
    frames = s.feed(data[700:])
    assert len(frames) == 3
    assert s.flush() is None


def test_splitter_preserves_every_byte_in_order():
    s = FrameSplitter()
    data = bytes(i % 251 for i in range(5000))
    out = b"".join(f for chunk in (data[:1], data[1:1001], data[1001:]) for f in s.feed(chunk))
    tail = s.flush()
    assert tail is not None and len(tail) == FRAME_BYTES
    assert (out + tail)[: len(data)] == data
    assert set((out + tail)[len(data) :]) <= {0}  # only zero padding was added


def test_flush_on_empty_splitter_is_none():
    assert FrameSplitter().flush() is None