"""Incremental ASR state advance and interim-buffer discipline (B6).

The bug this covers: when every split sentence was short enough to buffer,
_do_interim_asr returned before trimming the audio and before updating the echo
tail. The next pass re-recognized the same audio, produced the same fragment,
and appended it to _interim_pending again — so "はい。" showed up N times.
"""

import pytest

main = pytest.importorskip(
    "main", reason="main.py needs torch + PyQt6, which the offline job skips"
)


class Buffered:
    """Just enough of LiveTranslateApp to drive the fragment buffer."""

    _INTERIM_PENDING_MAX = main.LiveTranslateApp._INTERIM_PENDING_MAX
    _buffer_interim_fragment = main.LiveTranslateApp._buffer_interim_fragment
    _reset_interim_state = main.LiveTranslateApp._reset_interim_state

    def __init__(self):
        self._interim_pending = ""
        self._interim_active = False
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""


def test_the_same_fragment_is_not_buffered_twice():
    app = Buffered()
    app._buffer_interim_fragment("はい。")
    app._buffer_interim_fragment("はい。")
    assert app._interim_pending == "はい。"


def test_distinct_fragments_still_accumulate():
    app = Buffered()
    app._buffer_interim_fragment("はい。")
    app._buffer_interim_fragment("そうです。")
    assert app._interim_pending == "はい。そうです。"


def test_the_pending_buffer_is_bounded():
    app = Buffered()
    for i in range(200):
        app._buffer_interim_fragment(f"fragment-{i} ")
    assert len(app._interim_pending) <= Buffered._INTERIM_PENDING_MAX
    # The newest content is what survives.
    assert "fragment-199" in app._interim_pending


def test_reset_clears_every_interim_field():
    app = Buffered()
    app._interim_pending = "leftover"
    app._interim_active = True
    app._interim_committed_tail = "tail"
    app._last_interim_samples = 999
    app._last_interim_check_time = 5.0
    app._reset_interim_state()
    assert app._interim_pending == ""
    assert app._interim_active is False
    assert app._interim_committed_tail == ""
    assert app._last_interim_samples == 0
    assert app._last_interim_check_time == 0.0


def test_vad_counts_density_discards_so_the_pipeline_can_reset():
    """A discarded segment emits no vad_flush, so the counter is the only
    signal the pipeline gets that the utterance ended."""
    import numpy as np

    from vad_processor import VADProcessor

    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad.threshold = 0.5
    vad.mode = "silero"
    vad.discarded_segments = 0
    vad._speech_buffer = [np.zeros(512, dtype=np.float32) for _ in range(8)]
    vad._confidence_history = [0.0] * 8  # all below threshold -> density 0
    vad._speech_samples = 512 * 8
    vad._is_speaking = True
    vad._silence_counter = 0
    vad._was_trimmed = False

    assert vad._flush_segment() is None
    assert vad.discarded_segments == 1
    assert vad._speech_buffer == []


def test_a_voiced_segment_is_not_counted_as_discarded():
    import numpy as np

    from vad_processor import VADProcessor

    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad.threshold = 0.5
    vad.mode = "silero"
    vad.discarded_segments = 0
    vad._speech_buffer = [np.zeros(512, dtype=np.float32) for _ in range(8)]
    vad._confidence_history = [0.9] * 8
    vad._speech_samples = 512 * 8
    vad._is_speaking = True
    vad._silence_counter = 0
    vad._was_trimmed = False

    assert vad._flush_segment() is not None
    assert vad.discarded_segments == 0
