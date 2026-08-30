"""One get_audio() contract for all three capture backends (A4 / A10).

`None` means "no block right now"; a terminal failure must raise. A backend
that returns None for a dead device produces the app's worst failure mode:
still "Running", never any audio.
"""

import queue
from dataclasses import dataclass

import numpy as np
import pytest

from audio_capture_base import AudioCaptureBase, enqueue_latest
from platform_permissions import CaptureRuntimeError


@dataclass
class Metrics:
    callback_blocks: int = 0
    output_blocks: int = 0
    dropped_blocks: int = 0
    queue_depth: int = 0
    restart_count: int = 0
    last_error: str | None = None


def test_enqueue_latest_drops_the_oldest_block_when_full():
    q = queue.Queue(maxsize=2)
    m = Metrics()
    for i in range(4):
        enqueue_latest(q, i, m)
    assert [q.get_nowait(), q.get_nowait()] == [2, 3]
    assert m.dropped_blocks == 2


def test_enqueue_latest_survives_a_concurrent_drain():
    """get_nowait() raising queue.Empty used to escape into the Windows read
    loop, which has no outer handler."""
    q = queue.Queue(maxsize=1)
    q.put_nowait("old")
    m = Metrics()

    class RacyQueue:
        def put_nowait(self, item):
            raise queue.Full

        def get_nowait(self):
            raise queue.Empty

        def qsize(self):
            return 0

    assert enqueue_latest(RacyQueue(), "new", m) is False  # must not raise


class Backend(AudioCaptureBase):
    pass


def test_base_get_audio_returns_none_when_idle():
    assert Backend().get_audio(timeout=0.01) is None


def test_base_get_audio_raises_after_a_terminal_failure():
    backend = Backend()
    backend.fail_terminally("device vanished")
    with pytest.raises(CaptureRuntimeError):
        backend.get_audio(timeout=0.01)


def test_restart_clears_the_terminal_state():
    backend = Backend()
    backend.fail_terminally("device vanished")
    backend.start()
    assert backend.get_audio(timeout=0.01) is None


def test_pyaudio_backend_does_not_swallow_terminal_failures():
    pytest.importorskip("pyaudio")
    from audio_capture_pyaudio import PyAudioCapture

    backend = PyAudioCapture.__new__(PyAudioCapture)
    AudioCaptureBase.__init__(backend)
    backend.fail_terminally("microphone read keeps failing")
    with pytest.raises(CaptureRuntimeError):
        backend.get_audio(timeout=0.01)


def test_sck_backend_still_reports_its_own_stream_error():
    pytest.importorskip("numpy")
    from audio_capture_sck import SCKAudioCapture

    backend = SCKAudioCapture.__new__(SCKAudioCapture)
    AudioCaptureBase.__init__(backend)
    backend._stream_error_value = "stream stopped"
    with pytest.raises(CaptureRuntimeError):
        backend.get_audio(timeout=0.01)


def test_a_normal_block_still_comes_through():
    backend = Backend()
    backend.start()
    backend.push_audio(np.zeros(backend.block_size, dtype=np.float32))
    item = backend.get_audio(timeout=0.5)
    assert item is not None
    audio, _ = item
    assert audio.shape == (backend.block_size,)
