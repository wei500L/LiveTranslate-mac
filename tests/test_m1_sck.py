import threading
from types import SimpleNamespace

import numpy as np
import pytest

import audio_capture_sck
from audio_capture_sck import SCKAudioCapture, sample_buffer_to_float32
from platform_permissions import CaptureRuntimeError, PlatformUnavailableError


def test_sample_buffer_fixture_interleaved_int16_is_mono_float32():
    fixture = SimpleNamespace(
        data=np.array([0, 32767, -32768, 0, 16384, -16384], dtype=np.int16).tobytes(),
        sample_format="int16",
        channels=2,
        sample_rate=48000,
        interleaved=True,
    )
    audio, rate = sample_buffer_to_float32(fixture)
    assert rate == 48000
    assert audio.dtype == np.float32
    assert np.allclose(audio, [0.5, -0.5, 0.0], atol=2e-4)


def test_sample_buffer_fixture_noninterleaved_float32_is_mono():
    fixture = {
        "data": np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float32),
        "sample_format": "float32",
        "channels": 2,
        "sample_rate": 16000,
        "interleaved": False,
    }
    audio, rate = sample_buffer_to_float32(fixture)
    assert rate == 16000
    assert np.allclose(audio, [0.5, 0.5, 0.5])


class FakeStream:
    def __init__(self, delegate, fixture=None, fail_start=False):
        self.delegate = delegate
        self.fixture = fixture
        self.fail_start = fail_start
        self.stopped = False

    def addStreamOutput_type_sampleHandlerQueue_error_(self, *args):
        self.output_type = args[1]

    def startCaptureWithCompletionHandler_(self, completion):
        if self.fail_start:
            completion(RuntimeError("start failed"))
            return
        if self.fixture is not None:
            self.delegate.stream_didOutputSampleBuffer_ofType_(self, self.fixture, 1)
        completion(None)

    def stopCaptureWithCompletionHandler_(self, completion):
        self.stopped = True
        completion(None)


def test_sck_lifecycle_reassembles_blocks_and_stop_is_idempotent(monkeypatch):
    fixture = {
        "data": np.ones(700, dtype=np.float32),
        "sample_format": "float32",
        "channels": 1,
        "sample_rate": 16000,
    }
    streams = []

    def factory(delegate):
        stream = FakeStream(delegate, fixture)
        streams.append(stream)
        return stream

    monkeypatch.setattr(audio_capture_sck, "ensure_screen_capture_permission", lambda request: None)
    capture = SCKAudioCapture(
        require_permission=True,
        stream_factory=factory,
    )
    capture.start()
    item = capture.get_audio(timeout=1)
    assert item is not None
    assert item[0].shape == (512,)
    assert capture.metrics()["output_blocks"] == 1
    capture.stop()
    capture.stop()
    assert streams[0].stopped
    assert not any(t.name == "sck-audio" and t.is_alive() for t in threading.enumerate())


def test_sck_start_failure_is_typed_and_leaves_no_worker(monkeypatch):
    monkeypatch.setattr(audio_capture_sck, "ensure_screen_capture_permission", lambda request: None)
    capture = SCKAudioCapture(
        require_permission=False,
        stream_factory=lambda delegate: FakeStream(delegate, fail_start=True),
    )
    with pytest.raises(CaptureRuntimeError, match="start failed"):
        capture.start()
    capture.stop()
    assert capture._worker_thread is None


def test_sck_missing_frameworks_is_explicit(monkeypatch):
    monkeypatch.setattr(audio_capture_sck, "_load_frameworks", lambda: (_ for _ in ()).throw(PlatformUnavailableError("missing")))
    with pytest.raises(PlatformUnavailableError):
        audio_capture_sck._load_frameworks()
