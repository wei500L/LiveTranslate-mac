import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import audio_capture_sck
from audio_capture_sck import SCKAudioCapture, sample_buffer_to_float32
from platform_permissions import (
    CaptureRuntimeError,
    PermissionDeniedError,
    PlatformUnavailableError,
)


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


def test_native_noninterleaved_sample_uses_audio_buffer_list(monkeypatch):
    planes = [
        np.array([1.0, 0.5, 0.0], dtype=np.float32).tobytes(),
        np.array([0.0, 0.5, 1.0], dtype=np.float32).tobytes(),
    ]

    class FakeCoreMedia:
        @staticmethod
        def CMSampleBufferGetFormatDescription(sample):
            return object()

        @staticmethod
        def CMAudioFormatDescriptionGetStreamBasicDescription(desc):
            return SimpleNamespace(
                mSampleRate=48000,
                mChannelsPerFrame=2,
                mFormatFlags=(1 | (1 << 5)),
                mBitsPerChannel=32,
            )

        @staticmethod
        def CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sample):
            return {"mBuffers": [{"mData": plane} for plane in planes]}

        @staticmethod
        def CMSampleBufferGetDataBuffer(sample):
            raise AssertionError("non-interleaved audio must not use CMBlockBuffer directly")

    monkeypatch.setattr(
        audio_capture_sck,
        "_load_frameworks",
        lambda: (FakeCoreMedia, object(), None),
    )
    audio, rate = sample_buffer_to_float32(SimpleNamespace())
    assert rate == 48000
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


def test_sck_mixes_16khz_microphone_without_using_sck_native_rate(monkeypatch):
    fixture = {
        "data": np.ones(1536, dtype=np.float32),
        "sample_format": "float32",
        "channels": 1,
        "sample_rate": 48000,
    }

    class FakeMic:
        def __init__(self):
            self.items = [(np.ones(512, dtype=np.float32), None)]

        def get_audio(self, timeout=0):
            return self.items.pop(0) if self.items else None

    monkeypatch.setattr(audio_capture_sck, "ensure_screen_capture_permission", lambda request: None)
    capture = SCKAudioCapture(
        require_permission=False,
        stream_factory=lambda delegate: FakeStream(delegate, fixture),
        mic_capture=FakeMic(),
    )
    capture.start()
    item = capture.get_audio(timeout=1)
    capture.stop()
    assert item is not None
    assert np.allclose(item[0], 2.0)


def test_sck_stream_error_stops_backend_and_is_propagated(monkeypatch):
    monkeypatch.setattr(audio_capture_sck, "ensure_screen_capture_permission", lambda request: None)
    capture = SCKAudioCapture(
        require_permission=False,
        stream_factory=lambda delegate: FakeStream(delegate),
    )
    capture.start()
    capture._delegate.stream_didStopWithError_(capture._stream, RuntimeError("device lost"))

    assert not capture._running
    assert capture._stream is None
    with pytest.raises(CaptureRuntimeError, match="device lost"):
        capture.get_audio(timeout=0)


def test_sck_missing_frameworks_is_explicit(monkeypatch):
    monkeypatch.setattr(audio_capture_sck, "_load_frameworks", lambda: (_ for _ in ()).throw(PlatformUnavailableError("missing")))
    with pytest.raises(PlatformUnavailableError):
        audio_capture_sck._load_frameworks()


def _stub_capture(monkeypatch, *, device="A", fail_on=()):
    """SCKAudioCapture whose start/stop are recorded instead of touching SCK."""
    cap = SCKAudioCapture.__new__(SCKAudioCapture)
    cap._device_name = device
    cap._running = True
    cap._metrics = SimpleNamespace(last_error=None)
    calls = []

    def fake_stop():
        calls.append(("stop", cap._device_name))
        cap._running = False

    def fake_start():
        calls.append(("start", cap._device_name))
        if cap._device_name in fail_on:
            raise RuntimeError(f"cannot open {cap._device_name}")
        cap._running = True

    monkeypatch.setattr(cap, "stop", fake_stop)
    monkeypatch.setattr(cap, "start", fake_start)
    return cap, calls


def test_set_device_with_the_same_name_does_not_restart_the_stream(monkeypatch):
    """The panel re-emits the whole settings dict on every auto-save, so an
    unchanged device name must not tear down and rebuild the SCK stream."""
    cap, calls = _stub_capture(monkeypatch, device="A")

    assert cap.set_device("A") is True
    assert calls == []
    assert cap._running is True


def test_set_device_failure_restores_the_previous_device(monkeypatch):
    cap, calls = _stub_capture(monkeypatch, device="A", fail_on={"B"})

    assert cap.set_device("B") is False
    # stopped A, failed to start B, then came back to A.
    assert calls == [("stop", "A"), ("start", "B"), ("start", "A")]
    assert cap._device_name == "A"
    assert cap._running is True
    assert cap._metrics.last_error


def test_set_device_reports_stopped_when_recovery_also_fails(monkeypatch):
    cap, calls = _stub_capture(monkeypatch, device="A", fail_on={"A", "B"})

    assert cap.set_device("B") is False
    assert cap._running is False
    assert "restore failed" in cap._metrics.last_error


# --- A permanently undecodable stream must be terminal, not silence ------------


def _capture_with_queue(buffers):
    """A capture whose worker is driven by hand over a prefilled queue."""
    capture = SCKAudioCapture(require_permission=False)
    # The real callback queue is bounded at 16; the escalation test needs more
    # bad buffers than that, and put() on a full queue blocks forever.
    capture._callback_queue = queue.Queue()
    for buffer in buffers:
        capture._callback_queue.put(buffer)
    capture._running = False  # drain the queue and exit, like a real stop
    return capture


def test_consecutive_decode_failures_escalate_to_a_terminal_error(monkeypatch):
    """The pyaudio and WASAPI backends escalate after N consecutive read
    failures; the SCK worker used to swallow every decode exception with a
    warning, so a stream delivering a format it could not decode produced
    permanent silence while the UI kept saying "Running"."""
    import audio_capture_sck

    def boom(_sample):
        raise RuntimeError("cannot decode this format")

    monkeypatch.setattr(audio_capture_sck, "sample_buffer_to_float32", boom)
    capture = _capture_with_queue([object() for _ in range(25)])
    capture._worker()

    with pytest.raises(CaptureRuntimeError, match="decode"):
        capture.get_audio(timeout=0)


def test_intermittent_decode_failures_do_not_escalate(monkeypatch):
    """A single odd buffer between good ones is ordinary; only a run of them
    is terminal."""
    import audio_capture_sck

    calls = iter([
        RuntimeError("odd buffer"),                     # failure 1
        (np.ones(512, dtype=np.float32), 16000),        # success resets
        RuntimeError("odd buffer"),                     # failure 1 again
        (np.ones(512, dtype=np.float32), 16000),
    ])

    def decode(_sample):
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(audio_capture_sck, "sample_buffer_to_float32", decode)
    capture = _capture_with_queue([object() for _ in range(4)])
    capture._worker()

    assert capture._terminal_error is None
    assert capture.get_audio(timeout=0) is not None or capture.metrics()[
        "output_blocks"
    ] > 0


class FakeMicBackend:
    """Stands in for PyAudioCapture: records lifecycle, needs no hardware."""

    def __init__(
        self,
        device=None,
        mic_device=None,
        sample_rate=16000,
        chunk_duration=0.032,
        require_permission=True,
        system_audio="disabled",
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self._device_name = device
        self._mic_device_name = mic_device
        self._running = False
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1
        self._running = True

    def stop(self):
        self.stop_calls += 1
        self._running = False

    def set_device(self, device_name):
        self._device_name = device_name
        return True

    def set_mic_device(self, device_name):
        self._mic_device_name = device_name
        return True


def _mic_class(monkeypatch):
    import audio_capture_pyaudio

    monkeypatch.setattr(audio_capture_pyaudio, "PyAudioCapture", FakeMicBackend)
    return FakeMicBackend


def test_mac_facade_switches_mic_only_and_system_audio(monkeypatch):
    """None = ScreenCaptureKit system audio, "__disabled__" = mic only; a mode
    flip rebuilds the backend and carries the mic selection across."""
    _mic_class(monkeypatch)
    capture = audio_capture_sck.MacAudioCapture(
        system_audio="disabled", mic_device="__default__"
    )
    assert isinstance(capture._backend, FakeMicBackend)

    # Same mode: passthrough, no rebuild.
    assert capture.set_device("__disabled__") is True
    assert isinstance(capture._backend, FakeMicBackend)

    # Mic only -> system audio (pipeline not running: swap without start).
    assert capture.set_device(None) is True
    assert isinstance(capture._backend, SCKAudioCapture)
    assert capture._backend._mic_device_name == "__default__"

    # System audio -> mic only: mic selection survives the rebuild.
    assert capture.set_device("__disabled__") is True
    assert isinstance(capture._backend, FakeMicBackend)
    assert capture._backend._mic_device_name == "__default__"


def test_mac_facade_mode_flip_round_trip_is_visible_to_change_detection(monkeypatch):
    """main.py decides whether to call set_device() by comparing the stored
    audio_device against the backend's _device_name.  A mic-only backend built
    by a mode flip must therefore carry the "__disabled__" sentinel — with
    device=None it would equal the SCK backend's value and flipping back to
    system audio became a silent no-op while the UI showed the new mode."""
    _mic_class(monkeypatch)
    # Constructed the way main.py does: the mic-only path always receives the
    # device sentinel alongside system_audio="disabled".
    capture = audio_capture_sck.MacAudioCapture(
        device="__disabled__", system_audio="disabled", mic_device="__default__"
    )

    for stored in (None, "__disabled__", None, "__disabled__"):
        old = capture._device_name  # exactly main.py's guard read
        assert old != stored, "mode flip would be invisible to main.py's guard"
        assert capture.set_device(stored) is True
    assert isinstance(capture._backend, FakeMicBackend)
    assert capture._backend._device_name == "__disabled__"


def test_mac_facade_running_switch_starts_new_backend_and_stops_old(monkeypatch):
    class FakeSCK:
        def __init__(self, mic_device=None, sample_rate=16000, chunk_duration=0.032):
            self._mic_device_name = mic_device
            self._running = False
            self.start_calls = 0

        def start(self):
            self.start_calls += 1
            self._running = True

        def stop(self):
            self._running = False

    monkeypatch.setattr(audio_capture_sck, "SCKAudioCapture", FakeSCK)
    _mic_class(monkeypatch)
    capture = audio_capture_sck.MacAudioCapture(
        system_audio="disabled", mic_device="__default__"
    )
    capture._backend._running = True  # pretend the pipeline is live
    old_backend = capture._backend

    assert capture.set_device(None) is True
    assert isinstance(capture._backend, FakeSCK)
    assert capture._backend.start_calls == 1
    assert old_backend.stop_calls == 1


def test_mac_facade_failed_switch_restores_previous_backend(monkeypatch):
    """A switch whose new backend cannot start must leave the previous mode
    running, not a stopped capture behind a UI that says otherwise."""

    class FailingSCK:
        def __init__(self, **kwargs):
            self._mic_device_name = kwargs.get("mic_device")

        def start(self):
            raise PermissionDeniedError("Screen Recording access is denied")

        def stop(self):
            pass

    monkeypatch.setattr(audio_capture_sck, "SCKAudioCapture", FailingSCK)
    _mic_class(monkeypatch)
    capture = audio_capture_sck.MacAudioCapture(
        system_audio="disabled", mic_device="__default__"
    )
    capture._backend._running = True
    previous = capture._backend

    assert capture.set_device(None) is False
    assert capture._backend is previous
    assert previous.stop_calls == 1
    assert previous.start_calls == 1  # restarted after the failed switch
    # The typed cause is retained so the UI can show permission guidance
    # instead of a silent mic-only capture behind a "Running" indicator.
    assert isinstance(capture._switch_error, PermissionDeniedError)


def test_mac_facade_switch_error_is_cleared_on_success(monkeypatch):
    class FakeSCK:
        def __init__(self, mic_device=None, sample_rate=16000, chunk_duration=0.032):
            self._mic_device_name = mic_device
            self._running = False
            self._device_name = None

        def start(self):
            self._running = True

        def stop(self):
            self._running = False

        def set_device(self, device_name):
            self._device_name = device_name
            return True

    monkeypatch.setattr(audio_capture_sck, "SCKAudioCapture", FakeSCK)
    _mic_class(monkeypatch)
    capture = audio_capture_sck.MacAudioCapture(
        device="__disabled__", system_audio="disabled", mic_device="__default__"
    )
    capture._switch_error = RuntimeError("stale failure")

    assert capture.set_device(None) is True
    assert capture._switch_error is None


def test_mac_facade_same_mode_device_name_is_passthrough(monkeypatch):
    """Auto-save calls set_device with the whole settings dict after every
    unrelated change; same-mode calls must not tear the SCK stream down."""
    monkeypatch.setattr(
        audio_capture_sck, "ensure_screen_capture_permission", lambda request: None
    )
    capture = audio_capture_sck.MacAudioCapture(system_audio="enabled", mic_device="__default__")
    assert capture.set_device(None) is True
    assert isinstance(capture._backend, SCKAudioCapture)
