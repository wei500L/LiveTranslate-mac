"""ScreenCaptureKit system-audio capture for macOS 13 and newer.

The module deliberately keeps all PyObjC imports lazy.  Importing the public
audio API therefore remains safe on Windows, Linux, and macOS machines where
the optional frameworks have not been installed yet.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Mapping

import numpy as np

from audio_capture_base import AudioCaptureBase
from platform_permissions import (
    CaptureRuntimeError,
    DeviceUnavailableError,
    PlatformUnavailableError,
    ensure_screen_capture_permission,
)

log = logging.getLogger("LiveTranslate.Audio.ScreenCaptureKit")

try:  # NSObject is required by Objective-C delegate dispatch on real macOS.
    from Foundation import NSObject as _NSObject
except ImportError:  # pragma: no cover - exercised on non-macOS CI
    class _NSObject:  # type: ignore[no-redef]
        pass


def _load_frameworks():
    try:
        import CoreMedia
        import ScreenCaptureKit

        try:
            import Quartz
        except ImportError:
            Quartz = None
        return CoreMedia, ScreenCaptureKit, Quartz
    except ImportError as exc:
        raise PlatformUnavailableError(
            "ScreenCaptureKit requires PyObjC CoreMedia and ScreenCaptureKit frameworks; "
            "install requirements-mac.txt"
        ) from exc


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _decode_pcm(data, sample_format="float32", *, channels=1, interleaved=True):
    """Decode fixture/native bytes or arrays to mono float32 samples."""
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], Mapping):
        data = [plane.get("data", plane.get("mData", b"")) for plane in data]
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (bytes, bytearray)):
        planes = [
            _decode_pcm(plane, sample_format, channels=1, interleaved=True)
            for plane in data
        ]
        if not planes:
            return np.empty(0, dtype=np.float32)
        length = min(p.size for p in planes)
        return np.mean(np.stack([p[:length] for p in planes]), axis=0).astype(np.float32)
    if isinstance(data, (bytes, bytearray, memoryview)):
        fmt = str(sample_format).lower()
        if fmt in {"int16", "s16", "pcm16", "signed16"}:
            arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif fmt in {"int32", "s32", "pcm32", "signed32"}:
            arr = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        elif fmt in {"uint8", "u8"}:
            arr = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            arr = np.frombuffer(data, dtype="<f4").astype(np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)
    if channels > 1:
        if not interleaved and arr.ndim == 2:
            arr = arr.mean(axis=0)
        else:
            usable = arr.size - arr.size % channels
            arr = arr.reshape(-1)[:usable].reshape(-1, channels).mean(axis=1)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _sample_buffer_fixture(sample_buffer):
    """Read the small dict/object fixture format used by offline tests."""
    data = _field(sample_buffer, "data")
    if data is None:
        data = _field(sample_buffer, "audio_buffers")
    if data is None:
        data = _field(sample_buffer, "audio")
    if data is None:
        return None
    channels = int(_field(sample_buffer, "channels", 1) or 1)
    rate = int(_field(sample_buffer, "sample_rate", 16000) or 16000)
    fmt = _field(sample_buffer, "sample_format", _field(sample_buffer, "format", "float32"))
    inferred_interleaved = not (
        isinstance(data, (list, tuple))
        or (isinstance(data, np.ndarray) and data.ndim == 2)
    )
    interleaved = bool(_field(sample_buffer, "interleaved", inferred_interleaved))
    return _decode_pcm(data, fmt, channels=channels, interleaved=interleaved), rate


def sample_buffer_to_float32(sample_buffer):
    """Convert a CMSampleBuffer (or an offline fixture) to ``(audio, rate)``.

    The fixture path is intentionally public: it lets CI validate format
    conversion without requiring a TCC grant or a running macOS capture.
    """
    fixture = _sample_buffer_fixture(sample_buffer)
    if fixture is not None:
        return fixture
    CoreMedia, _, _ = _load_frameworks()
    try:
        desc = CoreMedia.CMSampleBufferGetFormatDescription(sample_buffer)
        asbd = CoreMedia.CMAudioFormatDescriptionGetStreamBasicDescription(desc)
        if hasattr(asbd, "contents"):
            asbd = asbd.contents
        rate = int(round(float(asbd.mSampleRate)))
        channels = int(asbd.mChannelsPerFrame)
        flags = int(asbd.mFormatFlags)
        non_interleaved = bool(flags & (1 << 5))
        is_float = bool(flags & 1)
        bits = int(getattr(asbd, "mBitsPerChannel", 32) or 32)
        fmt = "float32" if is_float and bits <= 32 else ("int16" if bits <= 16 else "int32")

        block = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
        if block is None:
            return np.empty(0, dtype=np.float32), rate
        # CMBlockBufferGetDataPointer has had both tuple and out-parameter
        # signatures across PyObjC releases; accept either representation.
        result = CoreMedia.CMBlockBufferGetDataPointer(block, 0, None, None, None)
        raw = result[-1] if isinstance(result, tuple) else result
        return _decode_pcm(raw, fmt, channels=channels, interleaved=not non_interleaved), rate
    except Exception as exc:
        raise CaptureRuntimeError(f"Unable to decode ScreenCaptureKit sample buffer: {exc}") from exc


class _SCKStreamDelegate(_NSObject):
    """Objective-C delegate kept separate so fake streams can use the same API."""

    def __init__(self, owner):
        try:
            super().__init__()
        except TypeError:
            pass
        self.owner = owner

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
        self.owner._sample_callback(sample_buffer, output_type)

    # Some PyObjC versions expose the selector with a colon spelling in Python.
    def stream_didOutputSampleBuffer_ofType(self, stream, sample_buffer, output_type):
        self.owner._sample_callback(sample_buffer, output_type)

    def stream_didStopWithError_(self, stream, error):
        self.owner._stream_error(error)


class SCKAudioCapture(AudioCaptureBase):
    """Capture system audio with ScreenCaptureKit and normalize it via M0 base."""

    def __init__(
        self,
        device=None,
        mic_device=None,
        sample_rate=16000,
        chunk_duration=0.032,
        require_permission=True,
        stream_factory=None,
        content_provider=None,
        mic_capture=None,
    ):
        super().__init__(sample_rate, chunk_duration)
        self._device_name = device
        self._mic_device_name = mic_device
        self._require_permission = require_permission
        self._stream_factory = stream_factory
        self._content_provider = content_provider
        self._stream = None
        self._delegate = None
        self._worker_thread = None
        self._callback_queue = queue.Queue(maxsize=16)
        self._sample_handler_queue = None
        self._mic_capture = mic_capture
        self._owns_mic_capture = False
        self._mic_pending = np.empty(0, dtype=np.float32)
        self._stream_error_value = None

    def _sample_callback(self, sample_buffer, output_type=None):
        if not self._running or self._stop_event.is_set():
            return
        try:
            self._callback_queue.put_nowait(sample_buffer)
        except queue.Full:
            self._metrics.dropped_blocks += 1
            self._metrics.last_error = "ScreenCaptureKit callback queue is full"

    def _stream_error(self, error):
        self._stream_error_value = str(error)
        self._metrics.last_error = self._stream_error_value
        log.warning("ScreenCaptureKit stream stopped with error: %s", error)

    def _next_mic(self, count):
        if self._mic_capture is None:
            return None
        while self._mic_pending.size < count:
            item = self._mic_capture.get_audio(timeout=0)
            if item is None:
                break
            self._mic_pending = np.concatenate((self._mic_pending, np.asarray(item[0], dtype=np.float32)))
        if self._mic_pending.size < count:
            return None
        mic = self._mic_pending[:count]
        self._mic_pending = self._mic_pending[count:]
        return mic

    def _worker(self):
        while self._running or not self._callback_queue.empty():
            try:
                sample = self._callback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                audio, rate = sample_buffer_to_float32(sample)
                if audio.size:
                    mic = self._next_mic(audio.size)
                    self.push_audio(audio, mic_audio=mic, native_rate=rate)
            except Exception as exc:
                self._metrics.last_error = str(exc)
                log.warning("ScreenCaptureKit audio callback failed: %s", exc)

    @staticmethod
    def _async_result(call, timeout=10.0):
        done = threading.Event()
        error = []

        def completion(*args):
            if args and args[0] not in (None, True):
                error.append(args[0])
            done.set()

        try:
            call(completion)
        except TypeError:
            call()
            return
        if not done.wait(timeout):
            raise CaptureRuntimeError("Timed out waiting for ScreenCaptureKit operation")
        if error:
            raise CaptureRuntimeError(str(error[0]))

    def _build_stream(self):
        if self._stream_factory is not None:
            return self._stream_factory(self._delegate)
        CoreMedia, SCK, _ = _load_frameworks()
        del CoreMedia
        if self._content_provider is None:
            content_done = threading.Event()
            content_result = []

            def content_handler(*args):
                content_result.append(args)
                content_done.set()

            SCK.SCShareableContent.getShareableContentWithCompletionHandler_(content_handler)
            if not content_done.wait(10) or not content_result:
                raise DeviceUnavailableError("No display is available for ScreenCaptureKit")
            content, err = (content_result[0] + (None, None))[:2]
            if err:
                raise DeviceUnavailableError(str(err))
            displays = content.displays() if callable(getattr(content, "displays", None)) else content.displays
            if not displays:
                raise DeviceUnavailableError("No display is available for ScreenCaptureKit")
            display = displays[0]
        else:
            display = self._content_provider()
        filt = SCK.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(display, [], [])
        config = SCK.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setExclusivelyCapturesSystemAudio_(True)
        if hasattr(config, "setShowsCursor_"):
            config.setShowsCursor_(False)
        self._native_rate = 48000
        self._native_channels = 2
        if hasattr(config, "setSampleRate_"):
            config.setSampleRate_(self._native_rate)
        if hasattr(config, "setChannelCount_"):
            config.setChannelCount_(self._native_channels)
        return SCK.SCStream.alloc().initWithFilter_configuration_delegate_(filt, config, self._delegate)

    def _make_sample_handler_queue(self):
        """Create the queue SCK invokes for audio callbacks when available."""
        try:
            import dispatch

            return dispatch.dispatch_queue_create("LiveTranslate.ScreenCaptureKit", None)
        except (ImportError, AttributeError):
            # Fake streams and older PyObjC builds accept ``None``.  Real
            # builds normally provide the dispatch module through PyObjC.
            return None

    def start(self):
        if self._running:
            return
        if self._require_permission:
            ensure_screen_capture_permission(request=True)
        if self._mic_capture is None and self._mic_device_name not in (None, "__disabled__"):
            try:
                from audio_capture_pyaudio import PyAudioCapture

                self._mic_capture = PyAudioCapture(mic_device=self._mic_device_name, require_permission=self._require_permission)
                self._mic_capture.start()
                self._owns_mic_capture = True
            except Exception:
                self._mic_capture = None
                raise
        self._delegate = _SCKStreamDelegate(self)
        try:
            self._stream = self._build_stream()
            self._running = True
            self._stop_event.clear()
            self._sample_handler_queue = self._make_sample_handler_queue()
            self._worker_thread = threading.Thread(target=self._worker, name="sck-audio", daemon=True)
            self._worker_thread.start()
            if hasattr(self._stream, "addStreamOutput_type_sampleHandlerQueue_error_"):
                if self._stream_factory is not None:
                    output_type = 1
                else:
                    output_type = getattr(_load_frameworks()[1], "SCStreamOutputTypeAudio", 1)
                self._stream.addStreamOutput_type_sampleHandlerQueue_error_(self._delegate, output_type, self._sample_handler_queue, None)
            self._async_result(self._stream.startCaptureWithCompletionHandler_)
        except Exception as exc:
            self._metrics.last_error = str(exc)
            self.stop()
            if isinstance(exc, (PlatformUnavailableError, DeviceUnavailableError, CaptureRuntimeError)):
                raise
            raise CaptureRuntimeError(f"Failed to start ScreenCaptureKit: {exc}") from exc

    def stop(self):
        if not self._running and self._stream is None and self._worker_thread is None:
            return
        self._running = False
        self._stop_event.set()
        stream, self._stream = self._stream, None
        if stream is not None and hasattr(stream, "stopCaptureWithCompletionHandler_"):
            try:
                self._async_result(stream.stopCaptureWithCompletionHandler_, timeout=5)
            except Exception as exc:
                self._metrics.last_error = str(exc)
        if self._worker_thread:
            self._worker_thread.join(timeout=3)
            self._worker_thread = None
        self._callback_queue = queue.Queue(maxsize=16)
        self._sample_handler_queue = None
        self.flush()
        if self._mic_capture is not None:
            if self._owns_mic_capture:
                try:
                    self._mic_capture.stop()
                finally:
                    self._mic_capture = None
            else:
                # A caller-owned fake/alternate mic remains available across
                # SCK restarts; its lifecycle is controlled by that caller.
                self._mic_pending = np.empty(0, dtype=np.float32)
            self._owns_mic_capture = False

    def set_device(self, device_name):
        self._device_name = device_name
        # SCK captures the selected main display.  Device names are not WASAPI
        # identifiers; changing one requires a stream restart/content rebuild.
        if self._running:
            self.stop()
            try:
                self.start()
            except Exception:
                return False
        return True

    def set_mic_device(self, device_name):
        if self._mic_device_name == device_name:
            return True
        self._mic_device_name = device_name
        if self._running:
            self.stop()
            try:
                self.start()
            except Exception:
                return False
        return True


class MacAudioCapture:
    """Stable macOS facade selecting SCK system audio or the M0 mic backend."""

    def __init__(self, *args, system_audio="enabled", **kwargs):
        if system_audio == "disabled":
            from audio_capture_pyaudio import PyAudioCapture

            self._backend = PyAudioCapture(*args, system_audio="disabled", **kwargs)
        else:
            self._backend = SCKAudioCapture(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backend, name)

    @property
    def audio_queue(self):
        return self._backend.audio_queue

    def start(self):
        return self._backend.start()

    def stop(self):
        return self._backend.stop()

    def get_audio(self, timeout=1.0):
        return self._backend.get_audio(timeout)

    def set_device(self, device_name):
        return self._backend.set_device(device_name)

    def set_mic_device(self, device_name):
        return self._backend.set_mic_device(device_name)

    def metrics(self):
        return self._backend.metrics()


# Descriptive aliases keep integrations and tests from depending on the
# internal class abbreviation while retaining the concise public name.
ScreenCaptureKitAudioCapture = SCKAudioCapture
AudioCaptureSCK = SCKAudioCapture
