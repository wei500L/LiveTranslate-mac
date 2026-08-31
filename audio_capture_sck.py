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

# Consecutive undecodable buffers before the stream is declared dead. The
# pyaudio and WASAPI backends escalate the same way (READ_MAX_FAILURES); the
# SCK worker used to swallow every decode exception, so a stream delivering a
# format this decoder cannot read produced permanent silence while the UI kept
# saying "Running" — exactly the failure mode the get_audio() contract forbids.
_DECODE_MAX_FAILURES = 20

try:  # NSObject is required by Objective-C delegate dispatch on real macOS.
    from Foundation import NSObject as _NSObject
    import objc as _objc
except ImportError:  # pragma: no cover - exercised on non-macOS CI
    _objc = None

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
    if isinstance(data, (list, tuple)) and data and isinstance(
        data[0], (bytes, bytearray, memoryview)
    ):
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


def _audio_buffer_list_planes(buffer_list):
    """Extract the byte planes from common PyObjC AudioBufferList shapes."""
    if buffer_list is None:
        return None
    if isinstance(buffer_list, Mapping):
        buffers = buffer_list.get("mBuffers", buffer_list.get("buffers"))
        if buffers is None and any(key in buffer_list for key in ("mData", "data")):
            buffers = [buffer_list]
    else:
        buffers = getattr(buffer_list, "mBuffers", getattr(buffer_list, "buffers", None))
        if buffers is None and any(hasattr(buffer_list, key) for key in ("mData", "data")):
            buffers = [buffer_list]
    if buffers is None:
        if isinstance(buffer_list, (list, tuple)):
            buffers = buffer_list
        else:
            try:
                buffers = [buffer_list[index] for index in range(len(buffer_list))]
            except (TypeError, IndexError, AttributeError):
                return None
    try:
        buffers = list(buffers)
    except TypeError:
        buffers = [buffers]
    planes = []
    for buffer in buffers:
        data = _field(buffer, "mData", _field(buffer, "data"))
        if data is None:
            continue
        if isinstance(data, (bytes, bytearray, memoryview, np.ndarray)):
            planes.append(data)
    return planes or None


def _sample_buffer_audio_buffer_list(CoreMedia, sample_buffer, channels):
    """Best-effort bridge for PyObjC's AudioBufferList out-parameter API."""
    for name in ("audio_buffer_list", "audioBufferList", "buffer_list"):
        value = _field(sample_buffer, name)
        planes = _audio_buffer_list_planes(value)
        if planes:
            return planes

    getter = getattr(CoreMedia, "CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer", None)
    if getter is None:
        return None
    # PyObjC releases have exposed this C function both as a direct-return
    # helper and as an out-parameter function. Try the supported shapes and
    # inspect returned values rather than assuming one bridge representation.
    calls = (
        (sample_buffer,),
        (sample_buffer, None, None, 0, None, None, 0, None),
        (sample_buffer, None, 0, None, None, 0, None),
    )
    size_needed = 0
    for args in calls:
        try:
            result = getter(*args)
        except Exception:
            continue
        candidates = result if isinstance(result, tuple) else (result,)
        for candidate in candidates:
            planes = _audio_buffer_list_planes(candidate)
            if planes:
                return planes
            if isinstance(candidate, int) and candidate > size_needed:
                size_needed = candidate

    # The normal PyObjC binding exposes AudioBufferList as an in/out argument;
    # allocate the variable-length structure using CoreAudio's manual wrapper
    # and call the API again with the required size discovered above.
    if size_needed:
        try:
            import CoreAudio

            buffer_list = CoreAudio.AudioBufferList(max(1, int(channels)))
            result = getter(
                sample_buffer,
                None,
                buffer_list,
                size_needed,
                None,
                None,
                0,
                None,
            )
            planes = _audio_buffer_list_planes(buffer_list)
            if planes:
                return planes
            candidates = result if isinstance(result, tuple) else (result,)
            for candidate in candidates:
                planes = _audio_buffer_list_planes(candidate)
                if planes:
                    return planes
        except Exception:
            pass
    return None


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
        if isinstance(asbd, tuple):
            # Current PyObjC releases return the AudioStreamBasicDescription
            # struct as a plain tuple in field order rather than an object
            # with named attributes.
            sample_rate, _fmt_id, format_flags, _bpp, _fpp, _bpf, channels, bits = asbd[:8]
        else:
            sample_rate = asbd.mSampleRate
            channels = asbd.mChannelsPerFrame
            format_flags = asbd.mFormatFlags
            bits = getattr(asbd, "mBitsPerChannel", 32)
        rate = int(round(float(sample_rate)))
        channels = int(channels)
        flags = int(format_flags)
        non_interleaved = bool(flags & (1 << 5))
        is_float = bool(flags & 1)
        bits = int(bits or 32)
        fmt = "float32" if is_float and bits <= 32 else ("int16" if bits <= 16 else "int32")

        if non_interleaved:
            planes = _sample_buffer_audio_buffer_list(CoreMedia, sample_buffer, channels)
            if planes:
                return _decode_pcm(planes, fmt, channels=len(planes), interleaved=False), rate
            raise CaptureRuntimeError(
                "ScreenCaptureKit returned non-interleaved audio without an AudioBufferList"
            )

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

    def initWithOwner_(self, owner):
        if _objc is not None:
            self = _objc.super(_SCKStreamDelegate, self).init()
        else:  # pragma: no cover - only used by non-macOS fallback tests
            object.__init__(self)
        if self is not None:
            self.owner = owner
        return self

    def _did_output_sample(self, stream, sample_buffer, output_type):
        self.owner._sample_callback(sample_buffer, output_type)

    if _objc is not None:
        # PyObjC has no metadata for stream:didOutputSampleBuffer:ofType: (it
        # is absent from the SCStreamDelegate protocol listing on current
        # builds), so the inferred signature is "v@:@@@": the SCFrameOutputType
        # enum argument is then bridged as an object pointer, and the first
        # delivered audio buffer segfaults the whole process. Registering the
        # real signature pins the selector spelling too, which is why no
        # colon-less fallback method is needed anymore.
        stream_didOutputSampleBuffer_ofType_ = _objc.selector(
            _did_output_sample,
            selector=b"stream:didOutputSampleBuffer:ofType:",
            signature=b"v36@0:8@16@24i32",
        )
    else:  # pragma: no cover - non-macOS fallback without Objective-C
        stream_didOutputSampleBuffer_ofType_ = _did_output_sample

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
        self._running = False
        self._stop_event.set()
        log.warning("ScreenCaptureKit stream stopped with error: %s", error)
        # Tear down the native stream immediately. The capture loop will see
        # the retained typed error through get_audio() and can report it.
        try:
            self.stop()
        except Exception as exc:
            self._metrics.last_error = str(exc)
            log.warning("Failed to stop ScreenCaptureKit after stream error: %s", exc)

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
        failures = 0
        while self._running or not self._callback_queue.empty():
            try:
                sample = self._callback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                audio, rate = sample_buffer_to_float32(sample)
                if audio.size:
                    mic_count = max(1, int(round(audio.size * self.sample_rate / rate)))
                    mic = self._next_mic(mic_count)
                    self.push_audio(
                        audio,
                        mic_audio=mic,
                        native_rate=rate,
                        mic_native_channels=1,
                        mic_native_rate=self.sample_rate,
                    )
                    failures = 0
            except Exception as exc:
                failures += 1
                self._metrics.last_error = str(exc)
                log.warning(
                    "ScreenCaptureKit audio callback failed (%s/%s): %s",
                    failures, _DECODE_MAX_FAILURES, exc,
                )
                if failures >= _DECODE_MAX_FAILURES:
                    # A stream whose buffers keep failing to decode is
                    # terminally broken, not flaky: surface it through
                    # get_audio() so the pipeline stops instead of feeding on
                    # silence. _capture_loop calls stop() on this error, which
                    # tears the native stream down.
                    self.fail_terminally(
                        f"ScreenCaptureKit audio decode keeps failing: {exc}"
                    )
                    self._running = False
                    return

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
        # Removed from newer ScreenCaptureKit headers: on those systems
        # audio-only capture is expressed by attaching only the audio output
        # below, so the flag is simply unavailable and skipped.
        if hasattr(config, "setExclusivelyCapturesSystemAudio_"):
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
        # Offline/fake streams invoke delegates synchronously and do not need
        # an Objective-C dispatch queue. Real SCStream instances do.
        if self._stream_factory is not None:
            return None
        try:
            import dispatch
            create_queue = getattr(dispatch, "dispatch_queue_create")
            # PyObjC's dispatch binding wants a C string: bytes, not str.
            callback_queue = create_queue(b"LiveTranslate.ScreenCaptureKit", None)
            if callback_queue is None:
                raise RuntimeError("dispatch_queue_create returned nil")
            return callback_queue
        except (ImportError, AttributeError, RuntimeError) as exc:
            raise PlatformUnavailableError(
                "PyObjC libdispatch is required for ScreenCaptureKit callbacks; "
                "install requirements-mac.txt"
            ) from exc

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
        if hasattr(_SCKStreamDelegate, "alloc"):
            self._delegate = _SCKStreamDelegate.alloc().initWithOwner_(self)
        else:
            # Non-macOS fallback: no Objective-C alloc, but the same two-step
            # init contract. Passing the owner to the constructor raised
            # "takes no arguments", which is why the SCK tests only ever ran
            # where pyobjc happened to be installed.
            self._delegate = _SCKStreamDelegate().initWithOwner_(self)
        self._stream_error_value = None
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

    def get_audio(self, timeout=1.0):
        """Return a block or surface a terminal ScreenCaptureKit failure."""
        if self._stream_error_value is not None:
            raise CaptureRuntimeError(
                f"ScreenCaptureKit stream stopped with error: {self._stream_error_value}"
            )
        return super().get_audio(timeout)

    def set_device(self, device_name):
        # Idempotent: the control panel emits the *whole* settings dict on every
        # auto-save, so this is called after any unrelated slider or checkbox
        # change.  Without this short-circuit each of those tore down and
        # rebuilt the SCK stream (a multi-second stall plus an audio gap) for a
        # device that never changed.  set_mic_device below has always done this.
        if self._device_name == device_name:
            return True

        # SCK captures the selected main display.  Device names are not WASAPI
        # identifiers; changing one requires a stream restart/content rebuild.
        previous = self._device_name
        if not self._running:
            self._device_name = device_name
            return True

        # Only advance _device_name once the old stream is down, so the field
        # never advertises a device the backend is not actually capturing.
        self.stop()
        self._device_name = device_name
        try:
            self.start()
            return True
        except Exception as exc:
            self._metrics.last_error = str(exc)
            log.error("SCK device switch to %r failed: %s", device_name, exc)

        # Target failed — put the previous device back so capture keeps running
        # instead of leaving the backend stopped while the app still thinks it
        # is capturing.
        self._device_name = previous
        try:
            self.start()
            log.warning("Restored previous audio device %r", previous)
        except Exception as exc:
            self._metrics.last_error = f"restore failed: {exc}"
            log.error("Restoring previous audio device %r also failed: %s", previous, exc)
        return False

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
        # Typed error from the last failed mode switch, so callers can show an
        # actionable dialog (permission guidance) instead of parsing strings.
        self._switch_error = None
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
        # The macOS settings UI only ever stores "__disabled__" (mic only) or
        # None (ScreenCaptureKit system audio).  A mode flip is a backend
        # rebuild, not a device rename.
        want_sck = device_name != "__disabled__"
        if want_sck == isinstance(self._backend, SCKAudioCapture):
            return self._backend.set_device(device_name)
        self._switch_error = None
        return self._switch_backend(want_sck)

    def _switch_backend(self, want_sck):
        """Rebuild the backing capture, preserving mic and running state.

        A failed switch restores the previous backend so the app never ends up
        with a stopped capture while the UI still advertises the new mode.
        """
        previous = self._backend
        running = bool(getattr(previous, "_running", False))
        mic_device = getattr(previous, "_mic_device_name", None)
        if running:
            previous.stop()
        try:
            if want_sck:
                self._backend = SCKAudioCapture(
                    mic_device=mic_device,
                    sample_rate=previous.sample_rate,
                    chunk_duration=previous.chunk_duration,
                )
            else:
                from audio_capture_pyaudio import PyAudioCapture

                # The "__disabled__" sentinel matters: main.py detects mode
                # changes by comparing _device_name against the stored
                # setting, and the SCK backend carries None. A mic backend
                # built with device=None would make flipping back to system
                # audio a silent no-op.
                self._backend = PyAudioCapture(
                    device="__disabled__",
                    system_audio="disabled",
                    mic_device=mic_device,
                    sample_rate=previous.sample_rate,
                    chunk_duration=previous.chunk_duration,
                )
            if running:
                self._backend.start()
            self._switch_error = None
            return True
        except Exception as exc:
            log.error("macOS audio mode switch failed: %s", exc)
            self._switch_error = exc
            metrics = getattr(previous, "_metrics", None)
            if metrics is not None:
                metrics.last_error = str(exc)
            self._backend = previous
            try:
                if running:
                    previous.start()
            except Exception as restore_exc:
                log.error("Restoring the previous audio backend also failed: %s", restore_exc)
                return False
            return False

    def set_mic_device(self, device_name):
        return self._backend.set_mic_device(device_name)

    def metrics(self):
        return self._backend.metrics()


# Descriptive aliases keep integrations and tests from depending on the
# internal class abbreviation while retaining the concise public name.
ScreenCaptureKitAudioCapture = SCKAudioCapture
AudioCaptureSCK = SCKAudioCapture
