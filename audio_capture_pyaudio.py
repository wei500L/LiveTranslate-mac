"""CoreAudio/PyAudio microphone-only backend used by macOS M0."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from audio_capture_base import AudioCaptureBase
from platform_permissions import (
    DeviceUnavailableError,
    PlatformUnavailableError,
    ensure_microphone_permission,
)

log = logging.getLogger("LiveTranslate.Audio.PyAudio")


def _load_pyaudio():
    try:
        import pyaudio

        return pyaudio
    except ImportError as exc:
        raise PlatformUnavailableError(
            "PyAudio is required for microphone capture on macOS; install requirements-mac.txt"
        ) from exc


def list_input_devices():
    pa_module = _load_pyaudio()
    pa = pa_module.PyAudio()
    try:
        return [
            pa.get_device_info_by_index(i)["name"]
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
        ]
    finally:
        pa.terminate()


def list_output_devices():
    # System output capture is intentionally deferred to M1/SCK.
    return []


class PyAudioCapture(AudioCaptureBase):
    """Read a microphone stream in a daemon thread and normalize its buffers."""

    def __init__(
        self,
        device=None,
        mic_device=None,
        sample_rate=16000,
        chunk_duration=0.032,
        require_permission=True,
        system_audio="disabled",
    ):
        super().__init__(sample_rate, chunk_duration)
        self._device_name = device
        self._system_audio = system_audio
        self._mic_device_name = mic_device
        self._pa = None
        self._stream = None
        self._thread = None
        self._require_permission = require_permission

    def _find_device(self, name):
        pa = self._pa
        if name in (None, "__default__", "default"):
            return pa.get_default_input_device_info()
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("maxInputChannels", 0) > 0 and dev.get("name") == name:
                return dev
        raise DeviceUnavailableError(f"Microphone device not found: {name}")

    def start(self):
        if self._running:
            return
        if self._system_audio != "disabled":
            raise PlatformUnavailableError(
                "System audio capture on macOS requires the Phase M1 ScreenCaptureKit backend"
            )
        if self._mic_device_name is None:
            super().start()
            self._thread = threading.Thread(
                target=self._silence_loop, name="silent-audio-clock", daemon=True
            )
            self._thread.start()
            return
        if self._require_permission:
            ensure_microphone_permission(request=True)
        module = _load_pyaudio()
        self._pa = module.PyAudio()
        try:
            dev = self._find_device(self._mic_device_name)
            native_rate = int(dev.get("defaultSampleRate", self.sample_rate))
            channels = max(1, int(dev.get("maxInputChannels", 1)))
            frames = max(1, int(native_rate * self.chunk_duration))
            self._stream = self._pa.open(
                format=module.paFloat32,
                channels=channels,
                rate=native_rate,
                input=True,
                input_device_index=dev.get("index"),
                frames_per_buffer=frames,
            )
        except Exception as exc:
            self._metrics.last_error = str(exc)
            self._pa.terminate()
            self._pa = None
            if isinstance(exc, DeviceUnavailableError):
                raise
            raise DeviceUnavailableError(f"Failed to open microphone: {exc}") from exc
        self._native_rate, self._native_channels = native_rate, channels
        super().start()
        self._thread = threading.Thread(
            target=self._read_loop, name="mic-capture", daemon=True
        )
        self._thread.start()

    def _silence_loop(self):
        samples = np.zeros(self.block_size, dtype=np.float32)
        while self._running and not self._stop_event.wait(self.chunk_duration):
            self.push_audio(samples)

    def _read_loop(self):
        while self._running and not self._stop_event.is_set():
            try:
                data = self._stream.read(
                    max(1, int(self._native_rate * self.chunk_duration)),
                    exception_on_overflow=False,
                )
                audio = self.resample_to_mono(
                    np.frombuffer(data, dtype=np.float32),
                    self._native_channels,
                    self._native_rate,
                    self.sample_rate,
                )
                mic_rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
                self.push_audio(audio, mic_rms=mic_rms)
            except Exception as exc:
                self._metrics.last_error = str(exc)
                log.warning("Microphone read failed: %s", exc)
                time.sleep(0.05)

    def set_device(self, device_name):
        """Set system-audio mode; M0 supports only the disabled sentinel."""
        self._device_name = device_name
        if device_name == "__disabled__":
            self._system_audio = "disabled"
            return True
        self._metrics.last_error = "System audio selection requires ScreenCaptureKit"
        return False

    def set_mic_device(self, device_name):
        if device_name == self._mic_device_name:
            return True
        was_running = self._running
        if was_running:
            self.stop()
        self._mic_device_name = device_name
        if was_running:
            self._metrics.restart_count += 1
            self.start()
        return True

    def stop(self):
        if not self._running and self._stream is None:
            return
        super().stop()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None

    def get_audio(self, timeout=1.0):
        try:
            return self.audio_queue.get(timeout=timeout)
        except Exception:
            return None

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
