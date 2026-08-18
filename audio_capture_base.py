"""Platform-independent audio normalization and queueing."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, asdict

import numpy as np

log = logging.getLogger("LiveTranslate.Audio")


@dataclass
class CaptureMetrics:
    callback_blocks: int = 0
    output_blocks: int = 0
    dropped_blocks: int = 0
    queue_depth: int = 0
    restart_count: int = 0
    last_error: str | None = None


class AudioCaptureBase:
    """Normalize arbitrary native callback buffers to 16k mono 512-sample blocks."""

    block_size = 512

    def __init__(self, sample_rate=16000, chunk_duration=0.032, queue_size=100):
        self.sample_rate = int(sample_rate)
        if self.sample_rate != 16000:
            raise ValueError("AudioCaptureBase output sample_rate must be 16000 Hz")
        self.chunk_duration = float(chunk_duration)
        self.audio_queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._running = False
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_mic = np.empty(0, dtype=np.float32)
        self._metrics = CaptureMetrics()
        self._lock = threading.RLock()

    @staticmethod
    def resample_to_mono(
        audio, native_channels=1, native_rate=16000, target_rate=16000
    ):
        if isinstance(audio, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(audio, dtype=np.float32)
        else:
            arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return np.empty(0, dtype=np.float32)
        if arr.ndim == 1 and native_channels > 1:
            usable = arr.size - arr.size % native_channels
            arr = arr[:usable].reshape(-1, native_channels)
        if arr.ndim > 1:
            if arr.shape[-1] == native_channels:
                arr = arr.reshape(-1, native_channels).mean(axis=1)
            elif arr.shape[0] == native_channels:
                arr = arr.reshape(native_channels, -1).mean(axis=0)
            else:
                raise ValueError(
                    f"audio shape {arr.shape} does not match {native_channels} channels"
                )
        arr = np.asarray(arr, dtype=np.float32)
        if int(native_rate) == int(target_rate) or arr.size < 2:
            return arr
        n_out = max(1, int(round(arr.size * target_rate / native_rate)))
        x = np.linspace(0, arr.size - 1, n_out, dtype=np.float32)
        return np.interp(x, np.arange(arr.size, dtype=np.float32), arr).astype(np.float32)

    def _resample_to_mono(self, audio, native_channels=1, native_rate=16000):
        """Compatibility wrapper used by platform backends."""
        return self.resample_to_mono(
            audio, native_channels, native_rate, self.sample_rate
        )

    def _enqueue(self, audio: np.ndarray, mic_rms: float | None):
        item = (np.asarray(audio, dtype=np.float32), mic_rms)
        try:
            self.audio_queue.put_nowait(item)
        except queue.Full:
            self._metrics.dropped_blocks += 1
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                return
        self._metrics.output_blocks += 1
        self._metrics.queue_depth = self.audio_queue.qsize()

    def push_audio(
        self,
        audio,
        *,
        mic_audio=None,
        mic_rms=None,
        native_channels=1,
        native_rate=16000,
    ):
        """Push a chunk; optional native format arguments normalize it first."""
        with self._lock:
            self._metrics.callback_blocks += 1
            main = self.resample_to_mono(
                audio, native_channels, native_rate, self.sample_rate
            )
            if mic_audio is not None:
                mic_audio = self.resample_to_mono(
                    mic_audio, native_channels, native_rate, self.sample_rate
                )
                self._pending_mic = np.concatenate(
                    (
                        self._pending_mic,
                        mic_audio.reshape(-1),
                    )
                )
            self._pending = np.concatenate((self._pending, main))
            while self._pending.size >= self.block_size:
                block = self._pending[: self.block_size]
                self._pending = self._pending[self.block_size :]
                rms = mic_rms
                if self._pending_mic.size:
                    mic = np.zeros(self.block_size, dtype=np.float32)
                    count = min(self.block_size, self._pending_mic.size)
                    mic[:count] = self._pending_mic[:count]
                    self._pending_mic = self._pending_mic[count:]
                    rms = float(np.sqrt(np.mean(mic * mic)))
                    block = block + mic
                self._enqueue(block, rms)

    def flush(self):
        """Drop an incomplete tail; VAD must never receive a short block."""
        with self._lock:
            self._pending = np.empty(0, dtype=np.float32)
            self._pending_mic = np.empty(0, dtype=np.float32)

    def metrics(self) -> dict:
        with self._lock:
            self._metrics.queue_depth = self.audio_queue.qsize()
            return asdict(self._metrics)

    def get_audio(self, timeout=1.0):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()
        self._running = False
        self.flush()

    def start(self):
        self._stop_event.clear()
        self._running = True


class FakeAudioCapture(AudioCaptureBase):
    """Deterministic offline source for pipeline tests and smoke checks."""

    def __init__(self, chunks=(), sample_rate=16000, chunk_duration=0.032, repeat=False):
        super().__init__(sample_rate, chunk_duration)
        self._chunks = list(chunks)
        self._repeat = repeat
        self._thread = None

    def start(self):
        if self._running:
            return
        super().start()

        def produce():
            while self._running and self._chunks:
                for item in self._chunks:
                    if not self._running:
                        break
                    if isinstance(item, tuple):
                        self.push_audio(item[0], mic_audio=item[1])
                    else:
                        self.push_audio(item)
                if not self._repeat:
                    break
            self._running = False

        self._thread = threading.Thread(target=produce, name="fake-audio", daemon=True)
        self._thread.start()

    def stop(self):
        super().stop()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
