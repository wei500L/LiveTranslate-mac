"""Platform-independent audio normalization and queueing."""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np

from platform_permissions import CaptureRuntimeError

log = logging.getLogger("LiveTranslate.Audio")


@dataclass
class CaptureMetrics:
    callback_blocks: int = 0
    output_blocks: int = 0
    dropped_blocks: int = 0
    queue_depth: int = 0
    restart_count: int = 0
    last_error: str | None = None


def enqueue_latest(audio_queue, item, metrics) -> bool:
    """Drop-oldest enqueue that never raises.

    Shared with the Windows WASAPI backend, which predates AudioCaptureBase and
    does not inherit from it. Both queue.Empty (a consumer beat us to the head)
    and queue.Full (a producer refilled it) are ordinary races here, and letting
    either escape kills the capture thread.
    """
    try:
        audio_queue.put_nowait(item)
    except queue.Full:
        metrics.dropped_blocks += 1
        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(item)
        except (queue.Empty, queue.Full):
            return False
    metrics.output_blocks += 1
    metrics.queue_depth = audio_queue.qsize()
    return True


class AudioCaptureBase:
    """Normalize arbitrary native callback buffers to 16k mono 512-sample blocks.

    **get_audio() contract, identical in all three backends.** The pipeline's
    _capture_loop distinguishes exactly two outcomes, so a backend must never
    blur them:

    * ``None`` — no block available within ``timeout``. Transient and normal;
      the loop keeps polling.
    * ``(audio, mic_rms)`` — one 512-sample block.
    * raises ``CaptureRuntimeError`` (or another ``PlatformCaptureError``) —
      the capture is terminally dead. The loop stops the pipeline and reports it.

    A backend that turns a terminal failure into ``None`` produces the worst
    failure mode this app has: the process keeps running, the UI keeps saying
    "Running", and no audio ever arrives again. Set ``_terminal_error`` from a
    backend thread to make the next get_audio() raise.
    """

    block_size = 512
    _resample_index_cache = {}
    _resample_cache_lock = threading.Lock()

    def __init__(self, sample_rate=16000, chunk_duration=0.032, queue_size=100):
        self.sample_rate = int(sample_rate)
        if self.sample_rate != 16000:
            raise ValueError("AudioCaptureBase output sample_rate must be 16000 Hz")
        self.chunk_duration = float(chunk_duration)
        self.audio_queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._running = False
        self._pending = deque()
        self._pending_samples = 0
        self._pending_mic = deque()
        self._pending_mic_samples = 0
        self._metrics = CaptureMetrics()
        self._lock = threading.RLock()
        # Set by a backend thread that has failed for good; get_audio() turns it
        # into the exception the pipeline expects.
        self._terminal_error: str | None = None

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
        cache_key = (arr.size, int(native_rate), int(target_rate), n_out)
        with AudioCaptureBase._resample_cache_lock:
            indices = AudioCaptureBase._resample_index_cache.get(cache_key)
            if indices is None:
                indices = (
                    np.linspace(0, arr.size - 1, n_out, dtype=np.float32),
                    np.arange(arr.size, dtype=np.float32),
                )
                if len(AudioCaptureBase._resample_index_cache) >= 32:
                    AudioCaptureBase._resample_index_cache.pop(
                        next(iter(AudioCaptureBase._resample_index_cache))
                    )
                AudioCaptureBase._resample_index_cache[cache_key] = indices
        return np.interp(indices[0], indices[1], arr).astype(np.float32)

    @staticmethod
    def _append_pending(chunks, data):
        chunk = np.asarray(data, dtype=np.float32).reshape(-1)
        if chunk.size:
            chunks.append(np.array(chunk, dtype=np.float32, copy=True))
        return chunk.size

    @staticmethod
    def _take_pending(chunks, count):
        output = np.empty(count, dtype=np.float32)
        offset = 0
        while offset < count and chunks:
            chunk = chunks[0]
            take = min(count - offset, chunk.size)
            output[offset:offset + take] = chunk[:take]
            offset += take
            if take == chunk.size:
                chunks.popleft()
            else:
                chunks[0] = chunk[take:]
        return output, offset

    def _resample_to_mono(self, audio, native_channels=1, native_rate=16000):
        """Compatibility wrapper used by platform backends."""
        return self.resample_to_mono(
            audio, native_channels, native_rate, self.sample_rate
        )

    def _enqueue(self, audio: np.ndarray, mic_rms: float | None):
        enqueue_latest(
            self.audio_queue,
            (np.asarray(audio, dtype=np.float32), mic_rms),
            self._metrics,
        )

    def push_audio(
        self,
        audio,
        *,
        mic_audio=None,
        mic_rms=None,
        native_channels=1,
        native_rate=16000,
        mic_native_channels=1,
        mic_native_rate=16000,
    ):
        """Push a chunk; optional native format arguments normalize it first."""
        with self._lock:
            self._metrics.callback_blocks += 1
            main = self.resample_to_mono(
                audio, native_channels, native_rate, self.sample_rate
            )
            if mic_audio is not None:
                mic_audio = self.resample_to_mono(
                    mic_audio,
                    mic_native_channels,
                    mic_native_rate,
                    self.sample_rate,
                )
                self._pending_mic_samples += self._append_pending(
                    self._pending_mic, mic_audio
                )
            self._pending_samples += self._append_pending(self._pending, main)
            while self._pending_samples >= self.block_size:
                block, _ = self._take_pending(self._pending, self.block_size)
                self._pending_samples -= self.block_size
                rms = mic_rms
                if self._pending_mic_samples:
                    mic = np.zeros(self.block_size, dtype=np.float32)
                    count = min(self.block_size, self._pending_mic_samples)
                    mic_part, consumed = self._take_pending(self._pending_mic, count)
                    mic[:consumed] = mic_part[:consumed]
                    self._pending_mic_samples -= consumed
                    rms = float(np.sqrt(np.mean(mic * mic)))
                    block = block + mic
                self._enqueue(block, rms)

    def flush(self):
        """Drop an incomplete tail; VAD must never receive a short block."""
        with self._lock:
            self._pending.clear()
            self._pending_samples = 0
            self._pending_mic.clear()
            self._pending_mic_samples = 0

    def metrics(self) -> dict:
        with self._lock:
            self._metrics.queue_depth = self.audio_queue.qsize()
            return asdict(self._metrics)

    def fail_terminally(self, reason: str):
        """Mark the capture dead; the next get_audio() raises."""
        self._terminal_error = reason
        self._metrics.last_error = reason
        log.error("Audio capture failed terminally: %s", reason)

    def get_audio(self, timeout=1.0):
        """See the class docstring for the three-outcome contract."""
        if self._terminal_error is not None:
            raise CaptureRuntimeError(self._terminal_error)
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
        self._terminal_error = None
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
