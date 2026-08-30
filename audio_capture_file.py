"""File-injection audio backend: replay a recording through the real pipeline.

The pipeline normally gets its audio from ScreenCaptureKit or a microphone,
neither of which can be reproduced in a test or driven from a script. This
backend presents the same contract as every other capture backend (see
``AudioCaptureBase``) while sourcing its blocks from an audio file, so VAD, the
ASR worker, the translator and the transcript writer all run for real.

Used by ``debug_pipeline.py``. It is not wired into the GUI's device list — the
capture backend is chosen by platform in ``main.py``, and this one is a
diagnostic tool rather than a user-facing input source.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np

from audio_capture_base import AudioCaptureBase

log = logging.getLogger("LiveTranslate.Audio.File")


def load_audio_16k_mono(path: str | Path) -> np.ndarray:
    """Decode any supported file to the pipeline's 16 kHz mono float32 format.

    Tries soundfile/librosa first (they handle mp3/m4a/flac), then falls back to
    the stdlib wave module so a plain WAV needs no optional dependency.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")

    try:
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
        return _resample(audio, rate)
    except Exception as exc:
        log.debug("soundfile could not read %s (%s); trying librosa", path, exc)

    try:
        import librosa

        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        return np.asarray(audio, dtype=np.float32)
    except Exception as exc:
        log.debug("librosa could not read %s (%s); trying wave", path, exc)

    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
        if width != 2:
            raise ValueError(f"unsupported WAV sample width: {width * 8} bit")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if handle.getnchannels() > 1:
            audio = audio.reshape(-1, handle.getnchannels()).mean(axis=1)
        return _resample(audio, handle.getframerate())


def _resample(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if int(rate) == 16000 or audio.size < 2:
        return audio
    count = max(1, int(round(audio.size * 16000 / rate)))
    return np.interp(
        np.linspace(0, audio.size - 1, count, dtype=np.float32),
        np.arange(audio.size, dtype=np.float32),
        audio,
    ).astype(np.float32)


class FileAudioCapture(AudioCaptureBase):
    """Replay a decoded recording as 512-sample blocks.

    ``realtime=True`` paces the blocks at wall-clock speed, which is what you
    want when testing timing-sensitive behaviour (interim ASR intervals,
    progressive silence). ``realtime=False`` pushes as fast as the consumer
    drains, which is what you want for a quick correctness run.

    After the file is exhausted it emits ``trailing_silence`` seconds of silence
    so VAD sees the utterance end and flushes the final segment, then reports
    ``finished``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        realtime: bool = True,
        trailing_silence: float = 2.0,
        loops: int = 1,
    ):
        super().__init__()
        self.path = Path(path)
        self.audio = load_audio_16k_mono(self.path)
        self.realtime = realtime
        self.trailing_silence = trailing_silence
        self.loops = max(1, loops)
        self._thread: threading.Thread | None = None
        self.finished = threading.Event()
        # Mirrors the fields main.py reads off a capture backend when settings
        # change; a debug run never changes them, but they must exist.
        self._device_name = f"file:{self.path.name}"
        self._mic_device_name = None

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def start(self):
        if self._running:
            return
        super().start()
        self.finished.clear()
        self._thread = threading.Thread(
            target=self._feed_loop, name="file-capture", daemon=True
        )
        self._thread.start()
        log.info(
            "Replaying %s (%.1fs, %s)",
            self.path.name, self.duration, "realtime" if self.realtime else "fast",
        )

    def stop(self):
        super().stop()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self.finished.set()

    def set_device(self, device_name):
        self._device_name = device_name
        return True

    def set_mic_device(self, device_name):
        self._mic_device_name = device_name
        return True

    def _feed_loop(self):
        block = self.block_size
        period = block / self.sample_rate
        silence = np.zeros(block, dtype=np.float32)
        next_deadline = time.monotonic()
        try:
            for _ in range(self.loops):
                for offset in range(0, len(self.audio), block):
                    if not self._running or self._stop_event.is_set():
                        return
                    chunk = self.audio[offset:offset + block]
                    if chunk.size < block:
                        chunk = np.pad(chunk, (0, block - chunk.size))
                    self._await_room()
                    self.push_audio(chunk)
                    next_deadline = self._pace(next_deadline, period)
            # Let VAD observe the end of speech and flush the last segment.
            for _ in range(int(self.trailing_silence / period)):
                if not self._running or self._stop_event.is_set():
                    return
                self._await_room()
                self.push_audio(silence)
                next_deadline = self._pace(next_deadline, period)
        finally:
            self.finished.set()
            log.info("Replay of %s finished", self.path.name)

    def _await_room(self):
        """Block until the consumer has drained enough of the queue.

        A live device cannot slow down, so AudioCaptureBase drops the oldest
        block when the queue fills. A file can wait — and must, or a fast replay
        silently discards most of the recording before VAD ever sees it.
        """
        limit = max(1, self.audio_queue.maxsize - 4)
        while self.audio_queue.qsize() >= limit:
            if not self._running or self._stop_event.is_set():
                return
            self._stop_event.wait(0.01)

    def _pace(self, deadline: float, period: float) -> float:
        if not self.realtime:
            return deadline
        deadline += period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            self._stop_event.wait(remaining)
        return deadline
