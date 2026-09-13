"""Session-scoped WAV recording with optional FFmpeg MP3 finalization."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("LiveTranslate.AudioRecorder")


@dataclass(frozen=True)
class AudioArtifacts:
    wav: str | None = None
    mp3: str | None = None
    status: str = "none"
    duration_seconds: float = 0.0
    error: str | None = None


class AudioRecorder:
    """A non-blocking PCM recorder owned by one transcript session.

    ``push`` only enqueues/copies a small frame.  A dedicated writer thread
    performs disk I/O, so a slow filesystem cannot stall the capture thread.
    The current capture contract supplies normalized 16 kHz mono frames; the
    same interface accepts native-rate frames when a backend exposes them.
    """

    def __init__(self, base_dir: Path, *, ffmpeg_path: str | None = None):
        self._base_dir = Path(base_dir)
        self._ffmpeg_path = ffmpeg_path
        self._lock = threading.RLock()
        self._queue: list[tuple[np.ndarray, int, int]] = []
        self._wake = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._session_id: str | None = None
        self._wav_path: Path | None = None
        self._mp3_path: Path | None = None
        self._wav: wave.Wave_write | None = None
        self._rate = 16000
        self._channels = 1
        self._samples_written = 0
        self._quality = "high"
        self._error: str | None = None
        self._status = "none"

    @property
    def active(self) -> bool:
        with self._lock:
            return self._session_id is not None and self._status == "recording"

    @property
    def wav_path(self) -> str | None:
        with self._lock:
            return str(self._wav_path) if self._wav_path else None

    @property
    def mp3_path(self) -> str | None:
        with self._lock:
            return str(self._mp3_path) if self._mp3_path else None

    def start(self, session_id: str, *, quality: str = "high") -> None:
        with self._lock:
            if self._session_id is not None:
                return
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._session_id = session_id
            self._quality = quality if quality in {"speech", "high"} else "high"
            self._wav_path = self._base_dir / f"livetrans_{session_id}_audio.wav"
            self._mp3_path = self._base_dir / f"livetrans_{session_id}_audio.mp3"
            self._queue.clear()
            self._stop_requested = False
            self._samples_written = 0
            self._error = None
            self._status = "recording"
            self._thread = threading.Thread(
                target=self._writer_loop,
                name=f"audio-recorder-{session_id}",
                daemon=True,
            )
            self._thread.start()

    def push(self, samples, sample_rate: int = 16000, channels: int = 1) -> None:
        with self._wake:
            if self._session_id is None or self._status != "recording":
                return
            arr = np.asarray(samples, dtype=np.float32)
            if arr.size == 0:
                return
            self._queue.append((np.array(arr, copy=True), int(sample_rate), int(channels)))
            # Bound memory during a disk outage. Dropping the oldest audio is
            # preferable to taking down the capture thread; the final status
            # advertises the degraded recording.
            if len(self._queue) > 200:
                self._queue.pop(0)
                self._error = "audio writer queue overflow"
            self._wake.notify()

    def finish(self, timeout: float = 30.0) -> AudioArtifacts:
        with self._wake:
            if self._session_id is None:
                return AudioArtifacts()
            self._stop_requested = True
            self._wake.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.1, timeout))
        with self._lock:
            if thread is not None and thread.is_alive():
                self._error = self._error or "audio writer did not finish before timeout"
                self._close_wav_locked()
                return self._artifacts_locked("failed")
            return self._artifacts_locked(self._status)

    def abort(self) -> AudioArtifacts:
        with self._wake:
            self._stop_requested = True
            self._queue.clear()
            self._wake.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        with self._lock:
            self._close_wav_locked()
            return self._artifacts_locked("failed")

    def _writer_loop(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._stop_requested:
                    self._wake.wait(timeout=0.5)
                if not self._queue and self._stop_requested:
                    break
                frame = self._queue.pop(0)
            try:
                self._write_frame(*frame)
            except Exception as exc:  # disk errors must not kill the app
                log.warning("Audio recording failed: %s", exc)
                with self._lock:
                    self._error = str(exc)
        with self._lock:
            self._close_wav_locked()
            if self._wav_path is None or self._samples_written == 0:
                self._status = "none"
                self._reset_locked()
                return
            if self._error:
                self._status = "failed"
            else:
                self._status = "encoding"
        if self._status == "encoding":
            self._encode_mp3()
        with self._lock:
            if self._status == "encoding":
                self._status = "ready" if self._mp3_path and self._mp3_path.is_file() else "failed"
            self._reset_locked(keep_artifacts=True)

    def _write_frame(self, samples, sample_rate: int, channels: int) -> None:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if channels > 1:
            usable = arr.size - arr.size % channels
            arr = arr[:usable].reshape(-1, channels).mean(axis=1)
            channels = 1
        if self._quality == "speech":
            sample_rate = 16000
        if self._wav is None:
            self._rate = max(8000, int(sample_rate))
            self._channels = max(1, int(channels))
            assert self._wav_path is not None
            self._wav = wave.open(str(self._wav_path), "wb")
            self._wav.setnchannels(self._channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self._rate)
        if sample_rate != self._rate and arr.size > 1:
            target = max(1, int(round(arr.size * self._rate / sample_rate)))
            x = np.linspace(0, arr.size - 1, target, dtype=np.float32)
            arr = np.interp(x, np.arange(arr.size, dtype=np.float32), arr).astype(np.float32)
        pcm = np.clip(arr, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        self._wav.writeframes(pcm.tobytes())
        self._samples_written += int(pcm.size)

    def _close_wav_locked(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass
            self._wav = None

    def _encode_mp3(self) -> None:
        assert self._wav_path is not None and self._mp3_path is not None
        ffmpeg = self._ffmpeg_path or self._bundled_ffmpeg() or shutil.which("ffmpeg")
        if not ffmpeg:
            with self._lock:
                self._error = "ffmpeg not found; WAV retained"
                self._status = "failed"
            return
        tmp = self._mp3_path.with_suffix(".mp3.tmp")
        bitrate = "128k" if self._quality == "high" else "64k"
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(self._wav_path),
                 "-codec:a", "libmp3lame", "-b:a", bitrate,
                 "-f", "mp3", str(tmp)],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"ffmpeg exited {proc.returncode}")
            tmp.replace(self._mp3_path)
        except Exception as exc:
            log.warning("MP3 encoding failed: %s", exc)
            with self._lock:
                self._error = str(exc)
                self._status = "failed"
            try:
                tmp.unlink()
            except OSError:
                pass

    @staticmethod
    def _bundled_ffmpeg() -> str | None:
        root = Path(__file__).resolve().parent
        names = ("ffmpeg.exe", "ffmpeg")
        for candidate in (root / "ffmpeg" / name for name in names):
            if candidate.is_file():
                return str(candidate)
        return None

    def _artifacts_locked(self, status: str) -> AudioArtifacts:
        duration = self._samples_written / self._rate if self._rate else 0.0
        return AudioArtifacts(
            wav=str(self._wav_path) if self._wav_path and self._wav_path.is_file() else None,
            mp3=str(self._mp3_path) if self._mp3_path and self._mp3_path.is_file() else None,
            status=status,
            duration_seconds=round(duration, 2),
            error=self._error,
        )

    def _reset_locked(self, *, keep_artifacts: bool = False) -> None:
        # Keep paths/status for the caller's finish() result, but release the
        # active session so a new recording can start immediately.
        self._session_id = None
        self._thread = None
        self._queue.clear()
        self._stop_requested = False
        if not keep_artifacts:
            self._wav_path = None
            self._mp3_path = None
