"""Optional live test against the real Soniox service.

Runs ONLY when both gates are open:
  SONIOX_API_KEY is set AND RUN_SONIOX_LIVE_TEST=1

Synthesizes a short Russian utterance with the macOS `say` TTS voice (no
bundled media, minimal cloud cost — a pure tone produces no tokens, real
speech is required) and streams it through a real SonioxServiceManager:
connect, provisional tokens, a committed segment, bounded shutdown.
Skipped otherwise — the default suite never touches the network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_LIVE = bool(os.environ.get("SONIOX_API_KEY")) and os.environ.get(
    "RUN_SONIOX_LIVE_TEST"
) == "1"

pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="live Soniox test: set SONIOX_API_KEY and RUN_SONIOX_LIVE_TEST=1",
)


def _synthesize_russian_speech(tmp_path) -> np.ndarray | None:
    """A few seconds of real Russian speech via the macOS `say` voice, or
    None when no Russian voice is available (the test skips)."""
    if sys.platform != "darwin":
        return None
    try:
        voices = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=10
        ).stdout
        if "ru_" not in voices:
            return None
        aiff = tmp_path / "soniox_live_test.aiff"
        subprocess.run(
            [
                "say", "-v", "Milena", "-o", str(aiff),
                "Привет! Это проверка облачного распознавания русской речи "
                "в реальном времени.",
            ],
            check=True, capture_output=True, timeout=30,
        )
        import soundfile as sf

        audio, sr = sf.read(str(aiff), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            # Linear resample, same approach as audio_capture_base.
            n = int(len(audio) * 16000 / sr)
            x = np.linspace(0, len(audio) - 1, n)
            audio = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
        return audio
    except Exception:
        return None


def test_live_stream_commits_segment(tmp_path):
    audio = _synthesize_russian_speech(tmp_path)
    if audio is None:
        pytest.skip("no Russian TTS voice available for the live test")

    from soniox_client import (
        SonioxRuntimeConfig,
        SonioxServiceManager,
        SonioxStatus,
        resolve_api_key,
    )
    from soniox_accumulator import SonioxSegment

    class Sink:
        def __init__(self):
            self.segments = []
            self.statuses = []
            self.errors = []
            self.lives = []
            self._lock = threading.Lock()

        def on_live(self, state):
            with self._lock:
                self.lives.append(state)

        def on_segments(self, segments):
            with self._lock:
                self.segments.extend(segments)

        def on_status(self, status):
            with self._lock:
                self.statuses.append(status)

        def on_error(self, message):
            with self._lock:
                self.errors.append(message)

        def on_metrics(self, metrics):
            pass

    sink = Sink()
    manager = SonioxServiceManager(
        SonioxRuntimeConfig(api_key=resolve_api_key({})), sink
    )
    manager.start()

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if manager.status() == SonioxStatus.LIVE:
            break
        time.sleep(0.2)
    assert manager.status() == SonioxStatus.LIVE, f"statuses={sink.statuses}"

    # Stream the utterance in 512-sample blocks (paced, not realtime-critical).
    for start in range(0, len(audio), 512):
        manager.feed(audio[start:start + 512])
        time.sleep(0.005)

    # finish (end-of-audio) and drain: the server finalizes the tail.
    ok = manager.finish_and_drain(timeout=10.0)
    assert ok, f"drain failed; statuses={sink.statuses} errors={sink.errors}"
    assert not sink.errors, sink.errors
    assert sink.segments, "no segment committed from the live stream"
    segment = sink.segments[0]
    assert isinstance(segment, SonioxSegment)
    # Russian transcription with the Chinese one-way translation attached.
    assert segment.original, "empty transcription"
    assert segment.translation, "empty translation"
    # Provisionals streamed before the commit (the live card's data source).
    assert sink.lives, "no provisional updates received"
