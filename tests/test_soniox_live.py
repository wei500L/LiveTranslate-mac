"""Optional live test against the real Soniox service.

Runs ONLY when both gates are open:
  SONIOX_API_KEY is set AND RUN_SONIOX_LIVE_TEST=1

Streams a few seconds of synthesized audio (no bundled media, minimal cost)
through a real SonioxServiceManager and asserts the full loop: connect,
receive tokens, commit at least one segment, bounded shutdown. Skipped
otherwise — the default suite never touches the network.
"""

from __future__ import annotations

import os
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


def test_live_stream_commits_segment():
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
            self._lock = threading.Lock()

        def on_live(self, state):
            pass

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

    # ~3s of a 220Hz tone with amplitude envelope (clearly voiced), fed in
    # 512-sample blocks like the capture loop does.
    t = np.arange(16000 * 3) / 16000.0
    tone = 0.5 * np.sin(2 * np.pi * 220 * t) * (
        0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    )
    tone = tone.astype(np.float32)
    for start in range(0, len(tone), 512):
        manager.feed(tone[start:start + 512])
        time.sleep(0.032)  # roughly realtime

    # finish (end-of-audio) and drain: the server finalizes the tail.
    ok = manager.finish_and_drain(timeout=10.0)
    assert ok, f"drain failed; statuses={sink.statuses} errors={sink.errors}"
    assert not sink.errors, sink.errors
    assert sink.segments, "no segment committed from the live stream"
    segment = sink.segments[0]
    assert isinstance(segment, SonioxSegment)
    assert segment.original  # some Russian transcription text
