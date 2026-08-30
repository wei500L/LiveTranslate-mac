"""Bounded, idempotent shutdown and the enqueue path (CALL_CHAIN_FIX_TODO A3).

LiveTranslateApp pulls in torch and Qt, so these tests drive the two pieces
that can be exercised standalone: the queue discipline of _enqueue_asr /
_drain_interim_duplicates, and the stop-once bookkeeping of stop(). Both are
bound to a stand-in object carrying only the attributes they touch.
"""

import queue
import threading

import pytest

main = pytest.importorskip(
    "main", reason="main.py needs torch + PyQt6, which the offline job skips"
)


class Pipeline:
    """Minimal stand-in exposing exactly what the methods under test use."""

    def __init__(self, maxsize=4):
        self._asr_queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()

    _enqueue_asr = main.LiveTranslateApp._enqueue_asr
    _requeue_stop_sentinel = main.LiveTranslateApp._requeue_stop_sentinel
    _drain_interim_duplicates = main.LiveTranslateApp._drain_interim_duplicates


def _drain(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def test_enqueue_drops_the_oldest_segment_when_the_queue_is_full():
    p = Pipeline(maxsize=2)
    p._enqueue_asr("vad_flush", "a")
    p._enqueue_asr("vad_flush", "b")
    p._enqueue_asr("vad_flush", "c")
    assert _drain(p._asr_queue) == [("vad_flush", "b"), ("vad_flush", "c")]


def test_enqueue_does_not_dereference_the_stop_sentinel():
    """The overflow path used to do dropped[0] on whatever it pulled out, so a
    queue holding the None sentinel raised TypeError onto the capture thread."""
    p = Pipeline(maxsize=1)
    p._stop_event.set()
    p._asr_queue.put_nowait(None)
    p._enqueue_asr("vad_flush", "seg")  # must not raise
    # Stopping: the segment is refused and the sentinel survives.
    assert _drain(p._asr_queue) == [None]


def test_enqueue_is_a_noop_once_stopping():
    p = Pipeline()
    p._stop_event.set()
    p._enqueue_asr("vad_flush", "seg")
    assert _drain(p._asr_queue) == []


def test_drain_interim_duplicates_keeps_the_first_non_interim_item():
    p = Pipeline(maxsize=8)
    p._asr_queue.put_nowait(("interim", None))
    p._asr_queue.put_nowait(("interim", None))
    p._asr_queue.put_nowait(("vad_flush", "seg"))
    p._drain_interim_duplicates()
    assert _drain(p._asr_queue) == [("vad_flush", "seg")]


def test_drain_interim_duplicates_puts_the_sentinel_back():
    p = Pipeline(maxsize=8)
    p._asr_queue.put_nowait(("interim", None))
    p._asr_queue.put_nowait(None)
    p._drain_interim_duplicates()
    assert _drain(p._asr_queue) == [None]


class StoppableApp:
    """Exercises stop()'s ordering and idempotence without any real resources."""

    def __init__(self):
        self.calls = []
        self._stopped = False
        self._running = True
        self._stop_event = threading.Event()
        self._asr_queue = queue.Queue(maxsize=1)
        self._capture_thread = None
        self._asr_thread = None
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        self._tl_executor = self._recorder("tl_executor.shutdown")
        self._extra_tl_executor = self._recorder("extra_executor.shutdown")
        self._transcript = self._recorder("transcript.close", attr="close")
        self._mem_periodic_timer = None
        self._mlx_service = self._recorder("mlx.stop", attr="stop")
        self._audio = self._recorder("audio.stop", attr="stop")

    def _recorder(self, label, attr="shutdown"):
        calls = self.calls

        class R:
            pass

        setattr(R, attr, lambda _self, *a, **k: calls.append(label))
        return R()

    def _flush_on_stop(self):
        self.calls.append("flush")

    def _reset_interim_state(self):
        self._interim_active = False
        self._interim_pending = ""

    def _shutdown_asr_worker(self):
        self.calls.append("asr_worker.shutdown")

    stop = main.LiveTranslateApp.stop
    _stop_step = main.LiveTranslateApp._stop_step


def test_stop_runs_cleanup_in_the_contracted_order():
    app = StoppableApp()
    app.stop()
    assert app.calls == [
        "audio.stop",
        "flush",
        "tl_executor.shutdown",
        "extra_executor.shutdown",
        "transcript.close",
        "asr_worker.shutdown",
        "mlx.stop",
    ]
    assert app._running is False
    assert app._stop_event.is_set()


def test_stop_is_idempotent_and_does_not_flush_twice():
    app = StoppableApp()
    app.stop()
    app.calls.clear()
    app.stop()
    assert "flush" not in app.calls
    assert "asr_worker.shutdown" in app.calls  # residual cleanup still runs


def test_stop_does_not_block_on_a_full_asr_queue():
    """A blocking put(None) here used to hang forever once the ASR thread was
    gone and the queue stayed full."""
    app = StoppableApp()
    app.calls.clear()
    app._asr_queue.put_nowait(("vad_flush", "seg"))
    done = threading.Event()

    def run():
        app.stop()
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert done.wait(5), "stop() blocked on the full queue"


def test_a_failing_cleanup_step_does_not_skip_the_rest():
    app = StoppableApp()

    def boom():
        raise RuntimeError("transcript is on fire")

    app._transcript.close = boom
    app.stop()
    assert "asr_worker.shutdown" in app.calls
    assert "mlx.stop" in app.calls
