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


class _NoSessionWriter:
    """TranscriptWriter stand-in: no session open, nothing recorded."""

    @staticmethod
    def active_session():
        return None


class Pipeline:
    """Minimal stand-in exposing exactly what the methods under test use.

    The enqueue path carries the session-lifecycle bookkeeping now: the
    producer fence (`_session_boundary_lock`), the identity snapshot
    (`_session_snapshot`: generation + writer session), the work tracker
    (`_SessionWorkTracker`, driven directly) and the per-item work ids. The
    queue items are 5-tuples ``(seg_type, segment, work_id, generation,
    expected_session)``; assertions compare the first two fields.
    """

    def __init__(self, maxsize=4):
        self._asr_queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._session_boundary_lock = threading.RLock()
        self._session_generation = 0
        self._transcript = _NoSessionWriter()
        self._session_work = main._SessionWorkTracker()
        self._session_work_lock = threading.Lock()
        self._session_work_seq = 0

    _enqueue_asr = main.LiveTranslateApp._enqueue_asr
    _requeue_stop_sentinel = main.LiveTranslateApp._requeue_stop_sentinel
    _drain_interim_duplicates = main.LiveTranslateApp._drain_interim_duplicates
    _session_snapshot = main.LiveTranslateApp._session_snapshot
    _next_session_work_id = main.LiveTranslateApp._next_session_work_id
    _release_queued_work = main.LiveTranslateApp._release_queued_work


def _drain(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def _payloads(q):
    """(seg_type, segment) of every queued item — the identity fields vary."""
    return [(item[0], item[1]) for item in _drain(q)]


def test_enqueue_drops_the_oldest_segment_when_the_queue_is_full():
    p = Pipeline(maxsize=2)
    p._enqueue_asr("vad_flush", "a")
    p._enqueue_asr("vad_flush", "b")
    p._enqueue_asr("vad_flush", "c")
    assert _payloads(p._asr_queue) == [("vad_flush", "b"), ("vad_flush", "c")]
    # The evicted victim's work count was released: nothing waits on it.
    assert p._session_work.pending_count(p._session_generation) == 2


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
    # Full item shape; a dropped interim's work count is released, so a
    # real item carries a work_id for _release_queued_work to key on.
    p._enqueue_asr("interim", None)
    p._enqueue_asr("interim", None)
    p._asr_queue.put_nowait(("vad_flush", "seg"))
    p._drain_interim_duplicates()
    assert _payloads(p._asr_queue) == [("vad_flush", "seg")]
    # Both dropped interims released their counts.
    assert p._session_work.pending_count(p._session_generation) == 0


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
        # Session-lifecycle state stop() touches: the boundary fence, the
        # state machine (IDLE already → no notify needed), the gate flag and
        # the work tracker (discard_all).
        self._session_boundary_lock = threading.RLock()
        self._session_state = main.SessionState.IDLE
        self._session_end_gating = False
        self._session_work = main._SessionWorkTracker()
        self._session_ui_callback = None
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
        self._transcript = self._transcript_stub()
        self._mem_periodic_timer = None
        self._mlx_service = self._recorder("mlx.stop", attr="stop")
        self._audio = self._recorder("audio.stop", attr="stop")

    def _transcript_stub(self):
        app = self

        class _Transcript:
            def set_recording(self, recording):
                app.calls.append("transcript.set_recording")

            def close(self):
                app.calls.append("transcript.close")

        return _Transcript()

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
        "transcript.set_recording",
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


# --- Pause must not splice audio across the gap -----------------------------


class PausableApp:
    """pause() over stubs: it runs on the Qt thread and only touches VAD,
    the ASR queue, the session bookkeeping and the overlay."""

    pause = main.LiveTranslateApp.pause
    _enqueue_asr = main.LiveTranslateApp._enqueue_asr
    _requeue_stop_sentinel = main.LiveTranslateApp._requeue_stop_sentinel
    _session_snapshot = main.LiveTranslateApp._session_snapshot
    _next_session_work_id = main.LiveTranslateApp._next_session_work_id
    _release_queued_work = main.LiveTranslateApp._release_queued_work

    def __init__(self, buffered_segment, interim_active=False, asr_ready=True):
        self._paused = False
        self._asr_ready = asr_ready
        self._interim_active = interim_active
        self._interim_pending = "pending text"
        self._overlay = None
        self._asr_queue = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._vad_lock = threading.RLock()
        self._session_boundary_lock = threading.RLock()
        self._session_generation = 0
        self._transcript = _NoSessionWriter()
        self._session_work = main._SessionWorkTracker()
        self._session_work_lock = threading.Lock()
        self._session_work_seq = 0
        self._vad = self._FakeVAD(buffered_segment)

    class _FakeVAD:
        def __init__(self, segment):
            self.segment = segment
            self.flushed = None

        def flush(self):
            self.flushed = "flush"
            segment, self.segment = self.segment, None
            return segment

        def force_flush(self):
            self.flushed = "force_flush"
            segment, self.segment = self.segment, None
            return segment

    def _reset_interim_state(self):
        self._interim_active = False
        self._interim_pending = ""


def test_pause_hands_off_the_in_flight_utterance():
    """Left in the buffer, it would be spliced onto whatever is said after the
    resume — however long the pause lasted."""
    app = PausableApp(buffered_segment="half a sentence")
    app.pause()
    assert app._paused is True
    assert app._vad.flushed == "flush"
    assert _payloads(app._asr_queue) == [("vad_flush", "half a sentence")]


def test_pause_uses_force_flush_while_interim_is_active():
    """A trimmed buffer is below min_speech_duration by construction, so a
    plain flush would drop the remainder instead of emitting it."""
    app = PausableApp(buffered_segment="remainder", interim_active=True)
    app.pause()
    assert app._vad.flushed == "force_flush"
    assert _payloads(app._asr_queue) == [("vad_flush", "remainder")]


def test_pause_with_an_empty_buffer_queues_the_cleanup_marker():
    """No audio of *this* pause reached the queue, but the interim state may
    still be owned by an earlier queued vad_flush — pause never resets it
    inline. It enqueues a no-audio cleanup marker instead: the ASR loop is a
    single-consumer FIFO, so the marker's handler runs after every pre-pause
    item and performs the one reset."""
    app = PausableApp(buffered_segment=None)
    app.pause()
    payloads = _payloads(app._asr_queue)
    assert payloads == [("vad_flush", None)]  # the marker, no audio
    # The state stays with the queued consumer until the marker runs.
    assert app._interim_pending == "pending text"


def test_pause_without_asr_still_empties_the_vad_buffer():
    """No worker to send it to; the buffer is still cleared by flush(), and
    the state hand-off still goes through the queue-ordered marker."""
    app = PausableApp(buffered_segment="orphan", asr_ready=False)
    app.pause()
    assert _payloads(app._asr_queue) == [("vad_flush", None)]  # marker only
    assert app._interim_active is False  # never was active in this scenario
    assert app._interim_pending == "pending text"  # consumer owns the reset
    assert app._vad.segment is None  # flush() emptied it regardless
