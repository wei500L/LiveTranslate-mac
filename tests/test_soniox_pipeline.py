"""Offline tests for the Soniox commit path and capture-loop integration.

Drives LiveTranslateApp methods bound to a stand-in object (the
tests/test_shutdown.py pattern) plus the real TranscriptWriter against
tmp_path, verifying the meeting-record exactly-once contract and the
identity guards.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

main = pytest.importorskip(
    "main", reason="main.py needs torch + PyQt6, which the offline job skips"
)

import logging  # noqa: E402

from transcript_writer import TranscriptWriter  # noqa: E402
from soniox_accumulator import SonioxLiveState, SonioxSegment  # noqa: E402


class RecordingOverlay:
    """Overlay facade stand-in: records the calls, emits nothing."""

    def __init__(self):
        self.messages = []      # (msg_id, timestamp, original, provider)
        self.live = []          # (msg_id, original, translation, final)
        self.settles = []       # (msg_id, timestamp, original, translation)
        self.notices = []       # (message, level, timeout)
        self.stats = []
        self.connections = []
        self._lock = threading.Lock()

    def add_message(self, msg_id, timestamp, original, source_lang, asr_ms,
                    provider=""):
        with self._lock:
            self.messages.append((msg_id, timestamp, original, provider))

    def update_live(self, msg_id, original, translation, final):
        with self._lock:
            self.live.append((msg_id, original, translation, final))

    def update_stats(self, *args):
        with self._lock:
            self.stats.append(args)

    def update_connection(self, status):
        with self._lock:
            self.connections.append(status)

    def settle_live_message(self, msg_id, timestamp, original, translation,
                            source_lang="ru"):
        with self._lock:
            self.settles.append((msg_id, timestamp, original, translation))

    def show_notice(self, message, level="error", timeout=6000):
        with self._lock:
            self.notices.append((message, level, timeout))


class FakeSubwin:
    def __init__(self):
        self.updated = []
        self.visible = True

    def isVisible(self):
        return True

    def update_text(self, original, translations):
        self.updated.append((original, dict(translations)))


class FakeManager:
    def __init__(self, state=main.SonioxStatus.LIVE):
        self.fed = []
        self.state = state

    def feed(self, chunk):
        self.fed.append(chunk)

    def status(self):
        return self.state


class FakeSonioxClient:
    """ASR shim stand-in: status/pid/manager surface of SonioxASREngine."""

    def __init__(self, status="ready"):
        self.status = status
        self.manager = FakeManager()

    def shutdown(self):
        pass


class StandIn:
    """The attributes _commit_soniox_segment / _soniox_engine_active touch."""

    def __init__(self, tmp_path, engine=None):
        self._transcript = TranscriptWriter(tmp_path)
        self._transcript.set_enabled(True)
        self._transcript.set_recording(True)
        self._session_boundary_lock = threading.RLock()
        self._session_generation = 1
        self._session_state = main.SessionState.ACTIVE
        self._soniox_anchor = None
        self._soniox_live_msg_id = None
        self._soniox_segment_started_at = None
        self._soniox_last_token_at = None
        self._soniox_committed = set()
        self._msg_id = 0
        self._asr_count = 0
        self._translate_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._overlay = RecordingOverlay()
        self._subwin = None
        self._asr = engine
        self._asr_type = "soniox" if engine is not None else None
        self._session_work = main._SessionWorkTracker()
        self._session_state_callbacks = []
        self._stop_event = threading.Event()

        # Bind the real methods.
        self._commit_soniox_segment = main.LiveTranslateApp._commit_soniox_segment.__get__(self)
        self._take_soniox_timing = main.LiveTranslateApp._take_soniox_timing.__get__(self)
        self._soniox_engine_active = main.LiveTranslateApp._soniox_engine_active.__get__(self)
        self._soniox_manager = main.LiveTranslateApp._soniox_manager.__get__(self)
        self._soniox_feed = main.LiveTranslateApp._soniox_feed.__get__(self)
        self._finalize_untranslated = main.LiveTranslateApp._finalize_untranslated.__get__(self)
        self._adopt_auto_opened_session = main.LiveTranslateApp._adopt_auto_opened_session.__get__(self)
        self._notify_session_state = self._notify

    def _notify(self, state, session_id=None, summary=None):
        self._session_state_callbacks.append((state, session_id))

    def _compute_cost(self):
        return 0.0


@pytest.fixture()
def app(tmp_path):
    return StandIn(tmp_path, engine=FakeSonioxClient())


def test_late_connect_failure_after_recording_end_is_ignored(app):
    """A terminal callback must not become a subtitle card after ENDING."""
    app._show_soniox_error = main.LiveTranslateApp._show_soniox_error.__get__(app)
    app._asr.manager.state = main.SonioxStatus.FAILED
    app._soniox_anchor = (app._session_generation, "session")
    app._session_state = main.SessionState.ENDING

    app._show_soniox_error("Soniox connection failed")
    assert app._overlay.notices == []
    assert app._overlay.settles == []

    app._session_state = main.SessionState.ACTIVE
    app._show_soniox_error("Soniox connection failed")
    assert len(app._overlay.notices) == 1
    assert app._overlay.settles == []


def test_commit_writes_original_and_translation_exactly_once(app, tmp_path):
    session_id = app._transcript.begin_session()
    app._session_work.begin(app._session_generation)
    app._soniox_anchor = (app._session_generation, session_id)

    # Two distinct segments (the accumulator cannot emit the same segment
    # twice — each <end> commits once and clears state; two commits here
    # represent two endpoints).
    app._commit_soniox_segment(SonioxSegment(original="Привет мир", translation="你好世界"))
    app._commit_soniox_segment(SonioxSegment(original="Вторая фраза", translation="第二句"))

    all_txt = next(tmp_path.glob("livetrans_*_all.txt"))
    content = all_txt.read_text(encoding="utf-8")
    assert content.count("Привет мир") == 1
    assert content.count("你好世界") == 1
    assert content.count("Вторая фраза") == 1
    app._transcript.set_recording(False)
    summary = app._transcript.end_session()
    assert summary["entries"] == 2
    assert summary["translated"] == 2


def test_same_msg_id_recommit_is_refused_by_writer(app, tmp_path):
    """The exactly-once guard for a *replayed* commit carrying the same
    live-card msg_id: the second write finds no pending original and the
    writer drops it (the _complete contract: no orphan writes)."""
    session_id = app._transcript.begin_session()
    app._session_work.begin(app._session_generation)
    app._soniox_anchor = (app._session_generation, session_id)

    app._msg_id += 1
    app._soniox_live_msg_id = app._msg_id
    app._commit_soniox_segment(SonioxSegment(original="Один раз", translation="仅一次"))
    # Replay the same msg_id as if the live card never cleared.
    app._soniox_live_msg_id = app._msg_id
    app._commit_soniox_segment(SonioxSegment(original="Один раз", translation="仅一次"))

    all_txt = next(tmp_path.glob("livetrans_*_all.txt"))
    content = all_txt.read_text(encoding="utf-8")
    assert content.count("Один раз") == 1
    app._transcript.set_recording(False)
    summary = app._transcript.end_session()
    assert summary["entries"] == 1


def test_commit_logs_cloud_segment_timing(app, caplog):
    """Cloud latency is never shown in the UI; the log is where it lives.

    A segment that showed provisional text logs the two numbers measured
    from the sink's stamps; a segment that never did (fast <end>) logs no
    timing rather than a fabricated zero.
    """
    session_id = app._transcript.begin_session()
    app._session_work.begin(app._session_generation)
    app._soniox_anchor = (app._session_generation, session_id)

    sink = main._SonioxSink(app)
    sink.on_live(SonioxLiveState(original="Привет", translation=""))
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._commit_soniox_segment(
            SonioxSegment(original="Привет мир", translation="你好世界")
        )
    lines = [r.getMessage() for r in caplog.records if "Soniox segment" in r.getMessage()]
    assert len(lines) == 1
    assert "live " in lines[0] and "endpoint +" in lines[0]
    assert "Привет мир" in lines[0]

    # Stamps are consumed: a second segment with no provisionals of its own
    # cannot inherit the first one's numbers.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._commit_soniox_segment(
            SonioxSegment(original="Вторая", translation="第二")
        )
    lines = [r.getMessage() for r in caplog.records if "Soniox segment" in r.getMessage()]
    assert len(lines) == 1
    assert "live " not in lines[0] and "endpoint" not in lines[0]


def test_terminal_status_drops_the_pending_segment_timing(app):
    """A dangling live card settled by FINISHED/FAILED never commits, so its
    stamps must not leak into the next segment's log line."""
    sink = main._SonioxSink(app)
    sink.on_live(SonioxLiveState(original="Незаконченная", translation=""))
    assert app._soniox_segment_started_at is not None
    sink.on_status(main.SonioxStatus.FINISHED)
    assert app._soniox_segment_started_at is None
    assert app._soniox_last_token_at is None
    assert app._take_soniox_timing() == ""


def test_commit_without_anchor_is_display_only(app, tmp_path):
    app._soniox_anchor = None
    app._soniox_live_msg_id = None
    app._commit_soniox_segment(
        SonioxSegment(original="Без сессии", translation="无会话")
    )
    # No session was ever opened: no transcript files at all.
    assert list(tmp_path.glob("livetrans_*")) == []
    # And the overlay got the settle (final render, card reused/created).
    assert app._overlay.settles
    assert app._overlay.settles[-1][2] == "Без сессии"
    assert app._overlay.settles[-1][3] == "无会话"


def test_commit_with_stale_generation_is_display_only(app, tmp_path):
    session_id = app._transcript.begin_session()
    app._session_work.begin(app._session_generation)
    app._soniox_anchor = (app._session_generation + 5, session_id)
    app._commit_soniox_segment(
        SonioxSegment(original="Старое", translation="旧的")
    )
    all_txt = list(tmp_path.glob("livetrans_*_all.txt"))
    assert all_txt == [] or "Старое" not in all_txt[0].read_text("utf-8")


def test_commit_without_translation_finalizes_untranslated(app, tmp_path):
    session_id = app._transcript.begin_session()
    app._session_work.begin(app._session_generation)
    app._soniox_anchor = (app._session_generation, session_id)

    app._commit_soniox_segment(SonioxSegment(original="Только оригинал", translation=""))
    app._transcript.set_recording(False)
    summary = app._transcript.end_session()
    assert summary["entries"] == 1
    assert summary["untranslated"] == 1


def test_live_msg_id_reused_by_commit(app, tmp_path):
    session_id = app._transcript.begin_session()
    app._session_work.begin(app._session_generation)
    app._soniox_anchor = (app._session_generation, session_id)

    # The sink allocated a live card for the provisional phase.
    app._msg_id += 1
    app._soniox_live_msg_id = app._msg_id
    live_id = app._msg_id
    app._commit_soniox_segment(SonioxSegment(original="Привет", translation="你好"))
    # The live card id was consumed by the commit.
    assert app._soniox_live_msg_id is None
    assert app._msg_id == live_id


def test_capture_branch_feeds_only_when_engine_active(app):
    chunk = np.full(512, 0.001, dtype=np.float32)
    # Engine active: the manager receives the float chunk (conversion inside
    # manager.feed in production; the stand-in records the raw chunk).
    assert app._soniox_engine_active() is True
    app._soniox_feed(chunk)
    assert len(app._asr.manager.fed) == 1
    # Engine stopped: no feed.
    app._asr.status = "stopped"
    assert app._soniox_engine_active() is False
    app._soniox_feed(chunk)
    assert len(app._asr.manager.fed) == 1


def test_capture_branch_inactive_for_local_engines():
    stand_in = StandIn.__new__(StandIn)
    stand_in._asr_type = "whisper"
    stand_in._asr = None
    stand_in._asr_ready = True
    bound = main.LiveTranslateApp._soniox_engine_active.__get__(stand_in)
    assert bound() is False


def test_commit_goes_through_real_adoption_for_auto_open(app, tmp_path):
    # Legacy auto-open: recording with no explicit session — the writer
    # auto-opens one on the write. The anchor re-snapshots to the claimed
    # session so subsequent commits carry the right identity.
    app._session_state = main.SessionState.IDLE
    app._soniox_anchor = (app._session_generation, None)
    app._transcript.set_recording(True)

    app._commit_soniox_segment(
        SonioxSegment(original="Автооткрытие", translation="自动开启")
    )
    adopted = app._transcript.active_session()
    assert adopted is not None
    assert app._soniox_anchor == (app._session_generation, adopted)
    # The state machine was notified ACTIVE (adoption claim).
    assert app._session_state_callbacks
    assert app._session_state_callbacks[0][0] == main.SessionState.ACTIVE

    app._transcript.set_recording(False)
    app._transcript.end_session()


def test_subwin_receives_final_pair(app, tmp_path):
    session_id = app._transcript.begin_session()
    app._session_work.begin(app._session_generation)
    app._soniox_anchor = (app._session_generation, session_id)
    app._subwin = FakeSubwin()
    app._commit_soniox_segment(SonioxSegment(original="Окно", translation="窗口"))
    assert app._subwin.updated == [("Окно", {"zh": "窗口"})]
    app._transcript.set_recording(False)
    app._transcript.end_session()


class SinkApp:
    """The attributes _SonioxSink touches (a smaller stand-in than StandIn)."""

    def __init__(self, overlay):
        self._overlay = overlay
        self._msg_id = 0
        self._soniox_live_msg_id = None


def test_sink_finalizes_dangling_live_card_on_terminal_status():
    """Engine teardown (falling back to a local engine) must not leave the
    live card stuck in its provisional dim state."""
    overlay = RecordingOverlay()
    app = SinkApp(overlay)
    sink = main._SonioxSink(app)

    # The sink allocates the live card itself on the first provisional.
    from soniox_accumulator import SonioxLiveState

    sink.on_live(SonioxLiveState(original="Говорили", translation="正在讲"))
    assert overlay.live == []  # first provisional only creates the card
    assert len(overlay.messages) == 1
    assert app._soniox_live_msg_id == app._msg_id

    # Engine torn down -> FINISHED: the card is settled with final=True and
    # the residual id is gone.
    sink.on_status(main.SonioxStatus.FINISHED)
    assert app._soniox_live_msg_id is None
    assert overlay.live and overlay.live[-1] == (
        app._msg_id, "Говорили", "正在讲", True,
    )
    assert overlay.connections[-1] is None  # the indicator is cleared


def test_sink_failed_status_settles_card_but_keeps_chip():
    """Runtime failure settles the card AND keeps the failed indicator
    visible (the user must see the engine is down)."""
    overlay = RecordingOverlay()
    app = SinkApp(overlay)
    sink = main._SonioxSink(app)

    from soniox_accumulator import SonioxLiveState

    sink.on_live(SonioxLiveState(original="Обрыв", translation=""))
    sink.on_status(main.SonioxStatus.FAILED)
    assert app._soniox_live_msg_id is None
    assert overlay.live[-1][3] is True  # settled
    # FAILED keeps the chip visible ("failed"); FINISHED clears it (None).
    assert overlay.connections[-1] == "failed"


def test_sink_terminal_status_without_live_card_is_noop():
    overlay = RecordingOverlay()
    app = SinkApp(overlay)
    sink = main._SonioxSink(app)
    sink.on_status(main.SonioxStatus.FINISHED)
    assert overlay.live == []


def test_engine_shim_starts_its_manager():
    """Regression: SonioxASREngine used to build the manager but never call
    start() — the engine loaded, audio queued, and the connection never
    began (manager stuck in CONNECTING forever). The offline tests all
    passed because they started their own managers; only the real app path
    exposed it. The shim must come up running."""
    asr_soniox = pytest.importorskip(
        "asr_soniox", reason="asr_soniox needs the soniox package"
    )

    started = []

    class FakeManager:
        def __init__(self):
            self.status = "ready"
            self.shutdown_calls = 0

        def start(self):
            started.append(True)

        def shutdown(self, timeout=5.0):
            self.shutdown_calls += 1

    # asr_soniox binds the class at import time; patch its own namespace.
    original = asr_soniox.SonioxServiceManager
    asr_soniox.SonioxServiceManager = lambda config, sink: FakeManager()
    try:
        engine = asr_soniox.SonioxASREngine(
            api_key="k", sink=_NullSink()
        )
        assert started, "engine constructed without starting its manager"
        assert engine.status == "ready"
        engine.shutdown()
    finally:
        asr_soniox.SonioxServiceManager = original


class _NullSink:
    def on_live(self, s): pass
    def on_segments(self, s): pass
    def on_status(self, s): pass
    def on_error(self, m): pass
    def on_metrics(self, m): pass


def test_sink_does_not_open_card_for_empty_provisional():
    """Post-<end> cleared snapshots must not spawn an empty card stuck on
    'translating'."""
    overlay = RecordingOverlay()
    app = SinkApp(overlay)
    sink = main._SonioxSink(app)
    from soniox_accumulator import SonioxLiveState

    sink.on_live(SonioxLiveState(original="", translation=""))
    assert overlay.messages == []
    assert app._soniox_live_msg_id is None
    # Real speech opens the card as before.
    sink.on_live(SonioxLiveState(original="Речь", translation=""))
    assert len(overlay.messages) == 1
