"""Meeting-records page: rendering, filters, rename, providers, no blocking.

Drives the real widget tree offscreen. The summary worker's network path is
stubbed at the client factory; nothing here touches a real API.
"""

import json

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="records page needs Qt")

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ai_summary_service as svc
import meeting_records as records
from meeting_records_page import MeetingRecordsPage, _RENDER_BATCH


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _session_dir(tmp_path, count=2, entries=3):
    for i in range(count):
        stamp = f"2026010{i + 1}_090000"
        blocks = [
            f"**09:0{j}:00** · ru\n\nОригинал {j}\n\n> 译文 {j}\n" for j in range(entries)
        ]
        (tmp_path / f"livetrans_{stamp}_meeting.md").write_text(
            "# Meeting record\n\n" + "\n".join(blocks), encoding="utf-8"
        )
        (tmp_path / f"livetrans_{stamp}_all.txt").write_text(
            "".join(f"[09:0{j}:00] Оригинал {j}\n  -> 译文 {j}\n\n" for j in range(entries)),
            encoding="utf-8",
        )
        (tmp_path / f"livetrans_{stamp}_meta.json").write_text(
            json.dumps({
                "session": stamp, "started": f"2026-01-0{i + 1}T09:00:00",
                "entries": entries, "translated": entries, "untranslated": 0,
                "duration_seconds": 600, "asr_engine": "SenseVoice",
                "translation_model": "deepseek-chat",
                "source_language": "ru", "target_language": "zh",
            }),
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def page(app, tmp_path):
    _session_dir(tmp_path)
    settings = {
        "models": [
            {"id": "m1", "name": "Cloud", "model": "gpt", "api_base": "https://api.example.com/v1", "api_key": "sk-x"},
            {"id": "m2", "name": "Local", "model": "llama", "api_base": "http://127.0.0.1:1234/v1", "api_key": ""},
        ],
        "active_model": 0,
        "ai_summary_provider": "m2",
    }
    page = MeetingRecordsPage(tmp_path, settings)
    page.resize(1000, 700)
    page.refresh()
    app.processEvents()
    yield page
    page.cleanup()


# --- list and detail ------------------------------------------------------------


def test_sessions_are_listed_with_multiline_rows(page, tmp_path):
    assert page._list.count() == 2
    item = page._list.item(0)
    assert "\n" in item.text()  # multi-line, not one compressed row
    assert item.record["session"] == "20260102_090000"  # newest first


def test_search_and_filters(page):
    records.set_session_title(page._dir, "20260102_090000", "俄语课")
    page.refresh()
    page._search.setText("俄语课")
    assert page._list.count() == 1
    page._search.setText("不存在的词")
    assert page._list.count() == 0
    assert page._list_empty.isVisibleTo(page)
    page._search.clear()
    page._filter.setCurrentIndex(2)  # unsummarized
    assert page._list.count() == 2
    page._filter.setCurrentIndex(1)  # summarized: none have summaries
    assert page._list.count() == 0


def test_detail_shows_title_meta_and_structured_entries(page):
    assert page._title_label.text()
    assert "SenseVoice" in page._meta_label.text()
    assert len(page._entries) == 3
    assert page._entries[0]["translation"] == "译文 0"
    assert page._minutes_browser_stack.currentIndex() == 0  # guiding empty state


def test_full_record_renders_entries_incrementally(page, app):
    page._tabs.setCurrentIndex(1)
    app.processEvents()
    assert page._record_area._layout.count() - 1 == 3
    assert page._entries_rendered == 3


def test_toggle_original_rerenders_without_translations_listed_twice(page, app):
    page._show_original_btn.setChecked(True)
    app.processEvents()
    assert page._record_area._layout.count() - 1 == 3
    page._show_original_btn.setChecked(False)
    app.processEvents()
    assert page._record_area._layout.count() - 1 == 3


def test_long_meeting_renders_in_batches(page, app, monkeypatch):
    # A 300-entry meeting must not build 300 widgets on first paint.
    page._entries = [
        {"timestamp": f"00:{i // 60:02d}:{i % 60:02d}", "original": f"o{i}",
         "translation": f"t{i}"}
        for i in range(300)
    ]
    page._entries_rendered = 0
    page._record_area.reset()
    page._render_more_entries()
    assert page._entries_rendered == _RENDER_BATCH
    assert page._record_area._layout.count() - 1 == _RENDER_BATCH
    # Draining the rest via the scrollbar hook
    while page._entries_rendered < len(page._entries):
        page._render_more_entries()
    assert page._record_area._layout.count() - 1 == 300


def test_rename_updates_list_and_detail(page):
    item = page._list.item(0)
    page._list.setCurrentItem(item)
    # Drive the persistence path the rename handler uses (QInputDialog is
    # modal; the dialog-independent path is set_session_title + refresh).
    records.set_session_title(page._dir, item.record["session"], "新标题")
    page.refresh()
    assert page._list.item(0).text().splitlines()[0] == "新标题"


def test_narrow_width_switches_to_stacked_navigation(page, app):
    # The page is hidden here, so Qt defers resize events; drive the layout
    # the way a shown window would (resize + apply).
    page.resize(600, 700)
    page._apply_stacked_or_split()
    from PyQt6.QtCore import Qt

    assert page._splitter.orientation() == Qt.Orientation.Vertical
    # Selecting a session keeps a way back
    page._list.setCurrentRow(0)
    app.processEvents()
    assert page._back_btn.isVisibleTo(page)
    page._show_list()
    assert page._detail_stack.currentIndex() == 1


def test_wide_width_restores_split_navigation(page, app):
    page.resize(600, 700)
    page._apply_stacked_or_split()
    page.resize(1100, 700)
    page._apply_stacked_or_split()
    from PyQt6.QtCore import Qt

    assert page._splitter.orientation() == Qt.Orientation.Horizontal


# --- summary state ---------------------------------------------------------------


def test_summary_roundtrip_shows_ready_state(page):
    stamp = page._list.item(0).record["session"]
    entries = page._entries
    records.save_summary(page._dir, stamp, "# 纪要", {
        "provider_name": "Local", "source_hash": records.source_hash(entries),
    })
    page.refresh()
    page._list.setCurrentRow(0)
    page._load_session(page._list.item(0).record)
    assert page._minutes_browser_stack.currentIndex() == 1
    assert "纪要" in page._minutes_browser.toPlainText()
    assert page._generate_btn.text() != ""  # now says "Regenerate"


def test_stale_summary_is_flagged_in_list_and_detail(page):
    stamp = page._list.item(0).record["session"]
    entries = page._entries
    records.save_summary(page._dir, stamp, "# 旧纪要", {
        "source_hash": "not-the-real-hash",
    })
    page.refresh()
    row = page._list.item(0).text()
    # The stale badge is locale text (zh or en); the record field is the
    # locale-independent assertion.
    assert ("重新生成" in row) or ("regenerate" in row.lower())
    record = page._list.item(0).record
    assert record["summary_stale"] is True


# --- provider selection -------------------------------------------------------------


def test_provider_combo_lists_models_and_persists_choice(page):
    labels = [page._provider_combo.itemText(i) for i in range(page._provider_combo.count())]
    assert "Cloud" in labels and "Local" in labels
    # saved choice (m2, local) preselected
    assert page._provider_combo.currentData() == "m2"
    idx = next(i for i in range(page._provider_combo.count())
               if page._provider_combo.itemData(i) == "m1")
    page._provider_combo.setCurrentIndex(idx)
    assert page._settings["ai_summary_provider"] == "m1"


def test_generate_blocks_without_provider(page, monkeypatch):
    page._settings["ai_summary_provider"] = None
    page._populate_provider_combo()
    page._on_generate()
    assert page._worker is None
    assert page._status_label.text()  # a localized error message


def test_generate_blocks_with_missing_key(page):
    page._settings["ai_summary_provider"] = "m1"  # cloud entry has key sk-x; fake missing
    page._settings["models"][0]["api_key"] = ""
    page._on_generate()
    assert page._worker is None
    assert page._status_label.text()


# --- deletion --------------------------------------------------------------------


def test_delete_removes_session_and_summary_together(page, monkeypatch):
    stamp = page._list.item(0).record["session"]
    records.save_summary(page._dir, stamp, "# m", {})

    confirmed = {}
    monkeypatch.setattr(
        "meeting_records_page.QMessageBox",
        type("MB", (), {
            "warning": staticmethod(lambda *a, **k: confirmed.update(ret=page.__class__ and 16384) or 16384),
            "StandardButton": type("SB", (), {"Yes": 16384, "No": 65536}),
        }),
    )
    page._list.setCurrentRow(0)
    page._delete_session()
    assert not list(page._dir.glob(f"livetrans_{stamp}*"))
    assert page._list.count() == 1


# --- per-record permissions ------------------------------------------------------


class _FakeWriter:
    """Stands in for the app's TranscriptWriter for identity queries."""

    def __init__(self, active=None, ending=None, held=None):
        self._active = active
        self._ending = ending
        # The stamp whose files the writer holds, when it differs from the
        # active/ending answer (the half-dead-close case). Defaults to the
        # live stamp, mirroring the real writer.
        self._held = held
        self.renamed_with = None

    def active_session(self):
        return self._active

    def ending_session(self):
        return self._ending

    def held_session(self):
        if self._held is not None:
            return self._held
        return self._ending or self._active

    def rename_session(self, title, expected_session=None):
        self.renamed_with = (title, expected_session)
        return True


def test_ending_marks_only_its_own_row_and_gates_actions(page, monkeypatch):
    """A global ENDING paints exactly one row (the closing session's); the
    end button, AI minutes and delete all key on the record's own
    identity, never on the global state alone."""
    page._list.setCurrentRow(0)
    closing = page._list.item(0).record["session"]
    other = page._list.item(1).record["session"]

    # The closing meeting was a cached live row: the ENDING push takes the
    # lightweight in-memory path (no disk scan) and flips its flags there.
    listing = []
    monkeypatch.setattr(
        records, "list_sessions",
        lambda *a, **k: listing.append(a) or [],
    )
    page.set_transcript_writer(_FakeWriter(active=closing))
    page._record_for_session(closing)["is_active"] = True
    page._ending_session_id = closing
    page.on_session_state_changed("ending", closing)
    assert listing == []

    # Row identity: only the closing row is ending.
    by_stamp = {r["session"]: r for r in page._sessions}
    assert by_stamp[closing]["is_ending"] is True
    assert by_stamp[other]["is_ending"] is False

    # Permissions on the closing row.
    assert page._can_summarize(by_stamp[closing]) is False
    assert page._can_delete(by_stamp[closing]) is False
    assert page._can_rename(by_stamp[closing]) is False
    # Permissions on the unrelated history row are unaffected.
    assert page._can_summarize(by_stamp[other]) is True
    assert page._can_delete(by_stamp[other]) is True
    assert page._can_rename(by_stamp[other]) is True


def test_delete_of_live_or_ending_session_is_refused(page, monkeypatch):
    """The identity re-check inside _delete_session is the guard (hiding
    the menu entry is only a hint): a live or closing session must never
    be unlinked while the writer holds its files open."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    page.set_transcript_writer(_FakeWriter(active=stamp))
    page.refresh()

    popped = []
    monkeypatch.setattr(
        "meeting_records_page.QMessageBox",
        type("MB", (), {
            "warning": staticmethod(lambda *a, **k: popped.append(1) or 16385),
            "StandardButton": type("SB", (), {"Yes": 16384, "No": 65536}),
        }),
    )
    page._delete_session()
    # Refused before any confirmation could pass: files still exist.
    assert popped, "the refusal warning must have been shown"
    assert list(page._dir.glob(f"livetrans_{stamp}*"))


def test_delete_refused_on_stale_cache_when_writer_is_live(page, monkeypatch):
    """The cached record can lie: the refresh ran before the meeting
    started, so the row shows as a plain history record. The delete must
    still refuse — the authority is the writer queried at click time, not
    the record's is_active/is_ending snapshot."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    # Refresh WITHOUT a writer: the cached record is a closed history row.
    page.set_transcript_writer(None)
    page.refresh()
    assert page._can_delete(page._current_record()) is True

    # The meeting starts only after the refresh: the cache is now stale.
    page.set_transcript_writer(_FakeWriter(active=stamp))
    # No refresh: the cached flags still say "closed".

    popped = []
    monkeypatch.setattr(
        "meeting_records_page.QMessageBox",
        type("MB", (), {
            "warning": staticmethod(lambda *a, **k: popped.append(1) or 16385),
            "StandardButton": type("SB", (), {"Yes": 16384, "No": 65536}),
        }),
    )
    page._delete_session()
    assert popped, "the live-writer refusal must fire despite the stale cache"
    assert list(page._dir.glob(f"livetrans_{stamp}*"))


def test_delete_target_is_the_record_argument_not_current_item(page, monkeypatch):
    """The context menu deletes the row it was opened on. Right-clicking an
    unselected row must not delete the *current* row: _delete_session's
    explicit record argument is the target, identity held through the
    confirmation dialog."""
    # Select the newest row; target the OTHER one explicitly (the context
    # menu passes the right-clicked row's record).
    page._list.setCurrentRow(0)
    current_stamp = page._list.item(0).record["session"]
    target_stamp = page._list.item(1).record["session"]

    monkeypatch.setattr(
        "meeting_records_page.QMessageBox",
        type("MB", (), {
            "warning": staticmethod(lambda *a, **k: 16384),
            "StandardButton": type("SB", (), {"Yes": 16384, "No": 65536}),
        }),
    )
    page._delete_session(page._list.item(1).record)
    assert not list(page._dir.glob(f"livetrans_{target_stamp}*"))
    assert list(page._dir.glob(f"livetrans_{current_stamp}*"))


def test_end_button_only_on_the_live_meetings_own_detail(page):
    """The end button belongs to the live meeting's detail view: selecting
    a history row hides it; selecting the live row shows it. Clicking on a
    row that is no longer the live one (a race) emits nothing."""
    page._list.setCurrentRow(1)  # a history row
    history_stamp = page._list.item(1).record["session"]
    page.set_transcript_writer(_FakeWriter(active=history_stamp))
    page.refresh()

    page._list.setCurrentRow(0)  # the other (history) row
    page._load_session(page._list.item(0).record)
    assert page._end_recording_btn.isVisibleTo(page) is False

    page._list.setCurrentRow(1)  # the live row
    page._load_session(page._list.item(1).record)
    assert page._end_recording_btn.isVisibleTo(page) is True

    # A stale click (the meeting ended between show and click) emits
    # nothing and refreshes the button state instead.
    emitted = []
    page.end_recording_requested.connect(lambda sid: emitted.append(sid))
    page.set_transcript_writer(_FakeWriter(active=None))
    page.refresh()
    page._on_end_recording_clicked()
    assert emitted == []

    # The live click carries the meeting's identity: the app re-validates
    # it at handling time (main.py's active_session() comparison).
    page.set_transcript_writer(_FakeWriter(active=history_stamp))
    page.refresh()
    page._on_end_recording_clicked()
    assert emitted == [history_stamp]


def test_live_role_prefer_the_ending_state_over_writer_active(page):
    """In the window before end_session() starts, the writer still reports
    the session active while the state machine already announced ENDING —
    the app-level stamp must win, or the page would treat the closing
    record as merely active (end button re-enabled, export allowed)."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    # The writer is mid-END-flip: still reports active, not yet ending.
    page.set_transcript_writer(_FakeWriter(active=stamp, ending=None))
    page._ending_session_id = stamp
    page._session_state = "ending"
    assert page._live_session_role(stamp) == "ending"
    # Without the app-level push it is what the writer says.
    page._session_state = "active"
    assert page._live_session_role(stamp) == "active"
    # Another session's ENDING never paints this row.
    page._session_state = "ending"
    page._ending_session_id = "some_other_stamp"
    assert page._live_session_role(stamp) == "none"
    page._session_state = "idle"
    page._ending_session_id = None


# --- worker wiring ----------------------------------------------------------------


def test_worker_success_updates_ui_and_persists(page, monkeypatch):
    stamp = page._list.item(0).record["session"]
    page._list.setCurrentRow(0)

    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            return type("Resp", (), {"choices": [
                type("Ch", (), {"message": type("M", (), {"content": "# 生成纪要"})()})()]})()

    class _FakeClient:
        chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(svc, "make_openai_client", lambda *a, **k: _FakeClient())
    page._on_generate()
    assert page._worker is not None
    # Run synchronously (run() is the whole body of the thread).
    page._worker.run()
    loaded = records.load_summary(page._dir, stamp)
    assert loaded is not None
    assert loaded["content"] == "# 生成纪要"
    assert "api_key" not in json.dumps(loaded["meta"])


def test_worker_failure_shows_localized_error_keeps_old(page, monkeypatch):
    stamp = page._list.item(0).record["session"]
    records.save_summary(page._dir, stamp, "OLD", {})
    page._list.setCurrentRow(0)

    def boom(**kwargs):
        raise TimeoutError("t/o")

    class _FakeCompletions:
        create = staticmethod(boom)

    class _FakeClient:
        chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(svc, "make_openai_client", lambda *a, **k: _FakeClient())
    page._on_generate()
    page._worker.run()
    assert page._status_label.text()
    assert records.load_summary(page._dir, stamp)["content"] == "OLD"


def test_switching_sessions_cancels_running_worker(page, monkeypatch):
    page._list.setCurrentRow(0)

    class _FakeCompletions:
        def create(self, **kwargs):
            return type("Resp", (), {"choices": [
                type("Ch", (), {"message": type("M", (), {"content": "x"})()})()]})()

    class _FakeClient:
        chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(svc, "make_openai_client", lambda *a, **k: _FakeClient())
    page._on_generate()
    worker = page._worker
    # Selecting the other meeting cancels the in-flight generation
    page._list.setCurrentRow(1)
    assert worker._cancel.is_set()
    assert page._worker is None
    assert page._generate_btn.isEnabled()


# --- manual minutes editing (identity and conflict checks) -----------------------


def _loaded_summary_page(page, monkeypatch):
    """A page with one session selected and a committed summary loaded,
    with modal message boxes recorded instead of shown."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    records.save_summary(page._dir, stamp, "# original", {"provider_name": "X"})
    page._list.setCurrentRow(0)
    page._load_session(page._list.item(0).record)
    shown = []
    monkeypatch.setattr(
        "meeting_records_page.QMessageBox",
        type("MB", (), {
            "warning": staticmethod(lambda *a, **k: shown.append(a) or 16385),
            "critical": staticmethod(lambda *a, **k: shown.append(a) or 16385),
            "information": staticmethod(lambda *a, **k: shown.append(a) or 16385),
            "StandardButton": type("SB", (), {"Yes": 16384, "No": 65536}),
        }),
    )
    return stamp, shown


def _content_hash(text):
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_edit_save_lands_on_the_opened_session(page, monkeypatch):
    stamp, _shown = _loaded_summary_page(page, monkeypatch)
    assert page._save_edited_summary(stamp, _content_hash("# original"), "# edited")
    loaded = records.load_summary(page._dir, stamp)
    assert loaded["content"] == "# edited"
    assert loaded["meta"]["edited_by_user"] is True


def test_edit_save_refused_when_selection_changed_under_the_dialog(page, monkeypatch):
    """The editor was opened on meeting A; a background refresh moved the
    selection to meeting B while the dialog was open. The save must not
    land on B's files."""
    stamp_a, shown = _loaded_summary_page(page, monkeypatch)
    stamp_b = page._list.item(1).record["session"]
    # The dialog's event loop delivered a selection change.
    page._list.setCurrentRow(1)

    ok = page._save_edited_summary(stamp_a, _content_hash("# original"), "# edited")
    assert ok is False
    assert shown, "the identity refusal must be shown"
    # Neither meeting's summary was overwritten.
    assert records.load_summary(page._dir, stamp_a)["content"] == "# original"
    assert records.load_summary(page._dir, stamp_b) is None


def test_edit_save_refused_when_generation_overwrote_the_summary(page, monkeypatch):
    """A worker completed under the dialog and replaced the minutes: the
    manual save must refuse (conflict) instead of silently discarding the
    newer generation."""
    stamp, shown = _loaded_summary_page(page, monkeypatch)
    # The generation landed while the dialog was open.
    records.save_summary(page._dir, stamp, "# freshly generated", {})

    ok = page._save_edited_summary(stamp, _content_hash("# original"), "# edited")
    assert ok is False
    assert shown, "the conflict refusal must be shown"
    assert records.load_summary(page._dir, stamp)["content"] == "# freshly generated"


def test_edit_save_refused_when_summary_vanished_under_the_dialog(page, monkeypatch):
    """The summary pair was deleted (or broke) while the dialog was open:
    saving would silently resurrect the edited content as the committed
    minutes. That is a conflict, not a fresh save."""
    stamp, shown = _loaded_summary_page(page, monkeypatch)
    records.delete_summary(page._dir, stamp)

    ok = page._save_edited_summary(stamp, _content_hash("# original"), "# edited")
    assert ok is False
    assert shown, "the vanished-summary conflict must be shown"
    assert records.load_summary(page._dir, stamp) is None  # nothing resurrected


def test_edit_save_refused_while_generation_is_running(page, monkeypatch):
    """Single-writer rule: while a summary worker runs for this session a
    manual save must not race the worker's own save."""
    stamp, shown = _loaded_summary_page(page, monkeypatch)

    class _RunningWorker:
        session = stamp
        generation = 1

        @staticmethod
        def isRunning():
            return True

    page._worker = _RunningWorker()
    page._worker_session = stamp
    try:
        ok = page._save_edited_summary(
            stamp, _content_hash("# original"), "# edited"
        )
        assert ok is False
        assert shown, "the running-generation refusal must be shown"
    finally:
        page._worker = None
        page._worker_session = None
    assert records.load_summary(page._dir, stamp)["content"] == "# original"


def test_edit_button_disabled_while_generation_runs(page, monkeypatch):
    """The edit control must not advertise an action the single-writer rule
    refuses: while a worker runs, edit (and generate) are disabled and
    cancel is visible — and nothing re-enables them on a state push."""
    _loaded_summary_page(page, monkeypatch)

    class _RunningWorker:
        session = page._list.item(0).record["session"]
        generation = 1

        @staticmethod
        def isRunning():
            return True

    page._worker = _RunningWorker()
    page._worker_session = _RunningWorker.session
    try:
        page._update_action_availability()
        assert page._edit_btn.isEnabled() is False
        assert page._generate_btn.isEnabled() is False
        assert page._cancel_btn.isVisibleTo(page)
        # A state push (e.g. app-level refresh) must not re-enable them.
        page.on_session_state_changed("idle")
        assert page._edit_btn.isEnabled() is False
        assert page._generate_btn.isEnabled() is False
    finally:
        page._worker = None
        page._worker_session = None
    page._update_action_availability()
    assert page._edit_btn.isEnabled() is True


# --- ENDING export -----------------------------------------------------------------


def test_export_refused_while_session_is_ending(page, monkeypatch):
    """The seal may still be writing the final entries and the footer, so
    an export mid-close could read as complete while missing the last
    utterance — refuse until the close lands."""
    stamp, shown = _loaded_summary_page(page, monkeypatch)
    # The closing meeting's cached row is live: the push flips it in memory.
    page.set_transcript_writer(_FakeWriter(active=stamp))
    page._record_for_session(stamp)["is_active"] = True
    page.on_session_state_changed("ending", stamp)
    shown.clear()

    page._export("pdf_summary")
    assert shown, "the ending refusal must be shown"
    # An active record still exports (flagged as a snapshot elsewhere).
    page.set_transcript_writer(_FakeWriter(active=stamp, ending=None))
    page._ending_session_id = None
    page._session_state = "active"
    page._record_for_session(stamp)["is_ending"] = False
    page._record_for_session(stamp)["is_active"] = True
    shown.clear()
    # QFileDialog.getSaveFileName is monkeypatched to cancel: the point is
    # that the *guard* let the call through to the file dialog.
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )
    page._export("pdf_summary")
    assert shown == []


# --- rename routing ------------------------------------------------------------------


def test_rename_of_live_session_routes_through_writer_with_expected(page, monkeypatch):
    """A live session's rename goes through the writer's locked path and
    carries the session identity it was issued on, so an end+begin racing
    the dialog cannot retitle the new meeting."""
    from PyQt6.QtWidgets import QInputDialog

    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    writer = _FakeWriter(active=stamp)
    page.set_transcript_writer(writer)
    page.refresh()

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **k: ("新标题", True)),
    )
    page._rename_session(page._list.item(0))
    assert writer.renamed_with == ("新标题", stamp)


def test_rename_after_a_refresh_under_the_dialog_updates_the_live_row(page, monkeypatch):
    """The rename dialog pumps the event loop; a refresh under it clears
    the list and destroys the pre-dialog row's C++ object. Touching that
    item afterwards raises RuntimeError — the handler must re-locate the
    row by session identity instead."""
    from PyQt6.QtWidgets import QInputDialog

    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    stale_item = page._list.item(0)

    def dialog_that_refreshes_under_itself(*a, **k):
        # The nested event loop delivers a refresh while the dialog is up.
        page.refresh()
        return ("新标题", True)

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(dialog_that_refreshes_under_itself),
    )
    page._rename_session(stale_item)  # must not raise on the dead item
    fresh_item = page._list_item_for_session(stamp)
    assert fresh_item is not None
    assert fresh_item is not stale_item  # the rebuild replaced the row
    assert "新标题" in fresh_item.text()
    assert page._record_for_session(stamp)["title"] == "新标题"


def test_rename_refused_when_the_session_was_deleted_under_the_dialog(page, monkeypatch):
    """The meeting's files were deleted while the rename dialog was open:
    renaming would recreate a sidecar for a session that no longer exists
    (a phantom row in the list). Refuse instead."""
    from PyQt6.QtWidgets import QInputDialog
    from transcript_writer import _session_file_candidates

    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]

    def dialog_that_deletes_under_itself(*a, **k):
        # Delete through the exact file enumeration (the production delete
        # path), never a prefix glob — see delete_session.
        for path in _session_file_candidates(page._dir, stamp):
            if path.exists():
                path.unlink()
        page.refresh()
        return ("标题", True)

    shown = []
    monkeypatch.setattr(
        "meeting_records_page.QMessageBox",
        type("MB", (), {
            "warning": staticmethod(lambda *a, **k: shown.append(a) or 16385),
            "StandardButton": type("SB", (), {"Yes": 16384, "No": 65536}),
        }),
    )
    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(dialog_that_deletes_under_itself),
    )
    page._rename_session(page._list.item(0))
    assert shown, "the session-gone refusal must be shown"
    # No sidecar was resurrected for the deleted session.
    assert not list(_session_file_candidates(page._dir, stamp))


def test_delete_of_a_parent_stamp_spares_its_same_second_suffixed_sibling(page, monkeypatch):
    """[B1 regression] A bare stamp is the prefix of its same-second
    suffixed siblings (livetrans_X vs livetrans_X_01): the writer-ownership
    guard correctly allows deleting the closed parent record, and the
    deletion itself must then remove *only* the parent's exact files — the
    old ``livetrans_{stamp}_*`` prefix glob unlinked the sibling's whole
    file set too, destroying a second meeting (here: one still held by the
    writer) the user never asked to delete."""
    page._list.setCurrentRow(0)
    parent = page._list.item(0).record["session"]
    sibling = f"{parent}_01"
    # Real on-disk files for the same-second sibling — the full writer set,
    # the AI-summary pair and a staged tmp sibling — with distinct content
    # so survival is provable, not assumed.
    sibling_files = {
        f"livetrans_{sibling}_all.txt": "[09:00:00] sibling line\n",
        f"livetrans_{sibling}_original.txt": "[09:00:00] sibling line\n",
        f"livetrans_{sibling}_translation.txt": "[09:00:00] SIBLING TL\n",
        f"livetrans_{sibling}_meeting.md": "# Meeting record\n\nsibling\n",
        f"livetrans_{sibling}_meta.json": '{"session": "%s"}' % sibling,
        f"livetrans_{sibling}_summary.md": "# sibling minutes",
        f"livetrans_{sibling}_summary_meta.json": '{"provider_name": "X"}',
        f"livetrans_{sibling}_summary.md.tmp123.0123456789abcdef":
            "half-written",
    }
    for name, content in sibling_files.items():
        (page._dir / name).write_text(content, encoding="utf-8")
    # The writer holds the suffixed sibling's files (same-second session).
    page.set_transcript_writer(_FakeWriter(held=sibling))

    monkeypatch.setattr(
        "meeting_records_page.QMessageBox",
        type("MB", (), {
            "warning": staticmethod(lambda *a, **k: 16384),
            "StandardButton": type("SB", (), {"Yes": 16384, "No": 65536}),
        }),
    )
    # The parent's own row must not trip the writer-holds guard: the held
    # stamp is the sibling, and the bare parent stamp is only its PREFIX.
    assert page._writer_holds_session_files(parent) is False
    assert page._writer_holds_session_files(sibling) is True
    page._delete_session(page._record_for_session(parent))

    # The parent's own files are gone (the substring-guard version refused
    # the delete outright; the glob version deleted far too much).
    for kind in ("all", "original", "translation", "meeting", "meta"):
        suffix = {"meeting": ".md", "meta": ".json"}.get(kind, ".txt")
        assert not (page._dir / f"livetrans_{parent}_{kind}{suffix}").exists()
    # Every sibling file survives *with its content intact* — the deletion
    # of one history record must never destroy another meeting.
    for name, content in sibling_files.items():
        survived = page._dir / name
        assert survived.exists(), name
        assert survived.read_text(encoding="utf-8") == content, name


def test_export_button_disabled_in_ui_while_record_is_ending(page):
    """ENDING disables the export button in the UI (the handler's
    authoritative check stays as the guard); a closed record keeps it
    enabled."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    # The live row that the ENDING push flips in memory.
    page.set_transcript_writer(_FakeWriter(active=stamp))
    page._record_for_session(stamp)["is_active"] = True
    page.on_session_state_changed("ending", stamp)
    assert page._export_btn.isEnabled() is False

    page.set_transcript_writer(_FakeWriter(active=None, ending=None))
    page._record_for_session(stamp)["is_ending"] = False
    page._ending_session_id = None
    page.on_session_state_changed("idle")
    assert page._export_btn.isEnabled() is True


def test_refresh_with_preserved_selection_updates_detail_in_place(page):
    """A refresh that preserves the selection must still re-sync the
    detail pane's data (title, meta) — without switching the tab or
    resetting the record view, and without touching a running worker."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    # Move to the full-record tab and render entries.
    page._tabs.setCurrentIndex(1)
    rendered_before = page._entries_rendered
    layout_before = page._record_area._layout.count()

    records.set_session_title(page._dir, stamp, "刷新后的标题")
    page.refresh()

    assert page._title_label.text() == "刷新后的标题"
    # No tab switch, no record-view reset.
    assert page._tabs.currentIndex() == 1
    assert page._entries_rendered == rendered_before
    assert page._record_area._layout.count() == layout_before


def test_search_and_filter_rebuild_list_without_detail_reload(monkeypatch, page):
    """A search/filter change rebuilds the list only: the unchanged
    selection's minutes are not re-read from disk and not re-rendered (one
    keystroke, zero records-layer reads). A query that filters the
    selection out is a real selection change and loads the fallback row's
    detail; a genuine refresh still re-syncs the detail in place."""
    page._list.setCurrentRow(0)
    selected = page._current_session()
    assert selected == "20260102_090000"
    summary_reads = []
    real_load_summary = records.load_summary
    monkeypatch.setattr(
        records, "load_summary",
        lambda base_dir, stamp: (
            summary_reads.append(stamp), real_load_summary(base_dir, stamp))[1],
    )
    detail_loads = []
    monkeypatch.setattr(
        page, "_load_minutes",
        lambda record: detail_loads.append(record.get("session")),
    )

    # A query that keeps the selected row visible: list-only rebuild.
    page._search.setText("SenseVoice")
    assert page._list.count() == 2
    assert page._current_session() == selected  # selection survived
    assert summary_reads == []
    assert detail_loads == []

    # A filter switch with the row still visible: same.
    page._filter.setCurrentIndex(2)  # unsummarized: both rows stay
    assert page._list.count() == 2
    assert summary_reads == []
    assert detail_loads == []

    # A query that filters the selection out is a *real* selection change:
    # the fallback row 0 loads its detail through the normal path.
    page._search.setText("20260101")
    assert page._list.count() == 1
    assert page._current_session() == "20260101_090000"
    assert detail_loads == ["20260101_090000"]

    # A genuine refresh (fresh data) re-syncs the preserved selection.
    page.refresh()
    assert detail_loads == ["20260101_090000", "20260101_090000"]


def test_refresh_selecting_a_target_loads_only_the_target(monkeypatch, page):
    """refresh(select_session=...) lands on the target meeting with exactly
    one detail load: the refill must not first re-sync the previously
    selected meeting's detail (or load row 0 after a vanished selection)
    only to replace it with the target a moment later."""
    page._list.setCurrentRow(1)  # the older meeting, not the target
    older = page._current_session()
    target = page._list.item(0).record["session"]
    assert older != target
    detail_loads = []
    monkeypatch.setattr(
        page, "_load_minutes",
        lambda record: detail_loads.append(record.get("session")),
    )
    page.refresh(select_session=target)
    assert page._current_session() == target
    # Exactly one load, and it is the target — the older meeting's detail
    # was never loaded on the way.
    assert detail_loads == [target]


# --- refresh economics: writer injection and lightweight state pushes --------------


def test_same_writer_reinjection_does_not_defer_refresh(page, monkeypatch):
    """[per-1] Re-injecting the *same* writer (what every records-tab visit
    forwards through _refresh_transcripts) must not schedule a deferred
    refresh on top of the caller's synchronous one — tab entry would
    otherwise refresh twice from identical data."""
    scheduled = []
    monkeypatch.setattr(page, "_defer_refresh", lambda: scheduled.append(1))
    deferred_started = []
    real_start = page._deferred_refresh_timer.start
    monkeypatch.setattr(
        page._deferred_refresh_timer, "start",
        lambda *a: deferred_started.append(1) or real_start(),
    )
    page.set_transcript_writer(page._writer)  # same object
    page.set_transcript_writer(page._writer)
    assert scheduled == []
    assert deferred_started == []


def test_writer_change_refreshes_once(page, monkeypatch):
    """[per-2] A genuinely different writer updates the field and defers
    exactly one refresh; the synchronous refresh() that follows supersedes
    the pending deferred one (the timer is stopped), so one effective full
    refresh lands — not two."""
    calls = []
    real_refresh = page.refresh

    def counting_refresh(*a, **k):
        calls.append(a)
        return real_refresh(*a, **k)

    monkeypatch.setattr(page, "refresh", counting_refresh)
    page._deferred_refresh_timer.start()  # pending deferred refresh
    assert page._deferred_refresh_timer.isActive()
    writer = _FakeWriter(active=page._list.item(0).record["session"])
    page.set_transcript_writer(writer)  # change: defers
    assert page._writer is writer
    page.refresh()  # the caller's synchronous entry point
    # The direct refresh superseded the pending timer: one effective call.
    assert page._deferred_refresh_timer.isActive() is False
    assert calls == [()]


def test_tab_entry_calls_refresh_once(page, monkeypatch):
    """[per-3] The tab-entry path (set_transcript_writer with an unchanged
    writer, then the synchronous refresh — the two statements
    _refresh_transcripts runs) results in exactly one refresh call: the
    injection re-schedules nothing and the direct call supersedes any
    pending deferred timer."""
    calls = []
    real_refresh = page.refresh

    def counting_refresh(*a, **k):
        calls.append(a)
        return real_refresh(*a, **k)

    monkeypatch.setattr(page, "refresh", counting_refresh)
    # A pending deferred refresh from an earlier injection.
    page._deferred_refresh_timer.start()
    assert page._deferred_refresh_timer.isActive()
    page.set_transcript_writer(page._writer)  # unchanged: schedules nothing
    page.refresh()  # supersedes the pending timer
    assert page._deferred_refresh_timer.isActive() is False
    assert calls == [()]


def _lightweight_recorder(page, monkeypatch):
    """Patch records.list_sessions, records.load_summary and the page's
    full-refresh path with recorders, so a test can tell a lightweight
    in-memory update (no list_sessions, no summary/meta reads, no refresh
    call) from a full refresh."""
    full_refreshes = []
    monkeypatch.setattr(page, "refresh", lambda *a, **k: full_refreshes.append(a))
    listing = []
    real_list = records.list_sessions
    monkeypatch.setattr(
        records, "list_sessions",
        lambda *a, **k: listing.append(a) or real_list(*a, **k),
    )
    summary_reads = []
    real_load = records.load_summary
    monkeypatch.setattr(
        records, "load_summary",
        lambda *a, **k: summary_reads.append(a) or real_load(*a, **k),
    )
    return full_refreshes, listing, summary_reads


def test_pause_resume_pushes_stay_in_memory(page, monkeypatch):
    """[per-4] PAUSED and back to ACTIVE for the cached live row: no
    records.list_sessions call, no full refresh, the badge source flags
    (is_active) are unchanged and disk-derived flags are untouched.

    The summary-state keys are compared with .get(): a record without a
    committed summary pair legitimately *lacks* them (list_sessions only
    writes summary_stale/summary_edited when a summary exists), so a
    missing key before AND after the push is exactly the "untouched"
    contract — indexing would turn the absence into a KeyError."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    page.set_transcript_writer(_FakeWriter(active=stamp))
    page.refresh()
    before = dict(page._record_for_session(stamp))

    full_refreshes, listing, summary_reads = _lightweight_recorder(page, monkeypatch)
    page.on_session_state_changed("paused")
    page.on_session_state_changed("active")
    assert listing == []
    assert full_refreshes == []
    assert summary_reads == []

    after = page._record_for_session(stamp)
    for key in ("summary_stale", "summary_edited", "has_summary",
                "ended_cleanly", "interrupted", "is_active", "is_ending"):
        assert after.get(key) == before.get(key), key


def test_ending_push_stays_in_memory_and_marks_only_target(page, monkeypatch):
    """[per-5] ACTIVE→ENDING: no disk scan, no full refresh; only the target
    record flips to is_ending (is_active False, interrupted stays False);
    every other row keeps its flags."""
    page._list.setCurrentRow(0)
    target = page._list.item(0).record["session"]
    other = page._list.item(1).record["session"]
    page.set_transcript_writer(_FakeWriter(active=target))
    page.refresh()

    full_refreshes, listing, summary_reads = _lightweight_recorder(page, monkeypatch)
    page.on_session_state_changed("ending", target)
    assert listing == []
    assert full_refreshes == []
    assert summary_reads == []

    by_stamp = {r["session"]: r for r in page._sessions}
    assert by_stamp[target]["is_ending"] is True
    assert by_stamp[target]["is_active"] is False
    assert by_stamp[target]["interrupted"] is False
    assert by_stamp[other]["is_ending"] is False
    assert by_stamp[other]["is_active"] is False


def test_new_active_session_falls_back_to_full_refresh(page, monkeypatch):
    """[per-6] A first ACTIVE for a session the cache has never seen (the
    new meeting is not in the list yet): one full refresh re-reads it from
    disk rather than guessing an identity."""
    full_refreshes, _listing, _reads = _lightweight_recorder(page, monkeypatch)
    page.on_session_state_changed("active", "20261231_235959")
    assert full_refreshes == [()]


def test_idle_after_end_runs_full_refresh_and_selects_target(page, monkeypatch):
    """[per-7] IDLE carrying the just-ended session: one full refresh with
    select_session (the seal wrote new counts/status on disk), landing the
    selection on the ended meeting. An IDLE with no identity (the app-quit
    path) is also a full refresh: the records layer must re-read whatever
    the close wrote."""
    page._list.setCurrentRow(1)  # a different row is selected
    ended = page._list.item(0).record["session"]
    full_refreshes = []
    monkeypatch.setattr(
        page, "refresh",
        lambda select_session=None: full_refreshes.append(select_session),
    )
    page.on_session_state_changed("idle", ended)
    assert full_refreshes == [ended]
    # The status hint is shown for the just-ended meeting.
    assert page._status_label.text()
    # IDLE without an identity (app stop): one plain full refresh.
    page.on_session_state_changed("idle")
    assert full_refreshes == [ended, None]


def test_lightweight_push_does_not_cancel_running_worker(page, monkeypatch):
    """[per-8] A pause (or resume/ending) push during a running generation
    must not cancel the worker: _refill_list preserves the selection by
    identity, and the push path never reaches _load_session's cancel
    branch."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    page.set_transcript_writer(_FakeWriter(active=stamp))
    page.refresh()

    class _RunningWorker:
        session = stamp
        generation = 1
        cancelled = False

        def cancel(self):
            self.cancelled = True

        @staticmethod
        def isRunning():
            return True

    worker = _RunningWorker()
    page._worker = worker
    page._worker_session = stamp
    try:
        page.on_session_state_changed("paused")
        page.on_session_state_changed("active")
        page.on_session_state_changed("ending", stamp)
        assert worker.cancelled is False
        assert page._worker is worker  # ownership untouched
    finally:
        page._worker = None
        page._worker_session = None


def test_lightweight_push_preserves_stale_and_edited_badges(page, monkeypatch):
    """[per-9] The stale/edited (and ready) flags are disk-derived state:
    a lightweight pause/ending push must carry them through the list
    rebuild unchanged — rows and badges re-render, the data does not.

    The live row's flags are set directly on the cached record rather
    than through a writer + refresh: a session the writer reports active
    has its stale badge suppressed by design (list_sessions forces
    summary_stale=False while a meeting is still growing — a summary of a
    growing meeting is interim, not stale), so asserting True through
    that path would contradict the product semantics. The lightweight
    path never re-derives any of these flags; pinning them on the cache
    and asserting they survive the push is the real contract."""
    page._list.setCurrentRow(0)
    stamp = page._list.item(0).record["session"]
    # A committed (stale) summary and an edited flag on the record, plus
    # the live row the pushes will target.
    records.save_summary(page._dir, stamp, "# old", {"source_hash": "bogus"})
    page.refresh()
    live = page._record_for_session(stamp)
    assert live["has_summary"] is True
    # Stamp the cache the way a live session with a stale summary looks:
    # the summary exists, its hash no longer matches, the meeting is live.
    live["is_active"] = True
    live["summary_stale"] = True
    live["summary_edited"] = False

    full_refreshes, listing, summary_reads = _lightweight_recorder(page, monkeypatch)
    page.on_session_state_changed("paused")
    page.on_session_state_changed("active")
    page.on_session_state_changed("ending", stamp)
    assert listing == [] and full_refreshes == []
    assert summary_reads == []

    after = page._record_for_session(stamp)
    assert after["summary_stale"] is True
    assert after["summary_edited"] == live["summary_edited"]
    assert after["has_summary"] == live["has_summary"]


# --- temporary menus schedule their own release -------------------------------------


class _FakeAction:
    def setEnabled(self, _enabled):
        pass


class _RecordingMenu:
    """Stands in for QMenu: records every instance created by the page and
    whether deleteLater() was scheduled on it. exec() never opens a real
    popup (returns None = the cancel path), so the test drives the pure
    lifecycle contract: after the handler returns, the temporary menu it
    built must have a release scheduled — no reliance on Qt child counts
    or event-loop timing."""

    instances = []

    def __init__(self, parent=None):
        self.scheduled_delete = False
        _RecordingMenu.instances.append(self)

    def addAction(self, *_args):
        return _FakeAction()

    def addSeparator(self):
        return None

    def exec(self, *_args, **_kwargs):
        return None

    def deleteLater(self):
        self.scheduled_delete = True


def test_context_menu_schedules_release_after_invocation(page, monkeypatch):
    """Each right-click builds a QMenu(self) whose C++ ownership goes to the
    page; exec() hides but does not free it, so without deleteLater every
    invocation would leave the menu alive until page teardown. The handler
    must schedule the release on the cancel path too."""
    monkeypatch.setattr("meeting_records_page.QMenu", _RecordingMenu)
    _RecordingMenu.instances.clear()
    rect = page._list.visualItemRect(page._list.item(0))
    page._list_context_menu(rect.center())
    menus = _RecordingMenu.instances
    assert len(menus) == 1
    assert menus[0].scheduled_delete is True


def test_export_menu_schedules_release_after_invocation(page, monkeypatch):
    """The export dropdown is the same per-invocation menu (its actions
    even hold Python lambdas): after the click handler returns, the menu
    must have a release scheduled rather than surviving as a page child."""
    page._list.setCurrentRow(0)
    monkeypatch.setattr("meeting_records_page.QMenu", _RecordingMenu)
    _RecordingMenu.instances.clear()
    page._on_export_menu()
    menus = _RecordingMenu.instances
    assert len(menus) == 1
    assert menus[0].scheduled_delete is True


# --- app-level end entry (identity through the confirmation dialog) ---------------


def test_end_recording_session_re_verifies_the_named_meeting(monkeypatch):
    """The authoritative end entry re-checks the expected meeting right
    before the ENDING flip: the caller's confirmation dialog pumps a
    nested event loop, so an end plus a new begin can land under it. A
    request naming the old meeting must refuse rather than end the new
    one; a matching (or absent) identity proceeds."""
    main = pytest.importorskip(
        "main", reason="main.py needs torch + PyQt6, which the offline job skips"
    )

    app = object.__new__(main.LiveTranslateApp)
    app._session_state = main.SessionState.ACTIVE
    app._ending_thread = None
    app._session_generation = 7
    app._session_ui_callback = None

    class _Writer:
        def __init__(self, stamp):
            self._stamp = stamp

        def active_session(self):
            return self._stamp

    app._transcript = _Writer("20260905_100000")

    started = []

    class _FakeThread:
        def __init__(self, target=None, name=None, daemon=None):
            self._target = target

        def start(self):
            started.append(self._target)

    monkeypatch.setattr(main.threading, "Thread", _FakeThread)

    # A different meeting is open than the one the request named: refused
    # before the ENDING flip, nothing started.
    assert app.end_recording_session(expected_session="20260905_090000") is False
    assert started == []
    assert app._session_state == main.SessionState.ACTIVE

    # The matching identity flips to ENDING and starts the close.
    assert app.end_recording_session(expected_session="20260905_100000") is True
    assert len(started) == 1
    assert app._session_state == main.SessionState.ENDING
