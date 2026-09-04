"""The Meeting Records page: a master-detail records center.

Replaces the old transcripts tab (one-line list + raw Markdown dump) with:

* a session list with titles, time, duration, entry counts and summary state
  (search, filter, empty state, rename);
* a detail pane with three views — AI minutes (Markdown rendered), the full
  record (structured entries, translation primary / original secondary,
  incremental rendering) and session info;
* an action row: generate / regenerate / edit summary, export (PDF / MD / TXT),
  open folder, delete;
* a background ``SummaryWorker`` for AI minutes with progress, cancel, and
  every failure state mapped to an i18n message.

Orchestration only: the visual leaf components (list rows, record renderer,
minutes HTML) live in ``meeting_records_widgets.py``; data parsing and
persistence in ``meeting_records.py``; the AI pipeline and worker in
``ai_summary_service.py``; PDF drawing in ``pdf_exporter.py``.

Responsive: a QSplitter whose sizes persist in ``user_settings.json``; below a
width threshold the list and detail stack, with a back button in the detail
pane. The threshold is measured against this page's own contentsRect so the
panel's navigation column is not mistaken for page width. No fixed-width
layout — all columns are stretch/minimum based, so 125%/150% Windows scaling
and Retina both work.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import ai_summary_service as summary_service
import meeting_records as records
import pdf_exporter
import summary_task_registry
from ai_summary_service import (
    SummaryWorker,
    ensure_model_ids,
    provider_missing_key,
    provider_unsuitable,
    resolve_provider,
)
from i18n import t, get_lang
from meeting_records_widgets import (
    RecordScrollArea,
    SessionItem,
    minutes_html,
)
from transcript_writer import delete_session

log = logging.getLogger("LiveTranslate.RecordsUI")

# Below this width the master and detail panes stop sharing the row and the
# page flips to one-at-a-time navigation. Measured against the page's own
# contentsRect() — this page lives inside the ControlPanel next to a 170px
# navigation list, so a 900px panel leaves roughly 690px here; a threshold on
# the outer window width would flip to stacked mode in that perfectly usable
# split view.
_STACKED_WIDTH_THRESHOLD = 620
# Entries rendered per batch into the record view; the rest load when the
# scroll bar nears the bottom, so a 3-hour meeting never builds 800 widgets
# in one paint.
_RENDER_BATCH = 80


def _open_folder(path):
    path = str(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform.startswith("win"):
        os_startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])


def os_startfile(path):
    import os

    os.startfile(path)  # noqa: S102 - Windows-only API by design


class MeetingRecordsPage(QWidget):
    """The full meeting-records center, mounted as one ControlPanel page."""

    # The page persists its own settings through the panel's settings dict;
    # this signal asks the panel to save it (the panel owns the file).
    settings_updated = pyqtSignal(dict)
    # Request-only: the page never ends a session itself, it asks the app
    # (through the panel) so the one end-meeting implementation stays there.
    end_recording_requested = pyqtSignal()

    def __init__(self, transcripts_dir: Path, settings: dict, transcript_writer=None, parent=None):
        super().__init__(parent)
        self._dir = Path(transcripts_dir)
        self._settings = settings  # the panel's live settings dict
        # The live TranscriptWriter, when the app is running one. Its
        # active_session() decides which row is marked "still recording".
        # Injected through set_transcript_writer(); the constructor arg is
        # for the pre-app construction phase only (the panel is built before
        # the app object exists).
        self._writer = transcript_writer
        # App-level session state pushed from LiveTranslateApp ("idle" |
        # "active" | "paused" | "ending"); the page never derives it.
        self._session_state = "idle"
        self._sessions = []
        self._entries = []
        self._entries_rendered = 0
        self._worker = None
        self._worker_session = None
        self._worker_generation = 0
        self._summary_doc = None
        self._editing_summary = False
        self._cloud_notice_shown_for = set()
        self._suppress_selection = False
        # Application-level registry for summary workers: keeps strong
        # references past this page's destruction and gives the app one
        # cancel_all() at exit. Defaults to a page-local instance so the page
        # works standalone (tests, panel-only runs); the panel rebinds it to
        # the app's registry via set_summary_registry().
        self._task_registry = summary_task_registry.SummaryTaskRegistry()

        self._build_ui()
        self._apply_stacked_or_split()
        # Page-owned single-shot timer: replace every bare
        # QTimer.singleShot(0, ...) whose receiver was ambiguous. Parented
        # here, so a destroyed page drops pending callbacks with it.
        self._layout_settle_timer = QTimer(self)
        self._layout_settle_timer.setSingleShot(True)
        self._layout_settle_timer.setInterval(0)
        self._layout_settle_timer.timeout.connect(self._apply_stacked_or_split)
        # Deferred refresh posted from set_transcript_writer / session-state
        # callbacks: same ownership rules, one shot.
        self._deferred_refresh_timer = QTimer(self)
        self._deferred_refresh_timer.setSingleShot(True)
        self._deferred_refresh_timer.setInterval(0)
        self._deferred_refresh_timer.timeout.connect(self.refresh)

    # --- dependency injection -------------------------------------------

    def set_transcript_writer(self, writer):
        """Inject the live TranscriptWriter (called after app construction).

        Public on purpose: main.py used to reach into
        ``panel._transcript_writer`` and the page's private ``_writer``
        directly. Going through here means the page refreshes its active-row
        marking immediately when the writer arrives, instead of waiting for
        the next tab visit.
        """
        self._writer = writer
        self._defer_refresh()

    def set_summary_registry(self, registry):
        """Adopt the app's worker registry (strong refs + app-exit cancel)."""
        self._task_registry = registry

    def on_session_state_changed(self, state: str, session_id=None):
        """App-level session state push (SessionState constants).

        ACTIVE/PAUSED mark the still-growing row and gate AI minutes;
        ENDING shows the saving state; IDLE after an end refreshes the list
        and selects the just-finished meeting so the user can summarize it,
        with a note that listening stopped (model still loaded).
        """
        self._session_state = state
        recording = state in ("active", "paused")
        ending = state == "ending"
        self._end_recording_btn.setVisible(recording)
        self._end_recording_btn.setEnabled(not ending)
        if state == "idle" and session_id:
            # A meeting just completed: refresh onto it, and tell the user
            # the app stopped listening (the model stays loaded) so the
            # paused pipeline is not mistaken for a bug.
            self.refresh(select_session=session_id)
            if hasattr(self, "_status_label"):
                self._status_label.setText(t("session_end_paused_hint"))
        else:
            self.refresh()

    def _defer_refresh(self):
        """Refresh via the page-owned one-shot timer (see __init__)."""
        if not self._deferred_refresh_timer.isActive():
            self._deferred_refresh_timer.start()

    # --- UI construction ---------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setChildrenCollapsible(False)

        # -- left: session list -------------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(6)

        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText(t("records_search_placeholder"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refill_list)
        search_row.addWidget(self._search, 1)
        self._refresh_btn = QPushButton(t("btn_refresh"))
        self._refresh_btn.clicked.connect(self.refresh)
        search_row.addWidget(self._refresh_btn)
        left_layout.addLayout(search_row)

        filter_row = QHBoxLayout()
        self._filter = QComboBox()
        for key in ("records_filter_all", "records_filter_summarized",
                    "records_filter_unsummarized"):
            self._filter.addItem(t(key), key)
        self._filter.currentIndexChanged.connect(self._refill_list)
        filter_row.addWidget(self._filter, 1)
        left_layout.addLayout(filter_row)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setWordWrap(True)
        self._list.setMinimumWidth(240)
        self._list.setUniformItemSizes(False)
        self._list.currentItemChanged.connect(self._on_session_selected)
        self._list.itemDoubleClicked.connect(self._rename_session)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._list_context_menu)
        left_layout.addWidget(self._list, 1)

        self._list_empty = QLabel(t("records_empty"))
        self._list_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._list_empty.setProperty("role", "muted")
        left_layout.addWidget(self._list_empty)
        self._list_empty.hide()

        # -- right: detail -------------------------------------------------
        self._detail = QWidget()
        detail_layout = QVBoxLayout(self._detail)
        detail_layout.setContentsMargins(6, 0, 0, 0)
        detail_layout.setSpacing(6)

        self._back_btn = QPushButton(t("records_back_to_list"))
        self._back_btn.clicked.connect(self._show_list)
        self._back_btn.setProperty("role", "tertiary")
        self._back_btn.hide()
        detail_layout.addWidget(self._back_btn)

        self._detail_stack = QStackedWidget()
        self._detail_stack.addWidget(self._build_detail_content())
        self._detail_empty = self._build_empty_detail()
        self._detail_stack.addWidget(self._detail_empty)
        detail_layout.addWidget(self._detail_stack, 1)

        self._splitter.addWidget(left)
        self._splitter.addWidget(self._detail)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setSizes([300, 700])
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        root.addWidget(self._splitter, 1)

        self._save_splitter_timer = QTimer(self)
        self._save_splitter_timer.setSingleShot(True)
        self._save_splitter_timer.setInterval(500)
        self._save_splitter_timer.timeout.connect(self._save_splitter)

        self._apply_splitter_from_settings()

    def _build_detail_content(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header: title + subtitle meta line
        header = QVBoxLayout()
        self._title_label = QLabel("")
        self._title_label.setProperty("role", "detailTitle")
        self._title_label.setWordWrap(True)
        header.addWidget(self._title_label)
        self._meta_label = QLabel("")
        self._meta_label.setProperty("role", "muted")
        self._meta_label.setWordWrap(True)
        header.addWidget(self._meta_label)
        layout.addLayout(header)

        # View switcher + action row
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._tabs = QComboBox()
        for key in ("records_tab_summary", "records_tab_full", "records_tab_info"):
            self._tabs.addItem(t(key), key)
        self._tabs.setMinimumWidth(120)
        self._tabs.currentIndexChanged.connect(self._on_tab_changed)
        actions.addWidget(self._tabs)
        actions.addStretch()

        self._generate_btn = QPushButton(t("records_generate_summary"))
        self._generate_btn.setObjectName("primaryButton")
        self._generate_btn.clicked.connect(self._on_generate)
        actions.addWidget(self._generate_btn)

        # End the current recording: request-only (see end_recording_requested).
        # Visible only while a session is recording; the app-side state
        # machine drives visibility through on_session_state_changed().
        self._end_recording_btn = QPushButton(t("records_end_recording"))
        self._end_recording_btn.clicked.connect(
            self.end_recording_requested.emit
        )
        actions.addWidget(self._end_recording_btn)
        self._end_recording_btn.hide()

        self._cancel_btn = QPushButton(t("records_cancel_summary"))
        self._cancel_btn.clicked.connect(self._on_cancel_generate)
        self._cancel_btn.hide()
        actions.addWidget(self._cancel_btn)

        self._edit_btn = QPushButton(t("records_edit_summary"))
        self._edit_btn.clicked.connect(self._on_edit_summary)
        actions.addWidget(self._edit_btn)

        self._export_btn = QPushButton(t("records_export"))
        self._export_btn.clicked.connect(self._on_export_menu)
        actions.addWidget(self._export_btn)

        self._folder_btn = QPushButton(t("btn_open_transcripts"))
        self._folder_btn.clicked.connect(self._open_folder_current)
        actions.addWidget(self._folder_btn)

        self._delete_btn = QPushButton(t("btn_delete_record"))
        self._delete_btn.setObjectName("dangerButton")
        self._delete_btn.clicked.connect(self._delete_session)
        actions.addWidget(self._delete_btn)
        layout.addLayout(actions)

        self._status_label = QLabel("")
        self._status_label.setProperty("role", "statusLabel")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Body: one stacked page per tab
        self._body = QStackedWidget()

        # Tab 0: AI minutes
        minutes = QVBoxLayout()
        self._minutes_browser = QTextBrowser()
        self._minutes_browser.setObjectName("recordsMinutes")
        self._minutes_browser.setOpenExternalLinks(False)
        minutes.addWidget(self._minutes_browser, 1)

        self._minutes_empty = self._build_minutes_empty()
        self._minutes_browser_stack = QStackedWidget()
        self._minutes_browser_stack.addWidget(self._minutes_empty)
        self._minutes_browser_stack.addWidget(self._minutes_browser)
        self._minutes_browser_stack.setCurrentIndex(0)
        body_summary = QWidget()
        v = QVBoxLayout(body_summary)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._minutes_browser_stack)
        self._body.addWidget(body_summary)

        # Tab 1: full record
        record_page = QWidget()
        rv = QVBoxLayout(record_page)
        rv.setContentsMargins(0, 0, 0, 0)
        record_tools = QHBoxLayout()
        self._show_original_btn = QPushButton(t("records_show_original"))
        self._show_original_btn.setCheckable(True)
        self._show_original_btn.toggled.connect(self._on_toggle_original)
        record_tools.addWidget(self._show_original_btn)
        record_tools.addStretch()
        self._render_progress = QLabel("")
        self._render_progress.setProperty("role", "muted")
        record_tools.addWidget(self._render_progress)
        rv.addLayout(record_tools)
        self._record_area = RecordScrollArea()
        self._record_area.load_more_requested.connect(self._render_more_entries)
        rv.addWidget(self._record_area, 1)
        self._body.addWidget(record_page)

        # Tab 2: session info
        self._info_browser = QTextBrowser()
        self._info_browser.setObjectName("recordsInfo")
        self._body.addWidget(self._info_browser)

        layout.addWidget(self._body, 1)
        return page

    def _build_empty_detail(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(t("records_no_selection"))
        label.setProperty("role", "muted")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return widget

    def _build_minutes_empty(self) -> QWidget:
        """The guiding empty state when a session has no AI minutes yet."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("✦")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 26px; color: #e7b96f; background: transparent;")
        layout.addWidget(icon)
        heading = QLabel(t("records_summary_empty_title"))
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("font-size: 15px; font-weight: 600; background: transparent;")
        layout.addWidget(heading)
        hint = QLabel(t("records_summary_empty_hint"))
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #ada79d; background: transparent;")
        layout.addWidget(hint)

        controls = QHBoxLayout()
        controls.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls.setSpacing(6)
        self._template_combo = QComboBox()
        self._template_combo.addItem(t("records_template_meeting"), "meeting")
        self._template_combo.addItem(t("records_template_classroom"), "classroom")
        controls.addWidget(self._template_combo)
        self._minutes_lang_combo = QComboBox()
        controls.addWidget(self._minutes_lang_combo)
        self._provider_combo = QComboBox()
        controls.addWidget(self._provider_combo)
        # Connected once here: _populate_* re-run on every session load and
        # would otherwise stack duplicate connections, firing the change
        # handler N times for one user action.
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self._minutes_lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        layout.addLayout(controls)
        self._privacy_label = QLabel(t("records_privacy_note"))
        self._privacy_label.setWordWrap(True)
        self._privacy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._privacy_label.setStyleSheet("color: #ada79d; font-size: 11px; background: transparent;")
        layout.addWidget(self._privacy_label)
        return widget

    # --- provider / language combos ----------------------------------------

    def _populate_provider_combo(self):
        """Models available for summaries, plus a 'choose' prompt entry."""
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        models = self._settings.get("models") or []
        current = self._settings.get("ai_summary_provider")
        self._provider_combo.addItem(t("records_provider_pick"), None)
        selected = 0
        for entry in models:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("name") or entry.get("model") or "?")
            if provider_unsuitable(entry):
                # Metadata says translation-specialized; still selectable, but
                # the label says so instead of a silent poor result.
                label = f"{label} — {t('records_provider_unsuitable')}"
            self._provider_combo.addItem(label, entry.get("id"))
            if entry.get("id") == current:
                selected = self._provider_combo.count() - 1
        self._provider_combo.setCurrentIndex(selected)
        self._provider_combo.blockSignals(False)

    def _on_provider_changed(self, _index):
        data = self._provider_combo.currentData()
        self._settings["ai_summary_provider"] = data
        self.settings_updated.emit(self._settings)

    def _populate_lang_combo(self):
        """Output-language choices: follow UI / meeting, plus common languages."""
        self._minutes_lang_combo.blockSignals(True)
        self._minutes_lang_combo.clear()
        self._minutes_lang_combo.addItem(
            t("records_lang_follow"), summary_service.OUTPUT_LANG_FOLLOW
        )
        names = {
            "zh": "中文", "en": "English", "ru": "Русский", "ja": "日本語",
            "ko": "한국어", "fr": "Français", "de": "Deutsch", "es": "Español",
        }
        saved = self._settings.get("ai_summary_output_lang")
        selected = 0
        for code in ("zh", "en", "ru", "ja", "ko", "fr", "de", "es"):
            self._minutes_lang_combo.addItem(names[code], code)
            if code == saved:
                selected = self._minutes_lang_combo.count() - 1
        self._minutes_lang_combo.setCurrentIndex(selected)
        self._minutes_lang_combo.blockSignals(False)

    def _on_lang_changed(self, _index):
        self._settings["ai_summary_output_lang"] = (
            self._minutes_lang_combo.currentData()
        )
        self.settings_updated.emit(self._settings)

    # --- tab switch ------------------------------------------------------------

    def _on_tab_changed(self, index: int):
        self._body.setCurrentIndex(index)

    # --- responsive behaviour -------------------------------------------------

    def _apply_stacked_or_split(self):
        # contentsRect(): the width the page's own layout actually has, after
        # the panel's nav column and margins — not the outer window's.
        wide = self.contentsRect().width() >= _STACKED_WIDTH_THRESHOLD
        target = Qt.Orientation.Horizontal if wide else Qt.Orientation.Vertical
        if self._splitter.orientation() != target:
            self._splitter.setOrientation(target)
        self._back_btn.setVisible(not wide and self._detail_stack.currentIndex() == 0)

    def resizeEvent(self, event: QEvent):
        super().resizeEvent(event)
        # apply now (for the already-updated size) and once the layout has
        # settled (event.size() can lead widget.width() in some backends).
        # The page-owned timer replaces a bare QTimer.singleShot(0, self, ...)
        # — a page-destroyed pending callback dies with its parent here.
        self._apply_stacked_or_split()
        self._layout_settle_timer.start()

    def _on_splitter_moved(self):
        self._save_splitter_timer.start()

    def _save_splitter(self):
        if self._splitter.orientation() == Qt.Orientation.Horizontal:
            self._settings["records_splitter"] = self._splitter.sizes()
            self.settings_updated.emit(self._settings)

    def _apply_splitter_from_settings(self):
        sizes = self._settings.get("records_splitter")
        if isinstance(sizes, list) and len(sizes) == 2 and all(
            isinstance(s, (int, float)) and s >= 0 for s in sizes
        ):
            self._splitter.setSizes([int(s) for s in sizes])

    def _show_list(self):
        self._detail_stack.setCurrentIndex(1)
        self._back_btn.hide()

    def _on_session_selected(self, current, _previous):
        if self._suppress_selection:
            return
        if current is None:
            self._detail_stack.setCurrentIndex(1)
            return
        # Narrow layout: selecting shows the detail pane with a way back.
        self._detail_stack.setCurrentIndex(0)
        self._back_btn.setVisible(
            self._splitter.orientation() == Qt.Orientation.Vertical
        )
        self._load_session(current.record)

    # --- session list ------------------------------------------------------

    def refresh(self, select_session: str | None = None):
        # Migration (id stamping, legacy index→id) may change the settings
        # dict; persist it so the ids survive a crash before the next
        # arbitrary panel auto-save.
        if ensure_model_ids(self._settings):
            self.settings_updated.emit(self._settings)
        active = None
        if self._writer is not None:
            try:
                active = self._writer.active_session()
            except Exception:
                log.debug("Could not query active session", exc_info=True)
        try:
            self._sessions = records.list_sessions(self._dir, active_session=active)
        except Exception:
            log.error("Could not list meeting records", exc_info=True)
            self._sessions = []
        self._refill_list()
        # After an end, land the user on the meeting that just completed so
        # "generate AI minutes" is one click away.
        if select_session:
            for row in range(self._list.count()):
                item = self._list.item(row)
                if item and item.record.get("session") == select_session:
                    self._list.setCurrentRow(row)
                    break

    def _refill_list(self):
        """Rebuild list rows from the search box, filter and session state."""
        if not hasattr(self, "_list"):
            return
        self._suppress_selection = True
        self._list.clear()
        query = self._search.text().strip().lower()
        filter_key = self._filter.currentData() or "records_filter_all"
        visible = 0
        for record in self._sessions:
            if query and not self._record_matches(record, query):
                continue
            if filter_key == "records_filter_summarized" and not record.get("has_summary"):
                continue
            if filter_key == "records_filter_unsummarized" and record.get("has_summary"):
                continue
            self._list.addItem(SessionItem(record, self._session_state))
            visible += 1
        self._suppress_selection = False
        self._list_empty.setVisible(self._list.count() == 0)
        if self._list.count() == 0:
            self._detail_stack.setCurrentIndex(1)
            self._list_empty.setText(
                t("records_search_no_match") if (query or filter_key != "records_filter_all")
                else t("records_empty")
            )
        else:
            self._list.setCurrentRow(0)

    @staticmethod
    def _record_matches(record: dict, query: str) -> bool:
        haystack = " ".join(
            str(record.get(k) or "")
            for k in ("title", "session", "asr_engine", "translation_model")
        ).lower()
        return query in haystack

    def _list_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        rename = menu.addAction(t("records_rename"))
        menu.addSeparator()
        delete = menu.addAction(t("btn_delete_record"))
        action = menu.exec(self._list.mapToGlobal(pos))
        if action == rename:
            self._rename_session(item)
        elif action == delete:
            self._delete_session()

    def _rename_session(self, item=None):
        item = item or self._list.currentItem()
        if item is None:
            return
        record = item.record
        from PyQt6.QtWidgets import QInputDialog

        title, ok = QInputDialog.getText(
            self, t("records_rename"), t("records_rename_prompt"),
            text=records.session_title(record, self._dir),
        )
        if not ok:
            return
        title = title.strip()
        if not title:
            QMessageBox.warning(self, t("error_title"), t("records_rename_empty"))
            return
        if records.set_session_title(self._dir, record["session"], title):
            record["title"] = title
            item.setText(title)
            if record.get("session") == self._current_session():
                self._title_label.setText(title)

    def _current_session(self) -> str | None:
        item = self._list.currentItem()
        return item.record.get("session") if item else None

    def _current_record(self) -> dict | None:
        item = self._list.currentItem()
        return item.record if item else None

    # --- detail loading -------------------------------------------------------

    def _load_session(self, record: dict):
        worker = self._worker
        if worker is not None and worker.isRunning():
            # The old generation's results must not land on the new session.
            worker.cancel()
            self._worker = None
            self._worker_session = None
            # The registry holds the strong reference until run() ends, so
            # the draining thread needs no QApplication re-parenting: it
            # finishes on its own, its results land nowhere (the page's
            # generation check drops them), and the save itself is guarded
            # by the worker's own cancel check.
            self._cancel_btn.hide()
            self._generate_btn.setEnabled(True)
            self._generate_btn.setText(t("records_generate_summary"))
            self._status_label.setText("")

        title = records.session_title(record, self._dir)
        self._title_label.setText(title)
        self._meta_label.setText(self._meta_line(record))
        self._populate_info(record)
        self._entries = records.parse_session(self._dir, record)
        self._entries_rendered = 0
        self._record_area.reset()
        self._render_more_entries()
        self._load_minutes(record)
        self._tabs.setCurrentIndex(0)

    def _meta_line(self, record: dict) -> str:
        parts = []
        started = record.get("started") or ""
        try:
            parts.append(datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M"))
        except (TypeError, ValueError):
            if record.get("session"):
                parts.append(str(record["session"]))
        seconds = int(record.get("duration_seconds") or 0)
        if seconds:
            parts.append(pdf_exporter.format_duration(seconds))
        entries = record.get("entries")
        if entries is not None:
            parts.append(
                t("transcript_entries_one" if entries == 1 else "transcript_entries")
                .format(count=entries)
            )
        for key in ("asr_engine", "translation_model"):
            if record.get(key):
                parts.append(str(record[key]))
        state = self._summary_state_text(record)
        if state:
            parts.append(state)
        return "  ·  ".join(parts)

    def _summary_state_text(self, record: dict) -> str:
        # ENDING first: the writer stops reporting the session active while
        # it is being closed, and the state must not flip to "interrupted"
        # for the seconds the close takes.
        if self._session_state == "ending":
            return t("records_state_ending")
        if record.get("is_active"):
            if self._session_state == "paused":
                return t("records_state_paused")
            return t("records_state_active")
        if record.get("interrupted"):
            return t("records_state_interrupted")
        if self._worker is not None and self._worker_session == record.get("session"):
            return t("records_state_generating")
        if not record.get("has_summary"):
            return ""
        if record.get("summary_stale"):
            return t("records_state_stale")
        if record.get("summary_edited"):
            return t("records_state_edited")
        return t("records_state_ready")

    def _populate_info(self, record: dict):
        rows = [
            (t("records_info_session"), str(record.get("session") or "")),
            (t("records_info_started"), str(record.get("started") or "")),
            (t("records_info_ended"), str(record.get("ended") or "")),
            (
                t("records_info_duration"),
                pdf_exporter.format_duration(record.get("duration_seconds") or 0),
            ),
            (
                t("records_info_speech"),
                pdf_exporter.format_duration(record.get("speech_seconds") or 0),
            ),
            (
                t("records_info_entries"),
                str(record.get("entries") or 0),
            ),
            (t("records_info_translated"), str(record.get("translated") or 0)),
            (t("records_info_untranslated"), str(record.get("untranslated") or 0)),
            (t("records_info_asr"), str(record.get("asr_engine") or "")),
            (t("records_info_translation_model"), str(record.get("translation_model") or "")),
            (t("records_info_source_lang"), str(record.get("source_language") or "")),
            (t("records_info_target_lang"), str(record.get("target_language") or "")),
        ]
        lines = ["<table width='100%' cellspacing='6'>"]
        for i, (key, value) in enumerate(rows):
            if not value:
                continue
            bg = "background:rgba(255,255,255,0.03);" if i % 2 else ""
            lines.append(
                f"<tr><td style='color:#ada79d;width:38%;{bg}'>{key}</td>"
                f"<td style='{bg}'>{value}</td></tr>"
            )
        lines.append("</table>")
        self._info_browser.setHtml("".join(lines))

    # --- record rendering ---------------------------------------------------------

    def _render_more_entries(self):
        """Render the next batch of entries into the scroll area."""
        batch = self._entries[self._entries_rendered:
                              self._entries_rendered + _RENDER_BATCH]
        show_original = self._show_original_btn.isChecked()
        for entry in batch:
            self._record_area.add_entry(entry, show_original)
        self._entries_rendered += len(batch)
        total = len(self._entries)
        self._render_progress.setText(
            t("records_render_progress").format(
                shown=min(self._entries_rendered, total), total=total
            )
        )
        self._record_area.set_batch_done(self._entries_rendered >= total)

    def _on_toggle_original(self, checked: bool):
        self._record_area.reset()
        self._entries_rendered = 0
        self._render_more_entries()

    # --- AI minutes ------------------------------------------------------------------

    def _load_minutes(self, record: dict):
        loaded = records.load_summary(self._dir, record.get("session") or "")
        self._summary_doc = loaded
        if loaded is None:
            self._minutes_browser_stack.setCurrentIndex(0)
            self._populate_provider_combo()
            self._populate_lang_combo()
            self._generate_btn.setText(t("records_generate_summary"))
            self._edit_btn.setEnabled(False)
            self._status_label.setText("")
            return
        self._minutes_browser_stack.setCurrentIndex(1)
        self._minutes_browser.setHtml(minutes_html(loaded["content"]))
        self._populate_provider_combo()
        self._populate_lang_combo()
        meta = loaded.get("meta") or {}
        info = [
            meta.get("provider_name"),
            meta.get("model"),
            meta.get("generated_at"),
        ]
        state = []
        if record.get("summary_stale"):
            self._status_label.setProperty("stale", True)
            state.append(t("records_state_stale_detail"))
        if meta.get("edited_by_user"):
            state.append(t("records_state_edited"))
        suffix = f"  ·  {meta.get('template')}" if meta.get("template") else ""
        self._status_label.setText(
            "  ·  ".join(x for x in info if x) + suffix
            + ("  ·  " + " / ".join(state) if state else "")
        )
        self._generate_btn.setText(t("records_regenerate_summary"))
        self._edit_btn.setEnabled(True)


    # --- summary generation ----------------------------------------------------------

    def _on_generate(self):
        record = self._current_record()
        if record is None or not self._entries:
            self._status_label.setText(t("summary_empty_record"))
            return
        if record.get("is_active") and self._session_state != "idle":
            # The record is still growing (ACTIVE/PAUSED) or its final save is
            # in flight (ENDING); a summary now would be a mid-meeting
            # snapshot that reads as final minutes. The gate opens when the
            # end lands and the page refreshes to a closed session.
            self._status_label.setText(t("summary_active_session"))
            return
        if record.get("interrupted"):
            # An interrupted record is closed (nothing more will be appended
            # to it), so minutes are allowed — the state label alone is the
            # hint that the meeting ended abnormally.
            pass
        if self._worker is not None and self._worker.isRunning():
            return
        provider = resolve_provider(self._settings)
        if provider is None:
            self._status_label.setText(t("summary_need_provider"))
            return
        if provider_missing_key(provider):
            self._status_label.setText(t("summary_missing_key"))
            return
        if self._summary_doc is not None and (self._summary_doc.get("meta") or {}).get("edited_by_user"):
            confirm = QMessageBox.question(
                self, t("records_regenerate_summary"),
                t("records_regenerate_edited_confirm"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        self._notify_cloud_privacy(provider)

        output_lang = self._minutes_lang_combo.currentData() or summary_service.OUTPUT_LANG_FOLLOW
        default_lang = self._default_output_lang(record)
        self._worker_generation += 1
        self._worker = SummaryWorker(
            self._dir,
            record["session"],
            self._entries,
            provider,
            template=self._template_combo.currentData() or "meeting",
            output_lang=output_lang,
            default_output_lang=default_lang,
            generation=self._worker_generation,
            parent=self,
        )
        self._worker_session = record["session"]
        self._worker.progress.connect(self._on_summary_progress)
        self._worker.succeeded.connect(self._on_summary_succeeded)
        self._worker.failed.connect(self._on_summary_failed)
        self._worker.finished.connect(self._on_worker_finished)
        # Registry takes the strong reference (page destruction no longer
        # strands the thread), and the app's cancel_all at exit reaches it.
        self._task_registry.register(self._worker)
        self._cancel_btn.show()
        self._generate_btn.setEnabled(False)
        self._generate_btn.setText(t("records_state_generating"))
        self._status_label.setText(t("summary_progress_start"))
        self._worker.start()

    def _default_output_lang(self, record: dict) -> str:
        lang = self._minutes_lang_combo.currentData()
        if lang and lang != summary_service.OUTPUT_LANG_FOLLOW:
            return {"zh": "中文", "en": "English", "ru": "Русский",
                    "ja": "日本語", "ko": "한국어", "fr": "Français",
                    "de": "Deutsch", "es": "Español"}.get(lang, "中文")
        # Follow the meeting's target language — that is what the user reads.
        target = record.get("target_language") or ""
        mapping = {"zh": "中文", "en": "English", "ru": "Русский", "ja": "日本語",
                   "ko": "한국어", "fr": "Français", "de": "Deutsch", "es": "Español"}
        if target in mapping:
            return mapping[target]
        return "中文" if get_lang() == "zh" else "English"

    def _notify_cloud_privacy(self, provider: dict):
        """First-use-per-provider cloud notice (non-blocking once shown)."""
        api_base = str(provider.get("api_base") or "")
        is_local = any(
            h in api_base.lower()
            for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "host.docker.internal")
        )
        if is_local:
            return
        pid = provider.get("id")
        if pid in self._cloud_notice_shown_for:
            return
        self._cloud_notice_shown_for.add(pid)
        self._status_label.setText(t("records_privacy_note"))
        QMessageBox.information(self, t("records_privacy_title"), t("records_privacy_cloud"))

    def _on_summary_progress(self, stage: str, index: int, total: int):
        if self._generation_mismatch():
            return
        if stage == "part":
            if total > 1:
                self._status_label.setText(
                    t("summary_progress_part").format(index=index, total=total)
                )
            else:
                self._status_label.setText(t("summary_progress_start"))
        else:
            self._status_label.setText(t("summary_progress_merging"))

    def _generation_mismatch(self) -> bool:
        """True when the emitting worker is no longer the current generation.

        Both the worker identity and the generation counter are checked:
        switching meetings replaces the worker, and cancel-then-regenerate on
        the same meeting bumps the counter — an old worker's late signals must
        never update the current view or overwrite the newer summary.
        """
        worker = self.sender()
        if not isinstance(worker, SummaryWorker):
            worker = self._worker
        if worker is None or worker is not self._worker:
            return True
        return (
            worker.generation != self._worker_generation
            or self._worker_session != worker.session
        )

    def _on_summary_succeeded(self, content: str, meta: dict):
        if self._generation_mismatch():
            return  # user switched meetings or re-generated; keep the new state
        record = self._current_record()
        if record is None or record.get("session") != self._worker_session:
            return  # user switched meetings; the file is saved, list will show it
        record["has_summary"] = True
        record["summary_stale"] = False
        record["summary_edited"] = False
        item = self._list.currentItem()
        if item:
            item.record.update(record)
        self._load_minutes(record)
        self._meta_label.setText(self._meta_line(record))
        self._status_label.setText(t("summary_done"))

    def _on_summary_failed(self, kind: str, detail: str):
        if self._generation_mismatch():
            return
        message = t(kind)
        if detail and message == kind:  # missing translation key
            message = detail
        elif detail:
            message = f"{message} ({detail[:120]})"
        self._status_label.setText(message)

    def _on_worker_finished(self):
        worker = self.sender()
        if isinstance(worker, SummaryWorker) and worker is not self._worker:
            return  # an orphaned old worker finishing; the current one stays
        self._worker = None
        self._worker_session = None
        self._cancel_btn.hide()
        self._generate_btn.setEnabled(True)
        # A cancel leaves "Cancelling…" behind — run() emits nothing on that
        # path, this handler is the only place that learns it ended.
        if self._status_label.text() == t("summary_cancelling"):
            self._status_label.setText("")
        record = self._current_record()
        self._generate_btn.setText(
            t("records_regenerate_summary") if (record and record.get("has_summary"))
            else t("records_generate_summary")
        )
        # The registry drops its reference on finished; the QThread object
        # itself is freed here, on the GUI thread, after run() returned.
        if worker is not None:
            worker.deleteLater()

    def _on_cancel_generate(self):
        if self._worker is not None:
            self._status_label.setText(t("summary_cancelling"))
            self._worker.cancel()

    # --- summary editing ---------------------------------------------------------------

    def _on_edit_summary(self):
        if self._summary_doc is None:
            return
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle(t("records_edit_summary"))
        dlg.resize(600, 500)
        layout = QVBoxLayout(dlg)
        editor = QPlainTextEdit()
        editor.setPlainText(self._summary_doc["content"])
        layout.addWidget(editor, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        content = editor.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, t("error_title"), t("summary_error_empty"))
            return
        meta = dict(self._summary_doc.get("meta") or {})
        meta["edited_by_user"] = True
        meta["edited_at"] = datetime.now().isoformat(timespec="seconds")
        if records.save_summary(self._dir, self._current_session(), content, meta):
            record = self._current_record()
            if record:
                record["summary_edited"] = True
            self._load_minutes(self._current_record() or {})
            self._status_label.setText(t("summary_saved"))
        else:
            QMessageBox.critical(self, t("error_title"), t("summary_error_save"))

    # --- export --------------------------------------------------------------------------

    def _on_export_menu(self):
        record = self._current_record()
        if record is None:
            return
        has_summary = bool(record.get("has_summary"))
        menu = QMenu(self)
        if has_summary:
            menu.addAction(t("records_export_pdf_summary"), lambda: self._export("pdf_summary"))
            menu.addAction(t("records_export_pdf_full"), lambda: self._export("pdf_full"))
        else:
            act = menu.addAction(t("records_export_pdf_summary"))
            act.setEnabled(False)
            menu.addAction(t("records_export_pdf_full"), lambda: self._export("pdf_full"))
        menu.addSeparator()
        menu.addAction(t("records_export_md"), lambda: self._export("md"))
        menu.addAction(t("records_export_txt"), lambda: self._export("txt"))
        menu.exec(self._export_btn.mapToGlobal(
            self._export_btn.rect().bottomLeft()
        ))

    def _export(self, kind: str):
        record = self._current_record()
        if record is None:
            return
        session = record["session"]
        loaded = records.load_summary(self._dir, session)
        if kind in ("pdf_summary", "md_summary") and loaded is None:
            QMessageBox.information(
                self, t("records_export"), t("records_export_need_summary")
            )
            return

        title = records.session_title(record, self._dir)
        when = (record.get("started") or "")[:10]
        if kind == "pdf_summary":
            self._export_pdf(record, title, when, loaded["content"], None)
        elif kind == "pdf_full":
            summary = loaded["content"] if loaded else None
            if summary is None:
                QMessageBox.information(
                    self, t("records_export"), t("records_export_pdf_full_no_summary")
                )
            self._export_pdf(record, title, when, summary, self._entries)
        elif kind == "md":
            self._export_copy(record, title, when, "md",
                              loaded["content"] if loaded else None, self._read_record_file(record, "meeting"))
        elif kind == "txt":
            self._export_copy(record, title, when, "txt", None,
                              self._read_record_file(record, "all"))

    def _export_pdf(self, record, title, when, summary, entries):
        from PyQt6.QtWidgets import QFileDialog

        default = pdf_exporter.safe_filename(title, when)
        target, _ = QFileDialog.getSaveFileName(
            self, t("records_export"), default, "PDF (*.pdf)"
        )
        if not target:
            return
        meta_rows = self._pdf_meta_rows(record)
        ok = pdf_exporter.export_pdf(
            target,
            title=title,
            meta_rows=meta_rows,
            summary_markdown=summary,
            entries=entries,
            show_original=True,
        )
        if ok:
            self._status_label.setText(t("records_export_done"))
        else:
            QMessageBox.critical(
                self, t("error_title"), t("export_failed").format(error="PDF")
            )

    def _pdf_meta_rows(self, record: dict) -> list[tuple[str, str]]:
        rows = []
        if record.get("is_active"):
            # The export is a snapshot, not final minutes; the paper trail
            # should say so as plainly as the UI does.
            rows.append((t("records_pdf_active_flag"), t("records_state_active")))
        started = record.get("started") or ""
        try:
            rows.append((t("records_info_started"),
                         datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")))
        except (TypeError, ValueError):
            pass
        rows.append((t("records_info_duration"),
                     pdf_exporter.format_duration(record.get("duration_seconds") or 0)))
        rows.append((t("records_info_speech"),
                     pdf_exporter.format_duration(record.get("speech_seconds") or 0)))
        entries = record.get("entries")
        if entries is not None:
            rows.append((t("records_info_entries"), str(entries)))
        for key, label in (
            ("asr_engine", t("records_info_asr")),
            ("translation_model", t("records_info_translation_model")),
            ("source_language", t("records_info_source_lang")),
            ("target_language", t("records_info_target_lang")),
        ):
            if record.get(key):
                rows.append((label, str(record[key])))
        return rows

    def _export_copy(self, record, title, when, ext, summary, record_text):
        from PyQt6.QtWidgets import QFileDialog

        default = pdf_exporter.safe_filename(title, when, ext)
        target, _ = QFileDialog.getSaveFileName(
            self, t("records_export"), default,
            t("export_filter") if ext == "txt" else t("export_md_filter"),
        )
        if not target:
            return
        parts = [f"# {title}", ""]
        if summary:
            parts += ["## AI Minutes", "", summary, ""]
        if record_text:
            parts += ["## Record", "", record_text]
        try:
            Path(target).write_text("\n".join(parts), encoding="utf-8")
            self._status_label.setText(t("records_export_done"))
        except OSError as exc:
            QMessageBox.critical(
                self, t("error_title"), t("export_failed").format(error=str(exc))
            )

    def _read_record_file(self, record: dict, kind: str) -> str | None:
        path = (record.get("files") or {}).get(kind)
        if not path:
            return None
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    # --- folder / delete --------------------------------------------------------------------

    def _open_folder_current(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        _open_folder(self._dir)

    def _delete_session(self):
        record = self._current_record()
        if record is None:
            return
        session = record["session"]
        confirm = QMessageBox.warning(
            self,
            t("btn_delete_record"),
            t("dialog_delete_record").format(session=session),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        if self._worker is not None and self._worker_session == session:
            self._worker.cancel()
        records.delete_summary(self._dir, session)
        failures = delete_session(self._dir, session)
        if failures:
            QMessageBox.critical(
                self,
                t("error_title"),
                t("cache_delete_failed").format(paths="\n".join(failures)),
            )
        self.refresh()

    def cleanup(self):
        """Called when the panel closes: cancel the worker, keep old summaries.

        Cancellation is cooperative; a request already in flight can take up
        to the worker's timeout to return. The worker is *not* re-parented
        to the QApplication: the summary task registry holds the strong
        reference until ``run()`` ends, so the thread outlives this page
        safely. Results emitted after the page is gone are dropped by the
        generation check, and the save itself is guarded by the worker's own
        cancel check, so the old summary on disk is never overwritten.
        """
        worker = self._worker
        if worker is None:
            return
        worker.cancel()
        self._worker = None
        self._worker_session = None
