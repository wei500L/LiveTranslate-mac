"""Leaf widgets and renderers for the meeting-records center.

Split out of ``meeting_records_page.py`` so the page stays the orchestrator
it is meant to be: everything here is a self-contained visual component with
no thread, settings, network or file-write logic — the session-list row, the
incremental record renderer, the Markdown→HTML minutes renderer and the
record-state badge. The page imports and drives them.
"""

from __future__ import annotations

import re
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QListWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import pdf_exporter
from i18n import t

_BULLET = re.compile(r"^[-*] +")
_BOLD_MD = re.compile(r"\*\*(.+?)\*\*")
_CODE_MD = re.compile(r"`([^`]+)`")


def _state_badge(record: dict, session_state: str = "idle") -> str:
    """Per-record badge. The app-level ``session_state`` only refines the
    row that IS the current session (``is_active``/``is_ending`` on the
    record, matched by stamp); a global "ending" must never paint unrelated
    history rows. Interrupted (crash-left) sessions are labelled as such so
    a half-recorded meeting never reads as a completed one."""
    if record.get("is_ending"):
        return t("records_state_ending")
    if record.get("is_active"):
        if session_state == "paused":
            return t("records_state_paused")
        return t("records_state_active")
    if record.get("interrupted"):
        return t("records_state_interrupted")
    if record.get("summary_stale"):
        return t("records_state_stale")
    if record.get("summary_edited"):
        return t("records_state_edited")
    if record.get("has_summary"):
        return t("records_state_ready")
    return t("records_state_none")


class SessionItem(QListWidgetItem):
    """List entry with a two-to-three line information structure."""

    def __init__(self, record: dict, session_state: str = "idle"):
        title = record.get("title") or record.get("session", "?")
        super().__init__(title)
        self.record = record
        self.setData(Qt.ItemDataRole.UserRole, record)
        self.setText(self._compose_text(record, session_state))

    @staticmethod
    def _compose_text(record: dict, session_state: str = "idle") -> str:
        started = record.get("started") or ""
        try:
            when = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            when = str(record.get("session") or "?")
        lines = [record.get("title") or when]

        duration = pdf_exporter.format_duration(record.get("duration_seconds") or 0)
        line2 = when if when != lines[0] else ""
        if duration != "0:00":
            line2 = f"{line2}  ·  {duration}" if line2 else duration
        if line2:
            lines.append(line2)

        entries = record.get("entries")
        parts = []
        if entries is not None:
            parts.append(
                t("transcript_entries_one" if entries == 1 else "transcript_entries")
                .format(count=entries)
            )
        if record.get("translation_model"):
            parts.append(str(record["translation_model"]))
        state = _state_badge(record, session_state)
        if state:
            parts.append(state)
        if parts:
            lines.append("  ·  ".join(parts))
        return "\n".join(lines)


def minutes_html(markdown_text: str) -> str:
    """Render summary Markdown into styled HTML for the minutes browser.

    The model's output is escaped before any of our own tags are
    reintroduced: a meeting record (or a hallucinating model) containing
    ``<script>`` must never execute in the QTextBrowser.
    """
    import html as html_mod

    out = []
    in_list = False
    for raw in (markdown_text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if re_match := _BULLET.match(line):
            if not in_list:
                out.append("<ul style='margin:4px 0 4px 0;padding-left:18px;'>")
                in_list = True
            content = _inline_html(html_mod.escape(_BULLET.sub("", line)))
            out.append(f"<li>{content}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("### "):
            out.append(f"<h3>{html_mod.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(
                f"<h2 style='color:#e7b96f;'>{html_mod.escape(line[3:])}</h2>"
            )
        elif line.startswith("# "):
            out.append(f"<h1>{html_mod.escape(line[2:])}</h1>")
        else:
            out.append(f"<p>{_inline_html(html_mod.escape(line))}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def _inline_html(escaped: str) -> str:
    escaped = _BOLD_MD.sub(r"<b>\1</b>", escaped)
    escaped = _CODE_MD.sub(
        r"<code style='font-family:Menlo,monospace;background:rgba(255,255,255,0.06);'>\1</code>",
        escaped,
    )
    return escaped


class RecordScrollArea(QScrollArea):
    """Vertical record renderer with incremental loading.

    Entries are plain QFrames with labels — cheap to build in batches, and
    the scrollbar position triggers the next batch, so a 3-hour meeting
    never constructs hundreds of widgets up front.
    """

    load_more_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 8, 0)
        self._layout.setSpacing(2)
        self._layout.addStretch()
        self.setWidget(self._container)
        self._batch_done = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def reset(self):
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._batch_done = False

    def set_batch_done(self, done: bool):
        self._batch_done = done

    def add_entry(self, entry: dict, show_original: bool):
        translation = entry.get("translation")
        original = entry.get("original")
        if not translation and not original:
            return
        frame = QFrame()
        frame.setProperty("role", "recordEntry")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(1)

        ts = QLabel(entry.get("timestamp") or "")
        ts.setProperty("role", "timestamp")
        layout.addWidget(ts)

        body = translation or original
        text_label = QLabel(body)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(text_label)

        if translation is None and original:
            note = QLabel(t("records_untranslated"))
            note.setProperty("role", "muted")
            layout.addWidget(note)
        elif show_original and translation and original:
            orig_label = QLabel(original)
            orig_label.setProperty("role", "recordOriginal")
            orig_label.setWordWrap(True)
            layout.addWidget(orig_label)

        insert_at = self._layout.count() - 1  # before the trailing stretch
        self._layout.insertWidget(insert_at, frame)

    def _on_scroll(self, value):
        if self._batch_done:
            return
        bar = self.verticalScrollBar()
        if value >= bar.maximum() - 120:
            self.load_more_requested.emit()
