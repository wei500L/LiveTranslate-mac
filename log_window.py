import html
import logging
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor
from i18n import t
from platform_fonts import default_mono_font_family


class QLogHandler(logging.Handler):
    """Logging handler that emits to a Qt signal."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def emit(self, record):
        msg = self.format(record)
        self._signal.emit(msg, record.levelno)


class LogWindow(QWidget):
    """Real-time log viewer window."""

    log_signal = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_log"))
        self.setMinimumSize(700, 400)
        self.resize(900, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Log display
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setUndoRedoEnabled(False)
        self._text.document().setMaximumBlockCount(2000)
        self._text.setFont(QFont(default_mono_font_family(), 9))
        self._text.setStyleSheet("background-color: #111820; color: #edf4f7; border: 1px solid #30404e; border-radius: 9px; padding: 8px;")
        layout.addWidget(self._text)

        # Controls
        ctrl = QHBoxLayout()
        self._auto_scroll = QCheckBox(t("auto_scroll"))
        self._auto_scroll.setChecked(True)
        ctrl.addWidget(self._auto_scroll)

        self._show_debug = QCheckBox(t("show_debug"))
        self._show_debug.setChecked(False)
        ctrl.addWidget(self._show_debug)

        ctrl.addStretch()

        clear_btn = QPushButton(t("clear"))
        clear_btn.setObjectName("dangerButton")
        clear_btn.clicked.connect(self._text.clear)
        ctrl.addWidget(clear_btn)

        layout.addLayout(ctrl)

        # Connect signal
        self.log_signal.connect(self._append_log)

    def get_handler(self):
        handler = QLogHandler(self.log_signal)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        return handler

    @pyqtSlot(str, int)
    def _append_log(self, msg: str, level: int):
        if level < logging.INFO and not self._show_debug.isChecked():
            return

        color = {
            logging.DEBUG: "#808080",
            logging.INFO: "#d4d4d4",
            logging.WARNING: "#dcdcaa",
            logging.ERROR: "#f44747",
            logging.CRITICAL: "#ff0000",
        }.get(level, "#d4d4d4")

        # Highlight result lines — but only below WARNING, where the level
        # color is the signal the reader is scanning for. And match the message
        # shapes as they are actually logged ("Translate (410ms): ..."): every
        # line from the root logger carries the prefix "LiveTranslate: ", so
        # matching the bare word "Translate:" painted *all* of them — errors
        # included — in the translate color.
        if level < logging.WARNING:
            if "ASR [" in msg:
                color = "#9caf91"
            elif "Translate (" in msg:
                color = "#9cdcfe"
            elif "Speech segment" in msg:
                color = "#ce9178"

        # Escape the message: log lines are plain text, and model output in
        # particular is full of <think>-style tags that Qt would otherwise
        # swallow as HTML — hiding exactly the lines someone opens this window
        # to read.
        self._text.append(f'<span style="color:{color}">{html.escape(msg)}</span>')

        if self._auto_scroll.isChecked():
            self._text.moveCursor(QTextCursor.MoveOperation.End)
