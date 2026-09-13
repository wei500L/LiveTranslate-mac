import os
import threading

import psutil
from i18n import t, LANGUAGES
from PyQt6.QtCore import QPoint, QPropertyAnimation, QEasingCurve, QSize, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from platform_clickthrough import set_always_on_top, set_click_through
from platform_fonts import default_cjk_font_family, default_mono_font_family
from torch_backend import accelerator_memory

DEFAULT_STYLE = {
    "preset": "default",
    "bg_color": "#000000",
    "bg_opacity": 240,
    "header_color": "#1a1a2e",
    "header_opacity": 230,
    "border_radius": 8,
    "original_font_family": default_cjk_font_family(),
    "translation_font_family": default_cjk_font_family(),
    "original_font_size": 12,
    "translation_font_size": 23,
    "original_color": "#9a9aa8",
    "translation_color": "#ffffff",
    # Dimmer variants for streaming (provisional) text — cloud live card.
    "provisional_original_color": "#7a7a8c",
    "provisional_translation_color": "#b8b8c0",
    "window_opacity": 95,
    # Font sizes scale with the overlay's width by default (see
    # font_scale_for_width). Off means the sizes above are used literally,
    # exactly as before this setting existed.
    "scale_with_window": True,
}

_BASE = DEFAULT_STYLE

STYLE_PRESETS = {
    "default": dict(_BASE),
    "transparent": {
        **_BASE,
        "preset": "transparent",
        "bg_opacity": 120,
        "header_opacity": 120,
        "window_opacity": 70,
    },
    "compact": {
        **_BASE,
        "preset": "compact",
        "original_font_size": 9,
        "translation_font_size": 17,
    },
    "light": {
        **_BASE,
        "preset": "light",
        "bg_color": "#e8e8f0",
        "bg_opacity": 230,
        "header_color": "#c8c8d8",
        "header_opacity": 220,
        "original_color": "#555560",
        "translation_color": "#111111",
    },
    "dracula": {
        **_BASE,
        "preset": "dracula",
        "bg_color": "#282a36",
        "bg_opacity": 235,
        "header_color": "#44475a",
        "header_opacity": 230,
        "original_color": "#b4b4c2",
        "translation_color": "#f8f8f2",
    },
    "nord": {
        **_BASE,
        "preset": "nord",
        "bg_color": "#2e3440",
        "bg_opacity": 235,
        "header_color": "#3b4252",
        "header_opacity": 230,
        "original_color": "#d8dee9",
        "translation_color": "#eceff4",
    },
    "monokai": {
        **_BASE,
        "preset": "monokai",
        "bg_color": "#272822",
        "bg_opacity": 235,
        "header_color": "#3e3d32",
        "header_opacity": 230,
        "original_color": "#b8b8b0",
        "translation_color": "#f8f8f2",
    },
    "solarized": {
        **_BASE,
        "preset": "solarized",
        "bg_color": "#002b36",
        "bg_opacity": 235,
        "header_color": "#073642",
        "header_opacity": 230,
        "original_color": "#839496",
        "translation_color": "#eee8d5",
    },
    "gruvbox": {
        **_BASE,
        "preset": "gruvbox",
        "bg_color": "#282828",
        "bg_opacity": 235,
        "header_color": "#3c3836",
        "header_opacity": 230,
        "original_color": "#bdae93",
        "translation_color": "#fbf1c7",
    },
    "tokyo_night": {
        **_BASE,
        "preset": "tokyo_night",
        "bg_color": "#1a1b26",
        "bg_opacity": 235,
        "header_color": "#24283b",
        "header_opacity": 230,
        "original_color": "#a9b1d6",
        "translation_color": "#c0caf5",
    },
    "catppuccin": {
        **_BASE,
        "preset": "catppuccin",
        "bg_color": "#1e1e2e",
        "bg_opacity": 235,
        "header_color": "#313244",
        "header_opacity": 230,
        "original_color": "#a6adc8",
        "translation_color": "#cdd6f4",
    },
    "one_dark": {
        **_BASE,
        "preset": "one_dark",
        "bg_color": "#282c34",
        "bg_opacity": 235,
        "header_color": "#3e4452",
        "header_opacity": 230,
        "original_color": "#abb2bf",
        "translation_color": "#e5c07b",
    },
    "everforest": {
        **_BASE,
        "preset": "everforest",
        "bg_color": "#2d353b",
        "bg_opacity": 235,
        "header_color": "#343f44",
        "header_opacity": 230,
        "original_color": "#9da9a0",
        "translation_color": "#d3c6aa",
    },
    "kanagawa": {
        **_BASE,
        "preset": "kanagawa",
        "bg_color": "#1f1f28",
        "bg_opacity": 235,
        "header_color": "#2a2a37",
        "header_opacity": 230,
        "original_color": "#c8c3a6",
        "translation_color": "#dcd7ba",
    },
}


# Every value these hierarchy fields have *ever* shipped with and that has
# since been superseded (11/14 were the pre-redesign sizes, 10/15 the first
# two-line sizes, 12/17 the sizes before window scaling landed). A stored field still holding one of them was never
# customized by the user, so it is safe to upgrade to the current value;
# anything else is a deliberate choice and is left alone. A set rather than a
# single value because a style saved by an *interim* build must keep moving
# forward too — the very first migration shipped 10/15.
#
# Without this, apply_style's `{**DEFAULT_STYLE, **style}` merge lets a style
# saved by an older build pin the old look forever and new defaults never
# appear.
_SUPERSEDED_STYLE_VALUES = {
    "original_font_size": {11, 10},
    "translation_font_size": {14, 15, 17},
    "original_color": {"#cccccc", "#9a9aa8"},
}

# Presets whose superseded value differed from the superseded default.
_SUPERSEDED_PRESET_VALUES = {
    "compact": {"original_font_size": {9, 8}, "translation_font_size": {11, 12, 13}},
    "light": {"original_color": {"#333333"}},
    "dracula": {"original_color": {"#f8f8f2"}},
    "monokai": {"original_color": {"#f8f8f2"}},
    "gruvbox": {"original_color": {"#ebdbb2"}},
    "catppuccin": {"original_color": {"#cdd6f4"}},
    "everforest": {"original_color": {"#d3c6aa"}},
    "kanagawa": {"original_color": {"#dcd7ba"}},
}

#: Style keys that no longer exist; dropped on load rather than carried.
_REMOVED_STYLE_KEYS = ("timestamp_color",)


def migrate_style(style: dict) -> tuple[dict, bool]:
    """Carry a stored overlay style across the two-line hierarchy redesign.

    Returns ``(style, changed)``. Removed keys are dropped, and each
    hierarchy field still holding a superseded shipped value is upgraded to
    the current value for that preset; a field the user actually changed
    keeps its value. Run once per release that retunes the hierarchy — the
    superseded sets carry every generation, so a style that skipped one
    upgrade still catches up.

    Idempotent: a migrated (or newly saved) style comes back unchanged.
    """
    if not isinstance(style, dict):
        return style, False
    out = dict(style)
    changed = False
    for key in _REMOVED_STYLE_KEYS:
        if key in out:
            del out[key]
            changed = True

    preset_key = out.get("preset", "default")
    preset = STYLE_PRESETS.get(preset_key) or DEFAULT_STYLE
    preset_overrides = _SUPERSEDED_PRESET_VALUES.get(preset_key, {})
    for field, superseded_default in _SUPERSEDED_STYLE_VALUES.items():
        if field not in out:
            continue
        superseded = preset_overrides.get(field, superseded_default)
        if out[field] in superseded and out[field] != preset[field]:
            out[field] = preset[field]
            changed = True
    return out, changed


# Fixed accent colors shared by every message card. They are deliberately
# not style fields: the language tag and the hint are UI chrome, not
# user-tintable content colors.
_LANG_TAG_COLOR = "#e7b96f"
_HINT_COLOR = "#999"


def _hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


#: The window width at which the style's font sizes are used verbatim. The
#: overlay opens at this width, so it doubles as the "design width": the
#: number in the settings panel is what you see at the default window size.
FONT_SCALE_REFERENCE_WIDTH = 620
#: Scale bounds. The floor 0.7 covers the whole reachable shrink range (the
#: window's minimum width is 480 -> 0.774) with margin, and still leaves the
#: original line at 8pt; the ceiling 1.8 is 1116px wide / 41pt translation,
#: past which a two-line card eats the screen.
FONT_SCALE_MIN = 0.7
FONT_SCALE_MAX = 1.8
#: Floor for a derived size: no stored value may ever scale down to 0pt.
MIN_SCALED_FONT_PT = 4
#: Debounce after a resize before re-laying out the cards. Scaling itself is
#: cheap (measured 0.1ms for 50 cards' setFont); this exists so a drag does
#: not reflow every wrapping card on each frame, not to save CPU.
FONT_SCALE_DEBOUNCE_MS = 150
#: How often the batched live updates are drained to the cards. One frame:
#: the drain is a lock plus two dict swaps, and the render count is bounded
#: by the arrival rate (one render per msg_id per tick), not by this number.
LIVE_FLUSH_INTERVAL_MS = 16


def font_scale_for_width(width: int) -> float:
    """Font-size multiplier for a given overlay width, clamped.

    Width is the single input on purpose: the compact-mode height animation
    preserves width, so it cannot trigger a rescale at all.
    """
    if not width or width <= 0:
        return 1.0
    return min(
        FONT_SCALE_MAX, max(FONT_SCALE_MIN, width / FONT_SCALE_REFERENCE_WIDTH)
    )


def scaled_font_size(base_size: float, factor: float) -> int:
    """Round half up, deliberately not ``round()``: Python's banker's rounding
    turns 34.5 into 34, so a size could stay put — or shrink — as the window
    grows."""
    return max(MIN_SCALED_FONT_PT, int(base_size * factor + 0.5))


class ChatMessage(QWidget):
    """Single chat message widget: small original line + large translation
    line (the visual hierarchy — the translation is the primary content).

    Rendering is state-driven: the mutators (update_streaming /
    set_translation / update_live / apply_style) only change state fields,
    then one `_render()` derives both lines' HTML from that state. Local
    cards and cloud (live) cards share this single path; the only cloud-only
    visual is the provisional dim + cursor, driven by `_live_provisional`.
    """

    _current_style = DEFAULT_STYLE

    def __init__(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        source_lang: str,
        asr_ms: float,
        parent=None,
        provider: str = "",
    ):
        super().__init__(parent)
        self.msg_id = msg_id
        self._original = original
        self._translated = ""
        self._timestamp = timestamp
        self._source_lang = source_lang
        # asr_ms is accepted (the signal/slot arity and every caller pass it)
        # but deliberately not stored: latency is not displayed anywhere any
        # more, and local latency is measured in main.py's PERF log. Keeping a
        # dead field here invited the belief that something renders it.
        # The cloud provider id drives the no-translation hint wording
        # (soniox_no_translation vs same_language) when a segment settles
        # without a translation.
        self._provider = provider
        # Rendering state.
        self._live_provisional = False  # cloud card: recognition still running
        self._streaming_partial = None  # latest local streaming partial, if any
        self._settled = False  # a final translation outcome was recorded
        self._live_state = None  # last rendered cloud snapshot, for the above
        self.setObjectName("chatMessage")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        # Breathing room between the secondary original line and the
        # primary translation line.
        self._layout.setSpacing(4)

        s = self._current_style
        self._header_label = QLabel()
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._header_label.setTextFormat(Qt.TextFormat.RichText)
        self._header_label.setWordWrap(True)
        self._header_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._header_label)

        self._trans_label = QLabel()
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        self._trans_label.setTextFormat(Qt.TextFormat.RichText)
        self._trans_label.setWordWrap(True)
        self._trans_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._trans_label)
        self._render()

    # -- rendering ---------------------------------------------------------

    def _decorate(self, escaped_text: str, role: str) -> str:
        """Single extension point for per-role text decoration (e.g. future
        terminology highlighting on original/translation lines). Identity
        for now."""
        return escaped_text

    def _original_line_html(self, s) -> str:
        # Provisional text is carried by the dim color alone. A cursor glyph
        # was tried twice (a heavy ▍ block, then a thin │ bar) and dropped:
        # any trailing glyph crowds the last letter, and being part of the
        # same text run it is what wraps onto a new line first.
        color = (
            s["provisional_original_color"]
            if self._live_provisional
            else s["original_color"]
        )
        return (
            f'<span style="color:{_LANG_TAG_COLOR};">[{self._source_lang}]</span> '
            f'<span style="color:{color};">'
            f'{self._decorate(_escape(self._original), "original")}</span>'
        )

    def _translation_line_html(self, s) -> str:
        if self._streaming_partial is not None:
            return (
                f'<span style="color:{s["translation_color"]};">&gt; '
                f'{self._decorate(_escape(self._streaming_partial), "translation")}</span>'
            )
        if self._live_provisional:
            if self._translated:
                return (
                    f'<span style="color:{s["provisional_translation_color"]};">&gt; '
                    f'{self._decorate(_escape(self._translated), "translation")}</span>'
                )
            return (
                f'<span style="color:{_HINT_COLOR}; font-style:italic;">{t("translating")}</span>'
            )
        if self._translated:
            return (
                f'<span style="color:{s["translation_color"]};">&gt; '
                f'{self._decorate(_escape(self._translated), "translation")}</span>'
            )
        if self._settled:
            hint = (
                t("soniox_no_translation")
                if self._provider == "soniox"
                else t("same_language")
            )
            return (
                f'<span style="color:{_HINT_COLOR}; font-style:italic;">&gt; {hint}</span>'
            )
        return (
            f'<span style="color:{_HINT_COLOR}; font-style:italic;">{t("translating")}</span>'
        )

    def _render(self):
        s = self._current_style
        self._header_label.setText(self._original_line_html(s))
        self._trans_label.setText(self._translation_line_html(s))

    # -- state mutators ----------------------------------------------------

    def update_streaming(self, partial_text: str):
        """Render partial streaming text immediately.

        No throttle of its own: the overlay already batches these at
        LIVE_FLUSH_INTERVAL_MS and calls this at most once per card per
        tick, so a second 50ms single-shot here only added another tick of
        latency (up to 100ms before a local partial reached the screen).
        """
        self._streaming_partial = partial_text
        self._render()

    def _stop_streaming(self):
        """The one place that clears the streaming state."""
        self._streaming_partial = None

    def set_translation(self, translated: str, translate_ms: float):
        # translate_ms is accepted for the signal/slot arity but not stored —
        # see the note in __init__.
        self._translated = translated or ""
        self._settled = True
        self._stop_streaming()
        self._render()

    def apply_style(self, s: dict):
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        self._render()

    def apply_font_scale(self, factor: float) -> None:
        """Re-apply just the fonts at a window-derived scale.

        Deliberately narrow: this is the resize path, and the expensive part
        of apply_style is not the fonts (measured 0.1ms for 50 cards) but the
        container/header setStyleSheet re-polish and setWindowOpacity — ~70ms
        together, and the latter goes to the compositor on macOS.

        `_current_style` stays the *base* style here. Writing the derived
        size back into it would make the next rescale multiply an
        already-scaled value (23 -> 41 -> 74).
        """
        s = self._current_style
        self._header_label.setFont(
            QFont(
                s["original_font_family"],
                scaled_font_size(s["original_font_size"], factor),
            )
        )
        self._trans_label.setFont(
            QFont(
                s["translation_font_family"],
                scaled_font_size(s["translation_font_size"], factor),
            )
        )

    def update_live(self, original: str, translation: str, final: bool):
        """Cloud live card: render both lines in place, provisional (dim) or
        final (normal). Batched by the overlay's flush, so this runs at most
        once per card per tick, not per token."""
        self._original = original
        self._translated = translation
        self._live_provisional = not final
        if final:
            self._settled = True
        # The producer re-sends the whole snapshot on every token, so the
        # same (original, translation, final) triple arrives repeatedly. The
        # short-circuit is what makes a one-frame flush cadence affordable:
        # a render happens per *change*, not per tick. It must sit after the
        # state assignment and check whether a streaming partial is being
        # cleared — that clearing is itself a render input.
        had_partial = self._streaming_partial is not None
        self._stop_streaming()
        state = (original, translation, final)
        if state == self._live_state and not had_partial:
            return
        self._live_state = state
        self._render()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2a2a3a; color: #ddd; border: 1px solid #555; }
            QMenu::item:selected { background: #444; }
            QMenu::separator { height: 1px; background: #555; margin: 4px 0; }
        """)
        copy_orig = menu.addAction(t("copy_original"))
        copy_trans = menu.addAction(t("copy_translation"))
        copy_all = menu.addAction(t("copy_all"))
        menu.addSeparator()
        export_menu = menu.addMenu(t("export_menu"))
        export_orig = export_menu.addAction(t("export_original"))
        export_trans = export_menu.addAction(t("export_translation"))
        export_both = export_menu.addAction(t("export_all"))
        menu.addSeparator()
        clear_list = menu.addAction(t("clear_list"))
        try:
            action = menu.exec(event.globalPos())
            # Dispatch inside the try: the handlers run before the menu's
            # release is even scheduled. The menu is already hidden once
            # exec returns, so nothing below depends on it staying alive.
            if action == copy_orig:
                QApplication.clipboard().setText(self._original)
            elif action == copy_trans:
                QApplication.clipboard().setText(self._translated)
            elif action == copy_all:
                QApplication.clipboard().setText(f"{self._original}\n{self._translated}")
            elif action == clear_list:
                overlay = self.window()
                if hasattr(overlay, '_on_clear'):
                    overlay._on_clear()
            elif action in (export_orig, export_trans, export_both):
                mode = {export_orig: "original", export_trans: "translation", export_both: "both"}[action]
                overlay = self.window()
                if hasattr(overlay, "export_messages"):
                    overlay.export_messages(mode, parent=self)
        finally:
            # The popup is per-invocation, but QMenu(self) transfers the C++
            # ownership to this widget — a dropped Python reference does NOT
            # free it, so every right-click would otherwise leave the menu
            # (and its actions) alive as a child until teardown. Cancel,
            # selection and exception paths all land here.
            menu.deleteLater()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_BTN_CSS = """
    QPushButton {
        background: rgba(42, 39, 34, 230);
        border: 1px solid rgba(174, 143, 91, 100);
        border-radius: 6px;
        color: #e8dfd2;
        font-size: 11px;
        padding: 2px 10px;
    }
    QPushButton:hover {
        background: rgba(83, 64, 39, 240);
        border-color: #e7b96f;
        color: #fff;
    }
    QPushButton:pressed { background: rgba(32, 28, 24, 240); }
"""

_DANGER_BTN_CSS = _BTN_CSS.replace(
    "rgba(42, 39, 34, 230)", "rgba(83, 43, 42, 225)"
).replace("rgba(174, 143, 91, 100)", "rgba(205, 112, 104, 135)")

_BAR_CSS_TPL = """
    QProgressBar {{
        background: rgba(255,255,255,15);
        border: 1px solid rgba(255,255,255,30);
        border-radius: 3px;
        text-align: center;
        font-size: 8pt;
        color: #aaa;
    }}
    QProgressBar::chunk {{
        background: {color};
        border-radius: 2px;
    }}
"""


class MonitorBar(QWidget):
    """Compact system monitor displayed in the overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # MIC bar (hidden when mic is disabled)
        self._mic_lbl = QLabel("MIC")
        self._mic_lbl.setFixedWidth(26)
        self._mic_lbl.setFont(QFont(default_mono_font_family(), 8))
        self._mic_lbl.setStyleSheet("color: #a59c8e; background: transparent;")
        self._mic_lbl.setVisible(False)
        row1.addWidget(self._mic_lbl)

        self._mic_bar = QProgressBar()
        self._mic_bar.setRange(0, 100)
        self._mic_bar.setFixedHeight(14)
        self._mic_bar.setTextVisible(True)
        self._mic_bar.setFormat("%v%")
        self._mic_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#c586c0"))
        self._mic_bar.setVisible(False)
        row1.addWidget(self._mic_bar)

        rms_lbl = QLabel("RMS:")
        rms_lbl.setFixedWidth(26)
        rms_lbl.setFont(QFont(default_mono_font_family(), 8))
        rms_lbl.setStyleSheet("color: #a59c8e; background: transparent;")
        row1.addWidget(rms_lbl)

        self._rms_bar = QProgressBar()
        self._rms_bar.setRange(0, 100)
        self._rms_bar.setFixedHeight(14)
        self._rms_bar.setTextVisible(True)
        self._rms_bar.setFormat("%v%")
        self._rms_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#9caf91"))
        row1.addWidget(self._rms_bar)

        vad_lbl = QLabel("VAD:")
        vad_lbl.setFixedWidth(26)
        vad_lbl.setFont(QFont(default_mono_font_family(), 8))
        vad_lbl.setStyleSheet("color: #a59c8e; background: transparent;")
        row1.addWidget(vad_lbl)

        self._vad_bar = QProgressBar()
        self._vad_bar.setRange(0, 100)
        self._vad_bar.setFixedHeight(14)
        self._vad_bar.setTextVisible(True)
        self._vad_bar.setFormat("%v%")
        self._vad_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#dcdcaa"))
        row1.addWidget(self._vad_bar)

        layout.addLayout(row1)

        self._stats_label = QLabel()
        self._stats_label.setFont(QFont(default_mono_font_family(), 8))
        self._stats_label.setStyleSheet("color: #aaa197; background: transparent;")
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(interval=None)  # Prime the counter
        self._cpu = 0
        self._ram_mb = 0.0
        self._gpu_text = "N/A"
        self._asr_device = ""
        self._asr_count = 0
        self._tl_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0
        # Segments the VAD threw away as noise. Shown only when non-zero: a
        # discard emits nothing at all, so without this the user sees the app
        # simply not reacting and has no way to tell "not heard" from
        # "heard and dropped".
        self._dropped = 0
        # Cloud-engine connection status (None = hidden, local engines).
        self._connection = None

        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._update_system)
        self._sys_timer.start(1000)
        self._update_system()
        self._refresh_stats()

    def update_audio(self, rms: float, vad: float, mic_rms=None, dropped=None):
        self._rms_bar.setValue(min(100, int(rms * 500)))
        self._vad_bar.setValue(min(100, int(vad * 100)))
        # Absolute count, never an increment: _flush_monitor coalesces, so any
        # delta passed here would be lost whenever two pushes land in one 80ms
        # window. None means "caller has nothing to report" (the pause/device
        # -switch zeroing calls), which must not clear a real count.
        if dropped is not None and dropped != self._dropped:
            self._dropped = dropped
            self._refresh_stats()
        mic_active = mic_rms is not None
        if self._mic_lbl.isVisible() != mic_active:
            self._mic_lbl.setVisible(mic_active)
            self._mic_bar.setVisible(mic_active)
        if mic_active:
            self._mic_bar.setValue(min(100, int(mic_rms * 500)))

    def update_asr_device(self, device: str):
        self._asr_device = device
        self._refresh_stats()

    def update_pipeline_stats(
        self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0
    ):
        self._asr_count = asr_count
        self._tl_count = tl_count
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost = cost
        self._refresh_stats()

    def _update_system(self):
        try:
            self._cpu = int(self._proc.cpu_percent(interval=None) / os.cpu_count())
            self._ram_mb = self._proc.memory_info().rss / 1024 / 1024
        except Exception:
            pass
        memory = accelerator_memory(self._asr_device)
        if memory:
            alloc, _, label = memory
            self._gpu_text = f"{label} {alloc:.0f}MB" if alloc else label
        else:
            self._gpu_text = "N/A"
        self._refresh_stats()

    def update_connection(self, status):
        """Cloud connection status: None hides the indicator (local modes
        are unchanged); otherwise a colored localized label."""
        status = status or None
        if status != self._connection:
            self._connection = status
            self._refresh_stats()

    _CONNECTION_STYLES = {
        # status-value -> (i18n key, color)
        "connecting": ("soniox_status_connecting", "#e7b96f"),
        "live": ("soniox_status_live", "#9caf91"),
        "reconnecting": ("soniox_status_reconnecting", "#e7b96f"),
        "paused": ("soniox_status_paused", "#9a9aa5"),
        "failed": ("soniox_status_failed", "#e06c75"),
    }

    def _refresh_stats(self):
        total = self._prompt_tokens + self._completion_tokens
        tokens_str = f"{total / 1000:.1f}k" if total >= 1000 else str(total)
        dev_str = ""
        if self._asr_device:
            dev_color = "#9caf91" if "cuda" in self._asr_device.lower() else "#e7b96f"
            dev_str = (
                f'<span style="color:{dev_color};">{self._asr_device}</span> '
                f'<span style="color:#555;">|</span> '
            )
        conn_str = ""
        if self._connection is not None:
            key, color = self._CONNECTION_STYLES.get(
                self._connection, ("soniox_status_connecting", "#e7b96f")
            )
            conn_str = (
                f'<span style="color:{color};">[{t(key)}]</span> '
                f'<span style="color:#555;">|</span> '
            )
        cost_str = ""
        if self._cost > 0:
            from i18n import get_lang
            symbol = "¥" if get_lang() == "zh" else "$"
            cost_str = f' <span style="color:#fa5;">{symbol}{self._cost:.4f}</span>'
        # Same short-English-label convention as CPU/ASR/TL/Tok above, so it
        # needs no i18n key. Absent while zero: the normal case must look
        # exactly as it did before.
        dropped_str = ""
        if self._dropped > 0:
            dropped_str = (
                f' <span style="color:#555;">|</span> '
                f'<span style="color:#e06c75;">Drop</span> {self._dropped}'
            )
        self._stats_label.setText(
            f"{dev_str}"
            f"{conn_str}"
            f'<span style="color:#e7b96f;">CPU</span> {self._cpu}% '
            f'<span style="color:#e7b96f;">RAM</span> {self._ram_mb:.0f}MB '
            f'<span style="color:#e7b96f;">GPU</span> {self._gpu_text} '
            f'<span style="color:#555;">|</span> '
            f'<span style="color:#9caf91;">ASR</span> {self._asr_count} '
            f'<span style="color:#e7b96f;">TL</span> {self._tl_count} '
            f'<span style="color:#c8a982;">Tok</span> {tokens_str} '
            f'<span style="color:#666;">({self._prompt_tokens}\u2191{self._completion_tokens}\u2193)</span>'
            f'{cost_str}'
            f'{dropped_str}'
        )


class _DragArea(QWidget):
    """Small draggable area (title + grip)."""

    drag_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self._drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        if self._drag_pos:
            self._drag_pos = None
            self.drag_finished.emit()


_COMBO_CSS = """
    QComboBox {
        background: rgba(40, 38, 34, 225);
        border: 1px solid rgba(174, 143, 91, 90);
        border-radius: 5px;
        color: #d8d0c2;
        font-size: 11px;
        padding: 1px 6px;
    }
    QComboBox:hover { background: rgba(76, 60, 40, 230); color: #fff; }
    QComboBox::drop-down { border: none; width: 14px; }
    QComboBox::down-arrow { image: none; border: none; }
    QComboBox QAbstractItemView {
        background: #242321; color: #ddd4c5; selection-background-color: #5a452d;
    }
"""

_CHECK_CSS = (
    "QCheckBox { color: #a59c8e; background: transparent; spacing: 5px; }"
    "QCheckBox:hover { color: #eee6d8; }"
    "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #756a59; border-radius: 3px; background: rgba(20,20,20,120); }"
    "QCheckBox::indicator:checked { background: #c98b42; border-color: #e7b96f; }"
)


class DragHandle(QWidget):
    """Top bar: row1=title+buttons, row2=checkboxes+combos."""

    settings_clicked = pyqtSignal()
    subtitle_clicked = pyqtSignal()
    click_through_toggled = pyqtSignal(bool)
    topmost_toggled = pyqtSignal(bool)
    auto_scroll_toggled = pyqtSignal(bool)
    taskbar_toggled = pyqtSignal(bool)
    target_language_changed = pyqtSignal(str)
    source_language_changed = pyqtSignal(str)
    model_changed = pyqtSignal(int)
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    hide_clicked = pyqtSignal()
    quit_clicked = pyqtSignal()
    # Meeting-recording lifecycle button ("End this recording" / "Start new
    # recording"). Emits with no payload; the app-side state machine is the
    # single authority and pushes the label/state back through
    # set_session_state().
    session_toggle_clicked = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    position_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "full"
        self.setFixedHeight(64)
        self.setStyleSheet("background: rgba(28, 27, 25, 238); border: 1px solid rgba(231, 185, 111, 105); border-radius: 9px;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(2)

        # Row 1: drag title + action buttons
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(3)

        drag = _DragArea()
        drag.drag_finished.connect(self.position_changed)
        drag.setStyleSheet("background: transparent;")
        drag_layout = QHBoxLayout(drag)
        drag_layout.setContentsMargins(0, 0, 4, 0)
        drag_layout.setSpacing(6)

        title = QLabel("\u2630 LiveTranslate")
        title.setFont(QFont(default_mono_font_family(), 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #e2d6c3; background: transparent;")
        drag_layout.addWidget(title)
        drag_layout.addStretch()
        row1.addWidget(drag, 1)

        def _btn(text, tip=None):
            b = QPushButton(text)
            b.setFixedHeight(22)
            b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            b.setFont(QFont(default_mono_font_family(), 8))
            b.setStyleSheet(_BTN_CSS)
            if tip:
                b.setToolTip(tip)
            return b

        hide_btn = _btn(t("hide"))
        hide_btn.clicked.connect(self.hide_clicked.emit)
        row1.addWidget(hide_btn)

        self._subtitle_btn = _btn(t("subtitle"))
        self._subtitle_btn.clicked.connect(self.subtitle_clicked.emit)
        row1.addWidget(self._subtitle_btn)

        self._running = False
        self._start_stop_btn = _btn(t("paused"))
        self._start_stop_btn.clicked.connect(self._on_start_stop)
        row1.addWidget(self._start_stop_btn)

        self._clear_btn = _btn(t("clear"))
        self._clear_btn.clicked.connect(self.clear_clicked.emit)
        row1.addWidget(self._clear_btn)

        # Meeting-session button: label and colour follow the app-level
        # session state (IDLE/ACTIVE/PAUSED/ENDING), never local state.
        self._session_state = "idle"
        self._session_btn = _btn(t("session_btn_start"))
        self._session_btn.clicked.connect(self.session_toggle_clicked.emit)
        row1.addWidget(self._session_btn)

        # Mode toggle button
        self._mode_btn = _btn(t("mode_full"))
        self._mode_btn.clicked.connect(self._toggle_mode)
        row1.addWidget(self._mode_btn)

        settings_btn = _btn(t("settings"))
        settings_btn.clicked.connect(self.settings_clicked.emit)
        row1.addWidget(settings_btn)

        quit_btn = _btn(t("quit"))
        quit_btn.setStyleSheet(_DANGER_BTN_CSS)
        quit_btn.clicked.connect(self.quit_clicked.emit)
        row1.addWidget(quit_btn)

        outer.addLayout(row1)

        # Row 2 area: checkboxes (row 2a) + model/lang combos (row 2b)
        self._row2_widget = QWidget()
        self._row2_widget.setStyleSheet("background: transparent;")
        row2_outer = QVBoxLayout(self._row2_widget)
        row2_outer.setContentsMargins(0, 0, 0, 0)
        row2_outer.setSpacing(2)

        # Row 2a: checkboxes
        row2a = QHBoxLayout()
        row2a.setContentsMargins(0, 0, 0, 0)
        row2a.setSpacing(6)

        self._ct_check = QCheckBox(t("click_through"))
        self._ct_check.setFont(QFont(default_mono_font_family(), 8))
        self._ct_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._ct_check.setStyleSheet(_CHECK_CSS)
        self._ct_check.toggled.connect(self.click_through_toggled.emit)
        row2a.addWidget(self._ct_check)

        self._topmost_check = QCheckBox(t("top_most"))
        self._topmost_check.setFont(QFont(default_mono_font_family(), 8))
        self._topmost_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._topmost_check.setStyleSheet(_CHECK_CSS)
        self._topmost_check.setChecked(True)
        self._topmost_check.toggled.connect(self.topmost_toggled.emit)
        row2a.addWidget(self._topmost_check)

        self._auto_scroll = QCheckBox(t("auto_scroll"))
        self._auto_scroll.setFont(QFont(default_mono_font_family(), 8))
        self._auto_scroll.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._auto_scroll.setStyleSheet(_CHECK_CSS)
        self._auto_scroll.setChecked(True)
        self._auto_scroll.toggled.connect(self.auto_scroll_toggled.emit)
        row2a.addWidget(self._auto_scroll)

        self._taskbar_check = QCheckBox(t("taskbar"))
        self._taskbar_check.setFont(QFont(default_mono_font_family(), 8))
        self._taskbar_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._taskbar_check.setStyleSheet(_CHECK_CSS)
        self._taskbar_check.setChecked(False)
        self._taskbar_check.toggled.connect(self.taskbar_toggled.emit)
        row2a.addWidget(self._taskbar_check)

        row2a.addStretch()
        row2_outer.addLayout(row2a)

        # Row 2b: model + source language + target language combos (stretch to fill)
        row2b = QHBoxLayout()
        row2b.setContentsMargins(0, 0, 0, 0)
        row2b.setSpacing(4)

        _lbl_css = "color: #a59c8e; background: transparent;"
        _lbl_font = QFont(default_mono_font_family(), 8)
        _combo_font = QFont(default_mono_font_family(), 8)

        model_lbl = QLabel(t("model_label"))
        model_lbl.setFont(_lbl_font)
        model_lbl.setStyleSheet(_lbl_css)
        row2b.addWidget(model_lbl)

        self._model_combo = QComboBox()
        self._model_combo.setFixedHeight(18)
        self._model_combo.setFont(_combo_font)
        self._model_combo.setStyleSheet(_COMBO_CSS)
        self._model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._model_combo.currentIndexChanged.connect(self.model_changed.emit)
        row2b.addWidget(self._model_combo, 3)

        src_lbl = QLabel(t("source_label"))
        src_lbl.setFont(_lbl_font)
        src_lbl.setStyleSheet(_lbl_css)
        row2b.addWidget(src_lbl)

        self._source_lang = QComboBox()
        self._source_lang.setFixedHeight(18)
        self._source_lang.setFont(_combo_font)
        self._source_lang.setStyleSheet(_COMBO_CSS)
        self._source_lang.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        for code, native in LANGUAGES:
            label = t("asr_lang_auto") if code == "auto" else native
            self._source_lang.addItem(f"{code} - {label}", code)
        self._source_lang.currentIndexChanged.connect(
            lambda idx: self.source_language_changed.emit(
                self._source_lang.currentData() or "auto"
            )
        )
        row2b.addWidget(self._source_lang, 2)

        tgt_lbl = QLabel(t("target_label"))
        tgt_lbl.setFont(_lbl_font)
        tgt_lbl.setStyleSheet(_lbl_css)
        row2b.addWidget(tgt_lbl)

        self._target_lang = QComboBox()
        self._target_lang.setFixedHeight(18)
        self._target_lang.setFont(_combo_font)
        self._target_lang.setStyleSheet(_COMBO_CSS)
        self._target_lang.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        for code, native in LANGUAGES:
            if code == "auto":
                continue
            self._target_lang.addItem(f"{code} - {native}", code)
        self._target_lang.currentIndexChanged.connect(
            lambda idx: self.target_language_changed.emit(
                self._target_lang.currentData() or "zh"
            )
        )
        row2b.addWidget(self._target_lang, 2)

        row2_outer.addLayout(row2b)

        outer.addWidget(self._row2_widget)

    def _on_start_stop(self):
        if self._running:
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    # Amber for paused, green for subtitle-on. These states used to be built
    # with .replace() against strings that were no longer in _BTN_CSS, which
    # silently produced the default stylesheet — a paused overlay and an
    # enabled subtitle button were indistinguishable from their normal state
    # (the subtitle button's text never changes, so the color was the only
    # feedback it had).
    _PAUSED_CSS = _BTN_CSS.replace(
        "rgba(42, 39, 34, 230)", "rgba(94, 70, 30, 235)"
    ).replace(
        "rgba(174, 143, 91, 100)", "rgba(231, 185, 111, 160)"
    ).replace(
        "color: #e8dfd2", "color: #ffe2ae"
    )
    _SUBTITLE_ON_CSS = _BTN_CSS.replace(
        "rgba(42, 39, 34, 230)", "rgba(38, 74, 46, 235)"
    ).replace(
        "rgba(174, 143, 91, 100)", "rgba(140, 200, 150, 150)"
    ).replace(
        "color: #e8dfd2", "color: #d6efdb"
    )
    # Session button states: recording=green, paused=amber, ending=light
    # progress look, idle=neutral grey. Same replace-chain pattern as above.
    _SESSION_ACTIVE_CSS = _BTN_CSS.replace(
        "rgba(42, 39, 34, 230)", "rgba(38, 74, 46, 235)"
    ).replace(
        "rgba(174, 143, 91, 100)", "rgba(140, 200, 150, 150)"
    ).replace(
        "color: #e8dfd2", "color: #d6efdb"
    )
    _SESSION_PAUSED_CSS = _BTN_CSS.replace(
        "rgba(42, 39, 34, 230)", "rgba(94, 70, 30, 235)"
    ).replace(
        "rgba(174, 143, 91, 100)", "rgba(231, 185, 111, 160)"
    ).replace(
        "color: #e8dfd2", "color: #ffe2ae"
    )
    _SESSION_ENDING_CSS = _BTN_CSS.replace(
        "rgba(42, 39, 34, 230)", "rgba(58, 54, 48, 235)"
    ).replace(
        "rgba(174, 143, 91, 100)", "rgba(231, 185, 111, 120)"
    ).replace(
        "color: #e8dfd2", "color: #e7b96f"
    )
    _SESSION_IDLE_CSS = _BTN_CSS.replace(
        "rgba(42, 39, 34, 230)", "rgba(50, 48, 45, 235)"
    ).replace(
        "rgba(174, 143, 91, 100)", "rgba(120, 112, 100, 90)"
    ).replace(
        "color: #e8dfd2", "color: #b0a89a"
    )

    # state -> (label key, css). ENDING is disabled: the close is running.
    _SESSION_BUTTONS = {
        "idle": ("session_btn_start", None),          # uses _SESSION_IDLE_CSS
        "active": ("session_btn_end", "_SESSION_ACTIVE_CSS"),
        "paused": ("session_btn_end", "_SESSION_PAUSED_CSS"),
        "ending": ("session_btn_ending", "_SESSION_ENDING_CSS"),
    }

    def set_session_state(self, state: str):
        """Reflect the app-level session state onto the header button."""
        self._session_state = state
        entry = self._SESSION_BUTTONS.get(state, self._SESSION_BUTTONS["idle"])
        label_key, css_attr = entry
        self._session_btn.setText(t(label_key))
        css = (
            self._SESSION_IDLE_CSS if css_attr is None
            else getattr(self, css_attr)
        )
        self._session_btn.setStyleSheet(css)
        # "Ending…" is a progress state, not an action — a second click must
        # not re-enter the end path (the state machine guards, the button
        # stays honest too).
        self._session_btn.setEnabled(state != "ending")

    def set_target_language(self, lang: str):
        idx = self._target_lang.findData(lang)
        if idx >= 0:
            self._target_lang.blockSignals(True)
            self._target_lang.setCurrentIndex(idx)
            self._target_lang.blockSignals(False)

    def set_source_language(self, lang: str):
        idx = self._source_lang.findData(lang)
        if idx >= 0:
            self._source_lang.blockSignals(True)
            self._source_lang.setCurrentIndex(idx)
            self._source_lang.blockSignals(False)

    def set_source_language_enabled(self, enabled: bool):
        self._source_lang.setEnabled(enabled)

    def set_target_language_enabled(self, enabled: bool):
        # blockSignals so a programmatic "zh" (the Soniox lock) does not
        # fire target_language_changed back into the app.
        self._target_lang.blockSignals(not enabled)
        self._target_lang.setEnabled(enabled)

    def set_models(self, models: list, active_index: int = 0):
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m.get("name", m.get("model", "?")))
        if 0 <= active_index < self._model_combo.count():
            self._model_combo.setCurrentIndex(active_index)
        self._model_combo.blockSignals(False)

    @property
    def auto_scroll(self) -> bool:
        return self._auto_scroll.isChecked()

    def set_running(self, running: bool):
        self._running = running
        if running:
            self._start_stop_btn.setText(t("running"))
            self._start_stop_btn.setStyleSheet(_BTN_CSS)
        else:
            self._start_stop_btn.setText(t("paused"))
            self._start_stop_btn.setStyleSheet(self._PAUSED_CSS)

    def _toggle_mode(self):
        new_mode = "compact" if self._mode == "full" else "full"
        self._apply_mode(new_mode)
        self.mode_changed.emit(new_mode)

    def _apply_mode(self, mode: str):
        self._mode = mode
        compact = mode == "compact"
        self._row2_widget.setVisible(not compact)
        self._clear_btn.setVisible(not compact)
        self._subtitle_btn.setVisible(not compact)
        self._session_btn.setVisible(not compact)
        self._mode_btn.setText(t("mode_compact") if compact else t("mode_full"))
        self.setFixedHeight(24 if compact else 62)

    def set_mode(self, mode: str):
        if mode != self._mode:
            self._apply_mode(mode)
            # Announce it, exactly like the toggle button does: callers that
            # switch the mode programmatically (a menu, a restored window
            # state) must get the same downstream effects as a click — the
            # monitor bar hides, the app's own mode_changed fires. The old
            # apply-without-emit left the header compact and the monitor bar
            # still showing.
            self.mode_changed.emit(mode)

    def set_subtitle_checked(self, checked: bool):
        self._subtitle_btn.setStyleSheet(
            self._SUBTITLE_ON_CSS if checked else _BTN_CSS
        )


class SubtitleOverlay(QWidget):
    """Chat-style overlay window for displaying live transcription."""

    add_message_signal = pyqtSignal(int, str, str, str, float, str)
    settle_live_signal = pyqtSignal(int, str, str, str, str)
    update_translation_signal = pyqtSignal(int, str, float)
    update_connection_signal = pyqtSignal(str)
    update_streaming_signal = pyqtSignal(int, str)
    clear_signal = pyqtSignal()
    # Monitor signals (thread-safe)
    update_stats_signal = pyqtSignal(int, int, int, int, float)
    update_asr_device_signal = pyqtSignal(str)

    settings_requested = pyqtSignal()
    target_language_changed = pyqtSignal(str)
    source_language_changed = pyqtSignal(str)
    model_switch_requested = pyqtSignal(int)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    session_toggle_requested = pyqtSignal()
    subtitle_toggled = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    position_changed = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self._config = config
        # Mirrors the header checkbox (default on) so showEvent can re-pin
        # the native always-on-top state without reading UI widgets.
        self._topmost_enabled = True
        self._messages = {}
        self._max_messages = 50
        # Messages rotated out of the view. The cap is deliberate (memory and
        # render cost); the export just has to be honest about it.
        self._messages_dropped = 0
        self._transcript_paths = {}
        self._click_through = False
        # Scroll-follow state: whether the view is pinned to the bottom, and
        # the last maximum seen (see _on_scroll_value_changed for why the live
        # one is the wrong thing to compare against).
        self._follow_bottom = True
        self._scroll_max = 0
        self._height_before_compact = None
        self._mode_anim = None
        self._pos_save_timer = QTimer(self)
        self._pos_save_timer.setSingleShot(True)
        self._pos_save_timer.setInterval(500)
        self._pos_save_timer.timeout.connect(lambda: self.position_changed.emit())
        self._last_saved_geo = None
        # The user's style as configured, and the window-derived factor
        # currently rendered from it. `_base_style` is the source the rescale
        # reads; it is never written with derived values.
        self._base_style = dict(DEFAULT_STYLE)
        self._font_scale = 1.0
        # The sizes currently rendered, so a resize can tell whether it would
        # change anything at all (see _schedule_font_scale).
        self._scaled_sizes = (
            DEFAULT_STYLE["original_font_size"],
            DEFAULT_STYLE["translation_font_size"],
        )
        self._font_scale_timer = QTimer(self)
        self._font_scale_timer.setSingleShot(True)
        self._font_scale_timer.setInterval(FONT_SCALE_DEBOUNCE_MS)
        self._font_scale_timer.timeout.connect(self._apply_font_scale)
        self._update_lock = threading.Lock()
        self._latest_monitor = None
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(80)
        self._monitor_timer.timeout.connect(self._flush_monitor)
        self._monitor_timer.start()
        self._streaming_updates = {}
        # Cloud live-card updates, batched by the same 50ms timer: keys are
        # msg_ids, values (original, translation, final) tuples.
        self._live_updates = {}
        # Last style actually rendered, so an unchanged one is a no-op.
        self._applied_style = None
        self._streaming_timer = QTimer(self)
        self._streaming_timer.setInterval(LIVE_FLUSH_INTERVAL_MS)
        self._streaming_timer.timeout.connect(self._flush_streaming)
        self._streaming_timer.start()
        self._setup_ui()

        self.add_message_signal.connect(self._on_add_message)
        self.settle_live_signal.connect(self._on_settle_live)
        self.update_translation_signal.connect(self._on_update_translation)
        self.update_connection_signal.connect(self._on_update_connection)
        self.update_streaming_signal.connect(self._on_update_streaming)
        self.clear_signal.connect(self._on_clear)
        self.update_stats_signal.connect(self._on_update_stats)
        self.update_asr_device_signal.connect(self._on_update_asr_device)
        # Settle the scale for the width we were constructed at. main.py only
        # calls apply_style when a style was ever saved, and it restores the
        # archived window size before that — so without this a user who never
        # touched the style page would see base sizes in a wider window until
        # the first resize.
        self._apply_font_scale(force=True)

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowTitle("LiveTranslate")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        width = 620
        height = 500
        x = geo.right() - width - 20
        y = geo.bottom() - height - 60
        self.setGeometry(x, y, width, height)
        self.setMinimumSize(480, 200)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._container = QWidget()
        self._container.setStyleSheet(
            "background-color: rgba(14, 15, 16, 235); border: 1px solid rgba(231, 185, 111, 110); border-radius: 12px;"
        )

        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(0)

        # Drag handle
        self._handle = DragHandle()
        self._handle.settings_clicked.connect(self.settings_requested.emit)
        self._handle.subtitle_clicked.connect(self.subtitle_toggled.emit)
        self._handle.click_through_toggled.connect(self._set_click_through)
        self._handle.topmost_toggled.connect(self._set_topmost)
        self._handle.taskbar_toggled.connect(self._set_taskbar)
        self._handle.target_language_changed.connect(self.target_language_changed.emit)
        self._handle.source_language_changed.connect(self.source_language_changed.emit)
        self._handle.model_changed.connect(self.model_switch_requested.emit)
        self._handle.start_clicked.connect(self.start_requested.emit)
        self._handle.stop_clicked.connect(self.stop_requested.emit)
        self._handle.hide_clicked.connect(self.hide_requested.emit)
        self._handle.clear_clicked.connect(self._on_clear)
        self._handle.quit_clicked.connect(self.quit_requested.emit)
        self._handle.session_toggle_clicked.connect(
            self.session_toggle_requested.emit
        )
        self._handle.mode_changed.connect(self._on_mode_changed)
        self._handle.position_changed.connect(self.position_changed)
        container_layout.addWidget(self._handle)

        # Monitor bar (collapsible)
        self._monitor = MonitorBar()
        container_layout.addWidget(self._monitor)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                width: 6px; background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,60); border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self._msg_container = QWidget()
        self._msg_container.setStyleSheet("background: transparent;")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(0, 0, 0, 0)
        self._msg_layout.setSpacing(2)
        self._msg_layout.addStretch()

        self._scroll.setWidget(self._msg_container)
        container_layout.addWidget(self._scroll)

        grip_row = QHBoxLayout()
        grip_row.addStretch()
        # Follow content that grows after layout (the immediate scroll above
        # can only see the previous maximum).
        scrollbar = self._scroll.verticalScrollBar()
        scrollbar.rangeChanged.connect(self._on_scroll_range_changed)
        scrollbar.valueChanged.connect(self._on_scroll_value_changed)

        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        self._grip.setStyleSheet("background: transparent;")
        grip_row.addWidget(self._grip)
        container_layout.addLayout(grip_row)

        main_layout.addWidget(self._container)

        self._ct_timer = QTimer(self)
        self._ct_timer.timeout.connect(self._check_click_through)
        self._ct_timer.start(50)

    def _schedule_pos_save(self):
        if not self.isVisible():
            return
        geo = (self.x(), self.y(), self.width(), self.height())
        if geo != self._last_saved_geo:
            self._last_saved_geo = geo
            self._pos_save_timer.start()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._schedule_pos_save()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_pos_save()
        self._schedule_font_scale()

    def set_running(self, running: bool):
        self._handle.set_running(running)

    def set_session_state(self, state: str):
        self._handle.set_session_state(state)

    def _set_topmost(self, enabled: bool):
        self._topmost_enabled = enabled
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()
        # setWindowFlags() recreates the native window on macOS, dropping the
        # Space-pinning this re-asserts (see set_always_on_top).
        set_always_on_top(self, enabled)

    def _set_taskbar(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags &= ~Qt.WindowType.Tool
        else:
            flags |= Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.show()

    def _set_click_through(self, enabled: bool):
        self._click_through = enabled
        if self.isVisible():
            try:
                set_click_through(self, enabled)
            except Exception:
                pass

    def _check_click_through(self):
        if not self._click_through:
            return
        cursor = QCursor.pos()
        local = self.mapFromGlobal(cursor)
        scroll_top = self._scroll.mapTo(self, QPoint(0, 0)).y()
        in_header = 0 <= local.x() <= self.width() and 0 <= local.y() < scroll_top
        try:
            set_click_through(self, not in_header)
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        # Every show is a chance for macOS to hand back a fresh native window
        # (taskbar/top-most toggles, tray "Show Overlay"), so re-assert the
        # native pin along with click-through.
        if self._topmost_enabled:
            QTimer.singleShot(0, lambda: set_always_on_top(self, True))
        if self._click_through:
            QTimer.singleShot(0, self._check_click_through)
        else:
            try:
                set_click_through(self, False)
            except Exception:
                pass

    def _on_mode_changed(self, mode: str):
        compact = mode == "compact"
        self._monitor.setVisible(not compact)
        self.mode_changed.emit(mode)

        # Animate window height
        if self._mode_anim and self._mode_anim.state() != QPropertyAnimation.State.Stopped:
            self._mode_anim.stop()

        if compact:
            self._height_before_compact = self.height()
            target_h = self.minimumHeight()
        else:
            target_h = self._height_before_compact or 500

        actual_h = self.frameGeometry().height()
        if abs(actual_h - target_h) < 10:
            self.resize(self.width(), target_h)
        else:
            anim = QPropertyAnimation(self, b"size")
            anim.setDuration(200)
            anim.setStartValue(QSize(self.width(), actual_h))
            anim.setEndValue(QSize(self.width(), target_h))
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._mode_anim = anim
            anim.start()

    def set_mode(self, mode: str):
        self._handle.set_mode(mode)

    def set_subtitle_checked(self, checked: bool):
        self._handle.set_subtitle_checked(checked)

    def _flush_monitor(self):
        with self._update_lock:
            latest = self._latest_monitor
            self._latest_monitor = None
        if latest is not None:
            self._monitor.update_audio(*latest)

    def _flush_streaming(self):
        # Runs every LIVE_FLUSH_INTERVAL_MS whether or not anything arrived.
        # Reading the dicts without the lock is safe here: a truth test on a
        # dict is atomic under the GIL, and the worst case is deferring a
        # just-arrived snapshot to the next tick. Taking the lock on every
        # idle tick would be pure overhead.
        if not self._streaming_updates and not self._live_updates:
            return
        with self._update_lock:
            updates = self._streaming_updates
            self._streaming_updates = {}
            live = self._live_updates
            self._live_updates = {}
        for msg_id, partial_text in updates.items():
            self._on_update_streaming(msg_id, partial_text)
        for msg_id, (original, translation, final) in live.items():
            msg = self._messages.get(msg_id)
            if msg is not None:
                msg.update_live(original, translation, final)
        if live:
            # Immediately, not in 50ms: the follow-up used to be a deferred
            # jump, so the text appeared first and the view snapped after.
            # Growth that only lands after layout is followed by the
            # scrollbar's rangeChanged handler.
            self._scroll_to_bottom()

    @pyqtSlot(str)
    def _on_update_connection(self, status):
        self._monitor.update_connection(status or None)

    @pyqtSlot(int, int, int, int, float)
    def _on_update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost):
        self._monitor.update_pipeline_stats(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    @pyqtSlot(str)
    def _on_update_asr_device(self, device: str):
        self._monitor.update_asr_device(device)

    @pyqtSlot(int, str, str, str, float, str)
    def _on_add_message(self, msg_id, timestamp, original, source_lang, asr_ms,
                         provider=""):
        self._discard_card(msg_id)
        msg = ChatMessage(
            msg_id, timestamp, original, source_lang, asr_ms, provider=provider
        )
        self._append_card(msg)

    def _discard_card(self, msg_id: int) -> None:
        """Remove an existing card for this id (duplicate defense): a re-add
        with the same id used to leak the old widget into the layout."""
        old = self._messages.pop(msg_id, None)
        if old is not None:
            self._msg_layout.removeWidget(old)
            old.deleteLater()

    def _append_card(self, msg: "ChatMessage") -> None:
        self._messages[msg.msg_id] = msg
        self._msg_layout.addWidget(msg)
        # Every card is created here (_on_add_message and _on_settle_live both
        # route through it), so this is the one place a new card picks up the
        # current scale. No class-level factor: that would leak into the
        # tests that construct ChatMessage directly.
        if self._font_scale != 1.0:
            msg.apply_font_scale(self._font_scale)
        if len(self._messages) > self._max_messages:
            oldest_id = min(self._messages.keys())
            old_msg = self._messages.pop(oldest_id)
            self._msg_layout.removeWidget(old_msg)
            old_msg.deleteLater()
            # Remember that the view is no longer the whole session, so an
            # export can say so instead of quietly handing over a partial log.
            self._messages_dropped += 1
        self._scroll_to_bottom()

    @pyqtSlot(int, str, str, str, str)
    def _on_settle_live(self, msg_id, timestamp, original, translation,
                        source_lang):
        """Cloud commit: settle the live card into its final render — the
        card created during provisionals is REUSED (never a second card);
        a segment that never showed provisionals gets its card created here
        already-final."""
        msg = self._messages.get(msg_id)
        if msg is None:
            msg = ChatMessage(
                msg_id, timestamp, original, source_lang, 0.0,
                provider="soniox",
            )
            self._append_card(msg)
        msg.update_live(original, translation, final=True)

    @pyqtSlot(int, str, float)
    def _on_update_translation(self, msg_id, translated, translate_ms):
        with self._update_lock:
            self._streaming_updates.pop(msg_id, None)
        msg = self._messages.get(msg_id)
        if msg:
            msg.set_translation(translated, translate_ms)
            self._scroll_to_bottom()

    def _on_update_streaming(self, msg_id, partial_text):
        msg = self._messages.get(msg_id)
        if msg:
            msg.update_streaming(partial_text)

    @pyqtSlot()
    def _on_clear(self):
        for msg in self._messages.values():
            self._msg_layout.removeWidget(msg)
            msg.deleteLater()
        self._messages.clear()
        # Everything before this point was removed on purpose, not rotated
        # out by the cap — a later export must not claim truncation for it.
        self._messages_dropped = 0

    def _scale_factor_for_width(self) -> float:
        if not self._base_style.get("scale_with_window", True):
            return 1.0
        return font_scale_for_width(self.width())

    def _scaled_sizes_for(self, factor: float) -> tuple[int, int]:
        """The (original, translation) point sizes a factor would produce."""
        s = self._base_style
        return (
            scaled_font_size(s["original_font_size"], factor),
            scaled_font_size(s["translation_font_size"], factor),
        )

    def _schedule_font_scale(self):
        """Queue a rescale only if the width would change an actual size.

        Width is the only input: the compact-mode height animation resizes
        with `QSize(self.width(), h)`, so it cannot land here at all. The
        guard compares *derived sizes*, not factors: the factor moves
        continuously, but a 23pt line only moves a point after ~27px of
        width, so an ordinary drag never starts the timer.
        """
        if self._scaled_sizes_for(self._scale_factor_for_width()) == self._scaled_sizes:
            return
        # No cards means nothing to reflow, and the debounce would only make
        # the first frame after startup carry the wrong size (main.py restores
        # the archived window size right after construction). Apply at once;
        # cards created later pick the scale up in _append_card.
        if not self._messages:
            self._apply_font_scale()
            return
        self._font_scale_timer.start()

    def _apply_font_scale(self, force: bool = False) -> None:
        """Re-lay out every card's fonts for the current window width.

        Touches fonts only — never the container/header stylesheets or the
        window opacity (see ChatMessage.apply_font_scale).

        `_base_style` is the user's style and `_font_scale` the derived
        factor; `_applied_style` and `ChatMessage._current_style` must keep
        holding the *base* style, or the next panel save would find them
        unequal, re-run the full ~70ms path, and persist a derived size.
        """
        factor = self._scale_factor_for_width()
        sizes = self._scaled_sizes_for(factor)
        if not force and sizes == self._scaled_sizes:
            self._font_scale = factor
            return
        self._font_scale = factor
        self._scaled_sizes = sizes
        for msg in self._messages.values():
            msg.apply_font_scale(factor)

    def _scroll_to_bottom(self):
        if not self._handle.auto_scroll:
            return
        sb = self._scroll.verticalScrollBar()
        # A setText that just happened has not been laid out yet, so maximum()
        # may still describe the previous content. Growth that lands after
        # this is followed by _on_scroll_range_changed.
        sb.setValue(sb.maximum())

    def _on_scroll_value_changed(self, value: int):
        # Compare against the last known maximum, never the live one: our own
        # setValue below runs inside the rangeChanged handler, where the live
        # maximum may already have grown — comparing against it would read as
        # "the user scrolled up" and silently drop the follow.
        self._follow_bottom = value >= self._scroll_max - 2

    def _on_scroll_range_changed(self, _minimum: int, maximum: int):
        self._scroll_max = maximum
        if self._handle.auto_scroll and self._follow_bottom:
            self._scroll.verticalScrollBar().setValue(maximum)

    def apply_style(self, style: dict):
        s = {**DEFAULT_STYLE, **style}
        # Migrate old single font_family to split fields
        if "font_family" in s and "original_font_family" not in style:
            s["original_font_family"] = s["font_family"]
            s["translation_font_family"] = s["font_family"]
        # The control panel emits the *whole* settings dict on every auto-save,
        # so this is called when the user changes a VAD slider or a checkbox
        # too. Re-rendering every message costs ~59ms with a full 50-message
        # buffer, on the Qt thread, for a style that did not change.
        if s == self._applied_style:
            return
        self._applied_style = dict(s)
        self._base_style = dict(s)
        # Container background
        bg_rgba = _hex_to_rgba(s["bg_color"], s["bg_opacity"])
        self._container.setStyleSheet(
            f"background-color: {bg_rgba}; border-radius: {s['border_radius']}px;"
        )
        # Header background
        hdr_rgba = _hex_to_rgba(s["header_color"], s["header_opacity"])
        self._handle.setStyleSheet(f"background: {hdr_rgba}; border-radius: 4px;")
        # Window opacity
        self.setWindowOpacity(s["window_opacity"] / 100.0)
        # Update all existing messages
        ChatMessage._current_style = s
        for msg in self._messages.values():
            msg.apply_style(s)
        # The base sizes may have changed while the factor did not, and this
        # runs on startup too (an archived window width is rarely exactly the
        # reference width) — so the first frame already carries the right
        # size instead of being corrected on the first resize.
        self._apply_font_scale(force=True)

    def export_messages(self, mode: str, parent=None):
        """Export captured messages to a .txt file.

        mode: "original" | "translation" | "both"
        """
        if not self._messages:
            QMessageBox.information(parent or self, "LiveTranslate", t("export_empty"))
            return

        from datetime import datetime
        suffix = {"original": "original", "translation": "translation", "both": "all"}.get(mode, "all")
        default_name = f"livetrans_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.txt"
        path, _ = QFileDialog.getSaveFileName(
            parent or self,
            t("export_dialog_title"),
            default_name,
            t("export_filter"),
        )
        if not path:
            return

        lines = []
        for msg_id in sorted(self._messages.keys()):
            msg = self._messages[msg_id]
            ts = msg._timestamp
            orig = msg._original or ""
            trans = msg._translated or ""
            if mode == "original":
                lines.append(f"[{ts}] {orig}")
            elif mode == "translation":
                if trans:
                    lines.append(f"[{ts}] {trans}")
            else:
                lines.append(f"[{ts}] {orig}")
                if trans:
                    lines.append(f"  -> {trans}")
                lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines).rstrip() + "\n")
        except OSError as e:
            QMessageBox.critical(
                parent or self,
                "LiveTranslate",
                t("export_failed").format(error=str(e)),
            )
            return

        if self._messages_dropped:
            QMessageBox.information(
                parent or self,
                "LiveTranslate",
                t("export_truncated").format(
                    count=len(self._messages),
                    paths="\n".join(
                        str(p) for p in self._transcript_paths.values()
                    ) or t("export_no_transcript"),
                ),
            )

    def set_transcript_paths(self, paths: dict):
        """Where the complete session log lives, for the truncation notice."""
        self._transcript_paths = dict(paths or {})

    # Thread-safe public API
    def add_message(self, msg_id, timestamp, original, source_lang, asr_ms,
                    provider=""):
        self.add_message_signal.emit(
            msg_id, timestamp, original, source_lang, asr_ms, provider
        )

    def update_live(self, msg_id, original, translation, final: bool):
        """Cloud live card: batched in-place update (never a new message per
        token). Drained by the 50ms _streaming_timer flush."""
        with self._update_lock:
            self._live_updates[msg_id] = (original, translation, final)

    def settle_live_message(self, msg_id, timestamp, original, translation,
                            source_lang="ru"):
        """Cloud commit: settle the live card to its final render on the Qt
        thread — reuses the provisional card when it exists (no duplicate
        card per segment), creates an already-final card otherwise."""
        self.settle_live_signal.emit(
            msg_id, timestamp, original, translation, source_lang
        )

    def update_connection(self, status):
        """Cloud connection status for the monitor bar; None hides it."""
        self.update_connection_signal.emit(status or "")

    def update_translation(self, msg_id, translated, translate_ms):
        with self._update_lock:
            self._streaming_updates.pop(msg_id, None)
        self.update_translation_signal.emit(msg_id, translated, translate_ms)

    def update_streaming(self, msg_id, partial_text):
        with self._update_lock:
            self._streaming_updates[msg_id] = partial_text

    def update_monitor(self, rms, vad_conf, mic_rms=None, dropped=None):
        """Called from the capture thread; drained by _flush_monitor on Qt's.

        ``dropped`` is VADProcessor.discarded_segments, an absolute count --
        see MonitorBar.update_audio for why it must not be a delta. Leave it
        None when the caller is only zeroing the level meters.
        """
        with self._update_lock:
            self._latest_monitor = (rms, vad_conf, mic_rms, dropped)

    def update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0):
        self.update_stats_signal.emit(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    def update_asr_device(self, device: str):
        self.update_asr_device_signal.emit(device)

    def set_target_language(self, lang: str):
        self._handle.set_target_language(lang)

    def set_source_language(self, lang: str):
        self._handle.set_source_language(lang)

    def set_source_language_enabled(self, enabled: bool):
        self._handle.set_source_language_enabled(enabled)

    def set_target_language_enabled(self, enabled: bool):
        self._handle.set_target_language_enabled(enabled)

    def set_models(self, models: list, active_index: int = 0):
        self._handle.set_models(models, active_index)

    def clear(self):
        # _on_clear does the work; it also resets the drop counter.
        self.clear_signal.emit()
