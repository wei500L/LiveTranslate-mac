"""Subtitle window queueing, wrapping and settings isolation (B4 / B8 / B9).

The Qt widgets need no QApplication for these: the FIFO logic, the text
splitter and the settings merge are all driven through stand-ins or on plain
data.
"""

import pytest

subtitle_window = pytest.importorskip(
    "subtitle_window", reason="subtitle_window needs PyQt6"
)
subtitle_settings = pytest.importorskip("subtitle_settings")


class FakeTimer:
    """Stands in for the single-shot QTimer so the queue can be driven by hand."""

    instances = []

    def __init__(self, _parent=None):
        self.interval = 0
        self.callback = None
        self.started = False
        self.deleted = False
        FakeTimer.instances.append(self)

    def setSingleShot(self, _value):
        pass

    def setInterval(self, ms):
        self.interval = ms

    @property
    def timeout(self):
        outer = self

        class Signal:
            def connect(self, fn):
                outer.callback = fn

        return Signal()

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def deleteLater(self):
        self.deleted = True

    def fire(self):
        self.started = False
        self.callback()


class Window:
    """Only the members the queue methods touch."""

    _on_update_text = subtitle_window.SubtitleWindow._on_update_text
    _drain_pending_sentences = subtitle_window.SubtitleWindow._drain_pending_sentences
    _on_segment_timer = subtitle_window.SubtitleWindow._on_segment_timer
    _clear_segment_timer = subtitle_window.SubtitleWindow._clear_segment_timer
    _cancel_pending_segments = subtitle_window.SubtitleWindow._cancel_pending_segments

    def __init__(self, min_display_ms=1500):
        self._pending_sentences = []
        self._segment_timer = None
        self._min_display_ms = min_display_ms
        self._last_insert_time = 0.0
        self.inserted = []

    def _insert_sentence(self, original, translations):
        self.inserted.append(original)
        # Mimic the real method: mark the moment so the next one has to wait.
        self._last_insert_time = 1_000_000.0


@pytest.fixture(autouse=True)
def _fake_timer(monkeypatch):
    FakeTimer.instances = []
    monkeypatch.setattr(subtitle_window, "QTimer", FakeTimer)
    monkeypatch.setattr(subtitle_window.time, "monotonic", lambda: 1000.0)


def _send(window, text):
    window._on_update_text(text, "{}")


def test_the_first_sentence_shows_immediately():
    window = Window()
    _send(window, "first")
    assert window.inserted == ["first"]


def test_a_burst_of_sentences_is_shown_in_order_not_collapsed():
    """New finals used to cancel the pending one, so everything that arrived
    inside a minimum-display window was dropped."""
    window = Window()
    _send(window, "first")
    _send(window, "second")
    _send(window, "third")

    assert window.inserted == ["first"]
    assert [s[0] for s in window._pending_sentences] == ["second", "third"]

    window._segment_timer.fire()
    assert window.inserted == ["first", "second"]
    window._segment_timer.fire()
    assert window.inserted == ["first", "second", "third"]
    assert window._pending_sentences == []


def test_only_one_timer_is_armed_at_a_time():
    window = Window()
    _send(window, "a")
    _send(window, "b")
    _send(window, "c")
    _send(window, "d")
    armed = [t for t in FakeTimer.instances if t.started]
    assert len(armed) == 1


def test_clear_drops_the_queue_and_its_timer():
    window = Window()
    _send(window, "a")
    _send(window, "b")
    timer = window._segment_timer
    window._cancel_pending_segments()
    assert window._pending_sentences == []
    assert window._segment_timer is None
    assert timer.deleted
    # A stale timer must not write into a cleared window.
    window.inserted.clear()
    window._on_segment_timer()
    assert window.inserted == []


# --- B8: wrapping ---------------------------------------------------------


class FakeMetrics:
    """Monospace-ish metrics: every character is 10 units wide."""

    def __init__(self, _font=None):
        self.calls = 0

    def horizontalAdvance(self, text):
        self.calls += 1
        return len(text) * 10


class Splitter:
    split_text = subtitle_window._SubtitleTextWidget.split_text

    def __init__(self, width):
        self._font = None
        self._outline_width = 0
        self._outline_enabled = False
        self._width = width

    def width(self):
        return self._width


def test_wrapping_measures_a_logarithmic_number_of_prefixes(monkeypatch):
    """Measuring every prefix made this O(n^2) glyph shaping on the Qt thread,
    for every subtitle update and every resize."""
    metrics = []

    def make(_font):
        m = FakeMetrics()
        metrics.append(m)
        return m

    monkeypatch.setattr(subtitle_window, "QFontMetrics", make)
    text = "x" * 400
    segments = Splitter(width=200).split_text(text)

    assert "".join(segments) == text
    # 400 chars at 20 per line: a linear scan would be ~8000 measurements.
    assert metrics[0].calls < 400


def test_wrapping_still_prefers_punctuation_boundaries(monkeypatch):
    monkeypatch.setattr(subtitle_window, "QFontMetrics", lambda _f: FakeMetrics())
    text = "hello world, goodbye world, hello again"
    segments = Splitter(width=200).split_text(text)
    assert segments[0].endswith(",")
    assert "".join(s.rstrip(" ") for s in segments).replace(" ", "") == text.replace(
        " ", ""
    )


def test_short_text_is_not_split(monkeypatch):
    monkeypatch.setattr(subtitle_window, "QFontMetrics", lambda _f: FakeMetrics())
    assert Splitter(width=500).split_text("short") == ["short"]


# --- B9: state isolation --------------------------------------------------


def test_merge_settings_does_not_alias_the_caller_s_lines():
    incoming = {"lines": [{"type": "original", "enabled": True}]}
    merged = subtitle_window._merge_settings(
        subtitle_window.DEFAULT_SUBTITLE_WIN_SETTINGS, incoming
    )
    merged["lines"][0]["enabled"] = False
    assert incoming["lines"][0]["enabled"] is True


def test_merge_settings_does_not_alias_the_defaults():
    a = subtitle_window._merge_settings(
        subtitle_window.DEFAULT_SUBTITLE_WIN_SETTINGS, {}
    )
    b = subtitle_window._merge_settings(
        subtitle_window.DEFAULT_SUBTITLE_WIN_SETTINGS, {}
    )
    assert a["lines"] is not b["lines"]
    if a["lines"]:
        assert a["lines"][0] is not b["lines"][0]


class _Control:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def color(self):
        return self._value

    def text(self):
        return self._value

    def isChecked(self):
        return self._value

    def currentData(self):
        return self._value


class SettingsWidget:
    """SubtitleSettingsWidget's read/emit methods over stubbed controls."""

    _collect_settings = subtitle_settings.SubtitleSettingsWidget._collect_settings
    _lines = subtitle_settings.SubtitleSettingsWidget._lines
    _emit_settings = subtitle_settings.SubtitleSettingsWidget._emit_settings
    get_settings = subtitle_settings.SubtitleSettingsWidget.get_settings
    emit_settings = subtitle_settings.SubtitleSettingsWidget.emit_settings

    def __init__(self):
        self.emitted = []
        self.settings_changed = type(
            "S", (), {"emit": lambda _s, v: self.emitted.append(v)}
        )()
        self._settings = {"lines": [{"type": "original", "enabled": True}]}
        self._spacing_spin = _Control(8)
        self._width_spin = _Control(1000)
        self._bg_color_btn = _Control("#000000")
        self._bg_opacity_spin = _Control(50)
        self._border_radius_spin = _Control(4)
        self._win_bg_image_edit = _Control("")
        self._auto_hide_spin = _Control(5)
        self._hide_anim_combo = _Control("fade")
        self._hide_duration_spin = _Control(200)
        self._click_through_check = _Control(False)


def test_get_settings_is_a_pure_read():
    """It used to call _emit_settings(), so reading the settings propagated all
    the way to SubtitleWindow.apply_settings() and rebuilt every text widget."""
    widget = SettingsWidget()
    result = widget.get_settings()
    assert result["window_width"] == 1000
    assert widget.emitted == []


def test_emit_settings_still_broadcasts():
    widget = SettingsWidget()
    widget.emit_settings()
    assert len(widget.emitted) == 1
    assert widget.emitted[0]["window_width"] == 1000


def test_the_emitted_object_does_not_share_state_with_the_controls():
    """Emitting the internal dict handed SubtitleWindow the very list the
    settings controls keep editing."""
    widget = SettingsWidget()
    widget.emit_settings()
    sent = widget.emitted[0]
    assert sent is not widget._settings
    assert sent["lines"] is not widget._settings["lines"]
    sent["lines"][0]["enabled"] = False
    assert widget._settings["lines"][0]["enabled"] is True


def test_the_dialog_exposes_both_operations():
    assert hasattr(subtitle_settings.SubtitleSettingsDialog, "get_settings")
    assert hasattr(subtitle_settings.SubtitleSettingsDialog, "emit_settings")


# --- The line list must not disagree with itself ----------------------------


class LinesWidget:
    """SubtitleSettingsWidget's line-list operations over a bare settings dict."""

    _lines = subtitle_settings.SubtitleSettingsWidget._lines

    def __init__(self, settings):
        self._settings = settings


def test_a_missing_lines_key_yields_the_defaults_not_an_empty_list():
    """The display fell back to the defaults while every mutation fell back to
    [], so the widget showed rows no operation could act on."""
    widget = LinesWidget({})
    lines = widget._lines()
    assert lines, "expected the default line set"
    assert widget._settings["lines"] is lines, "must be written back, not a copy"


def test_the_accessor_returns_the_live_list():
    settings = {"lines": [{"type": "original"}]}
    widget = LinesWidget(settings)
    widget._lines().append({"type": "translation"})
    assert len(settings["lines"]) == 2


def test_a_non_list_lines_value_is_replaced():
    widget = LinesWidget({"lines": "corrupted"})
    assert isinstance(widget._lines(), list)
    assert widget._lines(), "expected the defaults"


def test_the_defaults_are_copied_not_shared():
    a = LinesWidget({})._lines()
    b = LinesWidget({})._lines()
    assert a is not b
    a[0]["enabled"] = not a[0].get("enabled", True)
    assert b[0].get("enabled", True) != a[0]["enabled"] or len(a) != len(b)
    # And the module default must be untouched.
    assert (
        subtitle_settings.DEFAULT_SUBTITLE_WIN_SETTINGS["lines"][0].get("enabled", True)
        is True
    )


class MovableLines(LinesWidget):
    """_move_line_up/_down over a stubbed list widget."""

    _move_line_up = subtitle_settings.SubtitleSettingsWidget._move_line_up
    _move_line_down = subtitle_settings.SubtitleSettingsWidget._move_line_down

    class _List:
        def __init__(self, row):
            self._row = row

        def currentRow(self):
            return self._row

        def setCurrentRow(self, row):
            self._row = row

    def __init__(self, settings, row):
        super().__init__(settings)
        self._lines_list = self._List(row)
        self.emits = 0

    def _refresh_lines_list(self):
        pass

    def _schedule_emit(self):
        self.emits += 1


def _named(*names):
    return [{"type": "translation", "lang": n} for n in names]


def test_moving_a_line_up_swaps_it_with_its_predecessor():
    settings = {"lines": _named("a", "b", "c")}
    widget = MovableLines(settings, row=2)
    widget._move_line_up()
    assert [ln["lang"] for ln in settings["lines"]] == ["a", "c", "b"]
    assert widget._lines_list.currentRow() == 1


def test_moving_the_first_line_up_does_nothing():
    settings = {"lines": _named("a", "b")}
    widget = MovableLines(settings, row=0)
    widget._move_line_up()
    assert [ln["lang"] for ln in settings["lines"]] == ["a", "b"]
    assert widget.emits == 0


def test_moving_up_a_row_past_the_end_does_not_raise():
    """`row > 0` alone indexed past the end when the widget's row outran the
    backing list."""
    settings = {"lines": _named("a", "b")}
    widget = MovableLines(settings, row=5)
    widget._move_line_up()          # must not raise IndexError
    assert [ln["lang"] for ln in settings["lines"]] == ["a", "b"]


def test_moving_a_line_down_swaps_it_with_its_successor():
    settings = {"lines": _named("a", "b", "c")}
    widget = MovableLines(settings, row=0)
    widget._move_line_down()
    assert [ln["lang"] for ln in settings["lines"]] == ["b", "a", "c"]
    assert widget._lines_list.currentRow() == 1


def test_moving_the_last_line_down_does_nothing():
    settings = {"lines": _named("a", "b")}
    widget = MovableLines(settings, row=1)
    widget._move_line_down()
    assert [ln["lang"] for ln in settings["lines"]] == ["a", "b"]
    assert widget.emits == 0
