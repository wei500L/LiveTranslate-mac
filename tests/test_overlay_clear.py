"""The overlay's Clear button must reset the drop counter (B14).

The view caps itself at 50 messages and counts what it rotated out, so an
export can honestly say "this is only the tail of the session". Clear empties
the view on purpose — after it, the counter must be zero again, or every
later export warns about truncation and points the user at transcript files
of a session they deliberately discarded.
"""

import pytest

pytest.importorskip("PyQt6", reason="subtitle_overlay needs PyQt6")

subtitle_overlay = pytest.importorskip("subtitle_overlay")


@pytest.fixture(scope="module")
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def overlay(app):
    return subtitle_overlay.SubtitleOverlay({})


def _fill(overlay, count, start=0):
    for i in range(start, start + count):
        overlay._on_add_message(i, "00:00:00", f"msg {i}", "zh", 1.0)


def test_rotation_past_the_cap_counts_drops(overlay):
    _fill(overlay, 60)
    assert len(overlay._messages) == 50
    assert overlay._messages_dropped == 10


def test_clear_resets_the_drop_counter(overlay):
    """The Clear button used to leave the counter at its old value, so every
    later export claimed it had been truncated by the cap."""
    _fill(overlay, 60)
    overlay._on_clear()
    assert overlay._messages == {}
    assert overlay._messages_dropped == 0


def test_the_public_clear_signal_path_also_resets(overlay):
    _fill(overlay, 60)
    overlay.clear()
    assert overlay._messages_dropped == 0


def test_fresh_rotation_after_clear_counts_from_zero(overlay):
    _fill(overlay, 60)
    overlay._on_clear()
    _fill(overlay, 56, start=100)
    assert overlay._messages_dropped == 6


# --- toggle states must be visible (B15) ------------------------------------


def test_paused_button_is_distinguishable_from_running(app):
    """set_running(False) used to compute its highlighted stylesheet with
    .replace() calls whose targets no longer existed in _BTN_CSS, so paused
    looked exactly like running."""
    handle = subtitle_overlay.DragHandle()
    handle.set_running(True)
    running = handle._start_stop_btn.styleSheet()
    handle.set_running(False)
    paused = handle._start_stop_btn.styleSheet()
    assert paused != running


def test_subtitle_button_shows_its_toggled_state(app):
    """The Subtitle button's text never changes, so the color wash is the only
    feedback that the subtitle window is on. It was a no-op for the same
    stale-replace reason."""
    handle = subtitle_overlay.DragHandle()
    off = handle._subtitle_btn.styleSheet()
    handle.set_subtitle_checked(True)
    on = handle._subtitle_btn.styleSheet()
    assert on != off
    handle.set_subtitle_checked(False)
    assert handle._subtitle_btn.styleSheet() == off
