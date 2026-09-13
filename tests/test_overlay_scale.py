"""Window-driven font scaling in the overlay (offscreen Qt).

The invariant under test throughout: the *base* style (what the settings
panel owns and what gets persisted) and the *derived* size (base x window
factor, what the cards render) are separate things. Letting a derived value
reach the base makes each rescale multiply an already-scaled size.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

subtitle_overlay = pytest.importorskip(
    "subtitle_overlay", reason="subtitle_overlay needs PyQt6"
)

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_overlay(app, width=620):
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.resize(width, 500)
    return overlay


def add_card(overlay, msg_id=1):
    overlay.add_message(msg_id, "12:00:00", "Точная верхняя грань", "ru", 0.0)
    overlay.update_translation(msg_id, "上确界", 100.0)
    for _ in range(3):
        QApplication.processEvents()
    return overlay._messages[msg_id]


def settle_scale(overlay):
    """Run the debounce path synchronously."""
    overlay._font_scale_timer.stop()
    overlay._apply_font_scale()


def test_default_layout_uses_the_reference_sizes(app):
    overlay = make_overlay(app, 620)
    card = add_card(overlay)
    assert overlay._font_scale == 1.0
    assert card._trans_label.font().pointSize() == 23
    assert card._header_label.font().pointSize() == 12


def test_a_wider_window_scales_both_lines(app):
    overlay = make_overlay(app, 620)
    card = add_card(overlay)
    overlay.resize(900, 500)
    settle_scale(overlay)
    assert card._header_label.font().pointSize() == 17
    assert card._trans_label.font().pointSize() == 33


def test_scale_growth_is_capped(app):
    overlay = make_overlay(app, 620)
    card = add_card(overlay)
    overlay.resize(3000, 500)
    settle_scale(overlay)
    assert overlay._font_scale == subtitle_overlay.FONT_SCALE_MAX
    assert card._trans_label.font().pointSize() == 41
    # And a resize far past the cap changes nothing further.
    overlay.resize(4000, 500)
    settle_scale(overlay)
    assert card._trans_label.font().pointSize() == 41


def test_apply_style_stores_base_sizes_not_scaled(app):
    """The core regression: a rescale must not feed its derived sizes back
    into the style the panel owns, or the next rescale multiplies an already
    multiplied value (23 -> 41 -> 74)."""
    overlay = make_overlay(app, 620)
    card = add_card(overlay)
    overlay.resize(900, 500)
    settle_scale(overlay)
    assert card._trans_label.font().pointSize() == 33  # derived, on screen

    # The panel now sends the same style again (any auto-save does this).
    overlay.apply_style(dict(subtitle_overlay.DEFAULT_STYLE))
    assert overlay._applied_style["translation_font_size"] == 23
    assert overlay._base_style["translation_font_size"] == 23
    assert subtitle_overlay.ChatMessage._current_style["translation_font_size"] == 23

    # Repeated forced rescales stay put instead of snowballing.
    for _ in range(3):
        overlay._apply_font_scale(force=True)
    assert card._trans_label.font().pointSize() == 33


def test_new_card_gets_the_current_scale(app):
    overlay = make_overlay(app, 620)
    overlay.resize(900, 500)
    settle_scale(overlay)
    card = add_card(overlay, msg_id=2)
    assert card._trans_label.font().pointSize() == 33


def test_scaling_does_not_touch_stylesheets_or_opacity(app):
    """The resize path is fonts only. The ~70ms cost of a full apply_style
    lives in the two setStyleSheet calls (subtree re-polish) and
    setWindowOpacity (compositor on macOS) — none of which a rescale needs."""
    overlay = make_overlay(app, 620)
    card = add_card(overlay)
    calls = {"container": 0, "handle": 0, "opacity": 0}

    original_container = overlay._container.setStyleSheet
    original_handle = overlay._handle.setStyleSheet
    original_opacity = overlay.setWindowOpacity

    def counting_container(*a, **k):
        calls["container"] += 1
        return original_container(*a, **k)

    def counting_handle(*a, **k):
        calls["handle"] += 1
        return original_handle(*a, **k)

    def counting_opacity(*a, **k):
        calls["opacity"] += 1
        return original_opacity(*a, **k)

    overlay._container.setStyleSheet = counting_container
    overlay._handle.setStyleSheet = counting_handle
    overlay.setWindowOpacity = counting_opacity
    try:
        overlay.resize(900, 500)
        settle_scale(overlay)
        overlay.resize(1200, 500)
        settle_scale(overlay)
    finally:
        overlay._container.setStyleSheet = original_container
        overlay._handle.setStyleSheet = original_handle
        overlay.setWindowOpacity = original_opacity

    assert calls == {"container": 0, "handle": 0, "opacity": 0}
    assert card._trans_label.font().pointSize() == 41  # the scale did happen


def test_a_resize_that_changes_no_derived_size_is_a_noop(app):
    overlay = make_overlay(app, 620)
    add_card(overlay)
    calls = []
    original = subtitle_overlay.ChatMessage.apply_font_scale
    subtitle_overlay.ChatMessage.apply_font_scale = (
        lambda self, factor: calls.append(factor)
    )
    try:
        overlay.resize(626, 500)
        overlay._font_scale_timer.stop()
        # The scheduler must not even arm the timer for a sub-point change.
        overlay._schedule_font_scale()
        assert not overlay._font_scale_timer.isActive()
        overlay._apply_font_scale()
    finally:
        subtitle_overlay.ChatMessage.apply_font_scale = original
    assert calls == []


def test_a_drag_is_debounced_into_a_single_rescale(app):
    overlay = make_overlay(app, 620)
    add_card(overlay)
    calls = []
    original = subtitle_overlay.ChatMessage.apply_font_scale
    subtitle_overlay.ChatMessage.apply_font_scale = (
        lambda self, factor: calls.append(factor)
    )
    try:
        for width in range(700, 900, 20):  # a drag
            overlay.resize(width, 500)
            overlay._schedule_font_scale()
        # Every width armed the timer, but nothing has fired yet — the whole
        # drag is still one pending rescale.
        assert calls == []
        overlay._font_scale_timer.stop()
        overlay._apply_font_scale()   # patched, so this is what records
    finally:
        subtitle_overlay.ChatMessage.apply_font_scale = original
    assert len(calls) == 1


def test_height_only_resize_never_rescales(app):
    """The compact animation drives b"size" with an unchanged width; scaling
    keys off width alone, so it cannot fire from there."""
    overlay = make_overlay(app, 620)
    add_card(overlay)
    calls = []
    original = subtitle_overlay.ChatMessage.apply_font_scale
    subtitle_overlay.ChatMessage.apply_font_scale = (
        lambda self, factor: calls.append(factor)
    )
    try:
        for height in (400, 300, 260, 200):
            overlay.resize(620, height)
    finally:
        subtitle_overlay.ChatMessage.apply_font_scale = original
    assert calls == []
    assert not overlay._font_scale_timer.isActive()


def test_switch_off_uses_the_base_sizes_at_any_width(app):
    overlay = make_overlay(app, 620)
    card = add_card(overlay)
    overlay.apply_style({**subtitle_overlay.DEFAULT_STYLE, "scale_with_window": False})
    overlay.resize(900, 500)
    settle_scale(overlay)
    assert overlay._font_scale == 1.0
    assert card._header_label.font().pointSize() == 12
    assert card._trans_label.font().pointSize() == 23
    # And the switch itself survives the style merge into the base.
    assert overlay._base_style["scale_with_window"] is False
