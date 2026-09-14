"""Overlay UI tests for the Soniox cloud mode (offscreen Qt).

Covers: timestamp/latency-chip removal, provisional colors, the
one-live-card invariant (no per-token message creation), the MonitorBar
connection status and the no-translation hint.
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

#: Glyphs that must never appear in a card. Two cursors were shipped and
#: removed (the heavy block and the thin bar); keeping them named here turns
#: "provisional text carries no cursor" into a regression guard.
CURSOR = "│"
OLD_CURSOR = "▍"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_message(app, provider="", original="Привет", asr_ms=320.0):
    msg = subtitle_overlay.ChatMessage(
        1, "12:00:00", original, "ru", asr_ms, provider=provider
    )
    return msg


def test_header_has_no_timestamp_or_latency_chips(app):
    """The timestamp and ASR/TL latency chips were removed from the UI
    (timestamps stay available for export; latency lives in PERF logs)."""
    msg = make_message(app, provider="", original="Привет", asr_ms=320.0)
    html = msg._header_label.text()
    assert "12:00:00" not in html
    assert "ASR" not in html
    assert "320" not in html
    msg.set_translation("你好", 850.0)
    trans = msg._trans_label.text()
    assert "TL" not in trans and "850" not in trans
    assert "你好" in trans


def test_header_keeps_language_tag(app):
    msg = make_message(app, provider="soniox")
    html = msg._header_label.text()
    assert "RU ·" in html
    assert "[ru]" not in html
    assert "Привет" in html


def test_translation_uses_typographic_hierarchy_not_console_prompt(app):
    msg = make_message(app)
    msg.set_translation("你好", 100.0)
    assert "&gt;" not in msg._trans_label.text()


def test_provisional_live_uses_dim_colors(app):
    style = dict(subtitle_overlay.DEFAULT_STYLE)
    msg = make_message(app, provider="soniox")
    subtitle_overlay.ChatMessage._current_style = style
    msg.update_live("Привет", "你好", final=False)
    html_orig = msg._header_label.text()
    assert style["provisional_original_color"] in html_orig
    assert style["original_color"] not in html_orig
    html_trans = msg._trans_label.text()
    assert style["provisional_translation_color"] in html_trans


def test_final_live_uses_normal_colors(app):
    style = dict(subtitle_overlay.DEFAULT_STYLE)
    msg = make_message(app, provider="soniox")
    subtitle_overlay.ChatMessage._current_style = style
    msg.update_live("Привет", "你好世界", final=True)
    assert style["original_color"] in msg._header_label.text()
    assert style["translation_color"] in msg._trans_label.text()
    assert "你好世界" in msg._trans_label.text()


def test_empty_final_translation_shows_hint_not_translating(app):
    msg = make_message(app, provider="soniox")
    msg.update_live("Привет", "", final=True)
    html = msg._trans_label.text()
    from i18n import t

    assert t("soniox_no_translation") in html
    assert t("translating") not in html


def test_provisional_without_translation_shows_translating(app):
    msg = make_message(app, provider="soniox")
    msg.update_live("Привет", "", final=False)
    html = msg._trans_label.text()
    from i18n import t

    assert t("translating") in html


class _OverlayHarness:
    """Drives SubtitleOverlay's batched live-update path without a shown
    window: the facade + flush logic is what matters."""

    def __init__(self, app):
        self.overlay = subtitle_overlay.SubtitleOverlay({})
        self.app = app


def test_no_new_message_per_token(app):
    harness = _OverlayHarness(app)
    overlay = harness.overlay
    overlay.add_message(10, "12:00:00", "При", "ru", 0.0, provider="soniox")
    # N provisional updates for the same card.
    for text in ("Прив", "Приве", "Привет"):
        overlay.update_live(10, text, "", final=False)
    # Drain the batched updates (the 50ms timer would do this on its own).
    overlay._flush_streaming()
    assert len(overlay._messages) == 1
    # And a final render plus a new segment's card.
    overlay.update_live(10, "Привет", "你好", final=True)
    overlay._flush_streaming()
    overlay.add_message(11, "12:00:05", "Нов", "ru", 0.0, provider="soniox")
    assert len(overlay._messages) == 2


def test_monitor_bar_connection_status(app):
    bar = subtitle_overlay.MonitorBar()
    assert not bar._rms_lbl.isVisible()
    assert not bar._vad_lbl.isVisible()
    assert bar._connection is None
    bar.update_connection("live")
    assert bar._connection == "live"
    html = bar._stats_label.text()
    from i18n import t

    assert t("soniox_status_live") in html
    bar.update_connection("failed")
    assert t("soniox_status_failed") in bar._stats_label.text()
    bar.update_connection(None)
    assert bar._connection is None
    assert t("soniox_status_failed") not in bar._stats_label.text()


def test_monitor_expand_button_has_clear_state_and_hides_cloud_vad(app):
    bar = subtitle_overlay.MonitorBar()
    assert bar._details_btn._expanded is False

    bar.update_connection("live")
    bar._details_btn.click()
    assert bar._details_btn._expanded is True
    assert bar._rms_bar.isHidden() is False
    assert bar._vad_bar.isHidden() is True

    bar._details_btn.click()
    assert bar._details_btn._expanded is False
    assert bar._rms_bar.isHidden() is True


def test_local_diagnostics_still_show_vad_when_expanded(app):
    bar = subtitle_overlay.MonitorBar()
    bar.update_connection(None)
    bar._details_btn.click()
    assert bar._vad_bar.isHidden() is False


def test_compact_status_shows_latest_latency(app):
    bar = subtitle_overlay.MonitorBar()
    bar.update_latency("asr", 78.4)
    bar.update_latency("translation", 425.6)
    assert "ASR 78ms" in bar._status_label.text()
    assert "TL 426ms" in bar._status_label.text()
    assert bar._stats_label.isHidden()


def test_local_mode_shows_no_connection_status(app):
    bar = subtitle_overlay.MonitorBar()
    html = bar._stats_label.text()
    from i18n import t

    assert t("soniox_status_live") not in html


def test_transient_notice_does_not_create_subtitle_card(app):
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.show_notice("Soniox connection failed", timeout=1000)
    for _ in range(5):
        app.processEvents()
    assert not overlay._notice_label.isHidden()
    assert overlay._notice_label.text() == "Soniox connection failed"
    assert overlay._messages == {}
    overlay._notice_timer.stop()


def test_provider_survives_the_signal_path(app):
    """Regression: the stale 5-arg @pyqtSlot on _on_add_message truncated the
    6th (provider) argument, so cloud cards lost their provider identity and
    later showed the wrong no-translation hint. The decorator must match the
    signal's arity."""
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.add_message(7, "12:00:00", "Тест", "ru", 320.0, provider="soniox")
    for _ in range(5):
        app.processEvents()
    msg = overlay._messages.get(7)
    assert msg is not None
    # The provider arrived intact through the queued-signal path...
    assert msg._provider == "soniox"
    # ...and drives the cloud-specific no-translation hint on settle.
    msg.set_translation("", 0.0)
    from i18n import t

    assert t("soniox_no_translation") in msg._trans_label.text()
    assert "12:00:00" not in msg._header_label.text()


def test_settle_reuses_the_provisional_card(app):
    """The commit must settle the SAME card the provisionals created — the
    old add_message-on-commit path leaked a second card per segment (the
    stale provisional one stayed in the layout)."""
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.add_message(10, "12:00:00", "При", "ru", 0.0, provider="soniox")
    for _ in range(5):
        app.processEvents()
    overlay.settle_live_message(
        10, "12:00:00", "Привет. Как у тебя?", "你好。你好吗？"
    )
    overlay._flush_streaming()
    for _ in range(5):
        app.processEvents()
    # Exactly one card for this id in the dict AND the visible layout.
    assert len(overlay._messages) == 1
    card_widgets = [
        overlay._msg_layout.itemAt(i).widget()
        for i in range(overlay._msg_layout.count())
        if overlay._msg_layout.itemAt(i).widget() is not None
        and hasattr(overlay._msg_layout.itemAt(i).widget(), "msg_id")
    ]
    same_id = [w for w in card_widgets if w.msg_id == 10]
    assert len(same_id) == 1
    # And it is settled: final text, final colors, no cursor.
    html = same_id[0]._header_label.text()
    assert "Как у тебя?" in html
    assert CURSOR not in html and OLD_CURSOR not in html
    trans = same_id[0]._trans_label.text()
    assert "你好吗？" in trans
    assert "翻译中" not in trans and "translating" not in trans


def test_settle_without_provisional_card_creates_final_card(app):
    """A fast <end> with no provisionals still gets its (already-final)
    card."""
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.settle_live_message(11, "12:01:00", "Быстрая фраза", "快句")
    for _ in range(5):
        app.processEvents()
    msg = overlay._messages.get(11)
    assert msg is not None
    assert "Быстрая фраза" in msg._header_label.text()
    assert "快句" in msg._trans_label.text()
    assert "ASR" not in msg._header_label.text()


def test_provisional_text_carries_no_cursor(app):
    """Provisional state is expressed by the dim colors alone. Two cursor
    glyphs were shipped and removed (they crowded the last letter and, being
    part of the text run, wrapped onto a new line first) — neither may come
    back."""
    style = dict(subtitle_overlay.DEFAULT_STYLE)
    msg = make_message(app, provider="soniox")
    subtitle_overlay.ChatMessage._current_style = style
    msg.update_live("Говорю", "在讲", final=False)
    html = msg._header_label.text()
    assert CURSOR not in html and OLD_CURSOR not in html
    # ...but the provisional state IS visible, as the dim color.
    assert style["provisional_original_color"] in html
    msg.update_live("Говорю", "在讲", final=True)
    html = msg._header_label.text()
    assert CURSOR not in html and OLD_CURSOR not in html
    assert style["original_color"] in html


def test_local_card_initial_state_is_not_dim(app):
    """Local (non-cloud) cards never set _live_provisional: the original
    line must use the normal original_color, with no cursor — only the
    cloud live path dims."""
    style = dict(subtitle_overlay.DEFAULT_STYLE)
    msg = make_message(app, provider="")
    subtitle_overlay.ChatMessage._current_style = style
    html = msg._header_label.text()
    assert style["original_color"] in html
    assert style["provisional_original_color"] not in html
    assert "▍" not in html


def test_local_and_cloud_cards_share_the_render_path(app):
    """update_streaming/set_translation (local) and update_live (cloud) all
    funnel through _render(): after any mutation both lines' HTML can be
    regenerated purely from state — e.g. apply_style must not lose the
    provisional dim or a settled empty-translation hint."""
    style = dict(subtitle_overlay.DEFAULT_STYLE)
    subtitle_overlay.ChatMessage._current_style = style
    # Cloud provisional card survives a style re-apply with its dim colors.
    cloud = make_message(app, provider="soniox")
    cloud.update_live("Говорю", "在讲", final=False)
    assert style["provisional_original_color"] in cloud._header_label.text()
    cloud.apply_style(style)
    assert style["provisional_original_color"] in cloud._header_label.text()
    assert CURSOR not in cloud._header_label.text()
    assert style["provisional_translation_color"] in cloud._trans_label.text()
    # Local card: a streaming partial is on screen immediately (no second
    # throttle inside the card) and survives a style re-apply.
    local = make_message(app, provider="")
    local.update_streaming("正在翻")
    assert "正在翻" in local._trans_label.text()
    local.apply_style(style)
    assert "正在翻" in local._trans_label.text()
    # Settled-without-translation hint survives a style re-apply.
    local.set_translation("", 100.0)
    from i18n import t

    assert t("same_language") in local._trans_label.text()
    local.apply_style(style)
    assert t("same_language") in local._trans_label.text()


def test_duplicate_add_with_same_id_does_not_leak_widget(app):
    """The overlay's own duplicate defense: re-adding the same msg_id (which
    the old commit path did every segment) replaces the card instead of
    stacking a second widget."""
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.add_message(20, "12:00:00", "Первый", "ru", 0.0, provider="soniox")
    overlay.add_message(20, "12:00:01", "Второй", "ru", 0.0, provider="soniox")
    for _ in range(5):
        app.processEvents()
    assert len(overlay._messages) == 1
    card_widgets = [
        overlay._msg_layout.itemAt(i).widget()
        for i in range(overlay._msg_layout.count())
        if overlay._msg_layout.itemAt(i).widget() is not None
        and getattr(overlay._msg_layout.itemAt(i).widget(), "msg_id", None) == 20
    ]
    assert len(card_widgets) == 1


def test_latest_two_cards_have_reading_hierarchy(app):
    overlay = subtitle_overlay.SubtitleOverlay({})
    for msg_id in (1, 2, 3):
        overlay._on_add_message(msg_id, "12:00:00", str(msg_id), "en", 0.0)
    assert overlay._messages[1]._focus_role == "history"
    assert overlay._messages[2]._focus_role == "previous"
    assert overlay._messages[3]._focus_role == "current"


def test_error_is_a_retryable_state_not_translation_text(app):
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay._on_add_message(5, "12:00:00", "hello", "en", 0.0)
    overlay._on_update_translation(5, "[error: timeout]", 0.0)
    msg = overlay._messages[5]
    from i18n import t

    assert msg._status == "error"
    assert msg._translated == ""
    assert t("translation_failed") in msg._trans_label.text()
    assert t("retry_translation") in msg._trans_label.text()


def test_retry_reuses_the_same_card_and_returns_to_translating(app):
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.add_message(6, "12:00:00", "hello", "en", 0.0)
    app.processEvents()
    overlay.update_translation(6, "[error: timeout]", 0.0)
    app.processEvents()
    requested = []
    overlay.retry_translation_requested.connect(lambda *args: requested.append(args))

    card = overlay._messages[6]
    card._request_retry()

    assert requested == [(6, "hello", "en")]
    assert overlay._messages[6] is card
    assert card._status == "translating"
    from i18n import t
    assert t("translating") in card._trans_label.text()
    assert overlay._subtitle_state.get(6).status.value == "translating"


def test_soniox_error_card_never_offers_in_place_retry(app):
    overlay = subtitle_overlay.SubtitleOverlay({})
    overlay.add_message(
        16, "12:00:00", "ошибка", "ru", 0.0, provider="soniox"
    )
    app.processEvents()
    overlay.update_translation(16, "[error: timeout]", 0.0)
    app.processEvents()
    requested = []
    overlay.retry_translation_requested.connect(lambda *args: requested.append(args))

    card = overlay._messages[16]
    card._request_retry()

    from i18n import t

    assert card._status == "error"
    assert card._error_retryable is False
    assert t("retry_translation") not in card._trans_label.text()
    assert requested == []
    state = overlay._subtitle_state.get(16)
    assert state.provider == "soniox"
    assert state.retryable is False


def test_scroll_updates_respect_manual_history_reading(app):
    overlay = subtitle_overlay.SubtitleOverlay({})
    sb = overlay._scroll.verticalScrollBar()
    sb.setRange(0, 100)
    sb.setValue(20)
    overlay._follow_bottom = False
    overlay._scroll_to_bottom()
    assert sb.value() == 20

    overlay._on_auto_scroll_toggled(True)
    app.processEvents()
    assert overlay._follow_bottom is True
    assert sb.value() == sb.maximum()
