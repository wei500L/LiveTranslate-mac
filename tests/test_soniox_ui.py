"""Overlay UI tests for the Soniox cloud mode (offscreen Qt).

Covers: latency chip suppression by provider, provisional colors, the
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


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_message(app, provider="", original="Привет", asr_ms=320.0):
    msg = subtitle_overlay.ChatMessage(
        1, "12:00:00", original, "ru", asr_ms, provider=provider
    )
    return msg


def test_cloud_provider_hides_asr_chip(app):
    msg = make_message(app, provider="soniox")
    html = msg._header_label.text()
    assert "ASR" not in html
    assert "320" not in html


def test_local_provider_keeps_asr_chip(app):
    msg = make_message(app, provider="")
    html = msg._header_label.text()
    assert "ASR 320ms" in html


def test_cloud_provider_hides_tl_chip(app):
    msg = make_message(app, provider="soniox")
    msg.set_translation("你好", 850.0)
    html = msg._trans_label.text()
    assert "TL" not in html
    assert "850" not in html
    assert "你好" in html


def test_local_provider_keeps_tl_chip(app):
    msg = make_message(app, provider="")
    msg.set_translation("你好", 850.0)
    html = msg._trans_label.text()
    assert "TL 850ms" in html


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


def test_local_mode_shows_no_connection_status(app):
    bar = subtitle_overlay.MonitorBar()
    html = bar._stats_label.text()
    from i18n import t

    assert t("soniox_status_live") not in html
