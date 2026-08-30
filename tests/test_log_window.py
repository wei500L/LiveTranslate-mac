"""The log viewer must render log lines verbatim.

QTextEdit.append() interprets its argument as rich text, and log lines are full
of angle brackets — model output above all, where a thinking model's
<think>...</think> block is exactly what someone opens the window to inspect.
"""

import pytest

pytest.importorskip("PyQt6", reason="log_window needs PyQt6")

log_window = pytest.importorskip("log_window")


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_angle_brackets_survive_rendering(app):
    window = log_window.LogWindow()
    window._append_log("Translate (100ms): <think>consider</think> done", 20)
    assert "<think>consider</think> done" in window._text.toPlainText()


def test_color_wrapping_does_not_change_the_text(app):
    window = log_window.LogWindow()
    window._append_log("plain & <simple> line", 20)
    assert "plain & <simple> line" in window._text.toPlainText()


def _rendered_colors(window):
    return window._text.toHtml()


def test_error_lines_keep_their_red(app):
    """Every line from the root logger carries the prefix "LiveTranslate: ",
    which ends in "Translate:" — matching that bare word for the translate
    highlight repainted ALL root-logger lines, errors included, in the
    translate blue, hiding exactly the color the reader scans for."""
    import logging

    window = log_window.LogWindow()
    window._append_log(
        "02:33 [ERROR] LiveTranslate: Start error: boom", logging.ERROR
    )
    html = _rendered_colors(window)
    assert "#f44747" in html  # red survives
    assert "#9cdcfe" not in html  # not repainted as a translate line


def test_warning_lines_keep_their_yellow(app):
    import logging

    window = log_window.LogWindow()
    window._append_log(
        "02:33 [WARNING] LiveTranslate: Translate error: boom", logging.WARNING
    )
    html = _rendered_colors(window)
    assert "#dcdcaa" in html
    assert "#9cdcfe" not in html


def test_result_line_highlights_match_the_logged_shape(app):
    """The translate highlight must match the message as it is actually
    logged — "Translate (410ms): ..." — not a guessed format."""
    import logging

    window = log_window.LogWindow()
    window._append_log(
        "02:33 [INFO] LiveTranslate: Translate (410ms): ok", logging.INFO
    )
    assert "#9cdcfe" in _rendered_colors(window)

    window2 = log_window.LogWindow()
    window2._append_log(
        "02:33 [INFO] LiveTranslate.ASRClient: ASR worker stopped", logging.INFO
    )
    # A line that merely mentions Translate-ish words is not a result line
    assert "#9cdcfe" not in _rendered_colors(window2)
