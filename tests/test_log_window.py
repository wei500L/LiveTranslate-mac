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
