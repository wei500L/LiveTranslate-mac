"""Putting a window in front of the user, from whatever state it is in.

Reported symptom: clicking Settings sometimes brought up no panel at all,
sometimes one that vanished again, sometimes one that came up greyed out
and unresponsive.

The first is the presentation path (platform_app.present_window and the
toggle keyed on window_is_foreground); the others are the MLX deferred-close
handshake in ControlPanel, whose flag used to outlive the user changing
their mind. Both halves are covered here.
"""

import pytest

pytest.importorskip("PyQt6", reason="control_panel needs PyQt6")


@pytest.fixture(scope="module")
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    """The real show/close handshake without ControlPanel.__init__'s config
    load, records page mount and background cache-scan thread."""
    from PyQt6.QtWidgets import QWidget

    import control_panel

    class _Panel(control_panel.ControlPanel):
        def __init__(self):
            QWidget.__init__(self)
            self._mlx_task = None
            self._mlx_health_task = None
            self._close_after_mlx_task = False
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            return super().close()

    return _Panel()


def test_reopening_the_panel_cancels_a_deferred_close(panel):
    """Closing during an MLX task leaves the panel disabled until the task
    finishes. Re-opening it has to undo that, or the user gets an inert
    window -- and the completion handler then closes it out from under them.
    """
    panel._close_after_mlx_task = True
    panel.setEnabled(False)

    panel.show()

    assert panel.isEnabled()
    assert panel._close_after_mlx_task is False


def test_deferred_close_fires_while_the_panel_is_still_up(panel, app):
    panel.show()
    assert panel.isVisible()
    panel._close_after_mlx_task = True

    panel._maybe_close_after_mlx_tasks()
    app.processEvents()

    assert panel.close_calls == 1


def test_deferred_close_spares_a_panel_the_user_hid_meanwhile(panel, app):
    """The user closed the panel, then hid it before the task finished. The
    pending close must not fire later and take the window with it."""
    panel.show()
    panel.hide()
    panel._close_after_mlx_task = True

    panel._maybe_close_after_mlx_tasks()
    app.processEvents()

    assert panel.close_calls == 0
    assert panel.isEnabled()
    assert panel._close_after_mlx_task is False


def test_native_window_lookup_degrades_on_the_offscreen_platform(app, monkeypatch):
    """The offscreen plugin fabricates a winId that is not an NSView pointer,
    and handing it to objc.objc_object() dereferences garbage -- a hard
    crash, not an exception any caller could catch. With no native window to
    control there is nothing to do but report unavailable, which is what
    keeps present_window() safe to call from a headless run."""
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QWidget

    import platform_clickthrough

    if QGuiApplication.platformName() != "offscreen":
        # The hazard is specific to a plugin that reports a winId with no
        # native window behind it; on a real display there is none to guard.
        pytest.skip("needs the offscreen platform")

    monkeypatch.setattr(platform_clickthrough.sys, "platform", "darwin")
    widget = QWidget()
    # Must be shown: an unshown widget has winId 0, which resolves to "no
    # NSWindow" and takes the ordinary graceful path instead of this one.
    widget.show()

    assert platform_clickthrough.set_visible_on_all_spaces(widget, True) is False
    with pytest.raises(platform_clickthrough.ClickThroughUnavailableError):
        platform_clickthrough._mac_ns_window(widget)
