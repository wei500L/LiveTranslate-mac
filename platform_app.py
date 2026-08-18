"""Small, lazy platform helpers for Qt application/window behavior.

The macOS AppKit import is deliberately kept inside functions so Linux and
Windows imports remain dependency-free.  The helpers are also usable from
offscreen tests where no native NSApplication is available.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("LiveTranslate.PlatformApp")


def is_macos(platform_name: str | None = None) -> bool:
    return (platform_name or sys.platform) == "darwin"


def set_dock_visible(visible: bool) -> bool:
    """Set the macOS Dock/Cmd-Tab policy; return False when unavailable."""
    if not is_macos():
        return False
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        from AppKit import NSApplicationActivationPolicyRegular

        policy = (
            NSApplicationActivationPolicyRegular
            if visible
            else NSApplicationActivationPolicyAccessory
        )
        app = NSApplication.sharedApplication()
        result = app.setActivationPolicy_(policy)
        return bool(result) if result is not None else True
    except (ImportError, AttributeError, RuntimeError) as exc:
        log.debug("macOS Dock policy unavailable: %s", exc)
        return False


def configure_application(app, *, dock_visible: bool = True) -> bool:
    """Apply cross-platform Qt defaults and the requested macOS Dock policy."""
    app.setQuitOnLastWindowClosed(False)
    return set_dock_visible(dock_visible)


def screen_available_geometry(widget=None):
    """Return the available geometry for a widget's current screen."""
    from PyQt6.QtWidgets import QApplication

    screen = None
    if widget is not None:
        screen = widget.screen()
        if screen is None:
            screen = QApplication.screenAt(widget.pos())
    screen = screen or QApplication.primaryScreen()
    return screen.availableGeometry() if screen else None


def position_is_visible(x: int, y: int, margin: int = 50) -> bool:
    """Check a saved top-left position against every attached display."""
    from PyQt6.QtWidgets import QApplication

    for screen in QApplication.screens():
        geo = screen.availableGeometry()
        if (
            geo.left() <= x + margin
            and x < geo.right()
            and geo.top() <= y + margin
            and y < geo.bottom()
        ):
            return True
    return False
