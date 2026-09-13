"""Small, lazy platform helpers for Qt application/window behavior.

The macOS AppKit import is deliberately kept inside functions so Linux and
Windows imports remain dependency-free.  The helpers are also usable from
offscreen tests where no native NSApplication is available.
"""

from __future__ import annotations

import logging
import sys

from platform_clickthrough import set_visible_on_all_spaces

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


def activate_app() -> bool:
    """Bring this application to the front; return whether it was applied.

    The overlay is deliberately shown without activating (it must never take
    focus from the video it is translating), so LiveTranslate is usually not
    the frontmost application when someone clicks Settings on it. A window
    raised while another application is frontmost is created *behind* that
    application -- visible to Qt, invisible to the user. Under the accessory
    activation policy (Dock icon hidden) the application cannot come forward
    on its own at all, so it has to be asked explicitly.
    """
    if not is_macos():
        return False
    try:
        from AppKit import NSApplication

        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        return True
    except (ImportError, AttributeError, RuntimeError) as exc:
        log.debug("macOS app activation unavailable: %s", exc)
        return False


def window_is_foreground(window) -> bool:
    """Whether a window is already somewhere the user can see and use it.

    A window that is merely *visible* still answers False here when it is
    minimized, covered by another application, or sitting on a different
    macOS Space. Every one of those states reads to the user as "it did not
    open", so a toggle keyed on visibility alone hides the very window the
    user is asking for -- and the reported symptom is exactly that: a click
    that appears to do nothing, with a second click that works. Each of
    these answered states must present the window instead of toggling it
    away.
    """
    return bool(
        window.isVisible()
        and not window.isMinimized()
        and window.isActiveWindow()
    )


def present_window(window, *, activate: bool = True) -> None:
    """Show a window and put it in front of the user, from whatever state.

    `show()` alone is what made Settings look broken: a minimized window
    stays minimized, a window opened while another application is fullscreen
    lands on the Space behind that fullscreen one (see
    set_visible_on_all_spaces), and nothing brings the application forward
    (see activate_app). Every step past `show()` is best-effort -- a missing
    native handle or an absent PyObjC must never cost us the show() that has
    already happened.

    Pass activate=False for a window that is already pinned on top of
    everything and is deliberately shown without focus (the subtitle overlay
    and subtitle window, which must never take focus from the video they are
    translating): those only need the minimize restore.
    """
    try:
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
    except Exception:
        log.debug("Could not show window", exc_info=True)
    # After show(): the native window has to exist before AppKit can be
    # asked to pin it to every Space.
    try:
        set_visible_on_all_spaces(window, True)
    except Exception:
        log.debug("Could not pin window to all Spaces", exc_info=True)
    try:
        window.raise_()
    except Exception:
        log.debug("Could not raise window", exc_info=True)
    if not activate:
        return
    try:
        window.activateWindow()
    except Exception:
        log.debug("Could not activate window", exc_info=True)
    activate_app()


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
