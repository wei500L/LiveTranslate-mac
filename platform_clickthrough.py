"""Cross-platform window mouse click-through helpers.

The platform-specific imports stay inside the functions so importing the UI on
Linux/macOS never attempts to load Win32 or PyObjC libraries.
"""

from __future__ import annotations

import sys


class ClickThroughUnavailableError(RuntimeError):
    """Raised when a native window handle cannot be controlled."""


def _window_handle(window):
    handle = window
    if hasattr(window, "windowHandle"):
        handle = window.windowHandle()
    if handle is None:
        raise ClickThroughUnavailableError("window handle is not available")
    return handle


def _mac_ns_window(window):
    native = _window_handle(window)
    if hasattr(native, "setIgnoresMouseEvents_"):
        return native
    try:
        import objc
        from ctypes import c_void_p
    except ImportError as exc:
        raise ClickThroughUnavailableError(
            "PyObjC AppKit is required for macOS click-through"
        ) from exc

    win_id = native.winId() if hasattr(native, "winId") else window.winId()
    view = objc.objc_object(c_void_p=int(win_id))
    ns_window = view.window() if hasattr(view, "window") else None
    if ns_window is None:
        raise ClickThroughUnavailableError("could not resolve NSWindow from Qt winId")
    return ns_window


def set_click_through(window, enabled: bool) -> bool:
    """Set native mouse transparency and return whether it was applied."""
    enabled = bool(enabled)
    if sys.platform == "win32":
        import ctypes

        hwnd = int(window.winId() if hasattr(window, "winId") else window)
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        transparent = 0x20
        new_style = style | transparent if enabled else style & ~transparent
        if new_style != style:
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, new_style)
        return True
    if sys.platform == "darwin":
        ns_window = _mac_ns_window(window)
        ns_window.setIgnoresMouseEvents_(enabled)
        return True
    return False


def get_click_through(window) -> bool | None:
    """Read native mouse transparency where the platform exposes it."""
    if sys.platform == "win32":
        import ctypes

        hwnd = int(window.winId() if hasattr(window, "winId") else window)
        return bool(ctypes.windll.user32.GetWindowLongW(hwnd, -20) & 0x20)
    if sys.platform == "darwin":
        ns_window = _mac_ns_window(window)
        if hasattr(ns_window, "ignoresMouseEvents"):
            return bool(ns_window.ignoresMouseEvents())
    return None


# Explicit aliases make the boundary convenient for callers that spell the
# feature as one word while keeping the readable canonical API above.
set_window_clickthrough = set_click_through
get_window_clickthrough = get_click_through


# AppKit window levels: Qt's WindowStaysOnTopHint maps to the floating level
# (3), which still sinks behind other floating panels. The modal-panel level
# (8) stays above those while staying below the menu bar (24) and the Dock.
_NS_MODAL_PANEL_WINDOW_LEVEL = 8
_NS_NORMAL_WINDOW_LEVEL = 0
_NS_WINDOW_COLLECTION_BEHAVIOR_CAN_JOIN_ALL_SPACES = 1 << 0
_NS_WINDOW_COLLECTION_BEHAVIOR_FULLSCREEN_AUXILIARY = 1 << 8


def set_always_on_top(window, enabled: bool) -> bool:
    """Pin a window above other applications' windows; return whether applied.

    On Windows Qt's WindowStaysOnTopHint is honored natively, so this is a
    no-op. On macOS the hint alone leaves two holes a subtitle overlay falls
    through during a meeting: a fullscreen app (Zoom, browsers) owns a
    separate Space the window never appears on, and an NSPanel can hide when
    the app deactivates. Joining all Spaces with the auxiliary fullscreen
    behavior, a level above floating, and hidesOnDeactivate=False restores
    the Windows-style always-on-top.
    """
    if sys.platform != "darwin":
        return False
    try:
        ns_window = _mac_ns_window(window)
    except Exception:
        # No native window yet (created lazily), or PyObjC missing: the Qt
        # hint still applies, just without the Space pinning.
        return False
    spaces = (
        _NS_WINDOW_COLLECTION_BEHAVIOR_CAN_JOIN_ALL_SPACES
        | _NS_WINDOW_COLLECTION_BEHAVIOR_FULLSCREEN_AUXILIARY
    )
    try:
        behavior = int(ns_window.collectionBehavior())
        if enabled:
            ns_window.setLevel_(_NS_MODAL_PANEL_WINDOW_LEVEL)
            if hasattr(ns_window, "setHidesOnDeactivate_"):
                ns_window.setHidesOnDeactivate_(False)
            ns_window.setCollectionBehavior_(behavior | spaces)
        else:
            ns_window.setLevel_(_NS_NORMAL_WINDOW_LEVEL)
            ns_window.setCollectionBehavior_(behavior & ~spaces)
    except Exception:
        return False
    return True
