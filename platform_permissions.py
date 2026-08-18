"""Best-effort platform permission checks with explicit error categories."""

from __future__ import annotations

import sys
import threading


class PlatformUnavailableError(RuntimeError):
    pass


class PermissionDeniedError(PermissionError):
    pass


class DeviceUnavailableError(RuntimeError):
    pass


class CaptureRuntimeError(RuntimeError):
    pass


def microphone_permission_status() -> str:
    """Return ``granted``, ``denied`` or ``unknown`` without prompting."""
    if sys.platform != "darwin":
        return "unknown"
    try:
        from AVFoundation import (
            AVCaptureDevice,
            AVMediaTypeAudio,
            AVAuthorizationStatusAuthorized,
            AVAuthorizationStatusDenied,
            AVAuthorizationStatusRestricted,
        )

        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        if status == AVAuthorizationStatusAuthorized:
            return "granted"
        if status in (AVAuthorizationStatusDenied, AVAuthorizationStatusRestricted):
            return "denied"
        return "unknown"
    except (ImportError, AttributeError):
        return "unknown"


def request_microphone_permission() -> bool:
    """Request microphone access on macOS; return the resulting best effort."""
    if sys.platform != "darwin":
        return True
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        # PyObjC invokes the completion handler asynchronously.  Do not let
        # callers initialize the audio backend until the user's choice is
        # available; otherwise first-launch permission prompts race PyAudio.
        completed = threading.Event()
        result = []

        def completion(granted):
            result.append(bool(granted))
            completed.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio, completion
        )
        if not completed.wait(timeout=60):
            # A callback timeout must not be treated as authorization.  A
            # status check handles environments where the callback is lost.
            return microphone_permission_status() == "granted"
        return result[0]
    except (ImportError, AttributeError) as exc:
        raise PlatformUnavailableError(
            "PyObjC AVFoundation is required for microphone permissions"
        ) from exc


def ensure_microphone_permission(*, request: bool = False) -> None:
    status = microphone_permission_status()
    if status == "denied":
        raise PermissionDeniedError(
            "Microphone access is denied; enable it in System Settings > Privacy & Security"
        )
    if request and status != "granted":
        if not request_microphone_permission():
            raise PermissionDeniedError(
                "Microphone access was not granted; enable it in System Settings > Privacy & Security"
            )


def screen_capture_permission_status() -> str:
    """Return the Screen Recording TCC state without displaying a prompt."""
    if sys.platform != "darwin":
        return "unknown"
    try:
        from Quartz import CGPreflightScreenCaptureAccess

        return "granted" if bool(CGPreflightScreenCaptureAccess()) else "denied"
    except (ImportError, AttributeError):
        # Older PyObjC releases do not expose the CGPreflight symbol.  Keep
        # this distinguishable from an explicit denial so callers can report
        # a dependency problem instead of claiming that access was denied.
        return "unknown"


def request_screen_capture_permission() -> bool:
    """Request Screen Recording access and return the best available result."""
    if sys.platform != "darwin":
        return True
    try:
        from Quartz import CGRequestScreenCaptureAccess

        requested = bool(CGRequestScreenCaptureAccess())
        if requested:
            return True
        return screen_capture_permission_status() == "granted"
    except (ImportError, AttributeError) as exc:
        raise PlatformUnavailableError(
            "PyObjC Quartz is required for Screen Recording permissions"
        ) from exc


def ensure_screen_capture_permission(*, request: bool = False) -> None:
    """Raise a typed error when Screen Recording access is unavailable."""
    status = screen_capture_permission_status()
    if status == "unknown":
        if sys.platform == "darwin" and request:
            # A few PyObjC versions expose the request symbol before the
            # preflight symbol.  Give those environments one explicit chance
            # to obtain access instead of failing before the system prompt.
            if request_screen_capture_permission():
                return
            raise PermissionDeniedError(
                "Screen Recording access was not granted; enable it in System Settings > Privacy & Security"
            )
        if sys.platform == "darwin":
            raise PlatformUnavailableError(
                "PyObjC Quartz does not expose Screen Recording permission APIs"
            )
        return
    if status == "denied" and request:
        if request_screen_capture_permission():
            return
    if status == "denied":
        raise PermissionDeniedError(
            "Screen Recording access is denied; enable it in System Settings > Privacy & Security"
        )
