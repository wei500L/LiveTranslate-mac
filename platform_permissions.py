"""Best-effort platform permission checks with explicit error categories."""

from __future__ import annotations

import sys


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

        # PyObjC's completion callback is asynchronous; this call only starts
        # the request.  A subsequent status check reflects the user's choice.
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio, lambda granted: None
        )
        return microphone_permission_status() != "denied"
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
        request_microphone_permission()
