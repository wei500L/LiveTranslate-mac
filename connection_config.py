"""Shared connection defaults and URL normalization.

Keeping endpoint defaults here prevents the UI, runtime, and helper clients
from silently drifting apart. Existing ``api_base`` and ``remote_asr_url``
settings remain supported for backward compatibility.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit


DEFAULT_TRANSLATION_API_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_REMOTE_ASR_URL = "http://127.0.0.1:8765"


def normalize_api_base(value: str | None) -> str:
    """Return an OpenAI-compatible base URL without duplicate slashes."""
    raw = str(value or "").strip() or DEFAULT_TRANSLATION_API_BASE
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def normalize_remote_asr_url(value: str | None) -> str:
    """Return the remote ASR server origin used by ``/health`` and ``/transcribe``."""
    raw = str(value or "").strip() or DEFAULT_REMOTE_ASR_URL
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def translation_api_base(configured: str | None = None) -> str:
    """Resolve the endpoint, allowing a one-line environment override."""
    return normalize_api_base(os.getenv("LIVETRANSLATE_API_BASE") or configured)


def translation_api_key(configured: str | None = None) -> str:
    return os.getenv("LIVETRANSLATE_API_KEY") or str(configured or "").strip()


def translation_model(configured: str | None = None) -> str:
    return os.getenv("LIVETRANSLATE_MODEL") or str(configured or "").strip()


def evaluate_model_list(model: str | None, model_ids) -> tuple[bool, str, dict]:
    """Turn a `/models` response into a UI-neutral connection outcome."""
    ids = {str(item) for item in (model_ids or []) if str(item)}
    if not ids:
        return False, "connection_models_empty", {}
    model = str(model or "").strip()
    if model and model not in ids:
        return False, "connection_model_missing", {"model": model}
    return True, "connection_success", {"count": len(ids)}


def classify_connection_error(error: Exception) -> tuple[str, dict]:
    """Map OpenAI/httpx failures to stable, translatable UI states."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    if name == "AuthenticationError" or status in (401, 403):
        return "connection_auth_failed", {}
    if name in ("APITimeoutError", "TimeoutException", "ReadTimeout") or isinstance(
        error, TimeoutError
    ):
        return "connection_timeout", {}
    if name in ("APIConnectionError", "ConnectError", "ConnectionError") or isinstance(
        error, ConnectionError
    ):
        return "connection_unreachable", {}
    if status == 404:
        return "connection_protocol_error", {}
    if status is not None:
        return "connection_server_error", {"status": status}
    return "connection_failed", {"error": str(error)}
