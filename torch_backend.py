"""Small capability layer shared by ASR, UI and worker cleanup code."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("LiveTranslate.Torch")


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def mps_available() -> bool:
    torch = _torch()
    try:
        return bool(torch and torch.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def normalize_device(device: str | None, *, for_ct2: bool = False) -> str:
    """Resolve a user-facing device, falling back safely when unavailable."""
    value = str(device or "").split(" (", 1)[0].strip().lower()
    if value.startswith("cuda"):
        torch = _torch()
        try:
            if torch is not None and torch.cuda.is_available():
                return value
        except (AttributeError, RuntimeError):
            pass
        return "cpu"
    if value == "mps":
        return "cpu" if for_ct2 or not mps_available() else "mps"
    return "cpu" if value in ("", "auto") else value


def available_devices() -> list[str]:
    if sys.platform == "darwin":
        return (["mps (Apple Silicon)"] if mps_available() else []) + ["cpu"]
    torch = _torch()
    devices = ["cpu"]
    try:
        if torch and torch.cuda.is_available():
            devices = [
                *(
                    f"cuda:{i} ({torch.cuda.get_device_name(i)})"
                    for i in range(torch.cuda.device_count())
                ),
                "cpu",
            ]
    except (AttributeError, RuntimeError):
        pass
    return devices


def device_supports_fp16(device: str) -> bool:
    value = str(device).lower()
    if not value.startswith("cuda"):
        return False
    torch = _torch()
    try:
        return bool(torch and torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def empty_cache(device: str | None = None) -> None:
    torch = _torch()
    if torch is None:
        return
    try:
        if device is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if mps_available() and hasattr(torch, "mps"):
                mps_empty = getattr(torch.mps, "empty_cache", None)
                if mps_empty:
                    mps_empty()
            return
        normalized = normalize_device(device)
        if normalized.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif normalized == "mps" and hasattr(torch, "mps"):
            empty = getattr(torch.mps, "empty_cache", None)
            if empty:
                empty()
    except (AttributeError, RuntimeError) as exc:
        log.debug("Unable to clear accelerator cache: %s", exc)


def accelerator_memory(device: str | None = None) -> tuple[float, float, str] | None:
    torch = _torch()
    if torch is None:
        return None
    normalized = normalize_device(device)
    try:
        if normalized.startswith("cuda") and torch.cuda.is_available():
            return (
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2,
                "CUDA",
            )
        if normalized == "mps":
            current = getattr(torch.mps, "current_allocated_memory", None)
            if current:
                used = current() / 1024**2
                return (used, used, "MPS")
            return (0.0, 0.0, "MPS")
    except (AttributeError, RuntimeError):
        return None
    return None
