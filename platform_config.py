"""Pure configuration migration and platform-default resolution."""

from __future__ import annotations

import sys

from platform_fonts import default_cjk_font_family


def normalize_config(
    config: dict,
    *,
    platform_name: str | None = None,
    mps_is_available: bool = False,
    cuda_is_available: bool = False,
) -> dict:
    """Normalize legacy settings in place while preserving unknown fields."""
    platform_name = platform_name or sys.platform
    audio = config.setdefault("audio", {})
    asr = config.setdefault("asr", {})
    subtitle = config.setdefault("subtitle", {})

    audio.setdefault("system_audio", "auto")
    audio.setdefault("mic_device", "auto")
    if audio.get("device") == "__disabled__":
        audio["system_audio"] = "disabled"
    elif audio.get("system_audio") == "auto":
        audio["system_audio"] = "enabled" if platform_name == "win32" else "disabled"
    if audio.get("mic_device") == "auto":
        audio["mic_device"] = "__default__" if platform_name == "darwin" else None

    selected = str(asr.get("device") or "auto").split(" (", 1)[0].lower()
    if selected == "auto" or (platform_name == "darwin" and selected.startswith("cuda")):
        if platform_name == "darwin" and mps_is_available:
            selected = "mps"
        elif platform_name == "win32" and cuda_is_available:
            selected = "cuda"
        else:
            selected = "cpu"
    elif selected == "mps" and (platform_name != "darwin" or not mps_is_available):
        selected = "cpu"
    elif selected.startswith("cuda") and not cuda_is_available:
        selected = "cpu"
    asr["device"] = selected
    if subtitle.get("font_family") in (None, "", "auto"):
        subtitle["font_family"] = default_cjk_font_family(platform_name)
    return config
