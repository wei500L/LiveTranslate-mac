"""Platform-aware Qt font family defaults."""

from __future__ import annotations

import sys


def default_ui_font_family(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    return ".AppleSystemUIFont" if platform_name == "darwin" else "Segoe UI"


def default_mono_font_family(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    return "Menlo" if platform_name == "darwin" else "Consolas"


def default_cjk_font_family(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    return "PingFang SC" if platform_name == "darwin" else "Microsoft YaHei"


def default_japanese_font_family(platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    if platform_name == "darwin":
        return "Hiragino Sans"
    return default_cjk_font_family(platform_name)


def font_defaults(platform_name: str | None = None) -> dict[str, str]:
    cjk = default_cjk_font_family(platform_name)
    return {
        "ui": default_ui_font_family(platform_name),
        "mono": default_mono_font_family(platform_name),
        "cjk": cjk,
        "japanese": default_japanese_font_family(platform_name),
    }
