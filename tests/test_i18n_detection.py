"""UI language auto-detection.

The regression these cover: on macOS the POSIX locale variables do not carry
the system language at all. A .app/Finder launch inherits no LANG, and
Terminal.app sets one from its own profile, so a Mac whose System Settings
language is Chinese reported en_US (or nothing) through every POSIX channel
and the app started in English.
"""

import plistlib

import pytest

import i18n


@pytest.fixture(autouse=True)
def _clear_locale_env(monkeypatch):
    """POSIX variables must not leak between cases."""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        monkeypatch.delenv(var, raising=False)


def _on_macos(monkeypatch, languages):
    monkeypatch.setattr(i18n.sys, "platform", "darwin")
    monkeypatch.setattr(i18n, "_macos_ui_languages", lambda: languages)


def test_macos_system_language_wins_over_a_terminal_supplied_locale(monkeypatch):
    """LANG=en_US from Terminal.app is not a language preference."""
    _on_macos(monkeypatch, ["zh-Hans-CN"])
    monkeypatch.setenv("LANG", "en_US.UTF-8")

    assert i18n._detect_system_lang() == "zh"


def test_macos_gui_launch_without_any_locale_variable(monkeypatch):
    """A Finder/.app launch inherits no LANG; AppleLanguages still decides."""
    _on_macos(monkeypatch, ["zh-Hant-TW"])

    assert i18n._detect_system_lang() == "zh"


def test_macos_english_system_stays_english_despite_a_chinese_fallback(monkeypatch):
    """Only the first entry is the user's actual preference."""
    _on_macos(monkeypatch, ["en-US", "zh-Hans-CN"])

    assert i18n._detect_system_lang() == "en"


def test_macos_unsupported_system_language_falls_back_to_english(monkeypatch):
    _on_macos(monkeypatch, ["ja-JP"])

    assert i18n._detect_system_lang() == "en"


def test_macos_unreadable_preference_falls_back_to_the_environment(monkeypatch):
    """No AppleLanguages (sandbox, fresh account) must not lose a zh LANG."""
    _on_macos(monkeypatch, [])
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")

    assert i18n._detect_system_lang() == "zh"


def test_non_macos_platforms_keep_using_the_environment(monkeypatch):
    """The macOS probe must not run on Windows/Linux."""
    monkeypatch.setattr(i18n.sys, "platform", "win32")
    monkeypatch.setattr(
        i18n, "_macos_ui_languages", lambda: pytest.fail("probed AppleLanguages")
    )
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")

    assert i18n._detect_system_lang() == "zh"

    monkeypatch.setenv("LANG", "en_US.UTF-8")

    assert i18n._detect_system_lang() == "en"


def test_apple_languages_reads_the_preference_file(monkeypatch, tmp_path):
    prefs = tmp_path / ".GlobalPreferences.plist"
    prefs.write_bytes(plistlib.dumps({"AppleLanguages": ["zh-Hans-CN", "en-US"]}))
    monkeypatch.setattr(i18n, "_GLOBAL_PREFS", str(prefs))

    assert i18n._macos_ui_languages() == ["zh-Hans-CN", "en-US"]


def test_apple_languages_uses_defaults_when_the_file_is_missing(monkeypatch, tmp_path):
    """cfprefsd can hold a newer value than the file; defaults(1) is the fallback."""
    monkeypatch.setattr(i18n, "_GLOBAL_PREFS", str(tmp_path / "absent.plist"))

    class _Result:
        stdout = '(\n    "zh-Hans-CN",\n    "en-US"\n)\n'

    monkeypatch.setattr(i18n.subprocess, "run", lambda *a, **k: _Result())

    assert i18n._macos_ui_languages() == ["zh-Hans-CN", "en-US"]


def test_apple_languages_survives_a_failing_defaults_call(monkeypatch, tmp_path):
    monkeypatch.setattr(i18n, "_GLOBAL_PREFS", str(tmp_path / "absent.plist"))

    def _boom(*args, **kwargs):
        raise OSError("defaults unavailable")

    monkeypatch.setattr(i18n.subprocess, "run", _boom)

    assert i18n._macos_ui_languages() == []


def test_detection_never_raises_on_a_broken_probe(monkeypatch):
    """This runs at import time, before any UI exists."""
    monkeypatch.setattr(i18n.sys, "platform", "darwin")

    def _boom():
        raise RuntimeError("preference daemon exploded")

    monkeypatch.setattr(i18n, "_macos_ui_languages", _boom)

    assert i18n._detect_system_lang() == "en"
