import locale
import logging
import os
import plistlib
import re
import subprocess
import sys

import yaml
from pathlib import Path

log = logging.getLogger("LiveTranslate.i18n")

_strings: dict = {}
_lang = "en"
_dir = Path(__file__).parent / "i18n"


_GLOBAL_PREFS = "~/Library/Preferences/.GlobalPreferences.plist"


def _macos_ui_languages() -> list[str]:
    """Return macOS's AppleLanguages preference order, empty list if unreadable.

    This is the only place the *system* language lives on macOS. POSIX locale
    variables do not carry it: a GUI launch (Finder, .app bundle) inherits no
    LANG at all, and Terminal.app sets one from its own profile rather than
    from System Settings -- so a Chinese Mac reports en_US or nothing through
    every POSIX channel. Reading the preference file directly costs ~1ms;
    `defaults read` is the fallback for the case where cfprefsd holds a newer
    value than the file on disk (a language changed since login).
    """
    path = Path(os.path.expanduser(_GLOBAL_PREFS))
    try:
        if path.exists():
            data = plistlib.loads(path.read_bytes())
            langs = data.get("AppleLanguages")
            if isinstance(langs, list) and langs:
                return [str(item) for item in langs]
    except Exception:
        log.debug("Reading %s failed; falling back to defaults(1)", path, exc_info=True)
    try:
        output = subprocess.run(
            ["defaults", "read", "-g", "AppleLanguages"],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout
        return re.findall(r'"?([A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*)"?\s*(?:,|\))', output)
    except Exception:
        log.debug("defaults read AppleLanguages failed", exc_info=True)
    return []


def _detect_system_lang() -> str:
    """Return 'zh' if the system language is Chinese, else 'en'.

    macOS is checked first and on its own terms: AppleLanguages is the setting
    the user actually changed in System Settings, while the POSIX variables
    below are either absent (GUI launch) or a terminal emulator's own default.
    Consulting the environment first made every Chinese Mac start in English.

    Elsewhere the environment is authoritative. locale.getdefaultlocale() is
    deprecated since 3.11 and slated for removal in 3.15, so read the
    environment first and only fall back to the locale module's supported API.
    """
    try:
        if sys.platform == "darwin":
            for tag in _macos_ui_languages():
                primary = tag.lower().split("-")[0]
                if primary == "zh":
                    return "zh"
                if primary:
                    # The first entry is the user's top preference; anything
                    # else means a non-Chinese UI regardless of what follows.
                    return "en"
        for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
            value = os.environ.get(var)
            if value:
                if value.lower().startswith("zh"):
                    return "zh"
                break
        else:
            code = locale.getlocale()[0] or ""
            if code.lower().startswith("zh") or code.startswith("Chinese"):
                return "zh"
    except Exception:
        log.debug("System language detection failed; defaulting to en", exc_info=True)
    return "en"


def set_lang(lang: str):
    """Load a locale table. Never raises: this runs at import time, before any
    UI exists, so a malformed YAML must not take the process down silently."""
    global _lang, _strings
    _lang = lang
    f = _dir / f"{lang}.yaml"
    if not f.exists():
        f = _dir / "en.yaml"
    try:
        loaded = yaml.safe_load(f.read_text("utf-8"))
    except Exception:
        log.error("Failed to load locale file %s; falling back to raw keys", f,
                  exc_info=True)
        loaded = None
    _strings = loaded if isinstance(loaded, dict) else {}


def get_lang() -> str:
    return _lang


def t(key: str) -> str:
    """Look up a UI string. A missing key returns the key itself so the UI still
    renders, but logs it so typos are visible during development."""
    value = _strings.get(key)
    if value is None:
        log.debug("Missing i18n key: %s (lang=%s)", key, _lang)
        return key
    return value


# Detect system language on import
set_lang(_detect_system_lang())

# Shared language list: (code, native_name)
LANGUAGES = [
    ("auto", None),  # display name comes from t("asr_lang_auto")
    ("ja", "日本語"),
    ("en", "English"),
    ("zh", "中文"),
    ("ko", "한국어"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("ru", "Русский"),
    ("pt", "Português"),
    ("it", "Italiano"),
    ("nl", "Nederlands"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("ar", "العربية"),
    ("th", "ไทย"),
    ("vi", "Tiếng Việt"),
    ("id", "Bahasa Indonesia"),
    ("ms", "Bahasa Melayu"),
    ("hi", "हिन्दी"),
    ("uk", "Українська"),
    ("cs", "Čeština"),
    ("ro", "Română"),
    ("el", "Ελληνικά"),
    ("hu", "Magyar"),
    ("sv", "Svenska"),
    ("da", "Dansk"),
    ("fi", "Suomi"),
    ("no", "Norsk"),
    ("he", "עברית"),
]

# Common languages shown directly in tray menu (no submenu)
COMMON_LANG_CODES = {"auto", "ja", "en", "zh", "ko", "fr", "de", "es", "ru"}
