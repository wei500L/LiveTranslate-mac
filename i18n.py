import locale
import logging
import os

import yaml
from pathlib import Path

log = logging.getLogger("LiveTranslate.i18n")

_strings: dict = {}
_lang = "en"
_dir = Path(__file__).parent / "i18n"


def _detect_system_lang() -> str:
    """Return 'zh' if the system locale is Chinese, else 'en'.

    locale.getdefaultlocale() is deprecated since 3.11 and slated for removal
    in 3.15, so read the environment first and only fall back to the locale
    module's supported API.
    """
    try:
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
