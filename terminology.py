"""Deterministic glossary parsing and safe rich-text highlighting.

The live subtitle path must never perform a network request or invoke another
model just to decorate a token.  This module therefore keeps matching small,
predictable and thread-safe: normalize Unicode/case/stress, require token
boundaries for source terms, and escape all user text before adding markup.
"""

from __future__ import annotations

import html
import unicodedata
from dataclasses import dataclass


_ARROW = "=>"


def normalize_term(value: str) -> str:
    """Normalize a term for matching while retaining the original display.

    Russian stress marks are combining characters.  Removing combining marks
    also makes a stressed and unstressed spelling compare equal without
    changing the text shown to the user.  ``ё`` is folded to ``е`` for the
    same reason: it is not a combining character (NFKC recomposes it), but
    recognition output and everyday typing freely mix both spellings, so a
    term written one way must match the other.
    """
    text = unicodedata.normalize("NFKC", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold().replace("ё", "е").strip()


def _is_cjk_char(char: str) -> bool:
    """Scripts written without spaces between words (CJK, kana, hangul)."""
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF  # hiragana + katakana
        or 0x3400 <= code <= 0x4DBF  # CJK extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK unified ideographs
        or 0xAC00 <= code <= 0xD7A3  # hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
    )


def _needle_can_match(needle: str) -> bool:
    """Reject lone CJK characters as match needles.

    Latin/Cyrillic needles are guarded by word boundaries, but CJK text has
    no spaces, so a single-character Chinese translation (``基``) would
    highlight inside completely unrelated words (``基本概念``) on the
    translation line.  Such entries stay in ``Glossary.entries`` — the
    source side still highlights and the hover title still shows the
    pairing — they are only excluded from the matching term tuples.
    """
    return not (len(needle) == 1 and _is_cjk_char(needle))


@dataclass(frozen=True)
class GlossaryEntry:
    original: str
    translation: str
    normalized_original: str
    normalized_translation: str


class Glossary:
    def __init__(self, entries: tuple[GlossaryEntry, ...] = ()):
        self.entries = entries
        # Keep the original order as a deterministic tie-breaker, but prefer
        # the longest match when terms overlap (``ряд`` vs ``рядом``).  The
        # old implementation ran one regex after another on already-marked
        # HTML, which could match inside its own ``<span title=...>`` and
        # produce nested/broken rich text.
        self._source_terms = tuple(
            (entry, index)
            for index, entry in enumerate(entries)
            if entry.normalized_original
            and _needle_can_match(entry.normalized_original)
        )
        self._translation_terms = tuple(
            (entry, index)
            for index, entry in enumerate(entries)
            if entry.normalized_translation
            and _needle_can_match(entry.normalized_translation)
        )

    @staticmethod
    def _normalized_with_map(text: str) -> tuple[str, list[int]]:
        """Return normalized text and a map back to source character offsets."""
        normalized_chars: list[str] = []
        index_map: list[int] = []
        for index, char in enumerate(text):
            normalized = unicodedata.normalize("NFKC", char)
            normalized = "".join(
                ch for ch in normalized if not unicodedata.combining(ch)
            ).casefold().replace("ё", "е")
            normalized_chars.extend(normalized)
            index_map.extend([index] * len(normalized))
        return "".join(normalized_chars), index_map

    @staticmethod
    def _is_word_char(char: str) -> bool:
        return bool(char) and (
            char == "_"
            or char.isascii() and char.isalnum()
            or "\u0400" <= char <= "\u04ff"
        )

    @staticmethod
    def _boundary_ok(text: str, start: int, stop: int) -> bool:
        """Require boundaries for Latin/Cyrillic terms, not for CJK text.

        Chinese and Japanese normally have no spaces, so a boundary rule for
        those scripts would hide perfectly valid phrases such as ``学习数列``.
        Latin/Cyrillic terms do need boundaries to avoid highlighting a term
        inside a longer word (``ряд`` in ``рядом``).
        """
        term_start = text[start] if start < len(text) else ""
        term_end = text[stop - 1] if stop else ""
        needs_start = Glossary._is_word_char(term_start)
        needs_end = Glossary._is_word_char(term_end)
        if needs_start and start > 0 and Glossary._is_word_char(text[start - 1]):
            return False
        if needs_end and stop < len(text) and Glossary._is_word_char(text[stop]):
            return False
        return True

    @classmethod
    def _find_matches(
        cls,
        text: str,
        terms: tuple[tuple[GlossaryEntry, int], ...],
        *,
        source: bool,
    ) -> list[tuple[int, int, GlossaryEntry]]:
        normalized, index_map = cls._normalized_with_map(text)
        if not normalized or not index_map:
            return []
        matches: list[tuple[int, int, GlossaryEntry, int]] = []
        for entry, order in terms:
            needle = entry.normalized_original if source else entry.normalized_translation
            if not needle:
                continue
            offset = 0
            while True:
                found = normalized.find(needle, offset)
                if found < 0:
                    break
                end = found + len(needle)
                if cls._boundary_ok(normalized, found, end):
                    start = index_map[found]
                    stop = index_map[end - 1] + 1
                    matches.append((start, stop, entry, order))
                # Move by one so overlapping candidates are still considered;
                # selection below chooses the longest non-overlapping range.
                offset = found + 1
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[3]))
        selected: list[tuple[int, int, GlossaryEntry]] = []
        end = -1
        for start, stop, entry, _ in matches:
            if start < end:
                continue
            selected.append((start, stop, entry))
            end = stop
        return selected

    def _highlight(self, text: str, *, source: bool) -> str:
        if not text:
            return ""
        matches = self._find_matches(
            text,
            self._source_terms if source else self._translation_terms,
            source=source,
        )
        if not matches:
            return html.escape(text, quote=True)
        parts = []
        cursor = 0
        for start, stop, entry in matches:
            parts.append(html.escape(text[cursor:start], quote=True))
            parts.append(
                self._span(
                    html.escape(text[start:stop], quote=True),
                    f"{entry.original} → {entry.translation}",
                )
            )
            cursor = stop
        parts.append(html.escape(text[cursor:], quote=True))
        return "".join(parts)

    @staticmethod
    def _span(content: str, label: str) -> str:
        title = html.escape(label, quote=True)
        return (
            f'<span class="term-highlight" title="{title}" '
            f'style="color:#ffd166; background-color:#493d24; '
            f'font-weight:700;">{content}</span>'
        )

    def highlight_original(self, text: str) -> str:
        return self._highlight(text, source=True)

    def highlight_translation(self, text: str) -> str:
        return self._highlight(text, source=False)


def parse_glossary(text: str | None) -> Glossary:
    """Parse ``original => translation`` lines; last duplicate wins."""
    entries: dict[str, GlossaryEntry] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or _ARROW not in line:
            continue
        original, translation = (part.strip() for part in line.split(_ARROW, 1))
        normalized_original = normalize_term(original)
        normalized_translation = normalize_term(translation)
        if not normalized_original or not translation:
            continue
        entries[normalized_original] = GlossaryEntry(
            original=original,
            translation=translation,
            normalized_original=normalized_original,
            normalized_translation=normalized_translation,
        )
    return Glossary(tuple(entries.values()))


def short_cjk_translations(text: str | None) -> list[str]:
    """Translations from a glossary text that the CJK rule keeps inert.

    Backs the panel's visible warning: a user entry with a single-character
    Chinese translation is still parsed and still highlights the source
    side, but it never highlights on the translation line — which must be
    told to the user, not dropped silently.
    """
    inert: list[str] = []
    for entry in parse_glossary(text).entries:
        if not _needle_can_match(entry.normalized_translation):
            if entry.translation not in inert:
                inert.append(entry.translation)
    return inert


EMPTY_GLOSSARY = Glossary()
