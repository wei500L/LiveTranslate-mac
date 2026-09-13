"""Pure token accumulator for the Soniox cloud streaming engine.

This module is the token state machine described in the Soniox realtime
contract; it has no I/O, no SDK import and no Qt dependency so it stays
unit-testable everywhere (CI installs neither soniox nor PyQt6 extras for
every job). Tokens arriving from ``soniox.types.Token`` are duck-typed: the
manager converts nothing, it passes the SDK objects straight through and this
module reads only ``text`` / ``is_final`` / ``translation_status``.

SDK contract assumed (verified against soniox 2.9.0):
  * every event carries only *new* tokens; provisional tokens are re-emitted
    in corrected form until they turn final. The manager additionally drops
    byte-identical duplicate events as a transport-level defense.
  * ``<end>`` marks a semantic endpoint, is always final and appears exactly
    once at the end of the finalized segment. Manual finalization emits
    ``<fin>``; we treat it as punctuation (it never enters either string).

Rules (each maps to a test in tests/test_soniox_accumulator.py):
  * final tokens append exactly once, routed by ``translation_status``:
    "original"/"none"/None -> original side, "translation" -> translation;
  * provisional tokens *replace* the previous provisional snapshot — they
    never append into the committed view;
  * ``<end>`` never appears in any string and commits the current segment
    exactly once (multiple ``<end>`` in one event commit multiple segments;
    a second ``<end>`` on empty state is a no-op);
  * ``finished=True`` commits trailing finals not yet followed by ``<end>``
    exactly once;
  * a segment must contain an alphanumeric character to be committed —
    punctuation-only segments are dropped but still reset the state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

END_TOKEN = "<end>"
FIN_TOKEN = "<fin>"

_TRANSLATION_STATUSES = ("translation",)
_ORIGINAL_STATUSES = (None, "original", "none")


@dataclass(frozen=True)
class SonioxToken:
    """Duck-typed mirror of ``soniox.types.Token``."""

    text: str
    is_final: bool = False
    translation_status: str | None = None


@dataclass(frozen=True)
class SonioxSegment:
    """One committed endpoint segment (final original + final translation)."""

    original: str
    translation: str  # "" when the segment ended without a translation


@dataclass(frozen=True)
class SonioxLiveState:
    """The live card's current render snapshot."""

    original: str
    translation: str


@dataclass(frozen=True)
class SonioxEventResult:
    live: SonioxLiveState
    committed: list[SonioxSegment] = field(default_factory=list)
    finished: bool = False


def _has_alnum(text: str) -> bool:
    return any(c.isalnum() for c in text)


class SonioxAccumulator:
    """Not thread-safe by design: exactly one supervisor thread feeds it."""

    def __init__(self) -> None:
        self._final_original = ""
        self._final_translation = ""
        self._prov_original = ""
        self._prov_translation = ""
        self._tail_committed = False

    @property
    def live(self) -> SonioxLiveState:
        # Provisional tokens replace, never append: the snapshot is the
        # committed finals plus the *current* provisional replacement.
        return SonioxLiveState(
            original=(self._final_original + self._prov_original).strip(),
            translation=(self._final_translation + self._prov_translation).strip(),
        )

    def process_event(
        self, tokens: Iterable[SonioxToken], finished: bool = False
    ) -> SonioxEventResult:
        committed: list[SonioxSegment] = []
        # A new event replaces the previous provisional snapshot wholesale.
        self._prov_original = ""
        self._prov_translation = ""

        for token in tokens:
            text = getattr(token, "text", "") or ""
            status = getattr(token, "translation_status", None)
            stripped = text.strip()
            if stripped == END_TOKEN:
                # Defensive: treat <end> as a boundary regardless of is_final
                # (the SDK emits it always-final, but filtering here keeps the
                # marker out of user-visible text on any contract drift).
                segment = self._commit_current()
                if segment is not None:
                    committed.append(segment)
                continue
            if stripped == FIN_TOKEN:
                # Manual-finalization marker: punctuation, never displayed.
                continue
            if getattr(token, "is_final", False):
                if status in _TRANSLATION_STATUSES:
                    self._final_translation += text
                elif status in _ORIGINAL_STATUSES:
                    self._final_original += text
                # Unknown status values are dropped rather than guessed into
                # either string.
            else:
                if status in _TRANSLATION_STATUSES:
                    self._prov_translation += text
                elif status in _ORIGINAL_STATUSES:
                    self._prov_original += text

        if finished:
            # Trailing finals with no <end> follow: commit exactly once.
            if not self._tail_committed:
                segment = self._commit_current()
                if segment is not None:
                    committed.append(segment)
                self._tail_committed = True

        return SonioxEventResult(live=self.live, committed=committed, finished=finished)

    def flush(self) -> SonioxSegment | None:
        """Hard boundary (pause / session end): commit the accumulated finals
        exactly once and clear the state."""
        self._prov_original = ""
        self._prov_translation = ""
        return self._commit_current()

    def reset(self) -> None:
        """Stream torn down (engine switch, reconnect discard): drop all."""
        self._final_original = ""
        self._final_translation = ""
        self._prov_original = ""
        self._prov_translation = ""
        self._tail_committed = False

    def _commit_current(self) -> SonioxSegment | None:
        original = self._final_original.strip()
        translation = self._final_translation.strip()
        self._final_original = ""
        self._final_translation = ""
        self._prov_original = ""
        self._prov_translation = ""
        # A new segment begins on the next token: the finished-tail guard
        # must re-arm so a (defensive) second finished event can commit again.
        self._tail_committed = False
        if not _has_alnum(original):
            # Punctuation-only / empty segment: nothing user-facing, but the
            # state is cleared so it cannot leak into the next segment.
            return None
        return SonioxSegment(original=original, translation=translation)
