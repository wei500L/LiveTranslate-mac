"""Shared, UI-independent subtitle state primitives.

The application has two render surfaces (the reading overlay and the OBS
window).  Keeping the state vocabulary here prevents one surface from treating
an interim result as a new sentence while the other treats it as a revision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum


class SubtitleStatus(str, Enum):
    PROVISIONAL = "provisional"
    TRANSLATING = "translating"
    FINAL = "final"
    NO_TRANSLATION = "no_translation"
    ERROR = "error"


@dataclass(frozen=True)
class SubtitleSegment:
    segment_id: int
    original: str
    translation: str = ""
    source_lang: str = "auto"
    target_lang: str = ""
    status: SubtitleStatus = SubtitleStatus.TRANSLATING
    timestamp: str = ""
    error_code: str = ""
    retryable: bool = False
    # Provider is part of the state because cloud streaming cards have a
    # different failure contract: Soniox has no replayable source segment,
    # so its errors must never advertise an in-place translation retry.
    provider: str = ""
    translations: dict[str, str] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.status in (SubtitleStatus.PROVISIONAL, SubtitleStatus.TRANSLATING)

    def revise(self, **changes) -> SubtitleSegment:
        return replace(self, **changes)

    def to_json(self) -> str:
        payload = dict(self.__dict__)
        payload["status"] = self.status.value
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> SubtitleSegment:
        data = json.loads(payload)
        data["status"] = SubtitleStatus(data.get("status", SubtitleStatus.TRANSLATING))
        return cls(**data)


class SubtitleStateStore:
    """Small ordered store used by tests and UI adapters.

    Updating an existing id never changes its position.  This is the invariant
    that keeps streaming corrections in-place and prevents scroll jitter.
    """

    def __init__(self, max_segments: int = 50):
        self.max_segments = max(1, int(max_segments))
        self._items: dict[int, SubtitleSegment] = {}
        self._order: list[int] = []

    def upsert(self, segment: SubtitleSegment) -> SubtitleSegment:
        if segment.segment_id not in self._items:
            self._order.append(segment.segment_id)
        self._items[segment.segment_id] = segment
        while len(self._order) > self.max_segments:
            old = self._order.pop(0)
            self._items.pop(old, None)
        return segment

    def get(self, segment_id: int) -> SubtitleSegment | None:
        return self._items.get(segment_id)

    def remove(self, segment_id: int) -> None:
        self._items.pop(segment_id, None)
        try:
            self._order.remove(segment_id)
        except ValueError:
            pass

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()

    def values(self) -> list[SubtitleSegment]:
        return [self._items[item] for item in self._order if item in self._items]

    def __len__(self) -> int:
        return len(self._order)
