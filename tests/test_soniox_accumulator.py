"""Offline tests for the Soniox token accumulator (no SDK, no Qt, no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soniox_accumulator import (
    END_TOKEN,
    FIN_TOKEN,
    SonioxAccumulator,
    SonioxSegment,
    SonioxToken,
)


def tok(text, is_final=False, status=None):
    return SonioxToken(text=text, is_final=is_final, translation_status=status)


def test_provisional_tokens_replace_never_append():
    acc = SonioxAccumulator()
    r1 = acc.process_event([tok("Прив"), tok("ет")])
    assert r1.live.original == "Привет"
    # Corrected provisional: the old snapshot must be replaced, not appended.
    r2 = acc.process_event([tok("Привет")])
    assert r2.live.original == "Привет"
    r3 = acc.process_event([tok("Привет мир")])
    assert r3.live.original == "Привет мир"


def test_provisional_replacement_includes_finals_prefix():
    acc = SonioxAccumulator()
    acc.process_event([tok("Привет", is_final=True, status="original")])
    r = acc.process_event([tok("ми")])
    assert r.live.original == "Приветми"
    r = acc.process_event([tok("мир")])
    assert r.live.original == "Приветмир"


def test_final_tokens_append_exactly_once():
    acc = SonioxAccumulator()
    acc.process_event(
        [
            tok("При", is_final=True, status="original"),
            tok("вет", is_final=True, status="original"),
        ]
    )
    # A later event re-emitting the same final text must not duplicate it:
    # finals are never re-processed because the SDK sends only new tokens;
    # the accumulator simply appends what it is given, once.
    acc.process_event([tok(" мир", is_final=True, status="original")])
    assert acc.live.original == "Привет мир"


def test_original_translation_routing_and_interleaving():
    acc = SonioxAccumulator()
    acc.process_event(
        [
            tok("Привет", is_final=True, status="original"),
            tok("你好", is_final=True, status="translation"),
            tok(" мир", is_final=True, status="original"),
            tok("世界", is_final=True, status="translation"),
        ]
    )
    assert acc.live.original == "Привет мир"
    assert acc.live.translation == "你好世界"


def test_none_status_and_missing_status_treated_as_original():
    acc = SonioxAccumulator()
    acc.process_event(
        [
            tok("A", is_final=True, status="none"),
            tok("B", is_final=True),  # status field absent/None
        ]
    )
    assert acc.live.original == "AB"
    assert acc.live.translation == ""


def test_translation_tokens_are_provisional_too():
    acc = SonioxAccumulator()
    acc.process_event([tok("你", status="translation"), tok("好", status="translation")])
    assert acc.live.translation == "你好"
    # Corrected translation replaces.
    acc.process_event([tok("您好", status="translation")])
    assert acc.live.translation == "您好"
    # Provisional translation never leaks into the final commit.
    seg = acc.flush()
    assert seg is None  # no *final* original either -> nothing to commit


def test_end_token_hidden_and_commits_once():
    acc = SonioxAccumulator()
    r = acc.process_event(
        [
            tok("Привет", is_final=True, status="original"),
            tok("你好", is_final=True, status="translation"),
            tok(END_TOKEN, is_final=True),
        ]
    )
    assert len(r.committed) == 1
    seg = r.committed[0]
    assert seg.original == "Привет"
    assert seg.translation == "你好"
    assert END_TOKEN not in seg.original
    assert END_TOKEN not in seg.translation
    # State cleared: live is empty and a second <end> is a no-op.
    assert acc.live.original == ""
    r2 = acc.process_event([tok(END_TOKEN, is_final=True)])
    assert r2.committed == []


def test_multiple_end_tokens_in_one_event_commit_multiple_segments():
    acc = SonioxAccumulator()
    r = acc.process_event(
        [
            tok("Первое", is_final=True, status="original"),
            tok("一", is_final=True, status="translation"),
            tok(END_TOKEN, is_final=True),
            tok("Второе", is_final=True, status="original"),
            tok("二", is_final=True, status="translation"),
            tok(END_TOKEN, is_final=True),
        ]
    )
    assert [(s.original, s.translation) for s in r.committed] == [
        ("Первое", "一"),
        ("Второе", "二"),
    ]


def test_end_after_provisional_then_new_segment_tokens():
    acc = SonioxAccumulator()
    acc.process_event([tok("Привет", is_final=True, status="original")])
    acc.process_event([tok("мир"), tok(END_TOKEN, is_final=True)]).committed
    # Tokens after <end> belong to the next segment's live card.
    r = acc.process_event([tok("Новое")])
    assert r.live.original == "Новое"
    assert r.live.translation == ""


def test_finished_commits_trailing_finals_once():
    acc = SonioxAccumulator()
    acc.process_event(
        [
            tok("Последняя", is_final=True, status="original"),
            tok("фраза", is_final=True, status="original"),
            tok("最后一句", is_final=True, status="translation"),
        ]
    )
    r = acc.process_event([], finished=True)
    assert len(r.committed) == 1
    assert r.committed[0].original == "Последняяфраза"
    assert r.committed[0].translation == "最后一句"
    # Idempotent: repeated finished events do not re-commit.
    r2 = acc.process_event([], finished=True)
    assert r2.committed == []


def test_finished_after_end_does_not_double_commit():
    acc = SonioxAccumulator()
    acc.process_event(
        [
            tok("Готово", is_final=True, status="original"),
            tok(END_TOKEN, is_final=True),
        ]
    )
    r = acc.process_event([], finished=True)
    assert r.committed == []


def test_punctuation_only_segment_dropped_but_state_reset():
    acc = SonioxAccumulator()
    r = acc.process_event(
        [
            tok("...", is_final=True, status="original"),
            tok("!?", is_final=True, status="translation"),
            tok(END_TOKEN, is_final=True),
        ]
    )
    assert r.committed == []
    assert acc.live.original == ""


def test_empty_token_event_refreshes_live_only():
    acc = SonioxAccumulator()
    acc.process_event([tok("Сохранённое", is_final=True, status="original")])
    r = acc.process_event([])
    assert r.committed == []
    assert r.live.original == "Сохранённое"


def test_flush_commits_finals_once_and_clears():
    acc = SonioxAccumulator()
    acc.process_event(
        [
            tok("Пауза", is_final=True, status="original"),
            tok("暂停", is_final=True, status="translation"),
        ]
    )
    acc.process_event([tok("хвост")])  # provisional tail, dropped by flush
    seg = acc.flush()
    assert seg == SonioxSegment(original="Пауза", translation="暂停")
    assert acc.flush() is None
    assert acc.live.original == ""


def test_reset_drops_everything():
    acc = SonioxAccumulator()
    acc.process_event([tok("Сброс", is_final=True, status="original")])
    acc.reset()
    assert acc.live.original == ""
    assert acc.flush() is None


def test_fin_token_never_displayed():
    acc = SonioxAccumulator()
    r = acc.process_event(
        [
            tok("Фраза", is_final=True, status="original"),
            tok(FIN_TOKEN, is_final=True),
        ]
    )
    assert FIN_TOKEN not in r.live.original
    seg = acc.flush()
    assert seg is not None
    assert FIN_TOKEN not in seg.original


def test_unicode_cyrillic_and_cjk_preserved_verbatim():
    acc = SonioxAccumulator()
    text = "Термодинамика — 第一定律, ΔU = Q − A"
    acc.process_event([tok(text, is_final=True, status="original")])
    seg = acc.flush()
    assert seg.original == text


def test_empty_text_tokens_ignored():
    acc = SonioxAccumulator()
    r = acc.process_event([tok("", is_final=True), tok(None)])
    assert r.live.original == ""
