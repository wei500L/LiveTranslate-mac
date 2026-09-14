from subtitle_state import SubtitleSegment, SubtitleStateStore, SubtitleStatus


def test_revision_keeps_one_ordered_segment():
    store = SubtitleStateStore()
    first = SubtitleSegment(7, "Hel", status=SubtitleStatus.PROVISIONAL)
    store.upsert(first)
    store.upsert(first.revise(original="Hello", status=SubtitleStatus.TRANSLATING))
    store.upsert(first.revise(
        original="Hello", translation="你好", status=SubtitleStatus.FINAL
    ))

    assert len(store) == 1
    assert store.values()[0].translation == "你好"
    assert store.values()[0].status == SubtitleStatus.FINAL


def test_store_caps_oldest_without_reordering_revisions():
    store = SubtitleStateStore(max_segments=2)
    store.upsert(SubtitleSegment(1, "one"))
    store.upsert(SubtitleSegment(2, "two"))
    store.upsert(SubtitleSegment(1, "ONE"))
    store.upsert(SubtitleSegment(3, "three"))

    assert [item.segment_id for item in store.values()] == [2, 3]


def test_json_round_trip_preserves_status_and_multilingual_results():
    segment = SubtitleSegment(
        9, "hello", "你好", source_lang="en", target_lang="zh",
        status=SubtitleStatus.ERROR, error_code="timeout", retryable=True,
        translations={"zh": "你好", "ja": "こんにちは"},
    )
    restored = SubtitleSegment.from_json(segment.to_json())
    assert restored == segment
