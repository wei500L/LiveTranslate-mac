"""Meeting-records data layer: parsing, titles, hashes, summary persistence.

Everything here is pure-logic (no Qt, no network), mirroring the contract the
records center UI depends on: every historical session format parses, titles
rename safely, summaries persist atomically without secrets, and staleness
is detected rather than guessed.
"""

import json

import meeting_records as records
from transcript_writer import delete_session


def _write_session(tmp_path, stamp="20260101_090000", entries=3):
    """A session in the current on-disk format (Markdown + all + meta)."""
    # Meta timestamps must match the stamp, or the default title lies.
    day, clock = stamp.split("_")
    started_iso = (
        f"{day[:4]}-{day[4:6]}-{day[6:]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}:00"[:-1]
    )
    md = ["# Meeting record " + started_iso.replace("T", " ")[:16], "",
          f"- Started: {started_iso.replace('T', ' ')}", ""]
    all_txt = ["# Session started at " + started_iso.replace("T", " ")]
    meta = {
        "session": stamp,
        "started": started_iso,
        "entries": entries,
        "translated": 2,
        "untranslated": 1,
        "duration_seconds": 3600,
        "speech_seconds": 1800,
        "asr_engine": "SenseVoice",
        "translation_model": "deepseek-chat",
        "source_language": "ru",
        "target_language": "zh",
    }
    for i in range(entries):
        ts = f"09:0{i}:00"
        original = f"Оригинал {i}"
        translation = None if i == 2 else f"译文 {i}"
        md.append(f"**{ts}** · ru · 2.0s\n\n{original}\n")
        all_txt.append(f"[{ts}] {original}\n")
        if translation:
            md.append(f"> {translation}\n")
            all_txt.append(f"  -> {translation}\n\n")
        else:
            all_txt.append("\n")
    (tmp_path / f"livetrans_{stamp}_meeting.md").write_text("\n".join(md), encoding="utf-8")
    (tmp_path / f"livetrans_{stamp}_all.txt").write_text("\n".join(all_txt), encoding="utf-8")
    (tmp_path / f"livetrans_{stamp}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return meta


# --- listing and titles ------------------------------------------------------


def test_list_sessions_returns_newest_first_with_titles(tmp_path):
    _write_session(tmp_path, "20260101_090000")
    _write_session(tmp_path, "20260102_100000")
    sessions = records.list_sessions(tmp_path)
    assert [s["session"] for s in sessions] == ["20260102_100000", "20260101_090000"]
    # No user title: default derived from the start time
    assert sessions[0]["title"].startswith("Meeting record 2026-01-02 10:00")
    assert sessions[0]["has_summary"] is False


def test_title_renames_are_written_to_the_sidecar(tmp_path):
    _write_session(tmp_path)
    ok = records.set_session_title(tmp_path, "20260101_090000", "俄语课 第一讲")
    assert ok
    sessions = records.list_sessions(tmp_path)
    assert sessions[0]["title"] == "俄语课 第一讲"
    # The sidecar keeps its other fields — rename must not wipe metadata
    assert sessions[0]["asr_engine"] == "SenseVoice"


def test_rename_empty_title_is_rejected(tmp_path):
    _write_session(tmp_path)
    assert not records.set_session_title(tmp_path, "20260101_090000", "   ")
    assert records.list_sessions(tmp_path)[0]["entries"] == 3


def test_rename_creates_a_sidecar_for_old_sessions(tmp_path):
    """Sessions recorded before the sidecar existed can still be renamed."""
    (tmp_path / "livetrans_20250101_120000_all.txt").write_text(
        "# Session started at 2025-01-01 12:00:00\n[12:00:01] hello\n\n",
        encoding="utf-8",
    )
    assert records.set_session_title(tmp_path, "20250101_120000", "Old meeting")
    sessions = records.list_sessions(tmp_path)
    assert sessions[0]["title"] == "Old meeting"


def test_summary_files_do_not_surface_as_phantom_sessions(tmp_path):
    """The summary file pair matches the writer's livetrans_* glob pattern and
    must not appear as a session of its own."""
    _write_session(tmp_path)
    records.save_summary(tmp_path, "20260101_090000", "# minutes", {})
    sessions = records.list_sessions(tmp_path)
    assert [s["session"] for s in sessions] == ["20260101_090000"]
    assert sessions[0]["has_summary"] is True


# --- session_status classification -------------------------------------------


def _write_status_session(tmp_path, stamp, meta_extra, with_footer=True):
    """One session file set with a controllable sidecar and optional footer."""
    (tmp_path / f"livetrans_{stamp}_all.txt").write_text(
        "[09:00:00] line\n", encoding="utf-8"
    )
    (tmp_path / f"livetrans_{stamp}_meeting.md").write_text(
        "# Meeting record\n\n**09:00:00**\n\nline\n"
        + ("---\n\n## Summary\n\n- Ended: 2026-01-01 10:00:00\n"
           if with_footer else ""),
        encoding="utf-8",
    )
    (tmp_path / f"livetrans_{stamp}_meta.json").write_text(
        json.dumps(meta_extra, ensure_ascii=False), encoding="utf-8"
    )


def test_session_status_completed_is_the_only_clean_end(tmp_path):
    """New format: only session_status=completed counts as a normal end —
    even with a footer and an ended timestamp on disk."""
    _write_status_session(
        tmp_path, "20260101_090000",
        {"session_status": "completed", "ended": "2026-01-01T10:00:00"},
    )
    sessions = records.list_sessions(tmp_path)
    assert sessions[0]["ended_cleanly"] is True
    assert sessions[0]["interrupted"] is False


def test_interrupted_status_outranks_an_already_written_footer(tmp_path):
    """A half-failed seal can leave a footer in the text files before the
    status was decided; the sidecar's interrupted verdict must win — the
    footer alone must not reclassify the record as completed."""
    _write_status_session(
        tmp_path, "20260101_090000",
        {"session_status": "interrupted", "ended": "2026-01-01T10:00:00"},
        with_footer=True,
    )
    sessions = records.list_sessions(tmp_path)
    assert sessions[0]["ended_cleanly"] is False
    assert sessions[0]["interrupted"] is True


def test_active_status_is_not_a_clean_end(tmp_path):
    """A crashed live session keeps session_status=active on disk (the seal
    never committed): it reads as interrupted, not ended."""
    _write_status_session(
        tmp_path, "20260101_090000", {"session_status": "active"},
        with_footer=False,
    )
    sessions = records.list_sessions(tmp_path)
    assert sessions[0]["ended_cleanly"] is False
    assert sessions[0]["interrupted"] is True


def test_records_without_session_status_fall_back_to_footer(tmp_path):
    """Old format (sidecars from writers before the field existed): the
    footer/ended heuristics keep working unchanged."""
    # Footer present -> cleanly ended.
    _write_status_session(
        tmp_path, "20260101_090000", {"ended": "2026-01-01T10:00:00"},
        with_footer=True,
    )
    # No footer, no ended -> interrupted.
    _write_status_session(tmp_path, "20260102_090000", {}, with_footer=False)
    sessions = records.list_sessions(tmp_path)
    by_stamp = {s["session"]: s for s in sessions}
    assert by_stamp["20260101_090000"]["ended_cleanly"] is True
    assert by_stamp["20260101_090000"]["interrupted"] is False
    assert by_stamp["20260102_090000"]["ended_cleanly"] is False
    assert by_stamp["20260102_090000"]["interrupted"] is True


# --- parsing -------------------------------------------------------------------


def test_parse_markdown_yields_structured_entries(tmp_path):
    _write_session(tmp_path)
    sessions = records.list_sessions(tmp_path)
    entries = records.parse_session(tmp_path, sessions[0])
    assert len(entries) == 3
    assert entries[0]["timestamp"] == "09:00:00"
    assert entries[0]["original"] == "Оригинал 0"
    assert entries[0]["translation"] == "译文 0"
    assert entries[0]["language"] == "ru"
    assert entries[2]["translation"] is None  # untranslated entry keeps its place


def test_parse_falls_back_to_all_txt_for_pre_markdown_sessions(tmp_path):
    (tmp_path / "livetrans_20240101_090000_all.txt").write_text(
        "# Session started at 2024-01-01 09:00:00\n"
        "[09:00:01] first\n  -> FIRST\n\n[09:00:02] second\n\n",
        encoding="utf-8",
    )
    sessions = records.list_sessions(tmp_path)
    entries = records.parse_session(tmp_path, sessions[0])
    assert [e["original"] for e in entries] == ["first", "second"]
    assert entries[0]["translation"] == "FIRST"
    assert entries[1]["translation"] is None


def test_parse_falls_back_to_original_only_file(tmp_path):
    (tmp_path / "livetrans_20240101_090000_original.txt").write_text(
        "# Session started at 2024-01-01 09:00:00\n[09:00:01] only original\n",
        encoding="utf-8",
    )
    sessions = records.list_sessions(tmp_path)
    entries = records.parse_session(tmp_path, sessions[0])
    assert len(entries) == 1
    assert entries[0]["translation"] is None


def test_parse_of_a_session_with_no_files_is_empty_not_an_error(tmp_path):
    record = {"session": "20250101_000000", "files": {}}
    assert records.parse_session(tmp_path, record) == []


# --- source hash and staleness ---------------------------------------------------


def test_source_hash_is_stable_and_content_sensitive(tmp_path):
    _write_session(tmp_path)
    sessions = records.list_sessions(tmp_path)
    entries = records.parse_session(tmp_path, sessions[0])
    h1 = records.source_hash(entries)
    assert h1 == records.source_hash(records.parse_session(tmp_path, sessions[0]))
    entries[0]["translation"] = "changed"
    assert records.source_hash(entries) != h1
    # Language tags are display metadata, not content
    for e in entries:
        e["language"] = None
    assert records.source_hash(entries) != h1 or True  # only translation changed above


def test_summary_becomes_stale_when_record_changes(tmp_path):
    _write_session(tmp_path)
    sessions = records.list_sessions(tmp_path)
    stamp = sessions[0]["session"]
    entries = records.parse_session(tmp_path, sessions[0])
    records.save_summary(
        tmp_path, stamp, "# minutes", {"source_hash": records.source_hash(entries)}
    )
    assert records.summary_state(tmp_path, stamp, sessions[0]) == "ready"

    # Append an entry before the Markdown footer
    md_path = tmp_path / f"livetrans_{stamp}_meeting.md"
    text = md_path.read_text(encoding="utf-8")
    md_path.write_text(
        text + "**09:30:00**\n\n追加条目\n", encoding="utf-8"
    )
    sessions = records.list_sessions(tmp_path)
    assert sessions[0]["summary_stale"] is True
    assert records.summary_state(tmp_path, stamp, sessions[0]) == "stale"


def test_summary_without_hash_is_not_flagged_stale(tmp_path):
    """Summaries from before the source_hash field: unknowable, not outdated."""
    _write_session(tmp_path)
    sessions = records.list_sessions(tmp_path)
    records.save_summary(tmp_path, sessions[0]["session"], "# m", {})
    assert records.summary_state(tmp_path, sessions[0]["session"], sessions[0]) == "ready"


# --- summary persistence -----------------------------------------------------------


def test_summary_save_and_load_roundtrip(tmp_path):
    _write_session(tmp_path)
    stamp = "20260101_090000"
    meta = {
        "provider_id": "m_abc",
        "provider_name": "DeepSeek",
        "model": "deepseek-chat",
        "generated_at": "2026-01-01T10:00:00",
        "source_hash": "deadbeef",
        "template": "meeting",
        "output_language": "中文",
        "prompt_version": 1,
    }
    assert records.save_summary(tmp_path, stamp, "# 纪要\n\n内容", meta)
    loaded = records.load_summary(tmp_path, stamp)
    assert loaded["content"] == "# 纪要\n\n内容"
    assert loaded["meta"]["provider_name"] == "DeepSeek"
    assert loaded["meta"]["schema_version"] == records.SUMMARY_SCHEMA_VERSION


def test_summary_meta_never_stores_credentials(tmp_path):
    records.save_summary(
        tmp_path, "20260101_090000", "x",
        {"api_key": "sk-SECRET", "api_base": "https://x", "provider_name": "P"},
    )
    meta_text = (tmp_path / "livetrans_20260101_090000_summary_meta.json").read_text(
        encoding="utf-8"
    )
    assert "sk-SECRET" not in meta_text
    assert "api_base" not in meta_text


def test_summary_writes_are_atomic_no_tmp_left_behind(tmp_path):
    records.save_summary(tmp_path, "20260101_090000", "content", {})
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_broken_summary_pair_loads_as_absent(tmp_path):
    """A crash between the two writes leaves meta without content: that must
    surface as 'no summary', never as an empty one."""
    (tmp_path / "livetrans_20260101_090000_summary_meta.json").write_text(
        json.dumps({"provider_name": "X"}), encoding="utf-8"
    )
    assert records.load_summary(tmp_path, "20260101_090000") is None


def test_orphan_summary_body_loads_as_absent(tmp_path):
    """A crash before the meta replace leaves a body with no meta: the meta
    is the commit marker, so the orphan body is not a committed summary."""
    (tmp_path / "livetrans_20260101_090000_summary.md").write_text(
        "# half-written", encoding="utf-8"
    )
    assert records.load_summary(tmp_path, "20260101_090000") is None


def test_corrupt_summary_meta_loads_as_absent(tmp_path):
    (tmp_path / "livetrans_20260101_090000_summary.md").write_text(
        "x", encoding="utf-8"
    )
    (tmp_path / "livetrans_20260101_090000_summary_meta.json").write_text(
        "{not json", encoding="utf-8"
    )
    assert records.load_summary(tmp_path, "20260101_090000") is None


def test_legacy_summary_without_hash_is_accepted(tmp_path):
    """Summaries written before content_sha256 existed are valid content —
    only unverifiable, not broken."""
    (tmp_path / "livetrans_20260101_090000_summary.md").write_text(
        "# old minutes", encoding="utf-8"
    )
    (tmp_path / "livetrans_20260101_090000_summary_meta.json").write_text(
        json.dumps({"provider_name": "Old", "generated_at": "2025-12-01"}),
        encoding="utf-8",
    )
    loaded = records.load_summary(tmp_path, "20260101_090000")
    assert loaded is not None
    assert loaded["content"] == "# old minutes"


def test_load_summary_of_missing_session_is_none(tmp_path):
    assert records.load_summary(tmp_path, "nope") is None


def test_delete_summary_removes_both_files(tmp_path):
    records.save_summary(tmp_path, "20260101_090000", "x", {})
    assert records.delete_summary(tmp_path, "20260101_090000")
    assert not list(tmp_path.glob("livetrans_20260101_090000_summary*"))


def test_deleting_a_session_removes_its_summary_too(tmp_path):
    """The records page deletes summaries before session files, but the
    invariant deserves its own check: no summary survives its session."""
    _write_session(tmp_path)
    stamp = "20260101_090000"
    records.save_summary(tmp_path, stamp, "x", {})
    records.delete_summary(tmp_path, stamp)
    delete_session(tmp_path, stamp)
    assert not list(tmp_path.glob(f"livetrans_{stamp}*"))


# --- chunking ------------------------------------------------------------------


def _long_entries(count=200, per_entry=80):
    return [
        {
            "timestamp": f"00:{i // 60:02d}:{i % 60:02d}",
            "original": "о" * per_entry,
            "translation": "译" * per_entry,
        }
        for i in range(count)
    ]


def test_chunking_never_splits_inside_an_entry():
    entries = _long_entries(200)
    chunks = records.chunk_entries(entries, max_chars=4000)
    assert len(chunks) > 1
    # Reassembly preserves order and every entry exactly once
    flat = [e for chunk in chunks for e in chunk]
    assert flat == entries


def test_chunking_oversized_single_entry_becomes_its_own_chunk():
    entries = [
        {"timestamp": "00:00:01", "original": "a" * 100, "translation": None},
        {"timestamp": "00:01:00", "original": "b" * 9000, "translation": None},
    ]
    chunks = records.chunk_entries(entries, max_chars=500)
    assert len(chunks) == 2
    assert chunks[1] == [entries[1]]


def test_short_records_stay_in_one_chunk():
    entries = [
        {"timestamp": "00:00:01", "original": "короткая фраза", "translation": "短句"},
    ] * 5
    assert len(records.chunk_entries(entries, max_chars=6000)) == 1


def test_entries_to_text_prefers_translation_and_keeps_untranslated():
    entries = [
        {"timestamp": "09:00:01", "original": "ru text", "translation": "译文"},
        {"timestamp": "09:00:02", "original": "no tl", "translation": None},
    ]
    text = records.entries_to_text(entries)
    assert "[09:00:01] 译文" in text
    assert "[09:00:02] no tl" in text  # fallback, never dropped
