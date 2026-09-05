"""Transcript ordering, completeness and session metadata.

The record is the durable artifact of a session, so the properties that matter
are: entries appear in the order they were said, nothing registered is ever
lost, and the file says how it was produced.
"""

import json

import pytest

from transcript_writer import TranscriptWriter, delete_session, read_session_meta


def _writer(tmp_path):
    writer = TranscriptWriter(tmp_path)
    writer.set_enabled(True)
    # Entry writes only auto-open a session while the pipeline records
    # (the legacy path); explicit begin/end is tested separately.
    writer.set_recording(True)
    return writer


def _entries(path):
    return [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("[")
    ]


def test_entries_are_written_in_utterance_order_not_completion_order(tmp_path):
    """Translations run on a worker pool and finish out of order. Writing each
    one as it landed produced a record whose lines were shuffled."""
    writer = _writer(tmp_path)
    for msg_id, text in ((1, "first"), (2, "second"), (3, "third")):
        writer.write_original(msg_id, f"00:00:0{msg_id}", text)

    # Completion order deliberately reversed.
    writer.write_translation(3, "THIRD")
    writer.write_translation(2, "SECOND")
    writer.write_translation(1, "FIRST")
    stamp = writer._session_ts  # saved: close clears it
    writer.close()

    all_lines = _entries(tmp_path / f"livetrans_{stamp}_all.txt")
    assert [line.split("] ", 1)[1] for line in all_lines] == [
        "first", "second", "third"
    ]
    tl_lines = _entries(tmp_path / f"livetrans_{stamp}_translation.txt")
    assert [line.split("] ", 1)[1] for line in tl_lines] == [
        "FIRST", "SECOND", "THIRD"
    ]


def test_a_late_translation_does_not_let_later_ones_jump_the_queue(tmp_path):
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "slow")
    writer.write_original(2, "00:00:02", "fast")
    writer.write_translation(2, "FAST")

    path = tmp_path / f"livetrans_{writer._session_ts}_all.txt"
    assert _entries(path) == []  # nothing may be emitted while #1 is outstanding

    writer.write_translation(1, "SLOW")
    assert [line.split("] ", 1)[1] for line in _entries(path)] == ["slow", "fast"]
    writer.close()


def test_the_original_file_stays_immediate(tmp_path):
    """It needs no translation, and it is the file you tail during a session."""
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "spoken now")
    path = tmp_path / f"livetrans_{writer._session_ts}_original.txt"
    assert "spoken now" in path.read_text(encoding="utf-8")
    writer.close()


def test_close_flushes_entries_whose_translation_never_arrived(tmp_path):
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "answered")
    writer.write_original(2, "00:00:02", "abandoned")
    writer.write_translation(1, "ANSWERED")
    stamp = writer._session_ts  # saved: close clears it
    writer.close()

    text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "answered" in text
    assert "abandoned" in text  # would otherwise be lost with the process


def test_an_untranslated_entry_keeps_its_place(tmp_path):
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "one")
    writer.write_original(2, "00:00:02", "two")
    writer.write_original(3, "00:00:03", "three")
    writer.write_translation(3, "THREE")
    writer.finalize_no_translation(2)  # same language, no translation needed
    writer.write_translation(1, "ONE")
    stamp = writer._session_ts  # saved: close clears it
    writer.close()

    lines = _entries(tmp_path / f"livetrans_{stamp}_all.txt")
    assert [line.split("] ", 1)[1] for line in lines] == ["one", "two", "three"]


def test_no_entry_is_registered_twice(tmp_path):
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "first")
    writer.write_original(1, "00:00:02", "corrected")
    writer.write_translation(1, "FIRST")
    stamp = writer._session_ts  # saved: close clears it
    writer.close()
    lines = _entries(tmp_path / f"livetrans_{stamp}_all.txt")
    assert len(lines) == 1


def test_a_translation_without_an_original_is_discarded(tmp_path):
    """A translation whose original was never recorded has no session to
    belong to. The old behavior wrote an orphan "no original" line into
    whatever session was open — the exact channel through which a refused
    entry's late translation could land in the *next* meeting's files."""
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "recorded line")  # opens the session
    stamp = writer._session_ts
    result = writer.write_translation(99, "orphan")
    writer.close()
    assert result == TranscriptWriter.WRITE_SKIPPED
    text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "recorded line" in text
    assert "orphan" not in text


# --- meeting record content ------------------------------------------------


def test_the_meeting_record_carries_language_duration_and_provenance(tmp_path):
    writer = _writer(tmp_path)
    writer.set_session_info(
        asr_engine="SenseVoice Small",
        translation_model="some-model",
        source_language="ru",
        target_language="zh",
    )
    writer.write_original(
        1, "09:15:00", "Здравствуйте", language="ru", duration=2.5
    )
    writer.write_translation(1, "你好")
    stamp = writer._session_ts  # saved: close clears it
    writer.close()

    text = (
        tmp_path / f"livetrans_{stamp}_meeting.md"
    ).read_text(encoding="utf-8")
    assert "# Meeting record" in text
    assert "**09:15:00** · ru · 2.5s" in text
    assert "Здравствуйте" in text
    assert "> 你好" in text
    assert "## Summary" in text
    assert "SenseVoice Small" in text
    assert "some-model" in text


def test_the_plain_text_files_end_with_a_session_summary(tmp_path):
    writer = _writer(tmp_path)
    writer.write_original(1, "09:15:00", "hello")
    writer.write_translation(1, "hola")
    stamp = writer._session_ts  # saved: close clears it
    writer.close()
    for kind in TranscriptWriter.KINDS:
        text = (tmp_path / f"livetrans_{stamp}_{kind}.txt").read_text("utf-8")
        assert "# Session ended at" in text
        assert "1 entries" in text


def test_the_metadata_sidecar_describes_the_session(tmp_path):
    writer = _writer(tmp_path)
    writer.set_session_info(asr_engine="Whisper medium", target_language="en")
    writer.write_original(1, "09:15:00", "hello", duration=3.0)
    writer.write_translation(1, "hola")
    writer.write_original(2, "09:15:10", "again", duration=1.0)
    writer.finalize_no_translation(2)
    stamp = writer._session_ts  # saved: close clears it
    writer.close()

    meta = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert meta["entries"] == 2
    assert meta["translated"] == 1
    assert meta["untranslated"] == 1
    assert meta["speech_seconds"] == 4.0
    assert meta["asr_engine"] == "Whisper medium"
    assert meta["target_language"] == "en"


def test_a_restarted_session_starts_from_a_clean_slate(tmp_path, monkeypatch):
    """The app stops and restarts the pipeline with the same writer instance.
    close() used to keep the counters, speech time and session info, so the
    second meeting's sidecar and footer reported the first meeting's totals
    and engine."""
    import transcript_writer as tw
    from datetime import datetime as real_datetime, timedelta

    # A clock that advances 90s per now() call: two sessions opened from the
    # same instance land on different timestamps instead of merging.
    clock = {"t": real_datetime(2026, 1, 1, 9, 0, 0)}

    class FakeDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            clock["t"] += timedelta(seconds=90)
            return clock["t"]

    monkeypatch.setattr(tw, "datetime", FakeDatetime)

    writer = _writer(tmp_path)
    writer.set_session_info(asr_engine="whisper")
    writer.write_original(1, "10:00:00", "hello", duration=1.5)
    writer.write_translation(1, "你好")
    writer.write_original(2, "10:00:05", "world", duration=1.5)
    writer.finalize_no_translation(2)
    writer.close()

    writer.set_enabled(True)  # what a restart does
    writer.write_original(10, "11:00:00", "second session", duration=1.0)
    writer.finalize_no_translation(10)
    writer.close()

    sessions = read_session_meta(tmp_path)
    assert len(sessions) == 2
    second = sessions[0]  # newest first
    assert second["entries"] == 1
    assert second["speech_seconds"] == 1.0
    assert second["translated"] == 0
    assert "asr_engine" not in second  # the first session's engine must not leak

    footer = (tmp_path / f"livetrans_{second['session']}_all.txt").read_text("utf-8")
    assert "1 entries" in footer


def test_same_second_sessions_get_suffixed_stamps(tmp_path, monkeypatch):
    """Two sessions begun within the same second must not share a file set:
    the second gets an _01-suffixed stamp. The stamp probe checks the real
    file names (with extensions) — extension-less probing never collided
    and silently let the second session fail its exclusive create."""
    import transcript_writer as tw
    from datetime import datetime as real_datetime

    frozen = real_datetime(2026, 1, 1, 9, 0, 0)

    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(tw, "datetime", FrozenDatetime)

    first = TranscriptWriter(tmp_path)
    first.set_enabled(True)
    stamp_a = first.begin_session()
    first.write_original(1, "09:00:00", "first meeting")
    first.finalize_no_translation(1)
    first.end_session()

    second = TranscriptWriter(tmp_path)
    second.set_enabled(True)
    stamp_b = second.begin_session()
    second.write_original(1, "09:00:00", "second meeting")
    second.finalize_no_translation(1)
    summary = second.end_session()

    assert stamp_a == "20260101_090000"
    assert stamp_b == "20260101_090000_01", "same-second session must be suffixed"
    # Each meeting's file carries its own content only.
    first_all = (tmp_path / f"livetrans_{stamp_a}_all.txt").read_text("utf-8")
    second_all = (tmp_path / f"livetrans_{stamp_b}_all.txt").read_text("utf-8")
    assert "first meeting" in first_all and "second meeting" not in first_all
    assert "second meeting" in second_all and "first meeting" not in second_all
    assert summary["session"] == stamp_b


def test_live_session_sidecar_has_no_ended_and_seal_writes_it_once(tmp_path):
    """A live session's sidecar must not claim an end time (the records
    layer reads its absence as "still recording"), and the seal fixes one
    timestamp shared by the footer, the sidecar and the returned summary."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    writer.write_translation(1, "hola")
    # Saved before the seal: end_session clears the writer's session state.
    stamp = writer._session_ts

    live_meta = (tmp_path / f"livetrans_{stamp}_meta.json").read_text(
        "utf-8"
    )
    assert '"ended": null' in live_meta

    summary = writer.end_session()
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    footer = (
        tmp_path / f"livetrans_{stamp}_all.txt"
    ).read_text("utf-8")
    assert sealed["ended"] is not None
    assert sealed["ended"] == summary["ended"]
    # The footer's "Session ended at" line carries the same instant.
    ended_line = next(
        line for line in footer.splitlines() if line.startswith("# Session ended")
    )
    assert summary["ended"].replace("T", " ") in ended_line


def test_write_original_rejects_entries_from_another_session(tmp_path):
    """The write-time identity check: audio registered for session A must
    never land in session B's files, even if B is the session open when the
    write arrives (an end+begin raced the queue)."""
    writer = _writer(tmp_path)
    stamp_a = writer.begin_session()
    writer.write_original(1, "09:00:00", "belongs to A")
    writer.end_session()

    stamp_b = writer.begin_session()
    assert stamp_b != stamp_a
    # A straggler carrying A's identity arrives while B is open.
    writer.write_original(2, "09:00:05", "stale from A", session=stamp_a)
    all_text = (tmp_path / f"livetrans_{stamp_b}_all.txt").read_text("utf-8")
    assert "stale from A" not in all_text
    # A None expectation (legacy auto-open callers) is still accepted.
    writer.write_original(3, "09:00:06", "no expectation")
    all_text = (tmp_path / f"livetrans_{stamp_b}_all.txt").read_text("utf-8")
    assert "no expectation" in all_text
    writer.end_session()


def test_disabled_writer_records_nothing(tmp_path):
    writer = TranscriptWriter(tmp_path)
    writer.set_enabled(False)
    writer.write_original(1, "00:00:01", "secret")
    writer.write_translation(1, "secret")
    writer.close()
    assert list(tmp_path.glob("livetrans_*")) == []


# --- session listing -------------------------------------------------------


def test_sessions_are_listed_newest_first_with_their_metadata(tmp_path):
    for stamp, entries in (("20260101_090000", 2), ("20260102_090000", 1)):
        (tmp_path / f"livetrans_{stamp}_all.txt").write_text(
            "".join(f"[09:00:0{i}] line {i}\n\n" for i in range(entries)),
            encoding="utf-8",
        )
        (tmp_path / f"livetrans_{stamp}_meta.json").write_text(
            json.dumps({"entries": entries, "asr_engine": f"engine-{stamp}"}),
            encoding="utf-8",
        )

    sessions = read_session_meta(tmp_path)
    assert [s["session"] for s in sessions] == [
        "20260102_090000", "20260101_090000"
    ]
    assert sessions[0]["entries"] == 1
    assert sessions[1]["entries"] == 2
    assert sessions[1]["asr_engine"] == "engine-20260101_090000"
    assert sessions[0]["files"]["all"].endswith("livetrans_20260102_090000_all.txt")


def test_listing_survives_a_session_with_no_sidecar(tmp_path):
    """Sessions recorded before the sidecar existed, or ended by a crash."""
    (tmp_path / "livetrans_20250101_120000_all.txt").write_text(
        "# Session started at 2025-01-01 12:00:00\n[12:00:01] hello\n\n",
        encoding="utf-8",
    )
    sessions = read_session_meta(tmp_path)
    assert len(sessions) == 1
    assert sessions[0]["session"] == "20250101_120000"
    assert sessions[0]["entries"] == 1
    assert sessions[0]["started"].startswith("2025-01-01T12:00:00")


def test_deleting_a_session_removes_all_of_its_files(tmp_path):
    from transcript_writer import _session_file_candidates

    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "hello")
    writer.write_translation(1, "hola")
    stamp = writer._session_ts  # saved: close clears it
    writer.close()
    assert [p for p in _session_file_candidates(tmp_path, stamp) if p.exists()]

    assert delete_session(tmp_path, stamp) == []
    assert [p for p in _session_file_candidates(tmp_path, stamp) if p.exists()] == []


def _same_second_pair(tmp_path):
    """Two complete file sets sharing one second: the bare stamp and its
    ``_01`` suffixed sibling — the shape _unique_stamp produces when a
    second session begins within the same second."""
    bare = "20260101_090000"
    sibling = f"{bare}_01"
    sets = {}
    for stamp, marker in ((bare, "parent"), (sibling, "sibling")):
        files = {
            f"livetrans_{stamp}_all.txt": f"[09:00:00] {marker} line\n",
            f"livetrans_{stamp}_original.txt": f"[09:00:00] {marker} line\n",
            f"livetrans_{stamp}_translation.txt": f"[09:00:00] {marker} TL\n",
            f"livetrans_{stamp}_meeting.md": f"# Meeting record\n\n{marker}\n",
            f"livetrans_{stamp}_meta.json": '{"session": "%s"}' % stamp,
            f"livetrans_{stamp}_summary.md": f"# {marker} minutes",
            f"livetrans_{stamp}_summary_meta.json": '{"provider_name": "X"}',
            # A staged sibling a crashed atomic commit can leave behind.
            f"livetrans_{stamp}_summary.md.tmp123.0123456789abcdef":
                "half-written",
        }
        for name, content in files.items():
            (tmp_path / name).write_text(content, encoding="utf-8")
        sets[stamp] = files
    return bare, sibling, sets


def test_deleting_a_bare_stamp_spares_its_same_second_suffixed_sibling(tmp_path):
    """[B1 regression] delete_session must enumerate this session's exact
    files, never a ``livetrans_{stamp}_*`` prefix glob: a bare stamp is the
    *prefix* of its same-second suffixed siblings, and the glob unlinked the
    sibling's whole file set — original, translation, combined text,
    Markdown, meta, AI minutes and staged temps — along with the target's,
    destroying a second meeting the user never asked to delete."""
    bare, sibling, sets = _same_second_pair(tmp_path)

    assert delete_session(tmp_path, bare) == []

    # The bare stamp's own files are all gone, temp siblings included.
    for name in sets[bare]:
        assert not (tmp_path / name).exists(), name
    # Every _01 file survives with its content intact.
    for name, content in sets[sibling].items():
        survived = tmp_path / name
        assert survived.exists(), name
        assert survived.read_text(encoding="utf-8") == content, name


def test_deleting_the_suffixed_sibling_spares_the_bare_stamp(tmp_path):
    """The delete must be exact in the other direction too: removing the
    ``_01`` sibling never touches the bare parent stamp's files."""
    bare, sibling, sets = _same_second_pair(tmp_path)

    assert delete_session(tmp_path, sibling) == []

    for name in sets[sibling]:
        assert not (tmp_path / name).exists(), name
    for name, content in sets[bare].items():
        survived = tmp_path / name
        assert survived.exists(), name
        assert survived.read_text(encoding="utf-8") == content, name


def test_delete_session_summary_names_match_the_records_layer(tmp_path):
    """The summary file names delete_session enumerates must be exactly the
    records layer's _summary_paths — the two definitions are pinned together
    so neither can drift."""
    import meeting_records as records_module
    from transcript_writer import _session_file_candidates

    stamp = "20260101_090000"
    md, meta = records_module._summary_paths(tmp_path, stamp)
    enumerated = {p.name for p in _session_file_candidates(tmp_path, stamp)}
    assert md.name in enumerated
    assert meta.name in enumerated


def test_listing_an_absent_directory_is_empty(tmp_path):
    assert read_session_meta(tmp_path / "nope") == []


# --- The invariant every producer must respect ------------------------------


def test_one_unfinished_entry_stalls_every_later_one(tmp_path):
    """Documents *why* the pipeline must close out every message it registers.

    Because entries are released in order, a single one that never completes
    holds back all of its successors. A path that returns without finalizing
    (a translation discarded by a mid-session model switch, say) therefore
    freezes the whole record, not just its own line.
    """
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "never finished")
    writer.write_original(2, "00:00:02", "second")
    writer.write_translation(2, "SECOND")
    writer.write_original(3, "00:00:03", "third")
    writer.write_translation(3, "THIRD")

    path = tmp_path / f"livetrans_{writer._session_ts}_all.txt"
    assert _entries(path) == []  # all three are stuck behind #1

    writer.finalize_no_translation(1)
    assert len(_entries(path)) == 3
    writer.close()


class _SupersededApp:
    """Drives _translate_async down the "model switched mid-flight" path."""

    _translate_async = None  # bound below, after main imports

    def __init__(self, transcript, translated="TRANSLATED"):
        self._transcript = transcript
        self._overlay = None
        self._subwin = None
        self._translator_generation = 99          # newer than the request's
        self._translated = translated
        self.commits = []
        # _translate_async keys its session-work release to the msg's
        # generation (session_generation arg, falling back to the current
        # one) and goes through the tracker; the stub mirrors the app's
        # fields so that bookkeeping works. On a fresh tracker the
        # superseded request's release is a no-op (generation unknown),
        # which is exactly the stub's situation.
        from main import _SessionWorkTracker

        self._session_generation = 1
        self._session_work = _SessionWorkTracker()

    # The request was snapshotted under an older generation.
    def _commit_translation_result(self, msg_id, text, translated, generation):
        self.commits.append((msg_id, generation))
        return False                              # superseded

    # _translate_async (the wrapper) funnels into the real inner method,
    # which is bound onto the class below together with the wrapper.
    _translate_async_inner = None

    class _Stub:
        def __init__(self, outer):
            self._outer = outer
            self.last_usage = (0, 0)

        def translate_iter(self, text, source_lang):
            yield self._outer._translated


def test_a_superseded_translation_still_closes_out_its_entry(tmp_path):
    """A mid-session model switch discards the result for history purposes, but
    the entry must still be released or it stalls every later one."""
    main = pytest.importorskip("main")
    _SupersededApp._translate_async = main.LiveTranslateApp._translate_async
    _SupersededApp._translate_async_inner = (
        main.LiveTranslateApp._translate_async_inner
    )

    writer = _writer(tmp_path)
    app = _SupersededApp(writer)
    writer.write_original(1, "00:00:01", "in flight when the model changed")
    writer.write_original(2, "00:00:02", "after")
    writer.write_translation(2, "AFTER")

    path = tmp_path / f"livetrans_{writer._session_ts}_all.txt"
    assert _entries(path) == []                   # #2 waits behind #1

    app._translate_async(
        1, "in flight when the model changed", "en",
        request_translator=_SupersededApp._Stub(app), generation=1,
    )

    assert app.commits == [(1, 1)]                # it did go through the guard
    lines = _entries(path)
    assert len(lines) == 2, "the superseded entry did not unblock the record"
    assert lines[0].endswith("in flight when the model changed")
    writer.close()


def test_a_superseded_empty_translation_also_closes_out(tmp_path):
    main = pytest.importorskip("main")
    _SupersededApp._translate_async = main.LiveTranslateApp._translate_async
    _SupersededApp._translate_async_inner = (
        main.LiveTranslateApp._translate_async_inner
    )

    writer = _writer(tmp_path)
    app = _SupersededApp(writer, translated="")
    writer.write_original(1, "00:00:01", "empty result")
    app._translate_async(
        1, "empty result", "en",
        request_translator=_SupersededApp._Stub(app), generation=1,
    )
    path = tmp_path / f"livetrans_{writer._session_ts}_all.txt"
    assert len(_entries(path)) == 1
    writer.close()


def test_late_translation_of_a_refused_original_never_enters_new_session(tmp_path):
    """End+begin raced the queue: the original was refused (its session had
    already closed) and a *new* session is open when its translation
    returns. Neither the original nor the translation may land in the new
    meeting's files."""
    writer = _writer(tmp_path)
    stamp_a = writer.begin_session()
    writer.write_original(1, "09:00:00", "belongs to A")
    writer.end_session()

    stamp_b = writer.begin_session()
    # The straggler original (refused) and then its late translation.
    original_result = writer.write_original(
        2, "09:00:05", "stale from A", session=stamp_a
    )
    assert original_result == TranscriptWriter.WRITE_SESSION_MISMATCH
    translation_result = writer.write_translation(
        2, "late translation of stale audio", session=stamp_a
    )
    assert translation_result == TranscriptWriter.WRITE_SESSION_MISMATCH
    writer.end_session()

    for kind in ("all", "translation", "original"):
        text = (
            tmp_path / f"livetrans_{stamp_b}_{kind}.txt"
        ).read_text("utf-8")
        assert "stale from A" not in text, kind
        assert "late translation" not in text, kind


def test_meta_create_failure_rolls_back_the_whole_file_set(tmp_path, monkeypatch):
    """The initial sidecar is part of the transaction: when it cannot be
    created (here: the exclusive create collides with a pre-existing file),
    the text files and the Markdown record roll back and begin_session
    reports failure — no phantom session, no stray files."""
    import transcript_writer as tw
    from datetime import datetime as real_datetime

    writer = TranscriptWriter(tmp_path)
    writer.set_enabled(True)
    # Pre-seed a colliding stamp on disk (an interrupted earlier attempt).
    base = real_datetime.now().strftime("%Y%m%d_%H%M%S")
    (tmp_path / f"livetrans_{base}_meta.json").write_text("{}", encoding="utf-8")

    # Freeze the clock so the new session picks the same base stamp; the
    # suffixed probe then finds a free _01 stamp whose meta does not exist,
    # so instead sabotage the exclusive create itself.
    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.strptime(base, "%Y%m%d_%H%M%S")

    monkeypatch.setattr(tw, "datetime", FrozenDatetime)

    real_open = tw.os.open

    def failing_open(path, flags, *args, **kwargs):
        if str(path).endswith("_meta.json"):
            raise FileExistsError(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tw.os, "open", failing_open)

    result = writer.begin_session()
    assert result is None
    assert writer.has_active_session() is False
    # The bare stamp's meta is the pre-seeded file; the attempt's suffixed
    # text/Markdown files must not linger anywhere.
    leftovers = sorted(p.name for p in tmp_path.glob("livetrans_*"))
    assert leftovers == [f"livetrans_{base}_meta.json"], leftovers


def test_abort_session_after_failed_close_keeps_files_and_clears_state(tmp_path):
    """A close that raises mid-seal must leave the record as written (no
    extra footer/ended invented afterwards) and clear every piece of
    in-memory session state, so the next begin starts clean and the records
    layer classifies the meeting as interrupted (no footer)."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "kept line")
    stamp = writer._session_ts  # saved: close/abort clears it

    # Force the seal to fail after the state has been written to.
    import transcript_writer as tw

    def failing_footer():
        raise OSError("disk full")

    monkeypatch_footer = failing_footer
    original_footer = writer._write_summary_footer_locked
    writer._write_summary_footer_locked = monkeypatch_footer
    try:
        raised = False
        try:
            writer.end_session()
        except OSError:
            raised = True
        assert raised, "the seal failure must propagate to the caller"
    finally:
        writer._write_summary_footer_locked = original_footer

    summary = writer.abort_session()
    # The aborted session's snapshot describes the session that was.
    assert summary is not None and summary["session"] == stamp
    assert writer.has_active_session() is False
    assert writer.active_session() is None
    # Files survive exactly as far as they got: no footer was appended.
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "kept line" in all_text
    assert "# Session ended" not in all_text

    # The aborted session is marked interrupted in its sidecar, so a footer
    # that already landed elsewhere cannot make it read as completed.
    aborted_meta = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert aborted_meta["session_status"] == "interrupted"
    assert summary["session_status"] == "interrupted"

    # The writer is reusable: a fresh session starts from a clean slate.
    stamp2 = writer.begin_session()
    assert stamp2 is not None
    assert writer.write_original(1, "10:00:00", "new session") == (
        TranscriptWriter.WRITE_RECORDED
    )
    writer.end_session()


def test_failed_original_write_rolls_back_memory_and_counts(tmp_path, monkeypatch):
    """When the original line cannot reach the file, nothing about the entry
    may survive in memory: _pending/_order/_entry_sessions hold nothing the
    seal could re-emit, and speech_seconds never counted it."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "good line", duration=1.0)
    stamp = writer._session_ts  # saved: close clears it

    # Sabotage the file write for the next entry only.
    real_write = writer._write_locked

    def failing_write(kind, text):
        if "doomed line" in text:
            return False
        return real_write(kind, text)

    monkeypatch.setattr(writer, "_write_locked", failing_write)
    result = writer.write_original(2, "09:00:05", "doomed line", duration=2.0)
    monkeypatch.undo()

    assert result == TranscriptWriter.WRITE_FAILED
    # The seal must not resurrect the failed entry from pending state.
    assert 2 not in writer._pending
    assert 2 not in writer._entry_sessions
    assert 2 not in writer._order
    # Only the persisted entry's speech time counts.
    assert writer.summary()["speech_seconds"] == 1.0

    summary = writer.end_session()
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "good line" in all_text
    assert "doomed line" not in all_text, "a failed write must not reappear"
    assert summary["session_status"] == "completed"
    assert summary["entries"] == 1


def test_footer_write_failure_seals_as_interrupted(tmp_path, monkeypatch):
    """A footer that cannot be written degrades the seal: the session closes
    (no exception — the failure is a verdict, not a crash) and the final
    sidecar carries session_status=interrupted."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    writer.write_translation(1, "hola")
    stamp = writer._session_ts  # saved: close clears it

    def failing_footer():
        return False

    monkeypatch.setattr(
        writer, "_write_summary_footer_locked", failing_footer
    )
    summary = writer.end_session()
    monkeypatch.undo()

    assert summary["session_status"] == "interrupted"
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert sealed["session_status"] == "interrupted"
    assert sealed["ended"] is not None  # the seal moment is still recorded


def test_final_meta_write_failure_leaves_no_completed_verdict(tmp_path, monkeypatch):
    """When neither the final sidecar nor its interrupted retry can be
    written, the on-disk sidecar keeps the live "active" status — never a
    fabricated "completed" — and the returned summary says interrupted."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    stamp = writer._session_ts  # saved: close clears it

    monkeypatch.setattr(
        writer, "_write_meta_locked", lambda *a, **k: False
    )
    summary = writer.end_session()
    monkeypatch.undo()

    assert summary["session_status"] == "interrupted"
    # The on-disk sidecar is the live version (active, no ended): the seal
    # could not commit any verdict.
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert sealed["session_status"] == "active"
    assert sealed.get("ended") is None


def test_live_sidecar_carries_active_status(tmp_path):
    """A session opened (or auto-opened) writes session_status=active into
    its sidecar from the very first commit — the records layer's primary
    signal for "still recording"."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")  # auto-opens the session
    stamp = writer._session_ts
    meta = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert meta["session_status"] == "active"
    summary = writer.end_session()
    assert summary["session_status"] == "completed"


def test_rename_live_session_goes_through_the_writer_lock(tmp_path):
    """Renaming a live session is the writer's job: one lock, one sidecar
    writer. The rename lands in the sidecar immediately, and the seal's
    final commit still carries it (the whitelisted user-field merge)."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    stamp = writer._session_ts  # saved: close clears it

    assert writer.rename_session("课前准备") is True
    live = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert live["title"] == "课前准备"

    # The seal keeps the title: its meta write merges the user fields.
    summary = writer.end_session()
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert sealed["title"] == "课前准备"
    assert sealed["session_status"] == "completed"

    # Refused with no open session, and with an empty title.
    assert writer.rename_session("x") is False
    assert writer.begin_session() is not None
    assert writer.rename_session("   ") is False
    writer.end_session()


def test_rename_refused_while_ending(tmp_path):
    """ENDING refuses: the seal owns the sidecar then, and a concurrent
    rename write could overwrite the just-committed status."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    stamp = writer._session_ts  # saved: close clears it

    writer._ending = True  # the end_session window
    try:
        assert writer.rename_session("nope") is False
    finally:
        writer._ending = False
    writer.end_session()
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert "title" not in sealed


def test_rename_expected_session_closes_the_identity_chain(tmp_path):
    """The caller names the meeting it saw when the rename was issued. When
    the session open at write time is a different one (an end+begin raced
    the rename dialog), the write is refused — the new meeting must not
    inherit the title the user typed for the old one."""
    writer = _writer(tmp_path)
    first = writer.begin_session()
    assert first is not None
    writer.write_original(1, "09:00:00", "hello")
    assert writer.rename_session("对旧会议的标题", expected_session=first) is True
    writer.end_session()

    # A new session exists now; a rename still carrying the OLD stamp
    # must be refused, and the new session's sidecar must stay untitled.
    second = writer.begin_session()
    assert second is not None and second != first
    assert writer.rename_session("对旧会议的标题", expected_session=first) is False
    meta = json.loads(
        (tmp_path / f"livetrans_{second}_meta.json").read_text("utf-8")
    )
    assert "title" not in meta
    # The matching stamp still renames normally.
    assert writer.rename_session("新会议标题", expected_session=second) is True
    meta = json.loads(
        (tmp_path / f"livetrans_{second}_meta.json").read_text("utf-8")
    )
    assert meta["title"] == "新会议标题"
    # A wrong expected stamp also refuses while the writer holds no session.
    writer.end_session()
    assert writer.rename_session("无会话", expected_session=second) is False


def test_held_session_reports_the_exact_stamp_across_states(tmp_path):
    """held_session() is the exact file-ownership answer callers must use
    instead of path matching (a bare stamp is a prefix of its same-second
    suffixed siblings): the stamp while open or ending, still the stamp in
    the half-dead close neither active_session() nor ending_session()
    reports, and None once every resource is released."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    stamp = writer._session_ts
    assert writer.held_session() == stamp

    writer._ending = True  # the end_session window
    assert writer.active_session() is None
    assert writer.held_session() == stamp

    writer._ending = False
    writer._session_open = False  # a half-dead close
    assert writer.active_session() is None
    assert writer.ending_session() is None
    assert writer.held_session() == stamp

    writer.abort_session()
    assert writer.held_session() is None


def test_completed_meta_commits_only_after_content_files_closed(tmp_path):
    """The completed verdict must be the last thing committed: when the
    final sidecar is written, every content handle must already be closed
    and durable (flush+fsync+close ran before it). Otherwise a later
    flush/close failure would leave a "completed" marker over unwritten
    buffers."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    writer.write_translation(1, "hola")
    stamp = writer._session_ts  # saved: close clears it

    order = []
    real_seal = writer._seal_content_files_locked
    real_meta = writer._write_meta_locked

    def sealing_seal():
        order.append("seal-start")
        ok = real_seal()
        order.append("seal-done" if ok else "seal-failed")
        return ok

    def ordered_meta(*args, **kwargs):
        closed = all(
            fp is None or fp.closed for fp in writer._files.values()
        )
        order.append(f"meta(content-closed={closed})")
        return real_meta(*args, **kwargs)

    writer._seal_content_files_locked = sealing_seal
    writer._write_meta_locked = ordered_meta

    summary = writer.end_session()

    assert order[0] == "seal-start"
    assert order[1] == "seal-done"
    assert order[2] == "meta(content-closed=True)"
    assert summary["session_status"] == "completed"
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert sealed["session_status"] == "completed"


def test_content_flush_failure_seals_as_interrupted(tmp_path, monkeypatch):
    """A content file that cannot be flushed/fsynced/closed degrades the
    seal: the verdict can only be interrupted, never completed."""
    import transcript_writer as tw

    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")
    stamp = writer._session_ts  # saved: close clears it

    real_fsync = tw.os.fsync

    def failing_fsync(fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(tw.os, "fsync", failing_fsync)

    summary = writer.end_session()
    monkeypatch.undo()

    assert summary["session_status"] == "interrupted"
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert sealed["session_status"] == "interrupted"


def test_opened_without_session_open_still_reports_resources(tmp_path):
    """A close that died between flags (_opened=True, _session_open=False)
    must still be recognized by has_open_resources — the check the IDLE
    broadcast gates on. has_active_session/has_open_session alone would
    both answer False and let the UI claim done over live handles."""
    writer = _writer(tmp_path)
    writer.write_original(1, "09:00:00", "hello")  # a real open session
    # Simulate the half-dead state a failed close leaves behind.
    writer._session_open = False
    assert writer.has_active_session() is False
    assert writer.has_open_session() is False
    assert writer._opened is True
    assert writer.has_open_resources() is True
    writer.abort_session()
    assert writer.has_open_resources() is False


def test_close_after_exception_can_start_a_new_session(tmp_path):
    """The pipeline-shutdown close() shares the abort fallback with the
    button-end path: a throwing seal is contained (logged, not raised),
    resources are released, and the writer can open the next session."""
    import transcript_writer as tw

    writer = TranscriptWriter(tmp_path)
    writer.set_enabled(True)
    writer.set_recording(True)
    assert writer.write_original(1, "09:00:00", "hello") == (
        TranscriptWriter.WRITE_RECORDED
    )
    stamp = writer._session_ts

    def exploding_footer():
        raise OSError("disk gone")

    monkeypatched = exploding_footer
    original = writer._write_summary_footer_locked
    writer._write_summary_footer_locked = monkeypatched
    try:
        # close() must swallow the failure (stop() has to continue) —
        # the abort fallback runs inside it.
        writer.close()
    finally:
        writer._write_summary_footer_locked = original

    assert writer.has_open_resources() is False
    sealed = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert sealed["session_status"] == "interrupted"

    # The next session opens and seals normally.
    stamp2 = writer.begin_session()
    assert stamp2 is not None and stamp2 != stamp
    assert writer.write_original(1, "10:00:00", "next meeting") == (
        TranscriptWriter.WRITE_RECORDED
    )
    summary = writer.end_session()
    assert summary["session_status"] == "completed"
    assert writer.has_open_resources() is False


# --- auto-open adoption keeps the session's opening work [B2] ----------------

import queue as _queue
import threading as _threading


class _AdoptionApp:
    """A minimal LiveTranslateApp stand-in driving the *real* session-work
    machinery — _SessionWorkTracker, _enqueue_asr, _process_segment_text,
    _adopt_auto_opened_session, _do_interim_asr and the stale-segment guard —
    against a real TranscriptWriter. The translation executor is sidestepped:
    the target language equals the source, so every entry takes the
    same-language finalize path and never submits a job.

    Only what the bound real methods touch is stubbed (ASR, the language
    setting, sentence splitting); the adoption itself, the tracker and the
    writer are the production code under test."""

    _bound = False

    def __init__(self, writer, main):
        self._main = main
        self._transcript = writer
        self._session_state = main.SessionState.IDLE
        self._session_generation = 0
        self._session_work = main._SessionWorkTracker()
        self._session_boundary_lock = _threading.RLock()
        self._session_work_lock = _threading.Lock()
        self._session_work_seq = 0
        self._stop_event = _threading.Event()
        self._asr_queue = _queue.Queue()
        # The ENDING gate _run_session_end raises around its flush.
        self._session_end_gating = False
        self._overlay = None
        self._subwin = None
        self._panel = None
        self._translator = None
        self._asr_config = None
        self._running = True
        self._paused = False
        # The ENDING-flush hand-off (_flush_for_session_end) gates on ASR
        # readiness; tests that need the not-ready branch flip it.
        self._asr_ready = True
        self._target_language = "ru"
        self._asr_count = 0
        self._msg_id = 0
        self._last_original = ""
        self._last_msg_id = 0
        # Interim-path state (used by the _do_interim_asr tests).
        self._vad_lock = _threading.Lock()
        self._vad = None
        self._interim_pending = ""
        self._interim_active = False
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        # Notified state changes: [(state, session_id), ...].
        self.notifications = []
        self._session_ui_callback = self._record_notification

    def _record_notification(self, state, session_id, summary=None):
        self.notifications.append((state, session_id))

    # --- stubs for what only the real app owns -------------------------------
    def _get_asr_language_setting(self):
        return "auto"

    def _run_asr(self, audio, kind, **kwargs):
        raise AssertionError("no ASR stub installed for this test")

    def _split_sentences(self, text, lang):
        return [text]

    # --- bound from LiveTranslateApp (see _make_adoption_app) ----------------
    _enqueue_asr = None
    _session_snapshot = None
    _next_session_work_id = None
    _requeue_stop_sentinel = None
    _release_queued_work = None
    _process_segment = None
    _process_segment_text = None
    _adopt_auto_opened_session = None
    _notify_session_state = None
    _finalize_untranslated = None
    _record_session_info = None
    _publish_transcript_paths = None
    begin_recording_session = None
    _strip_committed_overlap = None
    _is_short_utterance = None
    _is_substantial_echo = None
    _buffer_interim_fragment = None
    _do_interim_asr = None
    _reset_interim_state = None
    _process_interim_final = None
    _flush_for_session_end = None
    _enqueue_final_segment = None
    _run_session_end = None
    pause = None
    resume = None


def _make_adoption_app(tmp_path):
    main = pytest.importorskip("main")
    if not _AdoptionApp._bound:
        real = main.LiveTranslateApp
        for name in (
            "_enqueue_asr", "_session_snapshot", "_next_session_work_id",
            "_requeue_stop_sentinel", "_release_queued_work",
            "_process_segment", "_process_segment_text",
            "_adopt_auto_opened_session",
            "_notify_session_state", "_finalize_untranslated",
            "_record_session_info", "_publish_transcript_paths",
            "begin_recording_session", "_strip_committed_overlap",
            "_is_substantial_echo", "_buffer_interim_fragment",
            "_do_interim_asr", "_reset_interim_state",
            "_process_interim_final", "_flush_for_session_end",
            "_enqueue_final_segment", "_run_session_end",
            "pause", "resume",
        ):
            setattr(_AdoptionApp, name, getattr(real, name))
        # A staticmethod retrieved from the class is a plain function; store
        # it as a staticmethod again so the stub's calls keep its signature.
        setattr(
            _AdoptionApp, "_is_short_utterance",
            staticmethod(real._is_short_utterance),
        )
        for name in (
            "_ECHO_BOUNDARY", "_ECHO_MIN_UNSPACED", "_INTERIM_PENDING_MAX",
        ):
            setattr(_AdoptionApp, name, getattr(real, name))
        _AdoptionApp._bound = True
    writer = _writer(tmp_path)
    return _AdoptionApp(writer, main), writer, main


def test_items_enqueued_before_adoption_all_land_in_the_session(tmp_path):
    """[B2 regression] The first entry's write auto-opens the session and the
    adoption claims it — without bumping the generation: items enqueued
    *before* the auto-open carry that same generation and are this meeting's
    opening speech. The old bump made the stale-segment guard refuse them,
    silently dropping their audio (and, on the interim path, trimming it)."""
    app, writer, main = _make_adoption_app(tmp_path)

    # Two queue items enter while no session exists yet (the moments right
    # after the pipeline starts recording): both are counted from enqueue on.
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-1")
        app._enqueue_asr("vad_flush", "audio-2")
    item1 = app._asr_queue.get_nowait()
    item2 = app._asr_queue.get_nowait()
    gen = app._session_generation
    assert item1[2] is not None and item2[2] is not None  # counted ("pass")
    assert app._session_work.pending_count(gen) == 2

    # The first item is processed: its write auto-opens the session and the
    # adoption claims it — at the SAME generation, one ACTIVE notification.
    assert app._process_segment_text(
        "первая фраза", "ru", 100,
        generation=item1[3], expected_session=item1[4],
    ) is True
    stamp = writer.active_session()
    assert stamp is not None
    assert app._session_generation == gen
    assert app.notifications == [("active", stamp)]
    assert app._session_state == main.SessionState.ACTIVE

    # The second item — enqueued BEFORE the auto-open — still commits into
    # the same session instead of being discarded by the stale guard.
    assert app._process_segment_text(
        "вторая фраза", "ru", 100,
        generation=item2[3], expected_session=item2[4],
    ) is True
    writer.end_session()
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "первая фраза" in all_text
    assert "вторая фраза" in all_text


def test_ending_waits_for_work_enqueued_before_adoption(tmp_path):
    """The ENDING wait must cover queue work enqueued before the auto-open:
    the counts exist from enqueue on (register() auto-creates the not-yet-
    adopted generation), so wait_idle blocks on them instead of closing the
    session out from under in-flight recognition."""
    app, writer, main = _make_adoption_app(tmp_path)

    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-in-flight")
    item = app._asr_queue.get_nowait()
    gen = item[3]
    assert app._session_work.pending_count(gen) == 1

    # A first entry adopts the session at this same generation.
    assert app._process_segment_text(
        "фраза", "ru", 50, generation=gen, expected_session=None
    ) is True
    assert app._session_generation == gen
    assert app._session_state == main.SessionState.ACTIVE

    # The user ends the meeting: the end flips CLOSING and must wait for the
    # pre-adoption item still in flight.
    app._session_work.start_closing(gen)
    assert app._session_work.wait_idle(gen, timeout=0.0) is False
    # The ASR loop's finally releases the item's count — only then idle.
    app._session_work.release(gen, item[2])
    assert app._session_work.wait_idle(gen, timeout=0.0) is True


class _FakeInterimVAD:
    """Just enough VAD for _do_interim_asr: a peekable buffer and a
    recording trimmer. flush()/force_flush()/_reset() serve the ENDING
    hand-off tests — both flush variants return and clear the buffer
    (None when empty), matching the real VAD's contract for the paths
    _flush_for_session_end takes."""

    def __init__(self, seconds: float):
        self._samples = [0.0] * int(seconds * 16000)
        self.trimmed = 0

    def peek_buffer(self):
        if not self._samples:
            return None
        return self._samples, len(self._samples) / 16000

    def trim_front(self, samples):
        self.trimmed += samples
        self._samples = self._samples[samples:]

    def _flush_all(self):
        if not self._samples:
            return None
        segment, self._samples = self._samples, []
        return segment

    flush = _flush_all
    force_flush = _flush_all

    def _reset(self):
        self._samples = []


def test_interim_sentences_after_adoption_are_not_lost(tmp_path):
    """[B2 regression, interim path] An interim pass whose first sentence's
    write auto-opens and adopts the session must still commit its remaining
    sentences: adoption does not bump the generation, so nothing is refused
    and the committed prefix (trim + echo tail) covers what actually
    landed."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(4.0)
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Первое предложение. Второе предложение. Хвост",
         "language": "ru"},
        120,
    )
    app._split_sentences = lambda text, lang: [
        "Первое предложение.", "Второе предложение.", "Хвост"
    ]

    # The whole pass runs at the pre-adoption generation — the item that was
    # in flight when its first sentence opened the session.
    assert app._do_interim_asr(generation=0, expected_session=None) is True
    stamp = writer.active_session()
    assert stamp is not None
    assert app._session_generation == 0  # adoption did not bump
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "Первое предложение." in all_text
    assert "Второе предложение." in all_text
    assert app._vad.trimmed > 0
    assert app._interim_active is True


def test_interim_pass_entirely_refused_consumes_nothing(tmp_path):
    """[invariant 7] A pass every sentence of which the identity guard
    refuses (a real session boundary crossed between enqueue and processing)
    consumes nothing: no trim, no echo-tail update, no _interim_active —
    the audio stays in the buffer for a later pass instead of being lost."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(4.0)
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Третье предложение. Четвёртое предложение.",
         "language": "ru"},
        120,
    )
    app._split_sentences = lambda text, lang: [
        "Третье предложение.", "Четвёртое предложение.", "Хвост"
    ]
    tail_before = app._interim_committed_tail
    active_before = app._interim_active

    # A stale generation: every sentence is refused before any write.
    assert app._do_interim_asr(generation=42, expected_session=None) is False
    assert app._vad.trimmed == 0                    # audio kept for later
    assert app._interim_committed_tail == tail_before
    assert app._interim_active == active_before
    # Nothing was written anywhere.
    assert list(tmp_path.glob("livetrans_*_all.txt")) == []


def test_explicit_begin_still_refuses_pre_begin_queue_audio(tmp_path):
    """[invariant 3] Starting an explicit new recording still isolates the
    audio that was queued before it: the begin bumps the generation and the
    stale-segment guard refuses the old item — pre-begin speech never lands
    in the new meeting's files."""
    app, writer, main = _make_adoption_app(tmp_path)

    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-old")
    item = app._asr_queue.get_nowait()
    old_gen = item[3]
    assert old_gen == app._session_generation

    stamp_new = app.begin_recording_session()
    assert stamp_new is not None
    assert app._session_generation == old_gen + 1
    assert app.notifications == [("active", stamp_new)]

    assert app._process_segment_text(
        "старая фраза", "ru", 10,
        generation=old_gen, expected_session=item[4],
    ) is False
    all_text = (tmp_path / f"livetrans_{stamp_new}_all.txt").read_text("utf-8")
    assert "старая фраза" not in all_text
    writer.end_session()


def test_subtitle_only_records_nothing_and_drains_its_counts(tmp_path):
    """[invariant 5] Subtitle-only mode (the pipeline resumed after a meeting
    ended, no new session) recognises and displays but writes no meeting
    files, opens no session — and its queue-work counts still drain cleanly
    (nothing for an ENDING wait to hang on: there is no session to end)."""
    app, writer, main = _make_adoption_app(tmp_path)
    # A previous meeting ended: the writer's legacy auto-open is disarmed
    # (explicitly_ended) while the pipeline keeps recording.
    writer.begin_session()
    writer.end_session()
    assert writer.active_session() is None

    assert app._process_segment_text(
        "только субтитры", "ru", 10,
        generation=app._session_generation, expected_session=None,
    ) is True
    assert writer.active_session() is None      # no ghost session
    assert app._session_state == main.SessionState.IDLE
    assert app.notifications == []              # nothing adopted
    # The only files are the ended session's — nothing new was written.
    ended = list(tmp_path.glob("livetrans_*_all.txt"))
    assert len(ended) == 1

    # Subtitle-only queue items are counted (waitable if a session is
    # adopted at this generation later) and drain on release.
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio")
    item = app._asr_queue.get_nowait()
    assert item[2] is not None
    app._session_work.release(item[3], item[2])
    assert app._session_work.pending_count(item[3]) == 0


def test_adoption_and_explicit_begin_yield_one_authority(tmp_path):
    """[invariant 6] An adoption followed by an explicit begin (and vice
    versa) yields exactly one authoritative session, one ACTIVE notification
    and one live tracker generation — no double bump, no second tracker set,
    no residual auto-created entries."""
    app, writer, main = _make_adoption_app(tmp_path)

    # An entry auto-opens and adopts.
    assert app._process_segment_text(
        "фраза", "ru", 10, generation=0, expected_session=None
    ) is True
    stamp = writer.active_session()
    assert app.notifications == [("active", stamp)]

    # An explicit begin arriving now is refused — the session is claimed.
    assert app.begin_recording_session() is None
    assert app.notifications == [("active", stamp)]
    assert app._session_generation == 0

    # A second adopting entry is a no-op against the live session.
    assert app._process_segment_text(
        "вторая", "ru", 10, generation=0, expected_session=None
    ) is True
    assert app.notifications == [("active", stamp)]
    live = [
        g for g, s in app._session_work._gen_state.items()
        if s == app._session_work.OPEN
    ]
    assert live == [0]
    writer.end_session()

    # From scratch, an explicit begin wins first: a later adoption call
    # finds the live session and returns the live generation untouched.
    app2, writer2, _ = _make_adoption_app(tmp_path)
    stamp2 = app2.begin_recording_session()
    assert stamp2 is not None
    gen2 = app2._session_generation
    assert app2._adopt_auto_opened_session(1, gen2) == gen2
    assert app2.notifications == [("active", stamp2)]
    live2 = [
        g for g, s in app2._session_work._gen_state.items()
        if s == app2._session_work.OPEN
    ]
    assert live2 == [gen2]
    writer2.end_session()


# --- B2: the recognition-window boundary races [round 2] ---------------------


def test_begin_during_recognition_is_refused_at_the_fence(tmp_path):
    """[B2 regression, TOCTOU] The stale guard at _process_segment's entry
    passed (generation N was current), then an explicit begin landed *while
    the audio was still being recognized*. The next step used to be
    write_original(session=None), and the writer's wildcard accepted the
    pre-begin audio into the brand-new meeting. The boundary fence re-check
    must refuse it instead."""
    app, writer, main = _make_adoption_app(tmp_path)

    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-old")
    item = app._asr_queue.get_nowait()
    assert item[3] == app._session_generation  # N == current at enqueue
    assert item[4] is None                     # no session existed yet

    def begin_mid_asr(audio, kind, **kwargs):
        # The entry guard has already passed; the begin lands inside the
        # recognition window and opens a session at generation N+1.
        stamp = app.begin_recording_session()
        assert stamp is not None
        return {"text": "старая фраза", "language": "ru"}, 80

    app._run_asr = begin_mid_asr
    app._process_segment("audio-old", item[2], item[3], item[4])

    stamp_new = writer.active_session()
    assert stamp_new is not None
    assert app._session_generation == item[3] + 1
    # The new meeting's files never saw the pre-begin audio...
    all_text = (tmp_path / f"livetrans_{stamp_new}_all.txt").read_text("utf-8")
    assert "старая фраза" not in all_text
    # ...nothing of it was registered under the live generation...
    assert app._session_work.pending_count(app._session_generation) == 0
    # ...and the only ACTIVE notification is the begin's own.
    assert app.notifications == [("active", stamp_new)]
    writer.end_session()


def test_end_then_begin_during_recognition_refuses_the_stale_item(tmp_path):
    """[B2 regression, end→begin] The queue item was enqueued under an
    explicit session (generation N, stamp A); while its audio was being
    recognized the meeting ended (generation superseded and bumped, session
    sealed) and a new one began. The fence re-check refuses the straggler:
    the old meeting's late audio never lands in either meeting."""
    app, writer, main = _make_adoption_app(tmp_path)
    stamp_a = app.begin_recording_session()
    assert stamp_a is not None
    gen_a = app._session_generation

    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-a")
    item = app._asr_queue.get_nowait()
    assert item[3] == gen_a and item[4] == stamp_a

    def end_then_begin_mid_asr(audio, kind, **kwargs):
        # The end thread's tail, in its real order: the generation is
        # superseded and bumped, the writer seals session A, the state
        # returns to IDLE — then the user starts the next meeting.
        app._session_work.supersede(gen_a)
        with app._session_boundary_lock:
            app._session_generation += 1
        app._notify_session_state(main.SessionState.IDLE)
        writer.end_session()
        stamp_b = app.begin_recording_session()
        assert stamp_b is not None and stamp_b != stamp_a
        return {"text": "хвост прошлой встречи", "language": "ru"}, 80

    app._run_asr = end_then_begin_mid_asr
    app._process_segment("audio-a", item[2], item[3], item[4])

    stamp_b = writer.active_session()
    all_b = (tmp_path / f"livetrans_{stamp_b}_all.txt").read_text("utf-8")
    assert "хвост прошлой встречи" not in all_b
    all_a = (tmp_path / f"livetrans_{stamp_a}_all.txt").read_text("utf-8")
    assert "хвост прошлой встречи" not in all_a
    writer.end_session()


def test_begin_between_write_and_adoption_migrates_the_count(tmp_path):
    """[B2 regression] The entry recorded into a writer session the state
    machine did not track yet; an explicit begin then claimed that same
    session (begin_session returns an open session unchanged), bumped the
    generation and retired the old one's counts. The adoption helper must
    not answer the entry's stale generation just because a session is live:
    the count migrates to the live generation exactly once, so ENDING waits
    for the in-flight translation instead of sealing the meeting without
    it."""
    app, writer, main = _make_adoption_app(tmp_path)

    # The queue item's registration auto-creates generation 0 (the
    # pre-adoption opening work an ENDING wait must cover).
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio")
    item = app._asr_queue.get_nowait()
    assert app._session_work.pending_count(0) == 1

    # The entry's fenced write + registration, by hand: the writer
    # auto-opens a session and the msg count lands under generation 0.
    assert writer.write_original(7, "09:00:00", "запись", language="ru") == (
        TranscriptWriter.WRITE_RECORDED
    )
    assert writer._entry_sessions[7] is not None
    assert app._session_work.register_msg(0, 7) is True

    # The explicit begin claims that session, bumps to generation 1 and
    # retires generation 0's auto-created counts.
    stamp = app.begin_recording_session()
    assert stamp is not None
    assert stamp == writer._entry_sessions[7]  # same writer session
    assert app._session_generation == 1
    assert app._session_work.pending_count(0) == 0

    # The helper, entered while the state is already ACTIVE and the entry's
    # generation is stale: it must answer the live generation...
    assert app._adopt_auto_opened_session(7, 0) == 1
    # ...with the count migrated there exactly once.
    assert app._session_work.pending_count(1) == 1
    # The live generation's ENDING therefore waits for the translation:
    app._session_work.start_closing(1)
    assert app._session_work.wait_idle(1, timeout=0.0) is False
    app._session_work.release_msg(1, 7)
    assert app._session_work.wait_idle(1, timeout=0.0) is True
    writer.end_session()


def test_refused_interim_commit_keeps_the_buffered_pending(tmp_path):
    """[B2 regression, pending] The buffered fragments' audio was already
    trimmed away by the pass that buffered them, so when the sentence they
    were spliced into is refused by an identity guard, the pending must
    survive — the old code cleared it before the commit and the text was
    lost forever. A later pass commits it exactly once."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(4.0)
    app._interim_pending = "Да, "
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Новое предложение. Хвост", "language": "ru"}, 100,
    )
    app._split_sentences = lambda text, lang: [
        "Новое предложение.", "Хвост"
    ]

    # A stale generation: the spliced sentence is refused before any write.
    assert app._do_interim_asr(generation=42, expected_session=None) is False
    # The pending was never consumed, nothing was trimmed or written.
    assert app._interim_pending == "Да, "
    assert app._vad.trimmed == 0
    assert list(tmp_path.glob("livetrans_*_all.txt")) == []

    # A later pass at a current generation commits the pending with its own
    # sentence — exactly once, no duplication.
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Другое предложение. Хвост", "language": "ru"}, 100,
    )
    app._split_sentences = lambda text, lang: [
        "Другое предложение.", "Хвост"
    ]
    assert app._do_interim_asr(
        generation=app._session_generation, expected_session=None
    ) is True
    stamp = writer.active_session()
    assert stamp is not None
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "Да, Другое предложение." in all_text
    assert all_text.count("Да,") == 1
    assert app._interim_pending == ""
    assert app._vad.trimmed > 0


def test_interim_prefix_commit_with_later_refusal_consumes_only_the_prefix(
    tmp_path,
):
    """[B2 regression, pending] When earlier sentences of one interim pass
    commit and a later one is refused, only the committed prefix is
    consumed: the trim, the echo tail and _interim_active account for it
    alone, the refused sentence never reaches the record, and a pending
    absorbed by the committed prefix stays consumed."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(6.0)
    app._interim_pending = "Да, "
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Первое предложение. Второе предложение. Хвост",
         "language": "ru"}, 100,
    )
    app._split_sentences = lambda text, lang: [
        "Первое предложение.", "Второе предложение.", "Хвост"
    ]
    # The writer refuses the second sentence's write (a session boundary
    # landed under the pass); the first commits normally.
    real_write = writer.write_original
    writes = {"n": 0}

    def refuse_second_write(msg_id, *args, **kwargs):
        writes["n"] += 1
        if writes["n"] >= 2:
            return TranscriptWriter.WRITE_SESSION_MISMATCH
        return real_write(msg_id, *args, **kwargs)

    writer.write_original = refuse_second_write

    assert app._do_interim_asr(generation=0, expected_session=None) is True
    stamp = writer.active_session()
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    # The committed prefix (pending absorbed) landed; the refused sentence
    # did not.
    assert "Да, Первое предложение." in all_text
    assert "Второе предложение." not in all_text
    # The pending was consumed by the committed prefix — not resurrected.
    assert app._interim_pending == ""
    # Only the committed prefix participates in the accounting.
    assert app._interim_committed_tail == "Да, Первое предложение."
    assert app._interim_active is True
    assert app._vad.trimmed > 0


# --- the ENDING flush hands the interim state to the ASR loop ---------------


def _drain_vad_flush_like_the_asr_loop(app):
    """The ASR loop's vad_flush handling, verbatim in shape: the marker
    branch (no audio — the item exists so this ``finally`` runs), the
    dispatch on ``_interim_active`` (the flag must have survived the
    ENDING hand-off and the pause marker's ordering for the interim-final
    branch to run at all), the interim-state reset in the ``finally``, the
    queue count released afterwards. Keeping this replica here is the
    point — the ENDING hand-off and the pause marker rely on exactly this
    ownership, so the tests pin it against the real methods."""
    item = app._asr_queue.get_nowait()
    assert item[0] == "vad_flush"
    try:
        if item[1] is None:
            # pause()'s cleanup marker: nothing to recognize.
            pass
        elif app._interim_active:
            app._process_interim_final(item[1], item[2], item[3], item[4])
        else:
            app._process_segment(item[1], item[2], item[3], item[4])
    finally:
        app._reset_interim_state()
    if item[2] is not None:
        app._session_work.release(item[3], item[2])
    return item


def test_ending_flush_hands_interim_state_to_the_asr_loop(tmp_path):
    """[ENDING hand-off regression] The ENDING flush used to reset the
    interim state on the producer side, right after enqueueing the final
    segment — which sent the item down the plain ``_process_segment``
    branch (the loop dispatches on ``_interim_active``) and destroyed the
    buffered fragments before ``_process_interim_final`` could flush
    them: their audio was already trimmed away, so the text was lost
    forever. On a successful enqueue the producer must leave the state
    intact and let the loop's existing vad_flush ``finally`` own the one
    reset."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(4.0)
    stamp = app.begin_recording_session()
    assert stamp is not None
    gen = app._session_generation
    # The real ENDING order: CLOSING flips before the flush.
    app._session_work.start_closing(gen)
    app._interim_pending = "Да, "
    app._interim_committed_tail = "предыдущая фраза"
    app._interim_active = True
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Хвост фразы", "language": "ru"}, 60,
    )

    app._flush_for_session_end(gen)

    # The segment reached the queue and the producer did NOT reset: the
    # dispatch flag, the buffered fragments and the echo tail all survive
    # for the consumer, and the ENDING wait sees the item's count (it
    # gates the seal — the reset may only happen after this drains).
    assert app._asr_queue.qsize() == 1
    assert app._session_work.pending_count(gen) == 1
    assert app._interim_active is True
    assert app._interim_pending == "Да, "
    assert app._interim_committed_tail == "предыдущая фраза"

    item = _drain_vad_flush_like_the_asr_loop(app)
    assert item[3] == gen and item[4] == stamp

    # The buffered fragment reached the final record, spliced into the
    # final recognition...
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "Да, Хвост фразы" in all_text
    # ...and after the handler's finally the state is clean — the one
    # reset ran on the consumer side; nothing lingers for the next
    # session, and the drained count lets the seal proceed.
    assert app._interim_active is False
    assert app._interim_pending == ""
    assert app._interim_committed_tail == ""
    assert app._session_work.wait_idle(gen, timeout=0.0) is True
    writer.end_session()


def test_ending_flush_echo_tail_dedups_the_final_recognition(tmp_path):
    """[ENDING hand-off regression] The final flush's audio re-recognizes
    whatever the buffer still held when the meeting ended — which overlaps
    the interim-committed prefix. The committed tail must survive the
    hand-off so ``_process_interim_final`` can strip the repeat; the old
    producer-side reset wiped it and the committed sentence was written
    into the closing record twice."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(4.0)
    stamp = app.begin_recording_session()
    assert stamp is not None
    gen = app._session_generation
    # A sentence the interim path already committed (its audio was
    # trimmed; the tail is its only remaining trace).
    assert app._process_segment_text(
        "Первое предложение.", "ru", 100,
        generation=gen, expected_session=stamp,
    ) is True
    app._interim_committed_tail = "Первое предложение."
    app._interim_active = True
    app._session_work.start_closing(gen)
    # The final recognition replays the committed sentence plus the words
    # still being spoken when the meeting ended.
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Первое предложение. Хвост", "language": "ru"}, 60,
    )

    app._flush_for_session_end(gen)
    assert app._interim_committed_tail == "Первое предложение."
    _drain_vad_flush_like_the_asr_loop(app)

    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    # The committed sentence appears exactly once (its own entry, not a
    # replayed duplicate), and the genuinely new tail was kept.
    assert all_text.count("Первое предложение.") == 1
    assert "Хвост" in all_text
    writer.end_session()


def test_ending_flush_returns_none_keeps_state_for_an_older_queued_vad_flush(
    tmp_path,
):
    """[ENDING ownership regression] ``remaining is None`` proves only that
    *this* flush produced no segment — not that no earlier vad_flush (a
    pause hand-off, a natural VAD flush that raced the end gate) is still
    sitting in the ASR queue. That older item's consumer owns the interim
    state exactly like the final segment's does. The old code reset on the
    None path, so the still-queued consumer dispatched on a cleared
    ``_interim_active``, took the plain ``_process_segment`` branch and
    lost the buffered fragments (their audio was already trimmed away)
    together with the echo dedup."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(0.0)  # the ENDING flush finds nothing
    stamp = app.begin_recording_session()
    assert stamp is not None
    gen = app._session_generation

    # A vad_flush queued *before* the end (pause() hands off exactly like
    # this), already counted under the session's generation.
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-from-before-the-end")
    assert app._session_work.pending_count(gen) == 1
    app._interim_pending = "Да, "
    app._interim_committed_tail = "предыдущая фраза"
    app._interim_active = True
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Хвост фразы", "language": "ru"}, 60,
    )

    # The real ENDING order: CLOSING flips before the flush...
    app._session_work.start_closing(gen)
    app._flush_for_session_end(gen)

    # ...and the None path must NOT reset: the queued consumer still owns
    # the state (this is the pin — the old code reset right here).
    assert app._interim_active is True
    assert app._interim_pending == "Да, "
    assert app._interim_committed_tail == "предыдущая фраза"

    # The ASR loop consumes the older item with the state intact: the
    # buffered fragment splices into the final recognition.
    item = _drain_vad_flush_like_the_asr_loop(app)
    assert item[3] == gen and item[4] == stamp
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "Да, Хвост фразы" in all_text

    # The drain phase: with the count released the queue work is idle, and
    # the ENDING thread's one reset runs — a no-op after the consumer's
    # own finally, clean either way, never leaked into a next session.
    assert app._session_work.wait_idle(gen, timeout=0.0, queue_only=True) is True
    app._reset_interim_state()
    assert app._interim_active is False
    writer.end_session()


def test_ending_flush_refusals_leave_the_reset_to_the_drain_phase(tmp_path):
    """[ENDING ownership regression] Every path that leaves the flush's
    *own* audio unqueued (ASR not ready, a superseded generation, a stop
    already begun, the queue rejecting the item) used to reset the
    interim state on the spot. But an older queued vad_flush owns that
    state, so the flush must leave it alone on these paths too: the reset
    is the ENDING flow's (the queue-drain phase; the stop and abort paths
    carry their own), never the flush's."""
    app, writer, main = _make_adoption_app(tmp_path)
    gen = app._session_generation

    def dirty():
        app._interim_pending = "Да, "
        app._interim_committed_tail = "хвост"
        app._interim_active = True

    # An older vad_flush is queued: whatever the flush does with its own
    # audio, that item's consumer still owns the interim state.
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-old")

    # ASR not ready: the VAD buffer is dropped, but the interim state
    # stays the queued consumer's.
    app._vad = _FakeInterimVAD(4.0)
    app._asr_ready = False
    dirty()
    app._flush_for_session_end(gen)
    assert app._vad._samples == []          # the buffer went with it
    assert app._interim_active is True      # the state did not

    # A superseded generation (a quit raced the end): register_final
    # refuses, the segment is dropped — the state still stands.
    app._asr_ready = True
    app._vad = _FakeInterimVAD(4.0)
    dirty()
    app._flush_for_session_end(gen + 99)
    assert app._interim_active is True

    # A stop already begun: the enqueue refuses before even registering.
    app._vad = _FakeInterimVAD(4.0)
    dirty()
    app._stop_event.set()
    app._flush_for_session_end(gen)
    assert app._interim_active is True
    app._stop_event.clear()

    # The drain phase owns the one reset: while the older item is queued
    # the queue-only wait reports busy; once its consumer released, the
    # ENDING flow's reset runs.
    assert app._session_work.wait_idle(gen, timeout=0.0, queue_only=True) is False
    item = app._asr_queue.get_nowait()
    app._release_queued_work(item)
    assert app._session_work.wait_idle(gen, timeout=0.0, queue_only=True) is True
    app._reset_interim_state()
    assert app._interim_active is False
    assert app._interim_pending == ""


def test_run_session_end_resets_when_no_queue_work_and_nothing_remains(
    tmp_path,
):
    """[ENDING ownership regression] With no older queue work and no
    remaining VAD buffer there is no consumer to hand the state to — the
    drain phase owns the reset and must actually run it: a dirty state
    (buffered fragments, echo tail, dispatch flag) must not leak into the
    next session, and the seal still completes."""
    app, writer, main = _make_adoption_app(tmp_path)
    stamp = app.begin_recording_session()
    assert stamp is not None
    gen = app._session_generation
    app._vad = _FakeInterimVAD(0.0)
    app._interim_pending = "Да, "
    app._interim_committed_tail = "хвост"
    app._interim_active = True

    # Nothing is queued and the flush finds nothing: the end runs to
    # completion without blocking (the queue-only phase is instantly
    # idle) and resets on the way to the seal.
    summary = app._run_session_end(gen)

    assert summary is not None
    assert summary["session_status"] == "completed"
    assert app._interim_active is False
    assert app._interim_pending == ""
    assert app._interim_committed_tail == ""
    assert app._session_work.pending_count(gen) == 0


def test_run_session_end_resets_after_queue_consumers_and_before_the_seal(
    tmp_path,
):
    """[ENDING ownership regression] The one interim-state reset lives in
    _run_session_end's queue-drain phase: after every queued consumer of
    the generation returned (the tracker's queue counts — never
    Queue.empty(), which an already-taken item empties too early), and
    before end_session() and the IDLE broadcast that _work sends once
    _run_session_end returns. The queue-only phase exists so the end does
    not first burn the shared 30s budget on the network translations."""
    app, writer, main = _make_adoption_app(tmp_path)
    stamp = app.begin_recording_session()
    assert stamp is not None
    gen = app._session_generation

    # The older vad_flush is queued and counted; the ENDING flush will
    # find nothing (empty VAD buffer → the None path under repair).
    app._vad = _FakeInterimVAD(0.0)
    flush_ran = _threading.Event()
    real_force_flush = app._vad.force_flush

    def _force_flush_recording():
        result = real_force_flush()
        flush_ran.set()
        return result

    app._vad.force_flush = _force_flush_recording
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-from-before-the-end")
    app._interim_pending = "Да, "
    app._interim_active = True
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "Хвост фразы", "language": "ru"}, 60,
    )
    # The seal must observe an already-reset state: the reset runs before
    # end_session(), not after it.
    sealed_state = {}
    real_end_session = writer.end_session

    def _end_session_recording_state():
        sealed_state["interim_active"] = app._interim_active
        sealed_state["interim_pending"] = app._interim_pending
        return real_end_session()

    writer.end_session = _end_session_recording_state

    result = {}

    def _end():
        result["summary"] = app._run_session_end(gen)

    end_thread = _threading.Thread(target=_end, name="session-end-test")
    end_thread.start()

    # The ENDING flush has run and returned None; the end thread is now
    # blocked in the queue-drain wait on the older item's count. The state
    # must still be the consumer's — nothing has reset it.
    assert flush_ran.wait(timeout=5.0)
    assert app._interim_active is True
    assert app._interim_pending == "Да, "

    # The ASR loop consumes the older item; the release unblocks the drain
    # phase, which resets and proceeds to the seal.
    item = _drain_vad_flush_like_the_asr_loop(app)
    assert item[3] == gen and item[4] == stamp
    end_thread.join(timeout=5.0)
    assert not end_thread.is_alive()

    # The fragment landed (the consumer ran with the state intact)...
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "Да, Хвост фразы" in all_text
    # ...the seal completed and saw an already-clean interim state...
    assert sealed_state == {"interim_active": False, "interim_pending": ""}
    assert result["summary"]["session_status"] == "completed"
    # ...and nothing lingers for the next session.
    assert app._interim_active is False
    assert app._interim_pending == ""
    assert app._interim_committed_tail == ""


def test_session_end_queue_timeout_still_resets_and_never_leaks(
    tmp_path, monkeypatch,
):
    """[ENDING ownership regression] When the queue drain misses the
    shared deadline (a stuck ASR worker, a dead consumer thread), the end
    still closes: the interim state is reset anyway — bounded, logged, and
    safe because the straggler fails the generation guards once the
    generation is superseded — so a dirty state cannot leak into the next
    session and ENDING can never park forever waiting to reset."""
    app, writer, main = _make_adoption_app(tmp_path)
    stamp = app.begin_recording_session()
    assert stamp is not None
    gen = app._session_generation
    app._vad = _FakeInterimVAD(0.0)
    app._interim_pending = "Да, "
    app._interim_active = True

    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-stuck")
    assert app._session_work.pending_count(gen) == 1

    # Shrink the shared budget so the drain misses it within the test.
    monkeypatch.setattr(main.SessionState, "ENDING_TIMEOUT_S", 0.05)
    summary = app._run_session_end(gen)

    # The end completed (never parked in ENDING) and the state is clean.
    assert summary is not None
    assert app._interim_active is False
    assert app._interim_pending == ""
    # The stuck item was neither consumed nor waited on forever; its late
    # release after the supersede is the idempotent no-op it must be.
    item = app._asr_queue.get_nowait()
    app._release_queued_work(item)
    assert app._session_work.pending_count(gen) == 0


# --- pause hands the interim state to the queue's FIFO order ---------------


def test_pause_with_an_older_queued_vad_flush_keeps_the_interim_state(tmp_path):
    """[pause ownership regression] ``remaining is None`` proves only that
    *this* pause's flush produced no segment — not that no earlier
    vad_flush (a natural split, a silence-feed flush, an earlier hand-off)
    is still queued unconsumed. That item's consumer owns the interim
    state. The old code reset in the else-branch, so the still-queued
    consumer dispatched on a cleared ``_interim_active``, took the plain
    ``_process_segment`` branch and lost the buffered fragments (their
    audio was already trimmed away) together with the echo dedup. pause()
    must leave the state to the consumer and enqueue its no-audio cleanup
    marker behind it, so the one reset runs on the consumer side in queue
    order."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(0.0)   # the pause flush finds nothing
    stamp = app.begin_recording_session()
    assert stamp is not None
    gen = app._session_generation

    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-from-before-the-pause")
    app._interim_pending = "Да, "
    app._interim_committed_tail = "предыдущая фраза"
    app._interim_active = True
    # The older item's recognition replays the committed prefix (the echo
    # the tail must strip) plus the words still being spoken.
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "предыдущая фраза Хвост", "language": "ru"}, 60,
    )

    app.pause()

    # The pause flushed nothing and must NOT reset: the older item's
    # consumer still owns every field of the state.
    assert app._interim_active is True
    assert app._interim_pending == "Да, "
    assert app._interim_committed_tail == "предыдущая фраза"
    # The cleanup marker landed *behind* the older item.
    assert app._asr_queue.qsize() == 2

    # The ASR loop consumes the older item first, with the state intact:
    # the echo is stripped against the surviving tail and the buffered
    # fragment is spliced in. (With the old reset, this took the plain
    # path and wrote "предыдущая фраза Хвост" raw, no fragment.)
    item = _drain_vad_flush_like_the_asr_loop(app)
    assert item[1] == "audio-from-before-the-pause"
    assert item[3] == gen and item[4] == stamp
    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    assert "Да, Хвост" in all_text
    assert "предыдущая фраза" not in all_text

    # Then the marker: no audio, its finally performs the one reset.
    marker = _drain_vad_flush_like_the_asr_loop(app)
    assert marker[1] is None
    assert app._interim_active is False
    assert app._interim_pending == ""
    assert app._interim_committed_tail == ""
    writer.end_session()


def test_pause_with_no_consumer_enqueues_the_marker_and_still_resets(tmp_path):
    """[pause ownership regression] With nothing queued and nothing to
    flush there is no consumer at all — the marker is the consumer: it
    carries no audio, so nothing is recognized, but its vad_flush finally
    performs the reset. The state cannot survive the pause (the next
    utterance would splice onto the old one), and the subtitle-only
    admission ("pass", no session) carries the marker like any item."""
    app, writer, main = _make_adoption_app(tmp_path)
    app._vad = _FakeInterimVAD(0.0)
    gen = app._session_generation
    app._interim_pending = "Да, "
    app._interim_committed_tail = "хвост"
    app._interim_active = True

    app.pause()

    # The marker is the only queue item, and it is counted (the ENDING
    # wait would cover it if an end raced the pause).
    assert app._asr_queue.qsize() == 1
    assert app._session_work.pending_count(gen) == 1

    marker = _drain_vad_flush_like_the_asr_loop(app)
    assert marker[1] is None
    assert app._session_work.pending_count(gen) == 0
    assert app._interim_active is False
    assert app._interim_pending == ""
    assert app._interim_committed_tail == ""


def test_pause_resume_starts_the_next_utterance_from_a_clean_slate(tmp_path):
    """[pause ownership regression] The pause semantics — "speech still in
    flight before the pause completes; after the resume a fresh utterance
    starts" — must hold by *ordering*, not by resetting early: the marker
    is enqueued at the tail, the ASR loop is a single-consumer FIFO, so
    the reset runs after every pre-pause item and before anything
    enqueued after the resume. The post-resume segment therefore sees a
    clean state: no pre-pause fragment spliced onto it, no echo stripping
    against the old utterance."""
    app, writer, main = _make_adoption_app(tmp_path)
    stamp = app.begin_recording_session()
    assert stamp is not None
    app._vad = _FakeInterimVAD(0.0)

    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-before-pause")
    app._interim_pending = "Да, "
    app._interim_committed_tail = "предыдущая фраза"
    app._interim_active = True

    app.pause()
    app.resume()

    # Post-resume speech lands behind the marker (the capture thread
    # produces through the same fence).
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-after-resume")

    def fake_asr(audio, kind, **kw):
        if kind == "interim_final":
            return {"text": "Хвост фразы", "language": "ru"}, 60
        return {"text": "Новая фраза", "language": "ru"}, 60

    app._run_asr = fake_asr

    # FIFO: the pre-pause remainder (state intact) → the marker (one
    # reset) → the post-resume audio (clean state, plain path).
    item1 = _drain_vad_flush_like_the_asr_loop(app)
    assert item1[1] == "audio-before-pause"
    marker = _drain_vad_flush_like_the_asr_loop(app)
    assert marker[1] is None
    item3 = _drain_vad_flush_like_the_asr_loop(app)
    assert item3[1] == "audio-after-resume"

    all_text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
    # The pre-pause remainder completed: fragment spliced, echo stripped.
    assert "Да, Хвост фразы" in all_text
    # The new utterance is exactly its own text — nothing spliced from or
    # deduped against the pre-pause utterance.
    assert "Новая фраза" in all_text
    assert "Да, Новая фраза" not in all_text
    writer.end_session()


def test_pause_failure_paths_do_not_leak_or_corrupt(tmp_path):
    """[pause ownership regression] The failure paths must close both
    ways: with an older consumer still queued (ASR not ready — the audio
    is dropped but the state is not the pause's to clear), and with no
    consumer ever coming (a stop already begun — both enqueues refuse, so
    the pause clears the state itself rather than leaking it across the
    pause→resume boundary)."""
    app, writer, main = _make_adoption_app(tmp_path)
    gen = app._session_generation

    def dirty():
        app._interim_pending = "Да, "
        app._interim_committed_tail = "хвост"
        app._interim_active = True

    # ASR not ready, buffer audio present, an older vad_flush queued: the
    # audio is dropped as before, but the state stays the older
    # consumer's and the marker lands behind it.
    app._vad = _FakeInterimVAD(4.0)
    app._asr_ready = False
    with app._session_boundary_lock:
        app._enqueue_asr("vad_flush", "audio-old")
    dirty()
    app._run_asr = lambda audio, kind, **kw: (None, 0)

    app.pause()

    assert app._vad._samples == []            # the buffer went with it
    assert app._interim_active is True        # the state did not
    assert app._asr_queue.qsize() == 2        # older item + marker
    _item = _drain_vad_flush_like_the_asr_loop(app)
    marker = _drain_vad_flush_like_the_asr_loop(app)
    assert marker[1] is None
    assert app._interim_active is False       # the consumer-side reset ran

    # A stop already begun: both enqueues refuse before registering, no
    # consumer will ever run — the pause resets inline (stop() carries
    # its own reset too; this is the bounded last resort).
    app._asr_ready = True
    app._vad = _FakeInterimVAD(4.0)
    app._stop_event.set()
    dirty()
    app.pause()
    assert app._interim_active is False
    assert app._interim_pending == ""
    assert app._asr_queue.empty()
    app._stop_event.clear()
