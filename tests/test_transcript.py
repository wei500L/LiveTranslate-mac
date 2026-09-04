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
    stamp = writer._session_ts
    result = writer.write_translation(99, "orphan")
    writer.close()
    assert result == TranscriptWriter.WRITE_SKIPPED
    text = (tmp_path / f"livetrans_{stamp}_all.txt").read_text("utf-8")
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
    writer = _writer(tmp_path)
    stamp = writer._session_ts
    writer.write_original(1, "00:00:01", "hello")
    writer.write_translation(1, "hola")
    writer.close()
    assert list(tmp_path.glob(f"livetrans_{stamp}_*"))

    assert delete_session(tmp_path, stamp) == []
    assert list(tmp_path.glob(f"livetrans_{stamp}_*")) == []


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
    stamp = writer._session_ts
    writer.write_original(1, "09:00:00", "kept line")

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
    stamp = writer._session_ts
    writer.write_original(1, "09:00:00", "good line", duration=1.0)

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
    stamp = writer._session_ts
    writer.write_original(1, "09:00:00", "hello")
    writer.write_translation(1, "hola")

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
    stamp = writer._session_ts
    writer.write_original(1, "09:00:00", "hello")

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
    stamp = writer._session_ts
    meta = json.loads(
        (tmp_path / f"livetrans_{stamp}_meta.json").read_text("utf-8")
    )
    assert meta["session_status"] == "active"
    summary = writer.end_session()
    assert summary["session_status"] == "completed"


def test_completed_meta_commits_only_after_content_files_closed(tmp_path):
    """The completed verdict must be the last thing committed: when the
    final sidecar is written, every content handle must already be closed
    and durable (flush+fsync+close ran before it). Otherwise a later
    flush/close failure would leave a "completed" marker over unwritten
    buffers."""
    writer = _writer(tmp_path)
    stamp = writer._session_ts
    writer.write_original(1, "09:00:00", "hello")
    writer.write_translation(1, "hola")

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
    stamp = writer._session_ts
    writer.write_original(1, "09:00:00", "hello")

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
