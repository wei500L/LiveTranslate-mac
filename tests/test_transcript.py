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
    writer.close()

    all_lines = _entries(tmp_path / f"livetrans_{writer._session_ts}_all.txt")
    assert [line.split("] ", 1)[1] for line in all_lines] == [
        "first", "second", "third"
    ]
    tl_lines = _entries(tmp_path / f"livetrans_{writer._session_ts}_translation.txt")
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
    writer.close()

    text = (tmp_path / f"livetrans_{writer._session_ts}_all.txt").read_text("utf-8")
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
    writer.close()

    lines = _entries(tmp_path / f"livetrans_{writer._session_ts}_all.txt")
    assert [line.split("] ", 1)[1] for line in lines] == ["one", "two", "three"]


def test_no_entry_is_registered_twice(tmp_path):
    writer = _writer(tmp_path)
    writer.write_original(1, "00:00:01", "first")
    writer.write_original(1, "00:00:02", "corrected")
    writer.write_translation(1, "FIRST")
    writer.close()
    lines = _entries(tmp_path / f"livetrans_{writer._session_ts}_all.txt")
    assert len(lines) == 1


def test_a_translation_without_an_original_is_still_recorded(tmp_path):
    writer = _writer(tmp_path)
    writer.write_translation(99, "orphan")
    writer.close()
    text = (tmp_path / f"livetrans_{writer._session_ts}_all.txt").read_text("utf-8")
    assert "orphan" in text


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
    writer.close()

    text = (
        tmp_path / f"livetrans_{writer._session_ts}_meeting.md"
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
    writer.close()
    for kind in TranscriptWriter.KINDS:
        text = (tmp_path / f"livetrans_{writer._session_ts}_{kind}.txt").read_text("utf-8")
        assert "# Session ended at" in text
        assert "1 entries" in text


def test_the_metadata_sidecar_describes_the_session(tmp_path):
    writer = _writer(tmp_path)
    writer.set_session_info(asr_engine="Whisper medium", target_language="en")
    writer.write_original(1, "09:15:00", "hello", duration=3.0)
    writer.write_translation(1, "hola")
    writer.write_original(2, "09:15:10", "again", duration=1.0)
    writer.finalize_no_translation(2)
    writer.close()

    meta = json.loads(
        (tmp_path / f"livetrans_{writer._session_ts}_meta.json").read_text("utf-8")
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
