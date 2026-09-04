"""Persistent transcript writer — the durable record of a session.

Three plain-text files plus a Markdown meeting record are written per session,
alongside a small JSON sidecar so the transcripts page can list sessions without
parsing every file.

**Ordering.** Translations run on a pool of worker threads and finish out of
order, so writing each one as it lands produced a record whose lines were in
completion order rather than the order things were said. Entries are registered
in utterance order by ``write_original`` and released only when every earlier
entry has completed, which is what makes this usable as a meeting record. The
``original`` file is still written immediately — it needs no translation and is
the one to tail while a session runs.
"""

import json
import logging
import os
import re
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

log = logging.getLogger("LiveTranslate.Transcript")


def _discard_quietly(path: Path) -> None:
    """Best-effort unlink used by roll-back paths."""
    try:
        Path(path).unlink()
    except OSError:
        pass


# Staged meta writes use a per-call UUID suffix (see
# TranscriptWriter._staged_meta_path): a pid+counter suffix is unique only
# within this module — meeting_records stages the *same* final file with the
# same name shape, and two independent counters can agree, letting two
# concurrent staged writes replace each other's temp file.

# Session stamps are "YYYYmmdd_HHMMSS" plus an optional "_NN" suffix used only
# when two sessions begin within the same second (end one meeting, start the
# next immediately). The suffix form must sort *after* the bare stamp of the
# same second and stay sortable with the old format; "20260904_101530_01" does
# both under plain string comparison.
_STAMP_SUFFIX_RE = re.compile(r"^(?P<base>\d{8}_\d{6})(?:_(?P<seq>\d{2}))?$")


def stamp_sort_key(stamp: str) -> tuple:
    """Sort key for session stamps: base stamp first, then sequence number.

    A bare "20260904_101530" sorts before "20260904_101530_01", and "_02"
    before "_10" (numeric, not lexicographic on the two digits — "10" < "02"
    as text would put the tenth session before the second).
    """
    match = _STAMP_SUFFIX_RE.match(stamp or "")
    if not match:
        # Unparseable stamp: keep it after valid ones, in a stable order.
        return (1, stamp or "")
    seq = match.group("seq")
    return (0, match.group("base"), int(seq) if seq else -1)


def _stamp_to_iso(stamp: str) -> str | None:
    """ISO start time from a session stamp, tolerating the "_NN" suffix."""
    match = _STAMP_SUFFIX_RE.match(stamp or "")
    if not match:
        return None
    try:
        return datetime.strptime(match.group("base"), "%Y%m%d_%H%M%S").isoformat(
            timespec="seconds"
        )
    except ValueError:
        return None


class TranscriptWriter:
    KINDS = ("original", "translation", "all")
    MARKDOWN_KIND = "meeting"
    META_KIND = "meta"

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        # _enabled is the *user preference* ("auto-save transcripts"), owned by
        # the settings panel. It is deliberately separate from whether a
        # session is open: ending a meeting closes the session without
        # touching the preference, and write_original only auto-opens a
        # session while the pipeline itself is running a recording.
        self._enabled = True
        # True while the pipeline is actively recording (started and not
        # stopped). Entry-writing methods may auto-open a session only when
        # this is set, so an entry that lands after end_session() can neither
        # reopen the closed meeting nor spawn a ghost session.
        self._recording = False
        # True between begin_session() and end_session()/close().
        self._session_open = False
        # True once end_session() has been requested, before it completes.
        # Late translations for the session are still accepted then — the
        # ENDING phase exists to let them land — but originals are not.
        self._ending = False
        self._lock = threading.Lock()
        self._files = {}
        self._paths = {}
        # Utterance order, and the two halves of each entry.
        self._order = deque()
        self._pending = {}    # msg_id -> dict(timestamp, original, language, duration)
        self._done = {}       # msg_id -> translation text, or None for untranslated
        self._opened = False
        self._session_ts = None
        self._session_started = None
        self._info = {}
        self._counts = {"entries": 0, "translated": 0, "untranslated": 0}
        self._speech_seconds = 0.0
        # Session tokens: every entry carries the stamp of the session it was
        # written for, so a translation landing after the next session began
        # is routed to (or dropped from) the right meeting rather than being
        # appended to whichever session happens to be open now.
        self._entry_sessions = {}
        # Once an explicit end_session() has closed a meeting, write_original
        # may not auto-open another one — only begin_session() (the user's
        # "Start new recording") may. Cleared by begin_session() and by
        # set_recording(True) (a fresh pipeline start re-arms the legacy
        # auto-open path).
        self._explicitly_ended = False
        # The seal timestamp, fixed once when the session is closed (footer,
        # sidecar and summary all read this); None while the session is live,
        # so a live session's sidecar carries no "ended" field at all.
        self._ended_at = None
        # Persistent session status (the sidecar's session_status field):
        # "active" from open, "completed" only when the seal wrote every
        # required file and the final sidecar, "interrupted" for an aborted
        # or degraded close. None = no session.
        self._session_status = None
        # Set the moment any required write fails (entry emit, footer, final
        # sidecar): the seal then records "interrupted" instead of
        # "completed" however far it got.
        self._write_degraded = False

    # --- session lifecycle ---------------------------------------------

    def set_enabled(self, enabled: bool):
        """Apply the user's auto-save preference. Does not close a session.

        Ending a meeting must not permanently flip this preference, and
        re-enabling it must not conjure a session the user never started:
        auto-open on enable only happens while the pipeline is recording.
        """
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._enabled:
                if enabled and self._recording and not self._session_open:
                    self._open_session_locked()
                return
            self._enabled = enabled
            if enabled and self._recording and not self._session_open:
                self._open_session_locked()

    def is_enabled(self) -> bool:
        return self._enabled

    def set_recording(self, recording: bool):
        """Mark whether the pipeline is running (its start/stop, not a meeting's).

        A session opened while recording stays open across the pipeline's
        pause/resume (same meeting); the recording flag only gates whether a
        stray entry may auto-open a session. A fresh pipeline start re-arms
        that legacy auto-open path — each start() begins a new recording.
        """
        with self._lock:
            self._recording = bool(recording)
            if recording:
                self._explicitly_ended = False

    def begin_session(self) -> str | None:
        """Open a new meeting session explicitly (the user asked for one).

        Returns the session stamp, or None when disabled or on failure. If a
        session is already open, it is returned unchanged — starting a new
        meeting while one runs is the caller's state-machine bug to handle,
        not this method's job to silently stack.
        """
        with self._lock:
            if not self._enabled:
                return None
            self._explicitly_ended = False
            if self._session_open:
                return self._session_ts
            self._ending = False
            self._ended_at = None
            self._open_session_locked()
            return self._session_ts if self._session_open else None

    def end_session(self) -> dict | None:
        """Close the current session: flush pending entries, footer, meta, files.

        Returns the final summary dict (the sidecar contents) so the caller
        can tell the records center which meeting just completed; None when
        no session was open or when a session is already ending (the second
        end request loses; the first one owns the close).
        """
        with self._lock:
            if not self._session_open:
                return None
            if self._ending:
                return None
            self._ending = True
            try:
                summary = self._close_locked()
                # A ghost session must not auto-open behind this close: only
                # begin_session() (or a fresh pipeline start) re-arms that.
                self._explicitly_ended = True
                return summary
            finally:
                self._ending = False
                self._session_open = False

    def is_ending(self) -> bool:
        """True while an end_session() close is in progress."""
        with self._lock:
            return self._ending

    def has_active_session(self) -> bool:
        """True when a meeting session is open (ACTIVE or PAUSED, not ENDING)."""
        with self._lock:
            return self._session_open and not self._ending

    def has_open_session(self) -> bool:
        """True while a session is open in *any* sense — recording or in the
        middle of a close. ``has_active_session`` deliberately hides ENDING
        (for UI labelling); this is the lifecycle question: is there a
        session whose fate is not yet decided?"""
        with self._lock:
            return self._session_open or self._ending

    def has_open_resources(self) -> bool:
        """True when anything of a session is still held: the logical open
        flag, a close in progress, the opened-file-set flag, live file
        handles, a session stamp or paths. This is the strictest check —
        e.g. ``_opened=True, _session_open=False`` (a close that died before
        finishing) is caught here even though has_active_session() and
        has_open_session() both answer False. The IDLE broadcast gates on
        this being False."""
        with self._lock:
            return bool(
                self._session_open
                or self._ending
                or self._opened
                or self._files
                or self._session_ts is not None
                or self._paths
            )

    def session_paths(self) -> dict:
        with self._lock:
            return dict(self._paths)

    def active_session(self) -> str | None:
        """The stamp of the session currently being recorded, if any.

        The records center needs to know a record is still growing (entries,
        duration, hash all move) so it can label it and refuse to summarize a
        half-finished meeting. This is authoritative state, not a guess from
        file mtimes. A session in ENDING is deliberately not reported: it is
        being finalized, and the page will refresh when it lands.
        """
        with self._lock:
            if self._session_open and not self._ending:
                return self._session_ts
            return None

    def ending_session(self) -> str | None:
        """The stamp of the session currently being finalized, if any."""
        with self._lock:
            return self._session_ts if self._ending and self._session_open else None

    def set_session_info(self, **info):
        """Record what produced this session (ASR engine, model, languages).

        Shown in the meeting record's header and in the transcripts list, so a
        record from six weeks ago still says how it was made.
        """
        with self._lock:
            self._info.update({k: v for k, v in info.items() if v is not None})
            if self._session_open:
                self._write_meta_locked()

    def rename_session(self, title: str, expected_session: str | None = None) -> bool:
        """Rename the open session, inside the writer's lock.

        This is the single coordinated path for renaming a *live* session:
        the sidecar then has one writer (the records-layer
        ``set_session_title`` would race this writer's periodic meta
        rewrites and the seal's final commit — read-modify-write against
        read-modify-write loses titles and can roll a committed status
        back). The title/title_set_at fields ride the same whitelisted
        merge every meta write uses, so they survive the seal. Refused
        while ENDING (the seal owns the sidecar) and when no session is
        open. Returns True when the sidecar now carries the new title.

        ``expected_session`` closes the identity chain: the caller names
        the meeting it saw when the rename was issued, and the write is
        refused when the session open now is a different one (an end+begin
        raced the dialog) — otherwise the new meeting would get the title
        the user typed for the old one.
        """
        title = (title or "").strip()
        if not title:
            return False
        with self._lock:
            if not self._session_open or self._ending:
                return False
            if (
                expected_session is not None
                and expected_session != self._session_ts
            ):
                log.info(
                    "Refusing rename for session %s: the open session is %s",
                    expected_session, self._session_ts,
                )
                return False
            # One complete write (summary + merged user fields + the new
            # title): the sidecar is never reduced to a fields-only stub,
            # so even a crash right here leaves a readable record.
            return self._write_meta_locked(
                extra_user_fields={
                    "title": title,
                    "title_set_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

    # The suffixes actually used on disk, matched with the real extensions —
    # _unique_stamp probes these exact paths. Probing extension-less names
    # always missed (the files carry .txt/.md/.json), so same-second sessions
    # collided silently.
    _STAMP_KIND_SUFFIXES = (
        ("all", ".txt"),
        ("original", ".txt"),
        ("translation", ".txt"),
        (MARKDOWN_KIND, ".md"),
        (META_KIND, ".json"),
    )

    def _stamp_files_exist(self, stamp: str) -> bool:
        return any(
            (self._base_dir / f"livetrans_{stamp}_{kind}{suffix}").exists()
            for kind, suffix in self._STAMP_KIND_SUFFIXES
        )

    def _unique_stamp(self, now: datetime) -> str:
        """A stamp for this second that collides with no existing session.

        Plain seconds-precision stamps meant two meetings started within one
        second (end one, immediately begin the next) shared a file set, and
        the second meeting appended to the first's files. The first session of
        a second keeps the bare stamp; later ones get _01, _02, … checked
        against files actually on disk (with their real extensions), not an
        in-memory counter, so restarts and crashes cannot resurrect a
        collision.
        """
        base = now.strftime("%Y%m%d_%H%M%S")
        if not self._stamp_files_exist(base):
            return base
        for seq in range(1, 100):
            stamp = f"{base}_{seq:02d}"
            if not self._stamp_files_exist(stamp):
                return stamp
        # 99 sessions in one second is not a real workload; fall back to a
        # timestamp-unique name rather than blocking recording.
        return f"{base}_{now.strftime('%f')}"

    def _open_session_locked(self):
        """Open one session's file set, all-or-nothing.

        Every file is created exclusively ("x") and its header written before
        it counts: a file whose header write fails is closed and unlinked by
        the create step itself, so no half-written file survives. The initial
        meta sidecar is part of the success condition — if its exclusive
        create fails (collision, permission, disk), the whole file set rolls
        back. ``_session_open`` is set only when the text files, the Markdown
        record and the meta all exist; any failure leaves no session and no
        stray files behind.
        """
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(f"Failed to create transcript dir {self._base_dir}: {e}")
            return
        now = datetime.now()
        self._session_started = now
        self._session_ts = self._unique_stamp(now)
        header_ts = now.strftime("%Y-%m-%d %H:%M:%S")

        opened = {}
        paths = {}

        def _create(kind: str, path: Path, header: str):
            """Create one file exclusively and write its header. On any
            failure the file this call created is closed and unlinked, so
            the caller only ever sees fully-written files in ``opened``."""
            fp = None
            try:
                # "x" (exclusive create): a stamp collision must refuse a
                # fresh file set, never append to another meeting's files.
                fp = open(path, "x", encoding="utf-8", buffering=1)
                fp.write(header)
            except Exception as e:
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass
                _discard_quietly(path)
                return f"{path}: {e}"
            opened[kind] = fp
            paths[kind] = str(path)
            return None

        failures = []
        for kind in self.KINDS:
            path = self._base_dir / f"livetrans_{self._session_ts}_{kind}.txt"
            err = _create(kind, path, f"# Session started at {header_ts}\n")
            if err:
                failures.append(err)

        md_path = self._base_dir / f"livetrans_{self._session_ts}_{self.MARKDOWN_KIND}.md"
        err = _create(
            self.MARKDOWN_KIND, md_path,
            f"# Meeting record {now.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"- Started: {header_ts}\n",
        )
        if err:
            failures.append(err)

        if not failures:
            # The initial meta is part of the transaction: without it the
            # listing layer cannot see the session, so a session whose
            # sidecar cannot be created is no session.
            self._files = opened
            self._paths = paths
            self._paths[self.META_KIND] = str(
                self._base_dir / f"livetrans_{self._session_ts}_{self.META_KIND}.json"
            )
            if not self._write_meta_locked(create_only=True):
                failures.append(f"{self._paths[self.META_KIND]}: create failed")

        if failures:
            # Roll back everything this call created: a partial file set
            # would present a "successful" session whose entries silently
            # vanish into closed or missing files. "x" guarantees nothing
            # pre-existing was touched, so unlinking is safe. The meta needs
            # no handling here: _write_meta_locked(create_only=True) cleans
            # up its own partial file, and on an O_EXCL collision the file
            # on disk belongs to another session and must stay.
            for kind, fp in opened.items():
                try:
                    fp.close()
                except Exception:
                    pass
                _discard_quietly(Path(paths[kind]))
            log.error(
                "Could not open the full file set for session %s; no session "
                "started: %s", self._session_ts, "; ".join(failures),
            )
            # Leave any previous session's state untouched; a caller that
            # relied on a stamp must observe None, not a phantom session.
            self._files = {}
            self._paths = {}
            self._session_ts = None
            self._session_started = None
            self._session_status = None
            self._write_degraded = False
            self._markdown_header_open = False
            return

        self._markdown_header_open = True
        self._opened = True
        self._session_open = True
        # The sidecar written by create_only carries session_status=active
        # (see _summary_locked) — the records layer treats a live session by
        # this field first.
        self._session_status = "active"
        self._write_degraded = False
        log.info(f"Transcripts -> {self._base_dir} (session {self._session_ts})")

    # --- entry recording -----------------------------------------------

    # write_original result states, consumed by the caller to decide what
    # happens to this segment's translation.
    WRITE_RECORDED = "recorded"          # entry is in the session's files
    WRITE_SKIPPED = "skipped"            # nothing recorded (disabled, closed,
                                         # auto-open refused): subtitle-only
    WRITE_SESSION_MISMATCH = "mismatch"  # entry's session is not the open one
    WRITE_FAILED = "failed"              # the file write itself errored

    def write_original(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        *,
        language: str | None = None,
        duration: float | None = None,
        session: str | None = None,
    ) -> str:
        """Record an original line, for the session it belongs to.

        ``session`` is the caller's expected session stamp (snapshotted when
        the audio entered the ASR queue). The check happens inside this
        writer's lock, the final authority: when the session open now is not
        the one this audio belongs to (an end and a new begin raced the
        queue), the entry is refused — the old audio must not land in the
        new meeting's files. ``None`` means "no expectation" (the legacy
        auto-open path, where the session is opened by this very call), and
        is always accepted.

        Returns one of the WRITE_* states. Callers must not submit the
        entry's translation into the session on anything but WRITE_RECORDED;
        WRITE_SESSION_MISMATCH additionally means the entry's *late
        translation* must be dropped outright, not merely left unwritten.
        """
        if not original:
            return self.WRITE_SKIPPED
        with self._lock:
            if not self._enabled:
                return self.WRITE_SKIPPED
            if session is not None and self._session_open:
                if session != self._session_ts:
                    log.info(
                        "Refusing entry for session %s: the open session is %s "
                        "(msg %s)", session, self._session_ts, msg_id,
                    )
                    return self.WRITE_SESSION_MISMATCH
            if not self._session_open:
                if (
                    self._ending
                    or not self._recording
                    or self._explicitly_ended
                ):
                    # After end_session() the files are closed and the meeting
                    # is complete: a late original must not reopen it or
                    # silently start a ghost session — only begin_session()
                    # (the user's "Start new recording") or a fresh pipeline
                    # start re-arms auto-open.
                    return self.WRITE_SKIPPED
                self._open_session_locked()
            if not self._session_open:
                return self.WRITE_SKIPPED
            if session is not None and session != self._session_ts:
                # The auto-open above created a *different* session than the
                # one the caller expected (an end+begin raced the entry):
                # refuse rather than write the old audio into it.
                log.info(
                    "Refusing entry for session %s: auto-open created session "
                    "%s (msg %s)", session, self._session_ts, msg_id,
                )
                return self.WRITE_SESSION_MISMATCH
            current = self._session_ts
            if msg_id not in self._pending:
                self._order.append(msg_id)
            self._pending[msg_id] = {
                "timestamp": timestamp,
                "original": original,
                "language": language,
                "duration": duration,
            }
            self._entry_sessions[msg_id] = current
            wrote = self._write_locked(
                "original", f"[{timestamp}] {original}\n"
            )
            if not wrote:
                # The entry is NOT on disk: roll back every trace of it so
                # the seal cannot resurrect it later from _pending/_order
                # (a re-emitted entry whose original line never made it to
                # the file would corrupt the record), and the speech time
                # never counted it.
                self._forget_session_entry(msg_id)
                return self.WRITE_FAILED
            if duration:
                # Counted only after the line is on disk: the summary must
                # reflect persisted speech, not attempted speech.
                self._speech_seconds += float(duration)
            return self.WRITE_RECORDED

    def write_translation(self, msg_id: int, translation: str,
                          session: str | None = None) -> str:
        """Record a translation for an entry. Returns the WRITE_* state.

        ``session`` is the same expected-session stamp the entry's original
        carried: a translation for a session that is not the open one is
        discarded (a late straggler after an end+begin race), never written
        into whichever session happens to be open now."""
        if not translation:
            return self.WRITE_SKIPPED
        return self._complete(msg_id, translation, session)

    def finalize_no_translation(self, msg_id: int,
                                session: str | None = None) -> str:
        """Mark a message complete without a translation (same-language or
        error). Same session semantics as write_translation."""
        return self._complete(msg_id, None, session)

    def _complete(self, msg_id: int, translation: str | None,
                  session: str | None = None) -> str:
        """Complete one entry. Only entries whose original was recorded in
        the currently-open session are completed; everything else is
        discarded:

        * no pending original — the original was refused (session mismatch,
          closed session, subtitle-only) or already flushed by the seal. The
          old behavior wrote an orphan "no original" line into the open
          session; that is precisely how a refused entry's late translation
          could land in the *next* meeting's files, so the orphan path is
          gone;
        * pending original from a different session than the one the caller
          expects, or from a session that is no longer open — the ENDING
          wait timed out and the seal flushed it, or a new session began.
          Completed meetings are immutable.
        """
        with self._lock:
            if not self._enabled:
                self._forget_locked(msg_id)
                return self.WRITE_SKIPPED
            entry_session = self._entry_sessions.get(msg_id)
            if entry_session is None:
                log.info(
                    "Discarding translation with no recorded original (msg %s)",
                    msg_id,
                )
                return self.WRITE_SKIPPED
            if session is not None and session != entry_session:
                # The caller expected this entry in another session: a
                # straggler from before an end+begin race.
                log.info(
                    "Discarding translation for session %s on entry of "
                    "session %s (msg %s)", session, entry_session, msg_id,
                )
                return self.WRITE_SESSION_MISMATCH
            if entry_session != self._session_ts or not self._session_open:
                # A result for a session that is already closed — the ENDING
                # wait timed out and end_session() flushed the entry, or the
                # next session has since begun. The entry is already on disk
                # as an untranslated original; if the user wants it
                # translated, that is a future "retranslate history" feature,
                # not this path.
                log.info(
                    "Discarding late translation for closed session %s (msg %s)",
                    entry_session, msg_id,
                )
                self._forget_session_entry(msg_id)
                return self.WRITE_SESSION_MISMATCH
            self._done[msg_id] = translation
            self._drain_locked()
            return self.WRITE_RECORDED

    def _forget_session_entry(self, msg_id: int):
        self._pending.pop(msg_id, None)
        self._done.pop(msg_id, None)
        self._entry_sessions.pop(msg_id, None)
        try:
            self._order.remove(msg_id)
        except ValueError:
            pass

    def _drain_locked(self):
        """Emit every entry whose turn has come, in utterance order."""
        while self._order and self._order[0] in self._done:
            msg_id = self._order.popleft()
            entry = self._pending.pop(msg_id, None)
            translation = self._done.pop(msg_id, None)
            self._entry_sessions.pop(msg_id, None)
            if entry is not None:
                self._emit_locked(entry, translation)

    def _emit_locked(self, entry: dict, translation: str | None) -> bool:
        """Emit one completed entry to every view. Returns True only when
        every write succeeded; the entry counts are updated only then, so
        entries/translated/untranslated reflect persisted content. A failed
        emit marks the session degraded — the seal will classify it
        interrupted instead of completed."""
        ts = entry["timestamp"]
        original = entry["original"]

        meta = []
        if entry.get("language"):
            meta.append(entry["language"])
        if entry.get("duration"):
            meta.append(f"{float(entry['duration']):.1f}s")
        suffix = f" · {' · '.join(meta)}" if meta else ""
        block = f"\n**{ts}**{suffix}\n\n{original}\n"
        if translation:
            block += f"\n> {translation}\n"

        ok = True
        if translation:
            ok = self._write_locked(
                "translation", f"[{ts}] {translation}\n"
            ) and ok
            ok = self._write_locked(
                "all", f"[{ts}] {original}\n  -> {translation}\n\n"
            ) and ok
        else:
            ok = self._write_locked(
                "all", f"[{ts}] {original}\n\n"
            ) and ok
        ok = self._write_locked(self.MARKDOWN_KIND, block) and ok

        if not ok:
            # Some view of this entry is missing from the files: the entry
            # is consumed (the caller already removed it from pending), the
            # counts must not include it, and the session can no longer be
            # sealed as completed.
            self._write_degraded = True
            log.warning(
                "Entry %s could not be written to every view; the session "
                "will be classified as interrupted", ts,
            )
            return False

        self._counts["entries"] += 1
        if translation:
            self._counts["translated"] += 1
        else:
            self._counts["untranslated"] += 1
        return True

    def _forget_locked(self, msg_id: int):
        self._pending.pop(msg_id, None)
        self._done.pop(msg_id, None)
        try:
            self._order.remove(msg_id)
        except ValueError:
            pass

    def _write_locked(self, kind: str, text: str) -> bool:
        fp = self._files.get(kind)
        if fp is None:
            return False
        try:
            fp.write(text)
            return True
        except (OSError, ValueError) as e:
            # OSError: the disk/stream failed. ValueError: the handle is in
            # an unusable file state (e.g. closed under us) — both are
            # environment failures that belong on the degraded/interrupted
            # path. Anything else (TypeError from bad arguments, ...) is a
            # programming error and propagates.
            log.warning(f"Transcript write failed ({kind}): {e}")
            return False

    # --- summary and close ---------------------------------------------

    def summary(self) -> dict:
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        """Session summary. ``ended`` is None while the session is live and is
        fixed once, at the seal, so the footer, the meta sidecar and the
        returned summary all carry the *same* end timestamp — a live
        session's sidecar must not claim an end time (the records layer uses
        its absence/presence to tell "still recording" from "ended", and a
        crash-left record with a pre-filled ``ended`` would pass as cleanly
        ended instead of interrupted).

        ``session_status`` mirrors the persistent sidecar field:
        "active" while open, "completed"/"interrupted" once the close has
        decided — the records layer trusts this field over any footer for
        records that carry it."""
        started = self._session_started
        if self._ended_at is not None:
            ended = self._ended_at
            duration = int((ended - started).total_seconds()) if started else 0
        else:
            # Live snapshot (monitor bar, mid-session sidecar rewrites): the
            # duration is measured to now, but "ended" stays absent.
            ended = None
            duration = (
                int((datetime.now() - started).total_seconds()) if started else 0
            )
        return {
            "session": self._session_ts,
            "session_status": self._session_status,
            "started": started.isoformat(timespec="seconds") if started else None,
            "ended": ended.isoformat(timespec="seconds") if ended else None,
            "duration_seconds": duration,
            "speech_seconds": round(self._speech_seconds, 1),
            "entries": self._counts["entries"],
            "translated": self._counts["translated"],
            "untranslated": self._counts["untranslated"],
            **self._info,
        }

    # Fields the UI persists into the sidecar that this writer does not own.
    # They survive every meta rewrite (rename mid-session, close, restart);
    # the merge is a whitelist so a corrupt or unexpected old JSON can never
    # smuggle arbitrary keys back into a freshly written sidecar.
    _USER_META_FIELDS = ("title", "title_set_at")

    def _write_meta_locked(self, create_only: bool = False,
                           extra_user_fields: dict | None = None) -> bool:
        """Write the sidecar. Returns True on success.

        ``create_only`` (first write of a session) uses a real exclusive
        create — ``open(path, "x")`` — so a stamp collision or a leftover
        file can never be overwritten, closing the exists-check→replace
        race the staged-replace path would leave. Regular rewrites (live
        updates, the seal) keep the atomic staged replace, merging the
        whitelisted user fields from the file on disk;
        ``extra_user_fields`` (rename_session) supplies new values that
        take precedence over the on-disk ones. The staged temp file gets a
        unique suffix per call — two concurrent staged writes must never
        replace each other's temp.
        """
        path = self._paths.get(self.META_KIND)
        if not path:
            return False
        meta = self._summary_locked()
        try:
            if create_only:
                # Write the staged copy first (so encoding/serialization
                # failures never create a half-written sidecar), then give
                # it the final name exclusively. A failure *after* the
                # exclusive open unlinks the file this call created; an
                # O_EXCL collision leaves the other session's file alone.
                tmp = self._staged_meta_path(path)
                try:
                    tmp.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    try:
                        fd = os.open(
                            path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                        )
                    except FileExistsError:
                        log.error(
                            "Metadata for session %s already exists; refusing "
                            "to overwrite it", self._session_ts,
                        )
                        return False
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as fp:
                            fp.write(tmp.read_text(encoding="utf-8"))
                    except OSError:
                        _discard_quietly(path)
                        raise
                    return True
                finally:
                    _discard_quietly(tmp)
            old = {}
            if Path(path).is_file():
                try:
                    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        old = loaded
                except (OSError, ValueError):
                    old = {}
            for key in self._USER_META_FIELDS:
                if key in old:
                    meta[key] = old[key]
            if extra_user_fields:
                for key in self._USER_META_FIELDS:
                    if key in extra_user_fields:
                        meta[key] = extra_user_fields[key]
            tmp = self._staged_meta_path(path)
            tmp.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
            return True
        except OSError as e:
            log.warning(f"Could not write transcript metadata: {e}")
            return False

    @staticmethod
    def _staged_meta_path(path) -> Path:
        """A per-call staged path. A fixed ``.tmp`` name shared by every
        writer (and by the records layer's rename) let two concurrent
        staged writes replace each other's temp; a pid+counter suffix is
        still only unique within this module (meeting_records stages the
        same final file with the same name shape), so the suffix carries a
        UUID — collision-free across modules and processes."""
        return Path(path).with_name(
            f"{Path(path).name}.tmp{os.getpid()}.{uuid.uuid4().hex[:10]}"
        )

    def close(self):
        """Close the writer outright (app stop / shutdown). Idempotent.

        This is the *pipeline* shutdown path. It shares the seal protocol
        with ``end_session`` — content files flush+close before the final
        status commit — and on any failure it runs the *same* abort
        fallback the button-end path uses (mark interrupted, release
        handles, clear state) instead of merely flipping booleans, so a
        throwing close cannot leave handles or session state behind. The
        failure is logged, not re-raised: stop() must continue reclaiming
        the rest of the app.
        """
        with self._lock:
            if not self._opened and not self._session_open:
                return
            self._ending = True
            try:
                self._close_locked()
            except Exception:
                log.error(
                    "Transcript close failed; aborting the session",
                    exc_info=True,
                )
                try:
                    self._abort_locked()
                except Exception:
                    log.error(
                        "Transcript abort after a failed close also failed",
                        exc_info=True,
                    )
            finally:
                self._ending = False

    def _close_locked(self) -> dict | None:
        """Flush and finalize the current session; shared by close() and
        end_session(). Returns the final sidecar contents, or None when there
        was no session to close.

        Seal protocol — the status commits last, after the content is
        verifiably on disk:
        1. flush pending entries as untranslated (each emit's failure marks
           the session degraded — the entry is consumed either way, so it
           can never reappear);
        2. fix the seal timestamp once, shared by footer/summary;
        3. write the text footers (a failure marks degraded);
        4. seal the *content* files: flush → fsync → close each handle; any
           failure (OSError, ValueError from a dead handle, ...) marks
           degraded — a "completed" marker must never precede this point;
        5. decide the status: ``completed`` only when nothing degraded;
        6. atomically commit the final sidecar (completed, or interrupted
           after a degraded seal); a failed commit downgrades to
           ``interrupted`` and retries once — a second failure leaves the
           on-disk sidecar at its live ``active`` status, which readers
           classify as interrupted;
        7. clear the in-memory state (the reset's own errors are logged —
           it is cleanup, not part of the verdict).
        """
        if not self._opened:
            return None
        # Anything still waiting on a translation that will never arrive
        # would otherwise be lost, and its original line would sit in the
        # original file with no counterpart in the record. A failed emit
        # marks the session degraded; the entry is consumed regardless.
        while self._order:
            msg_id = self._order.popleft()
            entry = self._pending.pop(msg_id, None)
            if entry is not None:
                self._emit_locked(entry, self._done.pop(msg_id, None))
                self._entry_sessions.pop(msg_id, None)
        if self._ended_at is None:
            self._ended_at = datetime.now()
        if not self._write_summary_footer_locked():
            self._write_degraded = True
        # Content seal BEFORE any status commit: once a handle fails to
        # flush/close, the session is interrupted regardless of what the
        # buffered writes claimed.
        if not self._seal_content_files_locked():
            self._write_degraded = True
        self._session_status = (
            "interrupted" if self._write_degraded else "completed"
        )
        # The final sidecar is the commit marker, written only after the
        # content files verifiably closed.
        if not self._write_meta_locked():
            self._write_degraded = True
            self._session_status = "interrupted"
            if not self._write_meta_locked():
                log.error(
                    "Could not write the final sidecar for session %s; the "
                    "record's status is unknown from the sidecar alone",
                    self._session_ts,
                )
        summary = self._summary_locked()
        self._reset_session_state_locked()
        return summary

    def _seal_content_files_locked(self) -> bool:
        """Flush+fsync+close every content file, reporting success.

        This is the content seal: True means every line written this
        session is in the files (flush), durable (fsync) and the handles
        are released. Any failure — including ValueError from writing or
        flushing a half-dead handle — is logged, marks the session
        degraded and still tries to close the remaining handles, but the
        return value is False and the seal therefore interrupted. The
        handles dictionary is left untouched for the reset tail.
        """
        ok = True
        for kind, fp in self._files.items():
            if fp is None:
                continue
            try:
                fp.flush()
                os.fsync(fp.fileno())
                fp.close()
            except (OSError, ValueError) as e:
                # ValueError: the handle is in an unusable state (e.g.
                # already closed under us) — a file-state error, not a
                # programming error, and it belongs on the interrupted
                # path. Other exceptions (TypeError, ...) propagate: a
                # genuine bug should not be misread as a disk failure.
                log.warning("Could not seal transcript file %s: %s", kind, e)
                ok = False
                try:
                    fp.close()
                except Exception:
                    pass
        return ok

    def _abort_locked(self) -> dict | None:
        """Lock-held core of the abort fallback. See abort_session()."""
        try:
            summary = self._summary_locked() if self._session_ts else None
            log.error(
                "Aborting session %s after a failed close; the record stays "
                "as written and is marked interrupted",
                self._session_ts,
            )
            self._session_status = "interrupted"
            # Mark the half-sealed record: the sidecar's status field is
            # what readers of this format trust (a footer may already be in
            # the text files from the failed seal). A failure here is
            # logged, not raised — the abort must still complete.
            if self._session_ts:
                try:
                    if not self._write_meta_locked():
                        log.error(
                            "Could not mark session %s interrupted in its "
                            "sidecar", self._session_ts,
                        )
                except Exception:
                    log.error(
                        "Marking session %s interrupted failed",
                        self._session_ts, exc_info=True,
                    )
            if summary is not None:
                summary["session_status"] = "interrupted"
            return summary
        finally:
            # Whatever happened above, the writer ends with no session:
            # handles closed, memory cleared, the next begin starts clean.
            # Swallow close-time errors — a half-dead handle must not
            # propagate out of an abort.
            try:
                self._reset_session_state_locked()
            except Exception:
                log.error(
                    "Session state reset failed during abort", exc_info=True
                )

    def abort_session(self) -> dict | None:
        """Best-effort abort of a session whose close failed. Never raises.

        The record is kept exactly as far as it got, and the sidecar is
        rewritten (best effort) with ``session_status=interrupted`` so no
        footer that already landed can make the half-sealed meeting read as
        completed — the status field outranks the footer for readers of
        this format. Handles are released and every piece of in-memory
        session state is cleared, in a ``finally`` so even an internal
        failure cannot leave a live session behind. Returns the last known
        summary snapshot (carrying the interrupted status), or None when
        there was no session.
        """
        with self._lock:
            if not self._session_open and not self._opened:
                return None
            return self._abort_locked()

    def _reset_session_state_locked(self):
        """The single tail for every close path (seal, pipeline close,
        abort): close any remaining handles (idempotent — the content seal
        already closed the healthy ones), clear paths/timestamps/entries
        and the per-session counters. Must leave the writer
        indistinguishable from "no session ever opened" so the next begin
        starts clean.

        This is cleanup, NOT part of the seal verdict: the verdict is
        decided by _seal_content_files_locked and the final sidecar commit
        before this runs, and this method's own failures are logged and
        swallowed — the on-disk status is the durable evidence either way.
        Failure evidence (the files, the sidecar) is never touched here."""
        for fp in self._files.values():
            if fp is None:
                continue
            try:
                fp.flush()
                fp.close()
            except (OSError, ValueError):
                # Already sealed/dead handle: nothing further to do, and no
                # reason to raise out of cleanup.
                pass
        self._files.clear()
        self._pending.clear()
        self._done.clear()
        self._order.clear()
        self._entry_sessions.clear()
        self._paths.clear()
        self._session_ts = None
        self._session_started = None
        self._ended_at = None
        self._session_status = None
        self._write_degraded = False
        self._markdown_header_open = False
        self._opened = False
        self._session_open = False
        self._ending = False
        # The writer outlives the session (the app stops and restarts with
        # the same instance), so per-session state must not leak into the
        # next one's counts, footer or sidecar.
        self._counts = {"entries": 0, "translated": 0, "untranslated": 0}
        self._speech_seconds = 0.0
        self._info = {}

    def _write_summary_footer_locked(self) -> bool:
        """Write the end-of-session footers. Returns True only when every
        footer landed; a False return marks the seal degraded (the session
        will be classified interrupted, not completed)."""
        summary = self._summary_locked()
        minutes, seconds = divmod(summary["duration_seconds"], 60)
        hours, minutes = divmod(minutes, 60)
        length = f"{hours}:{minutes:02d}:{seconds:02d}"
        # Same instant as the meta sidecar's "ended" (both read _ended_at,
        # fixed once in _close_locked before either is written).
        ended = (
            self._ended_at.strftime("%Y-%m-%d %H:%M:%S")
            if self._ended_at
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        footer = (
            f"\n# Session ended at {ended}\n"
            f"# Duration {length}, {summary['entries']} entries "
            f"({summary['translated']} translated)\n"
        )
        ok = True
        for kind in self.KINDS:
            ok = self._write_locked(kind, footer) and ok

        lines = [
            "",
            "---",
            "",
            "## Summary",
            "",
            f"- Ended: {ended}",
            f"- Duration: {length}",
            f"- Speech: {summary['speech_seconds']:.0f}s",
            f"- Entries: {summary['entries']} "
            f"({summary['translated']} translated, {summary['untranslated']} not)",
        ]
        for label, key in (
            ("ASR engine", "asr_engine"),
            ("Translation model", "translation_model"),
            ("Source language", "source_language"),
            ("Target language", "target_language"),
        ):
            if summary.get(key):
                lines.append(f"- {label}: {summary[key]}")
        ok = self._write_locked(
            self.MARKDOWN_KIND, "\n".join(lines) + "\n"
        ) and ok
        return ok


def read_session_meta(base_dir: Path) -> list[dict]:
    """List recorded sessions, newest first, for the transcripts page.

    Prefers the JSON sidecar; falls back to counting lines in the text file so
    sessions recorded before the sidecar existed — or ended by a crash — still
    show up.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []

    sessions: dict[str, dict] = {}
    for path in base_dir.glob("livetrans_*_meta.json"):
        stamp = path.name[len("livetrans_"):-len("_meta.json")]
        try:
            sessions[stamp] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.debug("Unreadable transcript metadata: %s", path, exc_info=True)

    for path in base_dir.glob("livetrans_*_all.txt"):
        stamp = path.name[len("livetrans_"):-len("_all.txt")]
        record = sessions.setdefault(stamp, {"session": stamp})
        record.setdefault("started", _stamp_to_iso(stamp))
        if "entries" not in record:
            try:
                text = path.read_text(encoding="utf-8")
                record["entries"] = sum(
                    1 for line in text.splitlines()
                    if line.startswith("[") and line.strip()
                )
            except OSError:
                record["entries"] = 0
        try:
            record["size_bytes"] = path.stat().st_size
        except OSError:
            record["size_bytes"] = 0

    for stamp, record in sessions.items():
        record["session"] = stamp
        record["files"] = {
            kind: str(base_dir / f"livetrans_{stamp}_{kind}.txt")
            for kind in TranscriptWriter.KINDS
            if (base_dir / f"livetrans_{stamp}_{kind}.txt").is_file()
        }
        markdown = base_dir / f"livetrans_{stamp}_{TranscriptWriter.MARKDOWN_KIND}.md"
        if markdown.is_file():
            record["files"][TranscriptWriter.MARKDOWN_KIND] = str(markdown)

    return sorted(
        sessions.values(),
        key=lambda r: stamp_sort_key(r.get("session") or ""),
        reverse=True,
    )


def delete_session(base_dir: Path, stamp: str) -> list[str]:
    """Remove every file of one session. Returns the paths that would not go."""
    failures = []
    for path in Path(base_dir).glob(f"livetrans_{stamp}_*"):
        try:
            path.unlink()
        except OSError as exc:
            failures.append(f"{path}: {exc}")
    return failures
