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
import re
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

log = logging.getLogger("LiveTranslate.Transcript")

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

        Failure rolls back: any file that could not be created exclusively is
        closed and unlinked, no half-open session is reported, and an
        existing session's meta is never overwritten. ``_session_open`` is
        set only when every required file (the three text kinds + the
        Markdown record) was created; the meta sidecar is written last via
        the exclusive-create path too.
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
        failures = []

        def _create(kind: str, path: Path, header: str):
            try:
                # "x" (exclusive create): a stamp collision must refuse a
                # fresh file set, never append to another meeting's files.
                fp = open(path, "x", encoding="utf-8", buffering=1)
                fp.write(header)
                opened[kind] = fp
                paths[kind] = str(path)
                return None
            except FileExistsError:
                return f"{path}: exists"
            except OSError as e:
                return f"{path}: {e}"

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
        self._markdown_header_open = self.MARKDOWN_KIND in opened

        if failures:
            # Roll back what did open: a partial file set would present a
            # "successful" session whose entries silently vanish into closed
            # or missing files. Unlink only what this call created ("x"
            # guarantees nothing pre-existing is touched).
            for kind, fp in opened.items():
                try:
                    fp.close()
                except Exception:
                    pass
                try:
                    Path(paths[kind]).unlink()
                except OSError:
                    log.debug(
                        "Could not unlink %s during session rollback",
                        paths[kind], exc_info=True,
                    )
            log.error(
                "Could not open the full file set for session %s; no session "
                "started: %s", self._session_ts, "; ".join(failures),
            )
            # Leave any previous session's state untouched; a caller that
            # relied on a stamp must observe None, not a phantom session.
            self._session_ts = None
            self._session_started = None
            self._markdown_header_open = False
            return

        self._files = opened
        self._paths = paths
        self._paths[self.META_KIND] = str(
            self._base_dir / f"livetrans_{self._session_ts}_{self.META_KIND}.json"
        )
        self._opened = True
        self._session_open = True
        self._write_meta_locked(create_only=True)
        log.info(f"Transcripts -> {self._base_dir} (session {self._session_ts})")

    # --- entry recording -----------------------------------------------

    def write_original(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        *,
        language: str | None = None,
        duration: float | None = None,
        session: str | None = None,
    ):
        """Record an original line, for the session it belongs to.

        ``session`` is the caller's expected session stamp (snapshotted when
        the audio entered the ASR queue). The check happens inside this
        writer's lock, the final authority: when the session open now is not
        the one this audio belongs to (an end and a new begin raced the
        queue), the entry is refused — the old audio must not land in the
        new meeting's files. ``None`` means "no expectation" (the legacy
        auto-open path, where the session is opened by this very call), and
        is always accepted.
        """
        if not original:
            return
        with self._lock:
            if not self._enabled:
                return
            if session is not None and self._session_open:
                if session != self._session_ts:
                    log.info(
                        "Refusing entry for session %s: the open session is %s "
                        "(msg %s)", session, self._session_ts, msg_id,
                    )
                    return
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
                    return
                self._open_session_locked()
            if not self._session_open:
                return
            if session is not None and session != self._session_ts:
                # The auto-open above created a *different* session than the
                # one the caller expected (an end+begin raced the entry):
                # refuse rather than write the old audio into it.
                log.info(
                    "Refusing entry for session %s: auto-open created session "
                    "%s (msg %s)", session, self._session_ts, msg_id,
                )
                return
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
            if duration:
                self._speech_seconds += float(duration)
            self._write_locked("original", f"[{timestamp}] {original}\n")

    def write_translation(self, msg_id: int, translation: str):
        if not translation:
            return
        self._complete(msg_id, translation)

    def finalize_no_translation(self, msg_id: int):
        """Mark a message complete without a translation (same-language or error)."""
        self._complete(msg_id, None)

    def _complete(self, msg_id: int, translation: str | None):
        with self._lock:
            if not self._enabled:
                self._forget_locked(msg_id)
                return
            entry_session = self._entry_sessions.get(msg_id)
            if entry_session is None:
                # No original in any remembered session — written before this
                # writer's first session, or dropped. Emit standalone while a
                # session is open; after the session closed there is no file
                # to attach it to.
                if not self._session_open or self._ending:
                    self._forget_locked(msg_id)
                    return
                if translation:
                    ts = datetime.now().strftime("%H:%M:%S")
                    self._write_locked("translation", f"[{ts}] {translation}\n")
                    self._write_locked("all", f"[{ts}] -> {translation}\n\n")
                    self._write_locked(
                        self.MARKDOWN_KIND, f"\n**{ts}** — _(no original)_\n\n> {translation}\n"
                    )
                return
            if entry_session != self._session_ts or not self._session_open:
                # A result for a session that is already closed — the ENDING
                # wait timed out and end_session() flushed the entry, or the
                # next session has since begun. Completed meetings are
                # immutable: late results are discarded rather than appended
                # after the Summary footer (which would corrupt the record's
                # order, counts and source_hash). The entry is already on
                # disk as an untranslated original; if the user wants it
                # translated, that is a future "retranslate history" feature,
                # not this path.
                log.info(
                    "Discarding late translation for closed session %s (msg %s)",
                    entry_session, msg_id,
                )
                self._forget_session_entry(msg_id)
                return
            self._done[msg_id] = translation
            self._drain_locked()

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

    def _emit_locked(self, entry: dict, translation: str | None):
        ts = entry["timestamp"]
        original = entry["original"]
        self._counts["entries"] += 1
        if translation:
            self._counts["translated"] += 1
            self._write_locked("translation", f"[{ts}] {translation}\n")
            self._write_locked("all", f"[{ts}] {original}\n  -> {translation}\n\n")
        else:
            self._counts["untranslated"] += 1
            self._write_locked("all", f"[{ts}] {original}\n\n")

        meta = []
        if entry.get("language"):
            meta.append(entry["language"])
        if entry.get("duration"):
            meta.append(f"{float(entry['duration']):.1f}s")
        suffix = f" · {' · '.join(meta)}" if meta else ""
        block = f"\n**{ts}**{suffix}\n\n{original}\n"
        if translation:
            block += f"\n> {translation}\n"
        self._write_locked(self.MARKDOWN_KIND, block)

    def _forget_locked(self, msg_id: int):
        self._pending.pop(msg_id, None)
        self._done.pop(msg_id, None)
        try:
            self._order.remove(msg_id)
        except ValueError:
            pass

    def _write_locked(self, kind: str, text: str):
        fp = self._files.get(kind)
        if fp is None:
            return
        try:
            fp.write(text)
        except OSError as e:
            log.warning(f"Transcript write failed ({kind}): {e}")

    # --- summary and close ---------------------------------------------

    def summary(self) -> dict:
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        def _summary_locked(self) -> dict:
        """Session summary. ``ended`` is None while the session is live and is
        fixed once, at the seal, so the footer, the meta sidecar and the
        returned summary all carry the *same* end timestamp — a live
        session's sidecar must not claim an end time (the records layer uses
        its absence/presence to tell "still recording" from "ended", and a
        crash-left record with a pre-filled ``ended`` would pass as cleanly
        ended instead of interrupted)."""
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

    def _write_meta_locked(self, create_only: bool = False):
        path = self._paths.get(self.META_KIND)
        if not path:
            return
        meta = self._summary_locked()
        try:
            old = {}
            if Path(path).is_file():
                try:
                    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        old = loaded
                except (OSError, ValueError):
                    old = {}
            if create_only and Path(path).exists():
                log.error(
                    "Metadata for session %s already exists; refusing to "
                    "overwrite it", self._session_ts,
                )
                return
            for key in self._USER_META_FIELDS:
                if key in old:
                    meta[key] = old[key]
            tmp = Path(path).with_name(Path(path).name + ".tmp")
            tmp.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError as e:
            log.warning(f"Could not write transcript metadata: {e}")

    def close(self):
        """Close the writer outright (app stop / shutdown).

        This is the *pipeline* shutdown path: whatever is pending is flushed
        as untranslated, the footer and final sidecar are written, files are
        closed and all session state is reset. Ending a *meeting* without
        stopping the pipeline is ``end_session()``; do not bind this to the
        "end recording" button.
        """
        with self._lock:
            self._ending = True
            try:
                self._close_locked()
            finally:
                self._ending = False
                self._session_open = False
                self._opened = False

    def _close_locked(self) -> dict | None:
        """Flush and finalize the current session; shared by close() and
        end_session(). Returns the final sidecar contents, or None when there
        was no session to close.

        The seal timestamp is fixed exactly once, *before* the footer and
        sidecar are written, so every artifact (txt footers, Markdown
        Summary, meta ``ended``, the returned summary) reports the same
        moment. All per-session state is reset in one place, including the
        ended timestamp, so nothing leaks into the next session.
        """
        if not self._opened:
            return None
        # Anything still waiting on a translation that will never arrive
        # would otherwise be lost, and its original line would sit in the
        # original file with no counterpart in the record.
        while self._order:
            msg_id = self._order.popleft()
            entry = self._pending.pop(msg_id, None)
            if entry is not None:
                self._emit_locked(entry, self._done.pop(msg_id, None))
                self._entry_sessions.pop(msg_id, None)
        if self._ended_at is None:
            self._ended_at = datetime.now()
        self._write_summary_footer_locked()
        self._write_meta_locked()
        summary = self._summary_locked()
        for fp in self._files.values():
            if fp is None:
                continue
            try:
                fp.flush()
                fp.close()
            except Exception:
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
        self._markdown_header_open = False
        # The writer outlives the session (the app stops and restarts with
        # the same instance), so per-session state must not leak into the
        # next one's counts, footer or sidecar.
        self._counts = {"entries": 0, "translated": 0, "untranslated": 0}
        self._speech_seconds = 0.0
        self._info = {}
        return summary

    def _write_summary_footer_locked(self):
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
        for kind in self.KINDS:
            self._write_locked(kind, footer)

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
        self._write_locked(self.MARKDOWN_KIND, "\n".join(lines) + "\n")


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
