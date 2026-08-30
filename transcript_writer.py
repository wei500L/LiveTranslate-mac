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
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

log = logging.getLogger("LiveTranslate.Transcript")


class TranscriptWriter:
    KINDS = ("original", "translation", "all")
    MARKDOWN_KIND = "meeting"
    META_KIND = "meta"

    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._enabled = True
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

    # --- session lifecycle ---------------------------------------------

    def set_enabled(self, enabled: bool):
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._enabled:
                if enabled and not self._opened:
                    self._open_session_locked()
                return
            self._enabled = enabled
            if enabled and not self._opened:
                self._open_session_locked()

    def is_enabled(self) -> bool:
        return self._enabled

    def session_paths(self) -> dict:
        with self._lock:
            return dict(self._paths)

    def set_session_info(self, **info):
        """Record what produced this session (ASR engine, model, languages).

        Shown in the meeting record's header and in the transcripts list, so a
        record from six weeks ago still says how it was made.
        """
        with self._lock:
            self._info.update({k: v for k, v in info.items() if v is not None})
            if self._opened:
                self._write_meta_locked()

    def _open_session_locked(self):
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(f"Failed to create transcript dir {self._base_dir}: {e}")
            return
        now = datetime.now()
        self._session_started = now
        self._session_ts = now.strftime("%Y%m%d_%H%M%S")
        header_ts = now.strftime("%Y-%m-%d %H:%M:%S")
        for kind in self.KINDS:
            path = self._base_dir / f"livetrans_{self._session_ts}_{kind}.txt"
            try:
                # line buffered so tail -f works; append mode in case session reopens
                fp = open(path, "a", encoding="utf-8", buffering=1)
                fp.write(f"# Session started at {header_ts}\n")
                self._files[kind] = fp
                self._paths[kind] = str(path)
            except OSError as e:
                log.error(f"Failed to open transcript file {path}: {e}")
                self._files[kind] = None

        md_path = self._base_dir / f"livetrans_{self._session_ts}_{self.MARKDOWN_KIND}.md"
        try:
            fp = open(md_path, "a", encoding="utf-8", buffering=1)
            fp.write(f"# Meeting record {now.strftime('%Y-%m-%d %H:%M')}\n\n")
            fp.write(f"- Started: {header_ts}\n")
            self._files[self.MARKDOWN_KIND] = fp
            self._paths[self.MARKDOWN_KIND] = str(md_path)
            self._markdown_header_open = True
        except OSError as e:
            log.error(f"Failed to open meeting record {md_path}: {e}")
            self._files[self.MARKDOWN_KIND] = None
            self._markdown_header_open = False

        self._paths[self.META_KIND] = str(
            self._base_dir / f"livetrans_{self._session_ts}_{self.META_KIND}.json"
        )
        self._opened = True
        self._write_meta_locked()
        log.info(f"Transcripts -> {self._base_dir}")

    # --- entry recording -----------------------------------------------

    def write_original(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        *,
        language: str | None = None,
        duration: float | None = None,
    ):
        if not original:
            return
        with self._lock:
            if not self._enabled:
                return
            if not self._opened:
                self._open_session_locked()
            if msg_id not in self._pending:
                self._order.append(msg_id)
            self._pending[msg_id] = {
                "timestamp": timestamp,
                "original": original,
                "language": language,
                "duration": duration,
            }
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
            if not self._opened:
                self._open_session_locked()
            if msg_id not in self._pending:
                # No matching original (it was written before this session, or
                # dropped). Emit standalone rather than losing it.
                if translation:
                    ts = datetime.now().strftime("%H:%M:%S")
                    self._write_locked("translation", f"[{ts}] {translation}\n")
                    self._write_locked("all", f"[{ts}] -> {translation}\n\n")
                    self._write_locked(
                        self.MARKDOWN_KIND, f"\n**{ts}** — _(no original)_\n\n> {translation}\n"
                    )
                return
            self._done[msg_id] = translation
            self._drain_locked()

    def _drain_locked(self):
        """Emit every entry whose turn has come, in utterance order."""
        while self._order and self._order[0] in self._done:
            msg_id = self._order.popleft()
            entry = self._pending.pop(msg_id, None)
            translation = self._done.pop(msg_id, None)
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
        started = self._session_started
        ended = datetime.now()
        return {
            "session": self._session_ts,
            "started": started.isoformat(timespec="seconds") if started else None,
            "ended": ended.isoformat(timespec="seconds"),
            "duration_seconds": int((ended - started).total_seconds()) if started else 0,
            "speech_seconds": round(self._speech_seconds, 1),
            "entries": self._counts["entries"],
            "translated": self._counts["translated"],
            "untranslated": self._counts["untranslated"],
            **self._info,
        }

    def _write_meta_locked(self):
        path = self._paths.get(self.META_KIND)
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._summary_locked(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            log.warning(f"Could not write transcript metadata: {e}")

    def close(self):
        with self._lock:
            if self._opened:
                # Anything still waiting on a translation that will never arrive
                # would otherwise be lost, and its original line would sit in the
                # original file with no counterpart in the record.
                while self._order:
                    msg_id = self._order.popleft()
                    entry = self._pending.pop(msg_id, None)
                    if entry is not None:
                        self._emit_locked(entry, self._done.pop(msg_id, None))
                self._write_summary_footer_locked()
                self._write_meta_locked()
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
            self._opened = False
            # The writer outlives the session (the app stops and restarts with
            # the same instance), so per-session state must not leak into the
            # next one's counts, footer or sidecar.
            self._counts = {"entries": 0, "translated": 0, "untranslated": 0}
            self._speech_seconds = 0.0
            self._info = {}

    def _write_summary_footer_locked(self):
        summary = self._summary_locked()
        minutes, seconds = divmod(summary["duration_seconds"], 60)
        hours, minutes = divmod(minutes, 60)
        length = f"{hours}:{minutes:02d}:{seconds:02d}"
        ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    return sorted(sessions.values(), key=lambda r: r["session"], reverse=True)


def _stamp_to_iso(stamp: str) -> str | None:
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
    except ValueError:
        return None


def delete_session(base_dir: Path, stamp: str) -> list[str]:
    """Remove every file of one session. Returns the paths that would not go."""
    failures = []
    for path in Path(base_dir).glob(f"livetrans_{stamp}_*"):
        try:
            path.unlink()
        except OSError as exc:
            failures.append(f"{path}: {exc}")
    return failures
