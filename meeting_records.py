"""Meeting-records data layer: parse, title, hash and summary persistence.

The UI never reads transcript files directly — it goes through this module so
that every session, including ones recorded before the Markdown record or the
JSON sidecar existed, becomes the same structured thing:

* ``list_sessions()``    — metadata rows for the list pane (newest first)
* ``parse_session()``    — structured ``(timestamp, original, translation)``
                           entries parsed from Markdown, the combined text file,
                           or the original-only file, whichever survives
* ``session_title()``    — user-set title, else a localized default
* ``set_session_title()``— rename, written to the sidecar (created on demand)
* ``load_summary()`` / ``save_summary()`` / ``delete_summary()`` — the AI
  summary file pair ``livetrans_{session}_summary.md`` +
  ``livetrans_{session}_summary_meta.json``; writes are atomic and never store
  credentials, and a summary whose ``source_hash`` no longer matches the
  record is reported as stale instead of being silently dropped.

Pure logic: no Qt, no network, so the offline test job covers all of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger("LiveTranslate.Records")

# Staged atomic writes use a per-call UUID suffix (see _staged_path): the
# writer module stages the *same* sidecar file with the same name shape, and
# pid+counter counters are per-module — two independent counters can agree,
# letting concurrent staged writes replace each other's temp file.

SUMMARY_KIND = "summary"
SUMMARY_META_KIND = "summary_meta"
SUMMARY_SCHEMA_VERSION = 1
DEFAULT_TITLE_FORMAT = "Meeting record {stamp}"
SOURCE_HASH_FIELDS = ("timestamp", "original", "translation")

# Meeting Markdown: "**09:15:00** · ru · 2.5s" then the original, then "> tl".
_ENTRY_HEAD = re.compile(r"^\*\*(\d{1,2}:\d{2}:\d{2})\*\*(.*)$")
# Combined text: "[09:15:00] original" optionally followed by "  -> tl".
_LINE_TS = re.compile(r"^\[(\d{1,2}:\d{2}:\d{2})\]\s?(.*)$")
_ARROW = "-> "

_CHARS_PER_CHUNK_TARGET = 6000
_CHUNK_MIN_CHARS = 1200  # below this, a single request is used without splitting


def needs_chunking(entries: list[dict]) -> bool:
    return _entry_chars_total(entries) > _CHARS_PER_CHUNK_TARGET + _CHUNK_MIN_CHARS


def _entry_chars_total(entries: list[dict]) -> int:
    return sum(_entry_chars(e) for e in entries)


# --- session listing and metadata --------------------------------------------

def list_sessions(base_dir: Path, active_session: str | None = None,
                  ending_session: str | None = None) -> list[dict]:
    """Session metadata rows, newest first, with title and summary state.

    Wraps ``read_session_meta`` and adds what the records center needs:
    a display title (sidecar ``title`` or a default derived from the stamp)
    and whether an AI summary exists for the session. Summary files are
    excluded up front — the ``livetrans_{stamp}_summary.md`` pair matches
    the writer's ``livetrans_*`` globs and would otherwise surface as a
    phantom session with no entries.

    ``active_session`` (the writer's current stamp) marks the row that is
    still being recorded: entries and duration there are a snapshot, not
    final, and the UI treats it accordingly. ``ending_session`` marks the
    row whose close is in flight — the writer stops reporting it as active
    the moment ``end_session()`` starts, so the app-level stamp keeps the
    record identifiable (and protected) through the seal.
    """
    from transcript_writer import read_session_meta

    base_dir = Path(base_dir)
    sessions = read_session_meta(base_dir)
    # `livetrans_{stamp}_summary_meta.json` matches the writer's meta glob and
    # would surface as a phantom session. The reader keys the stamp off the
    # file name, so filter on the filename, not the (possibly corrupt) JSON's
    # session field — a damaged summary sidecar must not become a meeting.
    sessions = [
        r for r in sessions
        if not str(r.get("session", "")).endswith("_" + SUMMARY_KIND)
    ]
    # Sessions whose only surviving file is the original (pre-sidecar, or the
    # richer files were lost) do not match the writer's _all.txt glob; pick
    # them up so nothing recorded is invisible in the list.
    known = {r.get("session") for r in sessions}
    for path in base_dir.glob(f"livetrans_*_original.txt"):
        stamp = path.name[len("livetrans_"):-len("_original.txt")]
        if stamp in known or stamp.endswith("_" + SUMMARY_KIND):
            continue
        record = {"session": stamp, "started": _stamp_to_iso(stamp), "entries": 0}
        try:
            text = path.read_text(encoding="utf-8")
            record["entries"] = sum(
                1 for line in text.splitlines()
                if line.startswith("[") and line.strip()
            )
        except OSError:
            pass
        sessions.append(record)
    # Sort by the structured key so "20260904_101530_02" lands after
    # "20260904_101530_01" and after the bare same-second stamp, numerically.
    from transcript_writer import stamp_sort_key

    sessions.sort(key=lambda r: stamp_sort_key(r.get("session") or ""), reverse=True)
    for record in sessions:
        record.setdefault("files", {})
        stamp = record["session"]
        for kind in ("original", "translation", "all", "meeting"):
            path = base_dir / (
                f"livetrans_{stamp}_{kind}.txt" if kind != "meeting"
                else f"livetrans_{stamp}_meeting.md"
            )
            if path.is_file():
                record["files"].setdefault(kind, str(path))
        record["title"] = _session_title(record, base_dir)
        record["is_active"] = bool(
            active_session and stamp == active_session
        )
        record["is_ending"] = bool(
            ending_session and stamp == ending_session
        )
        # Closed-state classification, read from the record itself rather
        # than from process state. Two formats:
        #
        # New format (session_status in the sidecar): the writer's explicit
        #   verdict — "completed" is the only "ended normally"; "active" and
        #   "interrupted" are not, however many footers or "ended" fields a
        #   half-failed seal happened to write before the status was decided
        #   (the status is committed last, so it outranks everything).
        #
        # Old format (no session_status): the Markdown footer ("## Summary")
        #   is the writer's end-of-session mark, and a sidecar "ended"
        #   timestamp is the same class of evidence. The writer sets "ended"
        #   exactly once at the seal (a live session's sidecar carries no
        #   "ended" at all), so a false "cleanly ended" for a crashed live
        #   session is not possible under the current writer; records from
        #   writers that pre-dated that invariant may still show an "ended"
        #   snapshot from a mid-session sidecar rewrite — the footer check
        #   stays authoritative for them.
        #
        # An ENDING row is neither: the close is in flight (the writer has
        #   stopped reporting it active, and the sidecar still says "active"
        #   until the seal commits). Classifying it "interrupted" for those
        #   seconds contradicted its own is_ending flag — one record could
        #   be simultaneously ending and interrupted, and any consumer
        #   keying on interrupted (filters, exports, permissions) would
        #   misread a meeting mid-close as a crash-left one.
        status = record.get("session_status")
        if status is not None:
            record["ended_cleanly"] = status == "completed"
        else:
            record["ended_cleanly"] = (
                _meeting_has_footer(base_dir, stamp)
                or bool(record.get("ended"))
            )
        record["interrupted"] = bool(
            not record["is_active"]
            and not record["is_ending"]
            and not record["ended_cleanly"]
        )
        # The committed state is "the pair loads and verifies" — a meta without
        # a matching body (interrupted commit) must not show as summarized.
        record["has_summary"] = load_summary(base_dir, record["session"]) is not None
        if record["has_summary"]:
            summary_meta = _summary_meta_path(base_dir, record["session"])
            try:
                meta = json.loads(summary_meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            record["summary_edited"] = bool(meta.get("edited_by_user"))
            # A session still being recorded grows by definition; a summary
            # snapshot of it is interim, not "stale" noise.
            record["summary_stale"] = (
                False if record["is_active"]
                else _summary_is_stale(meta, record, base_dir)
            )
    return sessions


def _meeting_has_footer(base_dir: Path, stamp: str) -> bool:
    """True when the meeting Markdown carries the end-of-session footer.

    A missing Markdown file (pre-Markdown era, or only the txt files
    survived) resolves to the sidecar's "ended" field instead — handled by
    the caller — so this only answers "is there a footer".
    """
    path = base_dir / f"livetrans_{stamp}_meeting.md"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "## Summary" in text


def _session_title(record: dict, base_dir: Path) -> str:
    title = record.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    stamp = record.get("session") or ""
    started = record.get("started") or ""
    try:
        when = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        when = stamp or "?"
    return DEFAULT_TITLE_FORMAT.format(stamp=when)


def _stamp_to_iso(stamp: str) -> str | None:
    """ISO start time from a session stamp, tolerating the "_NN" suffix."""
    from transcript_writer import _stamp_to_iso as _tw_stamp_to_iso

    return _tw_stamp_to_iso(stamp)


def session_title(record: dict, base_dir: Path) -> str:
    """Public title lookup for one record (creates nothing)."""
    return _session_title(record, Path(base_dir))


def set_session_title(base_dir: Path, stamp: str, title: str) -> bool:
    """Persist a user title into the session sidecar, creating it if needed.

    The sidecar is the only writable metadata store that already exists; old
    sessions without one get it on first rename rather than a new file type.

    Coordinated-writer rule: this path is for *closed* sessions only. A
    live session's rename must go through ``TranscriptWriter.rename_session``
    (the writer's lock) — two uncoordinated read-modify-write loops over
    the same sidecar lose titles and can roll a committed status back. The
    UI routes accordingly; this function cannot tell, so it stays safe for
    the closed case.
    """
    base_dir = Path(base_dir)
    title = (title or "").strip()
    if not title:
        return False
    meta_path = base_dir / f"livetrans_{stamp}_meta.json"
    try:
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
        meta["session"] = stamp
        meta["title"] = title
        meta["title_set_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write_json(meta_path, meta)
        return True
    except OSError as exc:
        log.warning("Could not save title for %s: %s", stamp, exc)
        return False


# --- structured parsing ------------------------------------------------------

def parse_session(base_dir: Path, record: dict) -> list[dict]:
    """Structured entries for one session, in order of appearance.

    Markdown is the richest source (original + translation + language). The
    combined text file serves sessions recorded before Markdown existed, and
    the original-only file is the last resort — its entries simply have no
    translation, which the UI marks as an untranslated state.
    """
    base_dir = Path(base_dir)
    files = record.get("files") or {}
    for kind, parser in (
        ("meeting", _parse_markdown),
        ("all", _parse_all_text),
        ("original", _parse_original_text),
    ):
        path = files.get(kind)
        if not path:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        entries = parser(text)
        if entries:
            return entries
    return []


def _parse_markdown(text: str) -> list[dict]:
    """Parse the meeting Markdown into structured entries.

    Entry shape: ``**HH:MM:SS** · lang · 1.2s`` header, original text, and a
    ``> translation`` block quote when the entry was translated. The footer
    (``## Summary``) and the session header are not entries.
    """
    entries = []
    current = None
    in_summary = False
    for line in text.splitlines():
        if line.startswith("## Summary"):
            in_summary = True
        if in_summary:
            continue
        head = _ENTRY_HEAD.match(line.strip())
        if head:
            if current is not None:
                _finish_entry(current, entries)
            current = {
                "timestamp": head.group(1),
                "meta": head.group(2).strip(" ·"),
                "original": [],
                "translation": None,
            }
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("> "):
            current["translation"] = stripped[2:].strip()
        elif stripped == ">":
            current["translation"] = ""
        elif stripped:
            if current["translation"] is not None:
                # A second prose block after the quote is a new paragraph of
                # the translation in the orphan-entry form ("_(no original)_")
                current["translation"] += " " + stripped
            else:
                current["original"].append(stripped)
    if current is not None:
        _finish_entry(current, entries)
    return entries


def _finish_entry(current: dict, entries: list):
    original = "\n".join(current["original"]).strip()
    if not original and current["translation"] is None:
        return
    language = _language_from_meta(current.get("meta") or "")
    entries.append(
        {
            "timestamp": current["timestamp"],
            "original": original,
            "translation": current["translation"],
            "language": language,
        }
    )


def _language_from_meta(meta: str) -> str | None:
    for part in meta.split("·"):
        part = part.strip()
        if re.fullmatch(r"[a-z]{2,3}(\.[A-Za-z]{2,4})?", part):
            return part
    return None


def _parse_all_text(text: str) -> list[dict]:
    """Parse the combined text file: ``[ts] original`` + ``  -> translation``."""
    entries = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        match = _LINE_TS.match(stripped)
        if match:
            if current is not None:
                entries.append(current)
            current = {
                "timestamp": match.group(1),
                "original": match.group(2).strip(),
                "translation": None,
                "language": None,
            }
            continue
        if current is not None and stripped.startswith(_ARROW):
            current["translation"] = stripped[len(_ARROW):].strip()
    if current is not None:
        entries.append(current)
    return entries


def _parse_original_text(text: str) -> list[dict]:
    """Last resort: original lines only, no translations."""
    entries = []
    for line in text.splitlines():
        match = _LINE_TS.match(line.strip())
        if match and match.group(2).strip():
            entries.append(
                {
                    "timestamp": match.group(1),
                    "original": match.group(2).strip(),
                    "translation": None,
                    "language": None,
                }
            )
    return entries


# --- source hash ---------------------------------------------------------------

def source_hash(entries: list[dict]) -> str:
    """Stable digest of the record content a summary was built from.

    Only (timestamp, original, translation) triples participate — the entry
    language tag and display metadata changes must not invalidate summaries.
    """
    digest = hashlib.sha256()
    for entry in entries:
        for field in SOURCE_HASH_FIELDS:
            value = entry.get(field) or ""
            digest.update(field.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\x1e")
        digest.update(b"\x1f")
    return digest.hexdigest()


# --- AI summary persistence ----------------------------------------------------

def _summary_paths(base_dir: Path, stamp: str) -> tuple[Path, Path]:
    base = Path(base_dir) / f"livetrans_{stamp}_summary"
    return (base.with_suffix(".md"), base.with_name(base.name + "_meta.json"))


def _summary_meta_path(base_dir: Path, stamp: str) -> Path:
    return _summary_paths(base_dir, stamp)[1]


def load_summary(base_dir: Path, stamp: str) -> dict | None:
    """Load ``{content, meta}`` for a session, or None when absent/broken.

    Broken pairs are reported as absent — the UI shows the empty state and the
    user can regenerate, rather than an error for a file they cannot fix.

    What counts as broken:
    * a meta whose ``content_sha256`` does not match the body — a commit
      interrupted between the two ``os.replace`` calls, i.e. new metadata
      over old text;
    * a body with **no readable meta** — the signature of a commit
      interrupted *before* the meta replace (an orphan body). Treating that
      as "committed" would present a half-written generation as the user's
      minutes;
    * a legacy meta **without** ``content_sha256`` is accepted: summaries
      written before the field existed are valid content, only unverifiable.
    """
    md_path, meta_path = _summary_paths(Path(base_dir), stamp)
    if not md_path.is_file():
        return None
    try:
        content = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not meta_path.is_file():
        # Orphan body: the meta is the commit marker (it is written last,
        # and it carries the body hash); without it this body was never
        # committed as a summary.
        log.warning("Summary body without meta for %s — treating as absent", stamp)
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Unreadable/corrupt meta: the commit marker does not verify, so the
        # pair does not count as a committed summary.
        log.warning("Unreadable summary meta for %s — treating as absent", stamp)
        return None
    if not isinstance(meta, dict):
        log.warning("Invalid summary meta for %s — treating as absent", stamp)
        return None
    stored = meta.get("content_sha256")
    if stored:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != stored:
            log.warning(
                "Summary body/meta hash mismatch for %s — treating as absent", stamp
            )
            return None
    return {"content": content, "meta": meta}


def save_summary(
    base_dir: Path,
    stamp: str,
    content: str,
    meta: dict,
) -> bool:
    """Write the summary pair as one commit, meta last. Meta is scrubbed.

    Commit protocol (single-replace-pair, no true cross-file transaction):
    both files are staged as uniquely-suffixed ``.tmp`` siblings first, so a
    failure before the commit phase leaves the old pair untouched — and two
    concurrent saves (a generation finishing while the user edits, or two
    workers) never share a staged name, so one save cannot commit the other
    save's body with its own meta. The body is then replaced, and the meta —
    which carries ``content_sha256`` — is replaced last. The meta is
    therefore the commit marker: readers verify its hash against the body,
    so a crash between the two replaces can never present new metadata over
    an old body as a valid summary. A crash the other way (meta replaced,
    body not) cannot occur; a leftover ``.tmp*`` from a crash matches no
    listing glob and is removed by ``delete_summary``.
    """
    base_dir = Path(base_dir)
    md_path, meta_path = _summary_paths(base_dir, stamp)
    safe_meta = {k: v for k, v in meta.items() if k not in _SENSITIVE_META_KEYS}
    safe_meta["schema_version"] = SUMMARY_SCHEMA_VERSION
    safe_meta["session"] = stamp
    safe_meta["content_sha256"] = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    md_tmp = _staged_path(md_path)
    meta_tmp = _staged_path(meta_path)
    try:
        md_tmp.write_text(content, encoding="utf-8")
        meta_tmp.write_text(
            json.dumps(safe_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_tmp.replace(md_path)
        meta_tmp.replace(meta_path)
        return True
    except OSError as exc:
        log.warning("Could not save summary for %s: %s", stamp, exc)
        _discard_quietly(md_tmp)
        _discard_quietly(meta_tmp)
        return False


_SENSITIVE_META_KEYS = ("api_key", "api_base", "key", "password", "token")


def delete_summary(base_dir: Path, stamp: str) -> bool:
    """Remove both summary files of one session. True if anything went away."""
    md_path, meta_path = _summary_paths(Path(base_dir), stamp)
    removed = False
    # Includes the staged siblings a crashed commit can leave behind —
    # both the legacy fixed ".tmp" names and the unique-suffixed staged
    # names (".tmp<pid>.<uuid>") — so a delete never strands half-written
    # files in the transcripts folder.
    candidates = [md_path, meta_path,
                  md_path.with_name(md_path.name + ".tmp"),
                  meta_path.with_name(meta_path.name + ".tmp")]
    for pattern in (f"{md_path.name}.tmp*", f"{meta_path.name}.tmp*"):
        candidates.extend(Path(base_dir).glob(pattern))
    for path in candidates:
        try:
            if path.is_file():
                path.unlink()
                removed = True
        except OSError as exc:
            log.warning("Could not delete %s: %s", path, exc)
    return removed


def _summary_is_stale(meta: dict, record: dict, base_dir: Path) -> bool:
    """Whether the record changed since the summary was generated.

    Missing hash (summaries from before the field existed) is *not* stale —
    we cannot know, and marking every old summary outdated on upgrade would
    be noise. Only a present hash that no longer matches counts.
    """
    stored = meta.get("source_hash")
    if not stored:
        return False
    entries = parse_session(base_dir, record)
    if not entries:
        return False  # cannot judge without a parseable record
    return source_hash(entries) != stored


def summary_state(base_dir: Path, stamp: str, record: dict) -> str:
    """One of 'none' | 'generating' is never stored here — this returns
    'none' | 'ready' | 'stale' | 'edited' for list/filter purposes."""
    loaded = load_summary(base_dir, stamp)
    if loaded is None:
        return "none"
    if _summary_is_stale(loaded["meta"], record, Path(base_dir)):
        return "stale"
    if loaded["meta"].get("edited_by_user"):
        return "edited"
    return "ready"


# --- chunking -------------------------------------------------------------------

def chunk_entries(entries: list[dict], max_chars: int = _CHARS_PER_CHUNK_TARGET
                   ) -> list[list[dict]]:
    """Split entries into chunks near ``max_chars``, never inside an entry.

    Backtracks to a previous complete entry when the next one would overflow
    the budget; a single entry larger than the whole budget becomes its own
    chunk so oversized content is never silently dropped.
    """
    chunks = []
    current: list[dict] = []
    current_chars = 0
    for entry in entries:
        size = _entry_chars(entry)
        if current and current_chars + size > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(entry)
        current_chars += size
    if current:
        chunks.append(current)
    return chunks


def _entry_chars(entry: dict) -> int:
    # Translation preferred (it is the summary input when present), original
    # as the fallback — so chunking reflects the real request size.
    text = entry.get("translation") or entry.get("original") or ""
    return len(str(text)) + len(entry.get("timestamp") or "") + 2


def entries_to_text(entries: list[dict]) -> str:
    """Render entries as the ``[HH:MM:SS] text`` block sent to the model.

    The translation is the content when present (summaries are built from the
    language the user reads); entries without one fall back to the original
    rather than being dropped, so a few untranslated lines never lose the
    surrounding context.
    """
    lines = []
    for entry in entries:
        text = entry.get("translation") or entry.get("original") or ""
        lines.append(f"[{entry.get('timestamp', '')}] {text}".strip())
    return "\n".join(lines)


# --- atomic writes --------------------------------------------------------------

def _discard_quietly(path: Path):
    try:
        path.unlink()
    except OSError:
        pass


def _atomic_write_json(path: Path, data: dict):
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _atomic_write_text(path: Path, content: str):
    # Unique staged name per call: a fixed ".tmp" shared with the writer's
    # staged meta writes let two concurrent writers replace each other's
    # temp file. (The coordinated-writer rule keeps the *final* file
    # single-writer; this keeps even the staging area collision-free.)
    tmp = _staged_path(path)
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _staged_path(path: Path) -> Path:
    """A per-call staged sibling. A pid+counter suffix is unique only
    within this module — transcript_writer stages the *same* sidecar file
    with the same name shape, and two independent counters can agree, so
    the suffix carries a UUID (collision-free across modules and
    processes)."""
    return path.with_name(
        f"{path.name}.tmp{os.getpid()}.{uuid.uuid4().hex[:10]}"
    )
