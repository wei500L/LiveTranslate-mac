"""Structured log capture and run reporting for pipeline diagnostics.

Logs already go to a file, but a file is the wrong shape for answering "did
this run work?": the interesting records are a handful of warnings buried in
thousands of DEBUG lines from the model stack. This captures everything to disk
at DEBUG while keeping the warnings, errors and stage results in memory, so a
run ends with a verdict instead of a scrollback.

Reused by ``debug_pipeline.py`` and available to any other diagnostic entry
point that wants the same treatment.
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

# Model and HTTP stacks emit thousands of DEBUG lines that drown the pipeline's
# own. They stay in the file at WARNING; the pipeline logger stays at DEBUG.
NOISY_LOGGERS = (
    "httpcore", "httpx", "openai", "filelock", "huggingface_hub",
    "funasr", "modelscope", "onnxruntime", "urllib3", "matplotlib",
    "numba", "PIL", "torch", "transformers",
)


class _CollectingHandler(logging.Handler):
    """Keep warnings and above in memory, and count everything by level."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self.counts = Counter()
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self.counts[record.levelname] += 1
            if record.levelno < logging.WARNING:
                return
            try:
                message = record.getMessage()
            except Exception:
                message = str(record.msg)
            detail = None
            if record.exc_info:
                detail = "".join(traceback.format_exception(*record.exc_info)).strip()
            self.records.append(
                {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": message,
                    "detail": detail,
                }
            )


class DiagnosticRecorder:
    """Own the logging setup for a diagnostic run and summarize its outcome."""

    def __init__(self, log_dir: Path, prefix: str = "diagnostic"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{prefix}_{stamp}.log"
        self.notes: list[str] = []
        self.problems: list[str] = []
        self.summary: dict = {}
        self._collector = _CollectingHandler()
        self._installed = False

    # --- setup ---------------------------------------------------------

    def install(self):
        if self._installed:
            return
        self._installed = True
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("  %(levelname)-7s %(name)s: %(message)s"))

        root.handlers = [file_handler, console, self._collector]
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)

        # An exception on a worker thread must land in the report, not vanish.
        def _thread_hook(hook_args):
            logging.getLogger("LiveTranslate").critical(
                f"Uncaught exception in thread {hook_args.thread}",
                exc_info=(
                    hook_args.exc_type, hook_args.exc_value, hook_args.exc_traceback
                ),
            )

        threading.excepthook = _thread_hook
        sys.excepthook = lambda *info: logging.getLogger("LiveTranslate").critical(
            "Uncaught exception", exc_info=info
        )

    # --- collection ----------------------------------------------------

    def note(self, text: str):
        self.notes.append(text)
        logging.getLogger("LiveTranslate.Debug").info(text)

    def problem(self, text: str):
        self.problems.append(text)
        logging.getLogger("LiveTranslate.Debug").error(text)

    @property
    def errors(self) -> list[dict]:
        return [r for r in self._collector.records if r["level"] != "WARNING"]

    @property
    def warnings(self) -> list[dict]:
        return [r for r in self._collector.records if r["level"] == "WARNING"]

    def finish(self, observer, app, *, expect_translation: bool = False):
        """Snapshot what the run produced, once the pipeline has stopped.

        A stage that was asked to do work and produced nothing is a failure, not
        a quiet OK — that silence is exactly the symptom this whole diagnostic
        exists to catch.
        """
        transcript_paths = {}
        try:
            transcript_paths = {
                kind: str(path) for kind, path in app._transcript.session_paths().items()
            }
        except Exception:
            pass
        self.summary = {
            "segments": len(observer.segments),
            "segment_seconds": round(sum(observer.segments), 1),
            "asr_results": observer.asr_results,
            "interim_results": observer.interim,
            "translations": observer.translations,
            "transcript_paths": transcript_paths,
            "log_counts": dict(self._collector.counts),
        }
        if not observer.segments:
            self.problem(
                "VAD produced no speech segments — check the audio level, "
                "vad_mode and vad_threshold"
            )
        elif not observer.asr_results:
            self.problem(
                f"{len(observer.segments)} segment(s) reached ASR but nothing was "
                f"recognized — check the ASR language against the audio"
            )
        elif expect_translation and not observer.translations:
            self.problem(
                f"{len(observer.asr_results)} recognized line(s) but no translation "
                f"landed — check the endpoint, key, model id and timeout"
            )
        if not transcript_paths:
            self.problem(
                "no transcript session was opened — auto_save_transcript is off "
                "or the transcripts directory is not writable"
            )

    # --- reporting -----------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "log_path": str(self.log_path),
            "notes": self.notes,
            "problems": self.problems,
            "warnings": self.warnings,
            "errors": self.errors,
            "summary": self.summary,
        }

    def report(self, stream=sys.stdout):
        write = lambda text="": print(text, file=stream)  # noqa: E731
        write()
        write("=" * 72)
        write("PIPELINE DIAGNOSTIC REPORT")
        write("=" * 72)

        if self.notes:
            write()
            write("Setup")
            for note in self.notes:
                write(f"  - {note}")

        summary = self.summary
        if summary:
            write()
            write("Capture -> VAD -> ASR")
            write(
                f"  segments: {summary['segments']} "
                f"({summary['segment_seconds']}s of speech)"
            )
            asr = summary["asr_results"]
            write(f"  recognized: {len(asr)}")
            for text, language, ms in asr:
                write(f"    [{language}] {ms:6.0f}ms  {text}")
            if summary["interim_results"]:
                write(f"  interim: {len(summary['interim_results'])}")
                for text, language, ms in summary["interim_results"]:
                    write(f"    [{language}] {ms:6.0f}ms  {text}")

            write()
            write("Translation")
            translations = summary["translations"]
            write(f"  translated: {len(translations)}")
            for source, translated in translations:
                write(f"    {source}")
                write(f"      -> {translated}")

            write()
            write("Transcript")
            if summary["transcript_paths"]:
                for kind, path in sorted(summary["transcript_paths"].items()):
                    try:
                        lines = sum(
                            1 for line in Path(path).read_text(encoding="utf-8").splitlines()
                            if line.strip() and not line.startswith("#")
                        )
                    except OSError:
                        lines = -1
                    write(f"  {kind:12} {lines:3} lines  {path}")
            else:
                write("  (no transcript session was opened)")

        if self.problems:
            write()
            write(f"Problems ({len(self.problems)})")
            for problem in self.problems:
                write(f"  ! {problem}")

        if self.errors:
            write()
            write(f"Errors ({len(self.errors)})")
            for record in self.errors:
                write(f"  ! [{record['logger']}] {record['message']}")
                if record["detail"]:
                    for line in record["detail"].splitlines()[-4:]:
                        write(f"      {line}")

        if self.warnings:
            write()
            write(f"Warnings ({len(self.warnings)})")
            seen = Counter()
            for record in self.warnings:
                key = (record["logger"], record["message"][:100])
                seen[key] += 1
                if seen[key] == 1:
                    write(f"  - [{record['logger']}] {record['message']}")
                elif seen[key] == 2:
                    write("    (repeated)")

        write()
        verdict = "FAIL" if (self.problems or self.errors) else "OK"
        write(f"Verdict: {verdict}")
        write("=" * 72)
