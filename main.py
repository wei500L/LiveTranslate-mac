"""LiveTranslate cross-platform real-time audio translation application."""

import sys
import signal
import logging
import threading
import queue
import gc
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from model_manager import (
    DEFAULT_FUNASR_MODEL,
    DEFAULT_GIGAAM_MODEL,
    apply_cache_env,
    funasr_display_name,
    funasr_supports_padding,
    get_missing_models,
    gigaam_display_name,
    gigaam_is_russian_only,
    gigaam_model_id,
    gigaam_revision,
    is_asr_cached,
    normalize_gigaam_model_key,
    ASR_DISPLAY_NAMES,
    MODELS_DIR,
    local_faster_whisper_display_name,
    migrate_funasr_settings,
    normalize_asr_engine_selection,
    normalize_funasr_model_key,
    resolve_custom_whisper_model,
)

# Set cache env BEFORE importing torch so TORCH_HOME is respected
apply_cache_env()

import os

# Keep torch initialization ahead of Qt for consistent backend startup.
import torch  # noqa: F401

from audio_capture import AudioCapture
from platform_permissions import (
    CaptureRuntimeError,
    MicrophonePermissionDeniedError,
    PermissionDeniedError,
    PlatformUnavailableError,
)
from torch_backend import (
    accelerator_memory,
    cuda_available,
    empty_cache,
    mps_available,
    normalize_device,
)
from platform_fonts import default_mono_font_family, default_ui_font_family
from platform_app import configure_application, set_dock_visible
from platform_config import normalize_config
from connection_config import (
    DEFAULT_REMOTE_ASR_URL,
    DEFAULT_TRANSLATION_API_BASE,
    normalize_remote_asr_url,
    translation_api_key,
)
from vad_processor import VADProcessor
from asr_client import ASRClient, ASRWorkerError, ASRWorkerExited, ASRWorkerTimeout
from asr_remote import RemoteASRError
from translator import RepetitionError, translator_from_model_config
from mlx_service import MLXServiceManager, is_hy_mt_model
from transcript_writer import TranscriptWriter

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSystemTrayIcon,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QIcon,
    QPixmap,
    QPainter,
    QColor,
    QFont,
    QFontDatabase,
)
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal

from subtitle_overlay import SubtitleOverlay
from subtitle_window import SubtitleWindow
from log_window import LogWindow
from control_panel import (
    ControlPanel,
    SETTINGS_FILE,
    _load_saved_settings,
    _save_settings,
)
from dialogs import (
    SetupWizardDialog,
    ModelDownloadDialog,
    _ModelLoadDialog,
)
from i18n import t, set_lang, LANGUAGES, COMMON_LANG_CODES
from ui_theme import apply_theme


def _audio_start_error(exc):
    """Turn platform capture failures into actionable localized guidance."""
    if isinstance(exc, MicrophonePermissionDeniedError):
        return t("error_microphone_permission")
    if isinstance(exc, PermissionDeniedError):
        return t("error_screen_permission")
    if isinstance(exc, PlatformUnavailableError):
        return t("error_screen_dependency").format(error=exc)
    return str(exc)

_NO_PENDING = object()


def setup_logging():
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"livetrans_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    for noisy in (
        "httpcore",
        "httpx",
        "openai",
        "filelock",
        "huggingface_hub",
        "funasr",
        "modelscope",
        "onnxruntime",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Log file: {log_file}")

    # FunASR/ModelScope spam the root logger; suppress after our own init log
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)

    _logger = logging.getLogger("LiveTranslate")

    def _excepthook(exc_type, exc_value, exc_tb):
        _logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        _logger.critical(
            f"Uncaught exception in thread {args.thread}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    return _logger


log = logging.getLogger("LiveTranslate")


class ASRProtocolError(RuntimeError):
    """An ASR backend returned something outside the agreed result contract."""


class TranslationUnavailable(RuntimeError):
    """No translation service is configured or running.

    Distinct from the RuntimeError a shut-down executor raises: that one means
    "we are exiting", this one means "the user needs to know translation is off".
    """


class SessionState:
    """Meeting-recording lifecycle, owned by LiveTranslateApp only.

    One authority for "is a meeting being recorded": the overlay button, the
    records page and the panel all read this through signals or getters and
    never keep their own copy of the state.

    States:
      IDLE    — no recording session (pipeline may still run);
      ACTIVE  — a session is being recorded;
      PAUSED  — recording paused, resuming continues the *same* session;
      ENDING  — the session is completing final ASR/translation/save work.

    The recording *session* is deliberately independent of the *pipeline*
    (capture/ASR/translation threads): pausing the pipeline pauses the
    meeting; ending the meeting ends the record but leaves the pipeline,
    windows and ASR model loaded; quitting the app stops everything.
    """

    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDING = "ending"

    # Wait up to this long for in-flight ASR/translation work during ENDING.
    # Bounded so the UI cannot hang on "Ending…" forever; work that misses
    # the deadline is closed out as untranslated by end_session().
    ENDING_TIMEOUT_S = 30.0

    @classmethod
    def is_recording(cls, state: str) -> bool:
        return state in (cls.ACTIVE, cls.PAUSED)


class _SessionWorkTracker:
    """Per-session in-flight work counts, waited on by the ENDING thread.

    Replaces the ``_asr_queue.empty() and not _session_msg_ids`` drain check,
    which raced the ASR thread: an item already *taken* from the queue makes
    the queue empty while recognition has not finished (no msg_id registered
    yet, nothing written yet) — the end thread would close the meeting and
    lose the last utterance.

    Model: every unit of session work is counted under exactly one
    ``generation`` (the session's token), and each generation has a
    linearized lifecycle inside this tracker's single lock/Condition:

    ``OPEN``       — accepting registrations (session recording);
    ``CLOSING``    — the end was requested: *ordinary* registrations are
                     refused, so ``wait_idle`` can prove that once the count
                     reaches zero it stays zero — the only new work during
                     CLOSING is the controlled final VAD flush, which
                     registers through ``register_final``;
    ``SUPERSEDED`` — the session is fully closed (end completed, timed out,
                     or the app quit): registrations and releases are
                     refused/ignored, the generation's records are dropped.

    This removes the remaining race where a capture thread that produced a
    segment but had not enqueued it yet registers *after* the ENDING thread
    started waiting: the registration happens under the same lock as the
    OPEN→CLOSING transition, so it either lands before it (counted, waited
    on) or is refused after it (dropped — the segment belongs to a meeting
    the user already ended). No Queue.empty(), no polling.

    Two kinds of work are counted independently and additively:

    * queue-item work (``register``/``register_final``/``release``, keyed by
      a unique work id): one count per item put on the ASR queue; released
      by the ASR loop's ``finally`` when processing that item returns,
      whatever terminal path the processing took;
    * translation work (``register_msg``/``release_msg``, keyed by msg id):
      one count per translation job, registered when recognition produces a
      message; released by the ``finally`` in ``_translate_async`` or inline
      on the paths that never submit a job.

    Exactly-once holds because ``release``/``release_msg`` are idempotent on
    absent ids. Generations are monotonic, so a late release keyed to an
    older generation simply no-ops.
    """

    # Generation lifecycle states (tracker-internal; not SessionState).
    OPEN = "open"
    CLOSING = "closing"
    SUPERSEDED = "superseded"

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # generation -> {work_id: kind} for pending work
        self._pending = {}
        # generation -> set of msg_ids whose translations are in flight
        self._msgs = {}
        # generation -> lifecycle state (OPEN/CLOSING/SUPERSEDED). Entries
        # are retired once superseded and drained, so the map stays
        # proportional to live sessions, not the session history.
        self._gen_state = {}
        # Generations auto-created by register() (see there) for periods with
        # no explicitly tracked session — subtitle-only speech, or the window
        # before a legacy auto-open session is adopted. At most one is live
        # at a time (the current generation); a later begin/auto-create
        # supersedes the older ones so the map stays bounded.
        self._auto_created = set()
        self._highest_generation = -1

    # --- registration -----------------------------------------------------

    def begin(self, generation: int) -> None:
        """A new session opened: its counts start empty and it is OPEN.

        A begin for a generation whose close is in flight (CLOSING) is a
        caller bug and is refused (defensive only — the state machine is
        ENDING then and LiveTranslateApp never begins it). Any other
        generation — never seen, retired after a supersede, or the
        auto-created entry an adoption is reusing in place — may be
        (re)begun: work ids and msg ids are unique for the process
        lifetime, so a stale release from the previous lifecycle can never
        touch the fresh counts.
        """
        with self._cond:
            if self._gen_state.get(generation) == self.CLOSING:
                return
            self._begin_locked(generation)

    def _begin_locked(self, generation: int) -> None:
        # Lock-held core of begin(); also used to adopt an auto-created
        # entry in place (its pre-adoption counts must survive, so maps are
        # never reset for a generation that is already tracked).
        self._gen_state[generation] = self.OPEN
        self._pending.setdefault(generation, {})
        self._msgs.setdefault(generation, set())
        self._auto_created.discard(generation)
        self._highest_generation = max(self._highest_generation, generation)
        # An auto-created entry for an older generation is obsolete the
        # moment a real session begins (its counts, if any, drain through
        # the ordinary terminal paths and the entry retires) — the map stays
        # proportional to live sessions.
        self._supersede_auto_created_locked(keep=generation)
        self._cond.notify_all()

    def _supersede_auto_created_locked(self, keep: int) -> None:
        for gen in list(self._auto_created):
            if gen == keep:
                continue
            self._auto_created.discard(gen)
            self._gen_state[gen] = self.SUPERSEDED
            self._pending.pop(gen, None)
            self._msgs.pop(gen, None)
            self._maybe_retire_locked(gen)

    def register(self, generation: int, work_id, kind: str = "asr") -> bool:
        """Count one *ordinary* unit of work (capture-thread segment).

        Returns False when the generation is CLOSING or SUPERSEDED — the
        caller must drop the segment, not enqueue it.

        An *unknown* generation is auto-created as OPEN so the count is
        waitable: items produced while no session is tracked (subtitle-only
        speech) still carry a count, because a legacy auto-open session
        adopted moments later reuses this same generation (adoption does not
        bump it — see LiveTranslateApp._adopt_auto_opened_session) and those
        in-flight items are that meeting's opening work. Without the count,
        an end racing them would close the session first and their speech
        would be refused at write time. If no session ever adopts the
        generation the counts still drain normally and nothing waits on
        them (an ENDING wait only ever names a session's generation).
        """
        with self._cond:
            state = self._gen_state.get(generation)
            if state == self.SUPERSEDED or state == self.CLOSING:
                return False
            if state is None:
                self._pending[generation] = {}
                self._msgs[generation] = set()
                self._gen_state[generation] = self.OPEN
                self._auto_created.add(generation)
                self._highest_generation = max(
                    self._highest_generation, generation
                )
                # A newer generation being tracked means this auto-create is
                # for a stale value (defensive only — the fence keeps enqueue
                # snapshots current); retire any older auto-created entries.
                self._supersede_auto_created_locked(keep=generation)
            self._pending[generation][work_id] = kind
            return True

    def register_final(self, generation: int, work_id) -> bool:
        """Count the *final VAD flush* — the only new work allowed while
        CLOSING. Linearized with the OPEN→CLOSING flip by the same lock, so
        a ``wait_idle`` already waiting still sees the new count (it
        re-checks under the lock on every wake).

        Returns False when the generation is SUPERSEDED or unknown (the end
        was superseded by a quit before the flush ran): drop the segment.
        """
        with self._cond:
            state = self._gen_state.get(generation)
            if state not in (self.OPEN, self.CLOSING):
                return False
            self._pending[generation][work_id] = "final"
            return True

    def register_msg(self, generation: int, msg_id) -> bool:
        """Count one translation job under a session generation.

        Refused once SUPERSEDED (the result would be discarded anyway, so
        waiting for it is pointless); still accepted while CLOSING — the
        ENDING phase exists to let those translations land.
        """
        with self._cond:
            state = self._gen_state.get(generation)
            if state is None or state == self.SUPERSEDED:
                return False
            self._msgs[generation].add(msg_id)
            return True

    def admit(self, generation: int) -> str:
        """Admission decision for an ordinary (capture-thread) segment.

        * ``"register"`` — the generation is OPEN: count the item (wait on
          it) and enqueue it;
        * ``"pass"``     — no session exists under this generation (nothing
          was ever begun, or the state was cleared by an app quit that was
          itself superseded): this is the subtitle-only mode — recognition
          runs for the live overlay, no meeting is recorded. The item is
          still counted (register() auto-creates the generation): if a
          legacy auto-open session is adopted at this generation moments
          later, the in-flight items are that meeting's opening work and
          the ENDING wait must see them; with no session ever adopted the
          counts drain normally and nothing waits on them;
        * ``"drop"``     — the generation is CLOSING/SUPERSEDED: the user
          ended this meeting; recognising the audio would produce a result
          with no meeting to land in (and its write would be refused by the
          writer). Drop the segment before the queue.

        SUPERSEDED generations do not linger in _gen_state (supersede
        retires the entry once its pending counts are cleared, which the
        supersede itself does), so "drop" here means CLOSING-or-closed; the
        processing-side stale-segment guard (generation vs the current one)
        covers the retired case.
        """
        with self._cond:
            state = self._gen_state.get(generation)
            if state == self.OPEN:
                return "register"
            if state is None:
                return "pass"
            return "drop"

    # --- release ----------------------------------------------------------

    def release(self, generation: int, work_id) -> None:
        """Release one work unit. Idempotent: an absent id is a no-op, so a
        timeout-forced release followed by the real terminal path cannot
        double-count."""
        with self._cond:
            gen_map = self._pending.get(generation)
            if gen_map is not None and gen_map.pop(work_id, None) is not None:
                self._maybe_retire_locked(generation)
                self._cond.notify_all()

    def release_msg(self, generation: int, msg_id) -> None:
        with self._cond:
            gen_set = self._msgs.get(generation)
            if gen_set is not None and msg_id in gen_set:
                gen_set.discard(msg_id)
                self._maybe_retire_locked(generation)
                self._cond.notify_all()

    # --- session boundaries ------------------------------------------------

    def start_closing(self, generation: int) -> None:
        """The end was requested: ordinary registrations for this generation
        are refused from here on. Must run *before* the end thread's flush,
        so the flush's ``register_final`` is the only late arrival."""
        with self._cond:
            if self._gen_state.get(generation) == self.OPEN:
                self._gen_state[generation] = self.CLOSING
                self._cond.notify_all()

    def supersede(self, generation: int) -> None:
        """The generation's session is fully over (end completed, timed out,
        or the app quit). Later registrations/releases for it are
        refused/ignored, and the ENDING wait is woken."""
        with self._cond:
            self._gen_state[generation] = self.SUPERSEDED
            self._pending.pop(generation, None)
            self._msgs.pop(generation, None)
            self._auto_created.discard(generation)
            self._maybe_retire_locked(generation)
            self._cond.notify_all()

    def discard_all(self) -> None:
        """Clear every generation (app quit: nothing is waited on anymore)."""
        with self._cond:
            self._pending.clear()
            self._msgs.clear()
            self._gen_state.clear()
            self._auto_created.clear()
            self._cond.notify_all()

    def _maybe_retire_locked(self, generation: int) -> None:
        """Drop a SUPERSEDED generation's bookkeeping once nothing refers to
        it, so long-running processes do not accumulate one entry per
        recorded meeting. OPEN/CLOSING generations are never retired."""
        if self._gen_state.get(generation) != self.SUPERSEDED:
            return
        if self._pending.get(generation) or self._msgs.get(generation):
            return
        self._gen_state.pop(generation, None)

    # --- waiting ----------------------------------------------------------

    def wait_idle(self, generation: int, timeout: float) -> bool:
        """Block until the generation has no pending work, or the timeout
        passes. Runs on the ENDING background thread only, never the Qt
        thread. Returns True when idle, False on timeout.

        Sound only after ``start_closing(generation)``: before it, an
        ordinary registration could legally land after this returns (the
        capture thread raced the end request). The end thread always calls
        start_closing before wait_idle; a caller that does not gets the
        old, racy semantics.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._gen_state.get(generation) == self.SUPERSEDED:
                    return True
                gen_work = self._pending.get(generation)
                gen_msgs = self._msgs.get(generation)
                if not gen_work and not gen_msgs:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)

    def pending_count(self, generation: int) -> int:
        with self._cond:
            return len(self._pending.get(generation) or {}) + len(
                self._msgs.get(generation) or {}
            )


def validate_asr_result(result, kind: str):
    """Normalize one ASR backend result into the shared contract.

    Both the local worker and RemoteASREngine must satisfy this; neither may
    return a half-valid structure of its own shape.

    Returns ``None`` for a legitimately empty result (silence), or a
    ``(text, language)`` tuple. Raises ASRProtocolError when the backend broke
    the contract, so a bad result is visible instead of crashing the consumer
    on a missing key.
    """
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ASRProtocolError(
            f"{kind}: expected dict result, got {type(result).__name__}"
        )
    text = result.get("text")
    if not isinstance(text, str):
        raise ASRProtocolError(
            f"{kind}: 'text' must be a string, got {type(text).__name__}"
        )
    language = result.get("language")
    if not isinstance(language, str) or not language:
        raise ASRProtocolError(f"{kind}: 'language' must be a non-empty string")
    text = text.strip()
    if not text:
        return None
    return text, language


class _CacheDeleteThread(QThread):
    """rmtree off the Qt thread: a multi-GB model tree can block it for seconds.

    Reports every failure back instead of swallowing it, so the user learns
    which paths survived rather than only the log doing.
    """

    done = pyqtSignal(list)  # [(path, error_message), ...]

    def __init__(self, paths):
        super().__init__()
        self._paths = list(paths)

    def run(self):
        # The caller shows a modal progress dialog with no cancel button and
        # closes it on `done`, so this signal has to be emitted on every path —
        # otherwise the app is stuck behind that dialog with no way out.
        failures = []
        try:
            import shutil

            for path in self._paths:
                try:
                    shutil.rmtree(path)
                    log.info(f"Deleted: {path}")
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    log.error(f"Failed to delete {path}: {exc}")
                    failures.append((str(path), str(exc)))
        except BaseException as exc:
            log.error("Cache delete failed", exc_info=True)
            failures.append(("(cache delete)", str(exc)))
        finally:
            self.done.emit(failures)


def create_app_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(60, 130, 240))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont(default_mono_font_family(), 28, QFont.Weight.Bold))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "LT")
    p.end()
    return QIcon(pix)


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return normalize_config(
        config,
        platform_name=sys.platform,
        mps_is_available=mps_available(),
        cuda_is_available=cuda_available(),
    )


class LiveTranslateApp:
    def __init__(self, config):
        self._config = config
        self._running = False
        self._paused = False
        self._asr_ready = False  # True when ASR model is loaded

        self._audio = AudioCapture(
            device=(
                "__disabled__"
                if config["audio"].get("system_audio") == "disabled"
                else config["audio"].get("device")
            ),
            mic_device=config["audio"].get("mic_device"),
            system_audio=config["audio"].get("system_audio", "disabled"),
            sample_rate=config["audio"]["sample_rate"],
            chunk_duration=config["audio"]["chunk_duration"],
        )
        self._vad = VADProcessor(
            sample_rate=config["audio"]["sample_rate"],
            threshold=config["asr"]["vad_threshold"],
            min_speech_duration=config["asr"]["min_speech_duration"],
            max_speech_duration=config["asr"]["max_speech_duration"],
            chunk_duration=config["audio"]["chunk_duration"],
        )
        self._asr_type = None
        self._asr = None
        self._asr_signature = None
        self._asr_config = None
        self._asr_error_count = 0
        self._asr_device = normalize_device(config["asr"]["device"])
        self._whisper_model_size = config["asr"]["model_size"]
        self._funasr_model_key = normalize_funasr_model_key(
            config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
        )
        self._gigaam_model_key = normalize_gigaam_model_key(
            config["asr"].get("gigaam_model", DEFAULT_GIGAAM_MODEL)
        )
        self._asr_lock = threading.RLock()
        self._vad_lock = threading.Lock()
        # Settings changed from the Qt thread are deferred here and applied by the
        # ASR thread before its next transcribe, so the UI never blocks on the
        # worker pipe (which may be busy with an in-flight cross-process call).
        # Padding is keyed by engine_type because one settings save updates both
        # the funasr and whisper padding and they must not clobber each other.
        self._asr_pending_lock = threading.Lock()
        self._asr_pending_language = _NO_PENDING
        self._asr_pending_padding = {}
        # Auto-restart bookkeeping for a worker that dies mid-session. _asr_generation
        # is bumped on every (de)activation so a slow background (re)start can detect
        # that a newer engine switch superseded it and discard its stale worker.
        self._asr_restart_state = None
        self._asr_restart_count = 0
        self._asr_restart_max = 3
        self._asr_generation = 0
        self._asr_recycling = False
        # Proactively recycle the worker once its RSS grows this far past the
        # post-load baseline, to bound native-side (FunASR/CTranslate2) leaks that
        # accumulate in the long-lived worker process.
        self._asr_worker_baseline_mb = None
        self._asr_recycle_delta_mb = 768
        self._last_speech_activity = time.monotonic()
        self._target_language = config["translation"]["target_language"]
        self._mlx_service = MLXServiceManager()
        self._mlx_service.translate = t
        # Bootstrap translator from config.yaml, replaced by the active model as
        # soon as the panel applies its settings. Built through the same factory
        # so it cannot drift from the runtime one. thinking_style is explicit
        # rather than inherited: config.yaml's default endpoint is a local
        # server, and those reject the unknown parameters "auto" would send.
        self._translator = translator_from_model_config(
            {
                "api_base": config["translation"].get("api_base"),
                "api_key": config["translation"].get("api_key"),
                "model": config["translation"].get("model"),
                "streaming": config["translation"]["streaming"],
                "system_prompt": config["translation"].get("system_prompt"),
                "thinking_style": "off",
            },
            target_language=self._target_language,
            max_tokens=config["translation"]["max_tokens"],
            temperature=config["translation"]["temperature"],
        )
        self._translator.set_context_turns(
            config["translation"].get("context_window", 0)
        )
        self._overlay = None
        self._subwin = None
        self._panel = None
        self._capture_thread = None
        self._asr_thread = None
        self._asr_queue = queue.Queue(maxsize=16)
        # Shutdown is expressed by an event, not only by a queue sentinel: a full
        # queue used to make stop()'s blocking put(None) wait forever, and the
        # sentinel itself could be dropped by _enqueue_asr's overflow path.
        self._stop_event = threading.Event()
        self._stopped = True
        self._translation_workers = 8
        self._tl_executor = None
        self._extra_tl_executor = None
        self._translation_lock = threading.RLock()
        self._translation_history = []
        self._translation_order = deque()
        self._translation_results = {}
        self._translator_generation = 0
        self._translation_stats_lock = threading.Lock()
        self._translation_latencies = deque(maxlen=100)
        self._asr_latencies = deque(maxlen=100)
        self._latency_counts = {"asr": 0, "translation": 0}
        self._translation_pending = 0

        self._transcript = TranscriptWriter(Path(__file__).parent / "transcripts")

        # Memory diagnostic state
        import psutil
        self._mem_proc = psutil.Process(os.getpid())
        self._mem_baseline_mb = self._mem_proc.memory_info().rss / 1024 / 1024
        self._mem_last_mb = self._mem_baseline_mb
        self._mem_asr_call_count = 0
        self._mem_periodic_timer = None
        # Memory ceiling: warn once when combined RSS (main + ASR worker) exceeds
        # threshold. The ASR backend now runs in a worker process and keeps
        # native-side workspaces/caches that Python GC cannot always reclaim, so the
        # ceiling must include the worker's RSS (see _mem_snapshot).
        self._mem_threshold_mb = 4096
        self._mem_warned = False
        self._mem_warning_callback = None
        # Tray-notification sink for conditions the user must act on even when
        # no window is open (e.g. the MLX service gave up restarting).
        self._notify_callback = None

        self._asr_count = 0
        self._translate_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._input_price = 0.0
        self._output_price = 0.0
        self._msg_id = 0
        self._last_original = ""
        self._last_msg_id = 0

        # Incremental ASR state
        self._incremental_enabled = False
        self._interim_interval = 2.0
        self._interim_pending = ""
        self._interim_active = False
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""

        # --- meeting-recording session lifecycle (SessionState) ---
        # One authority; the overlay button and the records page observe it
        # through session_state_changed, never their own copy.
        self._session_state = SessionState.IDLE
        self._session_generation = 0          # token stamped on each session
        self._ending_session_id = None        # stamp being finalized
        # Per-session in-flight work: registered when an audio segment enters
        # the ASR queue (before recognition), transferred to the msg_id when
        # recognition produces a translation job, released exactly once from
        # every terminal path. ENDING waits on the count, not on
        # Queue.empty() — see _SessionWorkTracker for why.
        self._session_work = _SessionWorkTracker()
        # Work ids are unique per queue item, independent of msg_id (a
        # segment may yield no message at all, or several interim ones).
        self._session_work_seq = 0
        self._session_work_lock = threading.Lock()
        # Producer fence for the session boundary: linearizes, under one
        # lock, the capture thread's "gate check → VAD → enqueue" against
        # the end thread's "gate → CLOSING" (and the begin thread's
        # generation bump + session open). Speech the VAD accepted before
        # the end is therefore enqueued *before* CLOSING flips and is kept;
        # audio after the end never reaches the VAD. The generation and the
        # writer's session stamp are snapshotted together under this lock,
        # so a queue item's identity pair can never straddle a boundary.
        # RLock: enqueueing holds it and re-enters via the snapshot helper.
        self._session_boundary_lock = threading.RLock()
        self._ending_thread = None
        self._session_ui_callback = None      # Qt-thread notifier set by main()
        # Capture-loop gate during ENDING: True means feed VAD no new audio
        # (the session is completing; the pipeline itself is not paused).
        # Written and read only under _session_boundary_lock.
        self._session_end_gating = False

    def set_overlay(self, overlay: SubtitleOverlay):
        self._overlay = overlay
        self._publish_transcript_paths()

    def _record_session_info(self):
        """Tell the transcript what produced it, for the meeting record header."""
        model = None
        if self._panel:
            try:
                active = self._panel.get_active_model()
                model = (active or {}).get("name")
            except Exception:
                model = None
        if model is None and self._translator is not None:
            model = self._translator._model
        self._transcript.set_session_info(
            asr_engine=self._asr_config.get("display_name") if self._asr_config else None,
            translation_model=model,
            source_language=self._get_asr_language_setting(),
            target_language=self._target_language,
        )

    def _publish_transcript_paths(self):
        """Tell the overlay where the complete session log is.

        The overlay keeps only the last 50 messages, so an export has to be able
        to point at the transcript for anything older.
        """
        if self._overlay is None:
            return
        try:
            self._overlay.set_transcript_paths(self._transcript.session_paths())
        except Exception:
            log.debug("Could not publish transcript paths", exc_info=True)

    def set_subtitle_window(self, subwin: SubtitleWindow):
        self._subwin = subwin

    # --- meeting-recording session lifecycle -----------------------------

    def set_session_ui_callback(self, callback):
        """Register the Qt-thread notifier (main() wires it to signals).

        State transitions decided on background threads must not touch Qt
        widgets directly; they call this, and the callback posts onto the Qt
        event loop. Guarded so a missing sink logs instead of crashing the
        worker.
        """
        self._session_ui_callback = callback

    def _notify_session_state(self, state: str, session_id: str | None = None,
                              summary: dict | None = None):
        self._session_state = state
        if self._session_ui_callback is None:
            log.debug("Session state %s with no UI sink", state)
            return
        try:
            self._session_ui_callback(state, session_id, summary)
        except Exception:
            log.error("Session state callback failed", exc_info=True)

    def session_state(self) -> str:
        return self._session_state

    def session_generation(self) -> int:
        return self._session_generation

    def begin_recording_session(self) -> str | None:
        """Start a new meeting record (the "Start new recording" button).

        Only valid while the pipeline runs — the records button is hidden
        otherwise, and refusing here keeps a stray caller from opening files
        with no ASR behind them. Returns the new session stamp or None.
        """
        if self._session_state == SessionState.ENDING:
            log.warning("begin_recording_session ignored: a session is ending")
            return None
        # Session first, pipeline second: create the writer session *before*
        # resuming capture, so nothing can be recorded into a session that
        # failed to open (no empty active meeting on a start failure). The
        # whole open+bump+ACTIVE-flip runs under the session boundary lock
        # (the producer fence), so a capture-thread snapshot can never
        # observe the new generation with the old session or vice versa —
        # and an ASR-thread adoption (the legacy auto-open path, same lock)
        # can never interleave between the session opening and the state
        # machine claiming it: exactly one of the two claims the session,
        # one ACTIVE notification, one tracker generation.
        with self._session_boundary_lock:
            if self._session_state != SessionState.IDLE:
                log.warning(
                    "begin_recording_session ignored: session already %s",
                    self._session_state,
                )
                return None
            session_id = self._transcript.begin_session()
            if session_id is None:
                # Auto-save disabled or the writer could not open files (the
                # open is all-or-nothing). Without a session there is nothing
                # to track; stay IDLE and keep the pipeline paused — resuming
                # it here would recognise and call the translation API while
                # nothing is being recorded.
                log.info(
                    "begin_recording_session: writer has no session (disabled?)"
                )
                return None
            self._session_generation += 1
            self._session_work.begin(self._session_generation)
            self._record_session_info()
            self._publish_transcript_paths()
            if not self._running:
                # A recording session requires a live pipeline; the app-level
                # start (main()'s handler) refuses earlier, this guard covers
                # a capture death flipping _running at any moment. The state
                # has not flipped to ACTIVE yet, so there is nothing to
                # un-notify: close what we opened, under the same fence.
                log.warning(
                    "begin_recording_session: pipeline not running; "
                    "closing the session"
                )
                self._transcript.end_session()
                self._session_work.supersede(self._session_generation)
                return None
            self._notify_session_state(SessionState.ACTIVE, session_id)
        if self._paused:
            self.resume()
        log.info("Recording session started: %s", session_id)
        return session_id

    def _adopt_auto_opened_session(self, msg_id, msg_generation: int) -> int:
        """The single authoritative legacy auto-open adoption.

        Called from the ASR thread right after an entry recorded into a
        writer session the state machine does not track — the writer
        auto-opened it because an entry arrived while the pipeline records
        and no explicit session exists. Both entry paths
        (``_process_segment`` / ``_process_segment_text``) go through here;
        the adoption must exist exactly once.

        The adopted session reuses the CURRENT generation — adoption never
        bumps it. Items already enqueued by the capture thread (before the
        auto-open) carry this same generation and are this meeting's
        opening speech: a bump would make the stale-segment guard refuse
        them and their audio would be lost. Those items are counted from
        enqueue on (see ``_enqueue_asr`` and
        ``_SessionWorkTracker.register``'s auto-create), so an ENDING wait
        covers them; the entry being processed right now registers its
        translation count here when its own earlier registration was
        refused (the generation had nothing tracked yet).

        Both entry paths call this with the boundary fence already held
        (their write → registration → adoption is one linearized section),
        and the fence is re-entered here so the helper is also correct
        standalone: every decision — adopt, fast-path, migrate — happens
        under the lock, comparing generations. An unlocked early return on
        "a session is live" is what this used to do, and it was wrong: an
        explicit begin that landed between the entry's registration and
        this call bumps the generation and retires the old one's counts, so
        the msg kept releasing (and ENDING kept not waiting) under a
        generation nothing tracks anymore.

        Returns the generation the caller must use for this msg's
        translation bookkeeping from here on (release paths key to it).
        """
        with self._session_boundary_lock:
            current = self._session_generation
            if self._session_state == SessionState.IDLE:
                adopted = self._transcript.active_session()
                if not adopted:
                    # The session this entry recorded into is already gone (an
                    # end landed between the write and the fence): nothing to
                    # adopt, and the tracker refuses counts for it — the msg's
                    # release will no-op either way.
                    return msg_generation
                if msg_generation != current:
                    # Defensive exactly-once: a caller reaching adoption with
                    # a stale generation must not leave the count it
                    # registered there. release_msg no-ops when the old
                    # generation is already retired.
                    self._session_work.release_msg(msg_generation, msg_id)
                # No bump (see the docstring). begin() adopts an auto-created
                # entry in place — its pre-adoption queue-item counts survive,
                # which is exactly what an ENDING wait must see.
                self._session_work.begin(current)
                # Idempotent when the caller already registered under this
                # same generation (a set add).
                self._session_work.register_msg(current, msg_id)
                self._notify_session_state(SessionState.ACTIVE, adopted)
                return current
            if msg_generation == current:
                # A tracked session is live under the entry's OWN generation
                # (an explicit session, or an adoption that already happened):
                # the entry's write landed in that session and its msg count
                # is already keyed correctly — nothing to adopt or migrate.
                # The identity condition is the generation comparison, not
                # merely "some session is live": a live session under a
                # different generation must take the migration branch below.
                return msg_generation
            # A tracked session is live under a DIFFERENT generation: an
            # explicit begin claimed the writer session this entry recorded
            # into (begin_session returns an open session unchanged), so the
            # entry belongs to the live meeting and its count must move
            # there — ENDING for the live generation has to wait for this
            # translation. release-then-register keeps the count exactly
            # once: the release drops the old registration (a no-op on a
            # generation the begin already retired), the register is a set
            # add under the live one.
            self._session_work.release_msg(msg_generation, msg_id)
            self._session_work.register_msg(current, msg_id)
            return current

    def end_recording_session(self, expected_session: str | None = None) -> bool:
        """Finish the current meeting without stopping the app (the
        "End this recording" button). Returns True when an ENDING began.

        ``expected_session`` is the meeting the *user* asked to end, named
        at click time. It is re-verified here, immediately before the state
        flips to ENDING: every caller reaches this method through a
        confirmation dialog, and the dialog pumps a nested event loop — an
        end and a new begin can land under it, so the session active by the
        time the dialog closes may be a different meeting. The check and
        the flip are one synchronous GUI-thread sequence, so nothing can
        interleave them; a mismatch refuses (the caller shows why) instead
        of ending whichever meeting happens to be active now. ``None``
        keeps the legacy "end whatever is open" behaviour for callers that
        genuinely have no identity to assert.

        The heavy work runs on a background thread: flushing the last VAD
        segment, letting in-flight ASR/translation finish (bounded), then
        closing the session's files. The Qt thread never blocks on it.
        """
        if self._session_state not in (SessionState.ACTIVE, SessionState.PAUSED):
            log.debug("end_recording_session ignored: state=%s", self._session_state)
            return False
        if self._ending_thread is not None and self._ending_thread.is_alive():
            log.warning("end_recording_session: an end is already in progress")
            return False

        generation = self._session_generation
        # Snapshot the session id: the writer still reports the open session
        # until end_session() runs.
        session_id = self._transcript.active_session()
        if (
            expected_session is not None
            and expected_session != session_id
        ):
            log.info(
                "end_recording_session refused: the request named session "
                "%s, the open session is %s (the state changed under the "
                "confirmation dialog)", expected_session, session_id,
            )
            return False
        self._ending_session_id = session_id
        self._notify_session_state(SessionState.ENDING, session_id)

        def _work():
            summary = None
            try:
                summary = self._run_session_end(generation)
            except Exception:
                # The close itself failed (I/O error in the flush, the seal,
                # anywhere). Keep whatever of the meeting is already on disk
                # and abort the writer's session. abort_session never raises
                # and resets its state in a finally, so after this call the
                # writer has no session — verified below before IDLE is
                # announced.
                log.error(
                    "Session end failed; aborting the session, record kept "
                    "as-is and marked interrupted",
                    exc_info=True,
                )
                summary = self._transcript.abort_session() or summary
                # _run_session_end's supersede never ran; retire the
                # generation here so the tracker drops its CLOSING entry.
                self._session_work.supersede(generation)
            # IDLE may only be announced once the writer verifiably holds
            # nothing of the session: a state machine claiming "done" over
            # live handles is a lie that strands the meeting's files open.
            # has_open_resources is the strictest check — it also catches a
            # half-dead close (_opened=True, _session_open=False) that
            # has_active_session() would miss. abort_session's finally
            # guarantees this on the failure path; _run_session_end's
            # end_session/reset does on the success path.
            if self._transcript.has_open_resources():
                log.critical(
                    "Writer still holds session resources after the end "
                    "path completed; refusing to announce IDLE (staying in "
                    "ENDING — the app may need a restart)"
                )
                return
            with self._session_boundary_lock:
                if generation != self._session_generation:
                    # Superseded by a quit (the generation moved under us).
                    # The stop() path owns the final close.
                    return
                self._ending_session_id = None
                # Advance the generation so any work the close missed (a
                # timed-out translation straggler, an ASR result still on the
                # worker) is unambiguously invalid for the *next* session:
                # the tracker already superseded this generation, and the
                # bumped generation means a straggler carrying the old one
                # fails the identity check at write time.
                self._session_generation += 1
            # end_session returns None when there was no open session (the
            # close raced a stop, or a legacy auto session never opened):
            # there is no summary to announce, only the state change.
            ended = summary.get("session") if summary else session_id
            # Meeting over: the pipeline stays *paused*. Audio capture and
            # translation must not keep running on unrecorded audio — the
            # user asked to end the recording, not to keep listening. The
            # ASR model stays loaded; "Start new recording" resumes capture
            # after its session opens. summary=None signals "kept as-is"
            # (close failure or no session); the UI treats it the same as a
            # completed end — the record is on disk and viewable.
            self._notify_session_state(SessionState.IDLE, ended, summary)
            log.info(
                "Recording session ended: %s%s",
                ended,
                "" if summary else " (no summary; close failed or no session)",
            )

        self._ending_thread = threading.Thread(
            target=_work, name="session-end", daemon=True
        )
        self._ending_thread.start()
        return True

    def _run_session_end(self, generation: int) -> dict | None:
        """The ENDING workload, on its own thread. Returns the final sidecar
        contents, or None when superseded (app quit / generation moved)."""
        if generation != self._session_generation:
            return None
        # 1) Stop accepting new content for this session, atomically under
        #    the session boundary lock (the producer fence). The capture
        #    thread holds the same lock across "gate check → VAD → enqueue",
        #    so one of two orders holds:
        #    * the fence ran first: any segment the VAD produced is already
        #      enqueued and counted under the still-OPEN generation — the
        #      end waits for it (the last utterance is kept, never dropped);
        #    * the end ran first: the gate is up, the capture thread feeds
        #      the VAD nothing, and the flush below drains what the VAD had
        #      already accepted before the fence.
        #    start_closing (OPEN→CLOSING) flips inside the same lock, so no
        #    ordinary registration can slip between the two.
        with self._session_boundary_lock:
            self._session_work.start_closing(generation)
            self._session_end_gating = True
        try:
            # 2) Flush the last VAD buffer into the ASR queue through the
            #    controlled final-registration entry (the only new work
            #    allowed while CLOSING), registered before it is enqueued.
            self._flush_for_session_end(generation)

            # 3) Wait for this session's in-flight work to reach zero — and,
            #    because the generation is CLOSING, for it to *stay* zero:
            #    no ordinary registration can land after this returns.
            idle = self._session_work.wait_idle(
                generation, SessionState.ENDING_TIMEOUT_S
            )
            if not idle:
                log.warning(
                    "Session end timed out after %ss with %d work item(s) "
                    "pending; closing with untranslated entries",
                    SessionState.ENDING_TIMEOUT_S,
                    self._session_work.pending_count(generation),
                )
            if generation != self._session_generation:
                return None

            # 4) Close: entries that never got their translation are flushed
            #    as untranslated, the footer and final sidecar are written,
            #    the files close. After this the session is immutable: late
            #    results are dropped by the writer's session routing.
            summary = self._transcript.end_session()
            # 5) The generation's session is over — every work unit that
            #    returns after this (a timed-out translation straggler) finds
            #    its generation superseded and is discarded, whatever its
            #    terminal path. Releasing an already-cleared generation is a
            #    no-op, so double releases cannot corrupt anything.
            self._session_work.supersede(generation)
            return summary
        finally:
            # Order matters: pause first, *then* lift the capture gate. The
            # gate exists so the ENDING flush owns the VAD buffer; between
            # "gate lifted" and "_paused observed by the capture thread" the
            # capture loop would keep consuming audio and feeding VAD — a
            # short re-listen window on a meeting the user just ended. With
            # _paused set first, the capture loop's own pause check drops
            # every chunk before the gate is even consulted.
            self._paused = True
            with self._session_boundary_lock:
                self._session_end_gating = False

    def _flush_for_session_end(self, generation: int):
        """Hand the last VAD buffer to the ASR thread, as pause() does.

        Runs on the ENDING thread; the segment is registered through
        ``register_final`` — the controlled entry that is the *only* new
        work a CLOSING generation accepts — before it is enqueued. If the
        generation was superseded before the flush ran (a quit raced the
        end), the segment is dropped: the stop() path owns that audio.

        The interim state is handed over with the segment and reset by the
        ASR loop's vad_flush ``finally`` — the same ownership pause()
        relies on. Resetting it here used to break both halves of the
        final processing: the loop's dispatch between
        ``_process_interim_final`` and ``_process_segment`` reads
        ``_interim_active``, and ``_process_interim_final`` itself needs
        ``_interim_pending`` (the buffered fragments whose audio was
        already trimmed away — this final write is their only remaining
        copy) and ``_interim_committed_tail`` (echo dedup against the
        committed prefix). Only the paths that leave the audio unqueued
        reset here — ASR not ready, nothing remaining, a superseded
        generation, a stop already begun, or the queue rejecting the item
        even after dropping one — because then no consumer will ever run
        the handler's reset, and a dirty interim state left behind would
        leak into the next session.
        """
        if not self._asr_ready:
            with self._vad_lock:
                self._vad._reset()
            self._reset_interim_state()
            return
        with self._vad_lock:
            remaining = (
                self._vad.force_flush() if self._interim_active else self._vad.flush()
            )
        if remaining is None:
            self._reset_interim_state()
            return
        if not self._enqueue_final_segment(generation, remaining):
            # Nothing reached the queue: the interim state is this
            # thread's to clear now (see the docstring).
            self._reset_interim_state()

    def _enqueue_final_segment(self, generation: int, segment) -> bool:
        """Enqueue the ENDING flush's segment with a final-registration.

        Split from _enqueue_asr so the two producers can never be confused:
        ordinary capture segments go through the tracker's ``admit``
        (register / pass / drop), the one final flush goes through
        ``register_final`` (allowed while CLOSING, refused once SUPERSEDED).
        The item carries the writer's current session stamp for the
        write-time identity check (run before end_session, so the session
        is still open here). Overflow handling mirrors _enqueue_asr: a
        dropped victim releases its count, a failed put releases the item's
        own count.

        Returns True only when the item actually reached the ASR queue, so
        the ASR loop's vad_flush handler — and with it the interim-state
        reset the handler owns — is guaranteed to run. False on every
        refusal (a stop already begun, the generation superseded, or the
        queue still full after dropping one victim); the caller then owns
        the interim state and must reset it itself.
        """
        if self._stop_event.is_set():
            return False
        # Snapshot under the boundary fence like the ordinary producer, so
        # the item's identity pair is consistent (ENDing state blocks a
        # concurrent begin, but the fence keeps the invariant uniform).
        with self._session_boundary_lock:
            expected_session = self._transcript.active_session()
            work_id = self._next_session_work_id()
            if not self._session_work.register_final(generation, work_id):
                log.debug("Dropping ENDING flush for superseded generation")
                return False
            item = (
                "vad_flush", segment, work_id, generation,
                expected_session,
            )
            try:
                self._asr_queue.put_nowait(item)
                return True
            except queue.Full:
                pass
        try:
            dropped = self._asr_queue.get_nowait()
        except queue.Empty:
            dropped = None
        if dropped is None:
            self._requeue_stop_sentinel()
        else:
            log.warning("ASR queue full, dropped %s segment", dropped[0])
            self._release_queued_work(dropped)
        try:
            self._asr_queue.put_nowait(item)
        except queue.Full:
            log.warning("ASR queue still full, dropping the ENDING flush")
            self._release_queued_work(item)
            return False
        return True

    def _next_session_work_id(self):
        """A unique work id for one queued ASR item. Independent of msg_id:
        a segment can yield no message (noise, language filter) or several
        (interim sentences), so the queue unit needs its own identity."""
        with self._session_work_lock:
            self._session_work_seq += 1
            return self._session_work_seq

    def _session_snapshot(self) -> tuple:
        """(generation, session stamp) read together under the boundary lock.

        Must be called with ``_session_boundary_lock`` held (RLock — the
        fence makes a begin/end and this snapshot linear), so the pair can
        never straddle a session boundary — the generation and the stamp
        describe the same meeting by construction. Every writer of
        _session_generation holds the boundary lock, so a plain read here is
        safe.
        """
        return self._session_generation, self._transcript.active_session()

    def _on_session_state_ui(self, state: str, session_id: str | None,
                             summary: dict | None):
        """Qt-thread reaction to a session-state change (wired via main()).

        Meeting over → IDLE: the pipeline stays paused (the ENDING thread
        already set _paused), and the overlay's run/pause button must say
        so — an IDLE session next to a "Running" overlay would invite a
        resume that recognises audio nobody is recording.
        """
        if self._overlay:
            self._overlay.set_session_state(state)
            if state == SessionState.IDLE and self._running and self._paused:
                self._overlay.set_running(False)
        if self._panel:
            self._panel.on_session_state_changed(state, session_id)

    def set_panel(self, panel: ControlPanel):
        self._panel = panel
        # One manager, not two. The panel starts the service and the app stops
        # it, so separate instances meant the stopping one had never spawned the
        # process: it could not reap its own child, held a stale version cache,
        # and knew the pid only through the pid file.
        self._mlx_service = panel._mlx_manager
        self._mlx_service.translate = t
        panel.settings_changed.connect(self._on_settings_changed)
        panel.model_changed.connect(self._on_model_changed)
        panel.mlx_service_state_changed.connect(self._on_mlx_service_state_changed)
        panel.mlx_health_checked.connect(self._on_mlx_probe_result)
        panel.models_list_changed.connect(self._on_models_list_changed)
        self._mlx_restart_pending = False
        # Edge tracking: the 5s probe used to call _disable_translator() on every
        # tick while the service was down, so each tick bumped the translator
        # generation and cleared the context history.
        self._mlx_available = None
        self._mlx_restart_count = 0
        self._mlx_restart_max = 3
        self._mlx_next_restart_at = 0.0
        self._mlx_monitor_timer = QTimer()
        self._mlx_monitor_timer.setInterval(5000)
        self._mlx_monitor_timer.timeout.connect(self._monitor_mlx_service)
        self._mlx_monitor_timer.start()

    def _monitor_mlx_service(self):
        """Keep an active managed model usable if its local server exits."""
        if not self._panel:
            return
        active = self._panel.get_active_model()
        if not is_hy_mt_model(active):
            self._mlx_restart_pending = False
            return
        self._panel.request_mlx_health_check()

    def _on_mlx_probe_result(self, running: bool):
        if not self._panel or not is_hy_mt_model(self._panel.get_active_model()):
            return
        if running:
            self._mlx_restart_pending = False
            self._mlx_restart_count = 0
            self._mlx_next_restart_at = 0.0
            if self._mlx_available is not True:
                self._mlx_available = True
                log.info("HY-MT local service is available")
            if self._translator is None:
                self._on_model_changed(self._panel.get_active_model())
            return

        # Only on the available -> unavailable edge. Repeating it every 5s
        # invalidated in-flight translations and wiped the context history over
        # and over while the service was simply down.
        if self._mlx_available is not False:
            self._mlx_available = False
            log.warning("HY-MT local service is unavailable; translation disabled")
            self._disable_translator()

        if self._mlx_restart_pending:
            return
        if self._mlx_restart_count >= self._mlx_restart_max:
            return  # gave up; the user must start it from Settings
        if time.monotonic() < self._mlx_next_restart_at:
            return
        self._mlx_restart_pending = self._panel.auto_start_mlx_service()
        if self._mlx_restart_pending:
            self._mlx_restart_count += 1
            # 10s, 20s, 40s: same shape as the ASR worker restart budget.
            self._mlx_next_restart_at = time.monotonic() + 10.0 * (
                2 ** (self._mlx_restart_count - 1)
            )
            log.info(
                "HY-MT auto-restart attempt %s/%s",
                self._mlx_restart_count,
                self._mlx_restart_max,
            )
            if self._mlx_restart_count >= self._mlx_restart_max:
                self._notify_user(t("mlx_restart_gave_up"))

    def _on_mlx_service_state_changed(self, running: bool):
        active = self._panel.get_active_model() if self._panel else None
        if not is_hy_mt_model(active):
            return
        if running:
            self._mlx_available = True
            self._mlx_restart_count = 0
            # Only when the translator actually needs building. The panel emits
            # model_changed and mlx_service_state_changed for the same start,
            # and both landing here built two Translators, bumped the generation
            # twice and cleared the context history twice.
            if self._translator is None:
                self._on_model_changed(active)
        elif self._mlx_available is not False:
            self._mlx_available = False
            self._disable_translator()

    def _on_models_list_changed(self, models: list, active_idx: int):
        if self._overlay:
            self._overlay.set_models(models, active_idx)

    def _on_settings_changed(self, settings):
        with self._vad_lock:
            self._vad.update_settings(settings)
        if "style" in settings and self._overlay:
            self._overlay.apply_style(settings["style"])
        if "asr_language" in settings:
            self._set_asr_language(settings["asr_language"])
            if self._overlay:
                self._overlay.set_source_language(settings["asr_language"])
        if "sensevoice_pad_seconds" in settings:
            self._set_asr_padding("funasr", settings["sensevoice_pad_seconds"])
        if "whisper_pad_seconds" in settings:
            self._set_asr_padding("whisper", settings["whisper_pad_seconds"])
        if any(
            key in settings
            for key in (
                "asr_engine",
                "asr_device",
                "whisper_model_size",
                "funasr_model",
                "gigaam_model",
                "hub",
            )
        ):
            self._switch_asr_engine(
                settings.get(
                    "asr_engine",
                    self._asr_type or self._config["asr"].get("asr_engine", "funasr"),
                )
            )
        # `settings` is the full settings dict on every auto-save, so a key being
        # present says nothing about the user having changed it. Compare values.
        if "audio_device" in settings:
            old_device = self._audio._device_name
            new_device = settings["audio_device"]
            if old_device != new_device:
                switched = self._audio.set_device(new_device)
                if switched is False:
                    log.warning(
                        "Audio device switch to %r failed (%s); keeping the previous device",
                        new_device,
                        self._audio.metrics().get("last_error") or "unknown error",
                    )
                    # A denied Screen Recording grant looks exactly like
                    # permanent silence otherwise: the pipeline keeps
                    # "Running" on the restored mic-only backend with no
                    # user-visible hint. Surface the typed cause.
                    switch_error = getattr(self._audio, "_switch_error", None)
                    if self._panel and isinstance(switch_error, BaseException):
                        QMessageBox.warning(
                            self._panel,
                            t("error_title"),
                            t("error_audio_switch").format(
                                error=_audio_start_error(switch_error)
                            ),
                        )
                    # set_device restores the old stream when it can. If that
                    # recovery also failed the backend is stopped, so stop the
                    # pipeline instead of reporting a running state with no audio.
                    if self._running and not getattr(self._audio, "_running", True):
                        self.stop()
                else:
                    # Discard whatever was buffered for the previous device.
                    with self._vad_lock:
                        self._vad._reset()
                    if self._overlay:
                        self._overlay.update_monitor(0.0, 0.0)
        if "mic_device" in settings:
            mic_changed = self._audio.set_mic_device(settings["mic_device"])
            if mic_changed is False:
                log.warning("Microphone switch failed; keeping the previous device")
                # PyAudioCapture attempts to restore the old stream.  If that
                # recovery also failed, stop the pipeline instead of leaving
                # its capture thread waiting on a dead backend.
                if self._running and not getattr(self._audio, "_running", True):
                    self.stop()
        if "incremental_asr" in settings:
            self._incremental_enabled = settings["incremental_asr"]
        if "interim_interval" in settings:
            self._interim_interval = settings["interim_interval"]
        if "target_language" in settings:
            self._target_language = settings["target_language"]
            if self._overlay:
                self._overlay.set_target_language(self._target_language)
        if "timeout" in settings and self._translator:
            self._translator.set_timeout(settings["timeout"])
        if "translation_workers" in settings:
            self._set_translation_workers(settings["translation_workers"])
        if "auto_save_transcript" in settings:
            self._transcript.set_enabled(settings["auto_save_transcript"])
            self._publish_transcript_paths()

    def _mark_asr_unavailable(self, reason: str, client=None):
        with self._asr_lock:
            current = client or self._asr
            if client is not None and self._asr is not client:
                return
            self._asr_ready = False
            self._asr = None
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
            self._asr_restart_state = None
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
        if current is not None:
            try:
                current.shutdown()
            except Exception:
                try:
                    current.terminate()
                except Exception:
                    pass
        log.warning(f"ASR worker unavailable: {reason}")
        if self._overlay:
            self._overlay.update_asr_device("ASR unavailable")

    def _shutdown_asr_worker(self):
        with self._asr_lock:
            client = self._asr
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
            self._asr_restart_state = None
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
        if client is not None:
            log.info(f"Shutting down ASR worker: pid={client.pid}")
            client.shutdown()

    def _set_asr_language(self, language: str):
        if self._asr_type == "gigaam" and gigaam_is_russian_only(
            self._gigaam_model_key
        ):
            language = "ru"
        with self._asr_pending_lock:
            self._asr_pending_language = language

    def _get_asr_language_setting(self) -> str:
        """Return the active language hint, including fixed-language engines."""
        configured = (
            self._panel.get_settings().get("asr_language", "auto")
            if self._panel
            else "auto"
        )
        if self._asr_type == "gigaam" and gigaam_is_russian_only(
            self._gigaam_model_key
        ):
            return "ru"
        return configured

    def _set_asr_padding(self, engine_type: str, pad_seconds):
        with self._asr_pending_lock:
            self._asr_pending_padding[engine_type] = pad_seconds

    def _apply_pending_asr_settings(self, client, asr_type, funasr_key):
        """Apply deferred language/padding on the ASR thread, just before a transcribe.
        A pending value is cleared only once delivered; worker-death exceptions
        propagate with the pending intact so the restarted worker re-applies it. The
        applied value is written back into the restart config so an auto-restart or
        recycle does not revert a runtime override to the engine-switch-time value."""
        with self._asr_pending_lock:
            language = self._asr_pending_language
            pad_seconds = self._asr_pending_padding.get(asr_type, _NO_PENDING)
        if language is not _NO_PENDING:
            try:
                client.set_language(language)
            except ASRWorkerError as exc:
                log.warning(f"ASR language update failed: {exc}")
            self._update_restart_config(language=language)
            self._clear_pending_language(language)
        if pad_seconds is not _NO_PENDING:
            if not (asr_type == "funasr" and not funasr_supports_padding(funasr_key)):
                try:
                    client.set_input_padding(pad_seconds)
                except ASRWorkerError as exc:
                    log.warning(f"ASR padding update failed: {exc}")
                self._update_restart_config(pad_seconds=pad_seconds)
            self._clear_pending_padding(asr_type, pad_seconds)

    def _clear_pending_language(self, language):
        with self._asr_pending_lock:
            if self._asr_pending_language is language:
                self._asr_pending_language = _NO_PENDING

    def _clear_pending_padding(self, asr_type, pad_seconds):
        with self._asr_pending_lock:
            if self._asr_pending_padding.get(asr_type) == pad_seconds:
                del self._asr_pending_padding[asr_type]

    def _update_restart_config(self, **kwargs):
        with self._asr_lock:
            if self._asr_restart_state and self._asr_restart_state.get("config"):
                self._asr_restart_state["config"].update(kwargs)

    def _load_engine_client(self, config: dict):
        """Build the ASR backend for a worker config. Local engines run in an isolated
        worker subprocess (ASRClient); remote-whisper is a thin in-process HTTP client
        that needs no subprocess isolation (no native deps, no GPU model to load)."""
        if config.get("engine_type") == "remote-whisper":
            from asr_remote import RemoteASREngine

            url = normalize_remote_asr_url(config.get("remote_asr_url") or DEFAULT_REMOTE_ASR_URL)
            engine = RemoteASREngine(server_url=url)
            language = config.get("language")
            if language:
                engine.set_language(language)
            return engine
        return self._load_asr_client(config)

    def _load_asr_client(self, worker_config: dict) -> ASRClient:
        # request_timeout bounds how long a hung worker can stall the realtime path
        # before it is killed and auto-restarted. VAD caps segments at a few seconds,
        # so 60s is generous for a healthy transcribe yet far below the old 120s.
        client = ASRClient(worker_config, request_timeout=60.0)
        try:
            client.start()
            client.wait_ready()
            return client
        except Exception:
            client.shutdown()
            raise

    def _on_target_language_changed(self, lang: str):
        self._target_language = lang
        log.info(f"Target language: {lang}")
        if self._translator:
            self._translator.set_target_language(lang)
        if self._panel:
            # Through the panel, not a snapshot of it: writing to the dict
            # get_settings() returns left the panel holding the old value, and
            # its next auto-save wrote that back over this one.
            self._panel.update_setting("target_language", lang)

    def _on_model_changed(self, model_config: dict):
        log.info(
            f"Switching translator: {model_config['name']} ({model_config['model']})"
        )
        if is_hy_mt_model(model_config):
            if not self._mlx_service.is_running():
                message = t("error_mlx_not_running")
                log.warning("HY-MT MLX service is not running; refusing automatic start")
                self._disable_translator()
                if self._panel:
                    QMessageBox.warning(self._panel, t("error_title"), message)
                return

        prompt = model_config.get("system_prompt")
        if not prompt and self._panel:
            prompt = self._panel.get_settings().get("system_prompt")
        if not prompt:
            prompt = self._config["translation"].get("system_prompt")
        timeout = 10
        if self._panel:
            timeout = self._panel.get_settings().get("timeout", 10)
        # Keep the local HY-MT path predictable for classroom real-time use:
        # streaming on, no reasoning, and a short completion budget.
        if is_hy_mt_model(model_config):
            model_config = dict(model_config)
            model_config["streaming"] = True
            model_config["no_system_role"] = True
            model_config["thinking_style"] = "off"
            overrides = dict(model_config.get("overrides") or {})
            overrides.update({"max_tokens": 128})
            model_config["overrides"] = overrides

        # Shared with the benchmark, so both issue the same request for the
        # same model. Two copies of this argument list is how they came to
        # resolve different thinking styles.
        new_translator = translator_from_model_config(
            model_config,
            target_language=self._target_language,
            system_prompt=prompt,
            timeout=timeout,
            max_tokens=self._config["translation"]["max_tokens"],
            temperature=self._config["translation"]["temperature"],
        )
        context_turns = model_config.get(
            "context_turns", self._config["translation"].get("context_window", 0)
        )
        new_translator.set_context_turns(context_turns)
        with self._translation_lock:
            self._translator = new_translator
            self._translator_generation += 1
            self._translation_history.clear()
            self._translation_order.clear()
            self._translation_results.clear()
            self._translation_pending = 0
        if self._panel:
            # Through the setter, so the max(4, min(16, ...)) clamp applies here
            # too. Assigning the field directly let a hand-edited
            # user_settings.json spin up an unbounded pool.
            self._set_translation_workers(
                self._panel.get_settings().get(
                    "translation_workers", self._translation_workers
                )
            )
        self._input_price = model_config.get("input_price", 0)
        self._output_price = model_config.get("output_price", 0)
        if self._running:
            self._record_session_info()

    def _disable_translator(self):
        """Disable translation until the selected local service is available."""
        with self._translation_lock:
            self._translator = None
            self._translator_generation += 1
            self._translation_history.clear()
            self._translation_order.clear()
            self._translation_results.clear()
            self._translation_pending = 0

    def _set_translation_workers(self, workers):
        workers = max(4, min(16, int(workers or 8)))
        if workers == getattr(self, "_translation_workers", None):
            return
        self._translation_workers = workers
        if not self._running:
            return
        new_executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="translate"
        )
        old_executor = getattr(self, "_tl_executor", None)
        self._tl_executor = new_executor
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=False)
        log.info("Translation pool configured: workers=%d", workers)

    def _snapshot_translation_request(self, msg_id, text, target_language=None):
        with self._translation_lock:
            base = self._translator
            if base is None:
                raise TranslationUnavailable("No translation service is running")
            generation = self._translator_generation
            request = base.fork_for_request(
                target_language=target_language or self._target_language,
                history_snapshot=list(self._translation_history),
            )
            if msg_id not in self._translation_order:
                insert_at = len(self._translation_order)
                for index, queued_id in enumerate(self._translation_order):
                    if msg_id < queued_id:
                        insert_at = index
                        break
                self._translation_order.insert(insert_at, msg_id)
            self._translation_pending += 1
            return request, generation

    def _commit_translation_result(self, msg_id, text, translated, generation):
        with self._translation_lock:
            if generation != self._translator_generation:
                # A model switch already zeroed _translation_pending and cleared
                # the order/results for this generation. Decrementing the counter
                # or removing an id here corrupts the *new* generation's
                # bookkeeping — msg_ids are monotonic, so anything still keyed
                # under this id belongs to the newer run.
                log.debug(
                    "Discarding translation result from generation %s (now %s)",
                    generation, self._translator_generation,
                )
                return False
            self._translation_results[msg_id] = (text, translated)
            while self._translation_order:
                head = self._translation_order[0]
                item = self._translation_results.get(head)
                if item is None:
                    break
                self._translation_order.popleft()
                self._translation_results.pop(head, None)
                src, result = item
                if result and self._translator._context_turns > 0:
                    self._translation_history.append((src, result))
                    keep = self._translator._context_turns + 2
                    if len(self._translation_history) > keep:
                        self._translation_history = self._translation_history[-self._translator._context_turns:]
                self._translation_pending = max(0, self._translation_pending - 1)
            return True

    def _finalize_untranslated(self, msg_id, reason: str, user_visible: bool):
        """Close out a message that will never receive a translation.

        Every path that called overlay.add_message()/transcript.write_original()
        must reach here or the overlay entry stays stuck on "translating" and the
        TranscriptWriter._pending entry leaks for the rest of the session.
        """
        log.warning("Message %s left untranslated: %s", msg_id, reason)
        self._transcript.finalize_no_translation(msg_id)
        if self._overlay and user_visible:
            self._overlay.update_translation(
                msg_id, f"[{t('error_translation_unavailable')}]", 0
            )

    def _submit_translation(self, msg_id, text, source_lang, extra_langs=None,
                            session_generation=None, expected_session=None):
        request_translator, generation = self._snapshot_translation_request(
            msg_id, text
        )
        try:
            return self._tl_executor.submit(
                self._translate_async,
                msg_id,
                text,
                source_lang,
                extra_langs,
                request_translator,
                generation,
                session_generation,
                expected_session,
            )
        except RuntimeError:
            self._commit_translation_result(msg_id, text, None, generation)
            raise

    def _switch_asr_engine(self, engine_type: str):
        settings = self._panel.get_settings() if self._panel else {}
        engine_type, funasr_model = normalize_asr_engine_selection(
            engine_type, settings.get("funasr_model", self._funasr_model_key)
        )
        gigaam_model = normalize_gigaam_model_key(
            settings.get("gigaam_model", self._gigaam_model_key)
        )
        device = settings.get("asr_device", self._asr_device)
        hub = "ms"
        download_proxy = "system"
        if self._panel:
            hub = settings.get("hub", "ms")
            download_proxy = settings.get("download_proxy", "system")

        model_size = self._config["asr"]["model_size"]
        if self._panel:
            model_size = settings.get("whisper_model_size", model_size)
        model_path = None
        cache_model_key = model_size
        if engine_type == "whisper":
            model_path = resolve_custom_whisper_model(model_size)
            if model_path:
                cache_model_key = model_path
        elif engine_type == "funasr":
            cache_model_key = funasr_model
        elif engine_type == "gigaam":
            cache_model_key = gigaam_model

        remote_asr_url = settings.get(
            "remote_asr_url",
            self._config["asr"].get("remote_asr_url", DEFAULT_REMOTE_ASR_URL),
        )

        compute = self._config["asr"]["compute_type"]
        if engine_type == "whisper":
            signature_model = cache_model_key
        elif engine_type == "funasr":
            signature_model = funasr_model
        elif engine_type == "remote-whisper":
            # URL is part of the identity so editing it triggers a reconnect.
            signature_model = remote_asr_url
        elif engine_type == "gigaam":
            signature_model = (
                f"{gigaam_model_id(gigaam_model)}@{gigaam_revision(gigaam_model)}"
            )
        else:
            signature_model = engine_type
        signature = (engine_type, signature_model, device, hub, compute)

        with self._asr_lock:
            current_asr = self._asr
            current_ready = (
                self._asr_ready
                and current_asr is not None
                and current_asr.status == "ready"
            )
            if current_ready and self._asr_signature == signature:
                return
            if not current_ready:
                self._asr_ready = False

        log.info(f"Switching ASR worker: {self._asr_type} -> {engine_type}")
        # Reset interim state for the engine boundary. The active worker is
        # stopped before the target worker starts loading.
        self._reset_interim_state()
        # Engine boundary: drop the in-flight buffer rather than emitting it
        # through a worker that is about to be torn down.
        with self._vad_lock:
            self._vad._reset()

        cached = is_asr_cached(engine_type, cache_model_key, hub)
        display_name = ASR_DISPLAY_NAMES.get(engine_type, engine_type)
        if engine_type == "whisper":
            display_model = (
                local_faster_whisper_display_name(model_size)
                if model_path
                else model_size
            ) or Path(model_size).name
            display_name = f"Whisper {display_model}"
        elif engine_type == "funasr":
            display_name = funasr_display_name(funasr_model)
        elif engine_type == "gigaam":
            display_name = gigaam_display_name(gigaam_model)

        parent = (
            self._panel if self._panel and self._panel.isVisible() else self._overlay
        )

        worker_config = {
            "engine_type": engine_type,
            "funasr_model": funasr_model,
            "gigaam_model": gigaam_model,
            "model_size": cache_model_key,
            "device": device,
            "compute_type": compute,
            "hub": hub,
            "language": (
                "ru"
                if engine_type == "gigaam" and gigaam_is_russian_only(gigaam_model)
                else settings.get(
                    "asr_language", self._config["asr"].get("language", "auto")
                )
            ),
            "pad_seconds": (
                settings.get(
                    "sensevoice_pad_seconds",
                    self._config["asr"].get("sensevoice_pad_seconds", 0.5),
                )
                if engine_type == "funasr"
                else settings.get(
                    "whisper_pad_seconds",
                    self._config["asr"].get("whisper_pad_seconds", 0.5),
                )
                if engine_type == "whisper"
                else None
            ),
            "download_root": str((MODELS_DIR / "huggingface" / "hub").resolve()),
            "display_name": display_name,
            "remote_asr_url": remote_asr_url,
        }
        target_state = {
            "type": engine_type,
            "signature": signature,
            "device": device,
            "funasr_model_key": funasr_model
            if engine_type == "funasr"
            else self._funasr_model_key,
            "gigaam_model_key": gigaam_model
            if engine_type == "gigaam"
            else self._gigaam_model_key,
            "whisper_model_size": model_size
            if engine_type == "whisper"
            else self._whisper_model_size,
            "config": worker_config,
            "display_name": display_name,
            "device_label": (
                remote_asr_url if engine_type == "remote-whisper" else device
            ),
        }

        if not cached:
            missing = get_missing_models(engine_type, cache_model_key, hub)
            missing = [m for m in missing if m["type"] != "silero-vad"]
            if missing:
                dlg = ModelDownloadDialog(
                    missing, hub=hub, proxy=download_proxy, parent=parent
                )
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    log.info(f"Download cancelled/failed: {engine_type}")
                    with self._asr_lock:
                        self._asr_ready = (
                            self._asr is not None and self._asr.status == "ready"
                        )
                    return

        with self._asr_lock:
            old_asr = self._asr
            old_config = dict(self._asr_config) if self._asr_config else None
            old_state = {
                "type": self._asr_type,
                "signature": self._asr_signature,
                "device": self._asr_device,
                "funasr_model_key": self._funasr_model_key,
                "gigaam_model_key": self._gigaam_model_key,
                "whisper_model_size": self._whisper_model_size,
                "config": old_config,
                "display_name": (old_config or {}).get("display_name"),
                "device_label": (
                    (old_config or {}).get("remote_asr_url")
                    if self._asr_type == "remote-whisper"
                    else self._asr_device
                ),
            }
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
            self._asr_restart_state = None
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1

        dlg = _ModelLoadDialog(
            t("loading_model").format(name=display_name), parent=parent
        )

        new_asr = [None]
        restored_asr = [None]
        load_error = [None]
        restore_error = [None]

        def _load():
            if old_asr is not None:
                log.info(f"Stopping old ASR worker before switch: pid={old_asr.pid}")
                old_asr.shutdown()
                self._release_memory_caches()
            try:
                new_asr[0] = self._load_engine_client(worker_config)
            except Exception as e:
                load_error[0] = str(e)
                # A remote server that is simply down is an expected, user-actionable
                # condition, not a bug, so skip the noisy traceback for it.
                expected = isinstance(e, ConnectionError)
                log.error(
                    f"Failed to load ASR worker: {e}", exc_info=not expected
                )
                if old_config:
                    try:
                        log.info("Restoring previous ASR worker after switch failure")
                        restored_asr[0] = self._load_engine_client(old_config)
                    except Exception as restore_exc:
                        restore_error[0] = str(restore_exc)
                        log.error(
                            f"Failed to restore previous ASR worker: {restore_exc}",
                            exc_info=True,
                        )

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

        poll_timer = QTimer()

        def _check():
            if not thread.is_alive():
                poll_timer.stop()
                dlg.accept()

        poll_timer.setInterval(100)
        poll_timer.timeout.connect(_check)
        poll_timer.start()

        dlg.exec()
        poll_timer.stop()

        def _activate_asr(client, state):
            with self._asr_lock:
                self._asr = client
                self._asr_type = state["type"]
                self._asr_signature = state["signature"]
                self._asr_device = state["device"]
                self._asr_config = dict(state["config"]) if state["config"] else None
                self._funasr_model_key = state["funasr_model_key"]
                self._gigaam_model_key = state["gigaam_model_key"]
                self._whisper_model_size = state["whisper_model_size"]
                self._asr_ready = True
                self._asr_error_count = 0
                self._asr_restart_state = dict(state)
                self._asr_restart_count = 0
                self._asr_worker_baseline_mb = None
                self._asr_generation += 1
            if self._running:
                self._record_session_info()

        if new_asr[0] is not None:
            _activate_asr(new_asr[0], target_state)
            if self._overlay:
                self._overlay.update_asr_device(
                    f"{display_name} [{target_state['device_label']}]"
                )
            log.info(f"ASR worker ready: {engine_type} on {device}")
            return

        if restored_asr[0] is not None:
            _activate_asr(restored_asr[0], old_state)
            restored_name = old_state.get("display_name") or old_state.get("type")
            if self._overlay:
                self._overlay.update_asr_device(
                    f"{restored_name} [{old_state.get('device_label', old_state['device'])}]"
                )
            QMessageBox.warning(
                parent,
                t("error_title"),
                t("error_load_asr").format(
                    error=(
                        f"{load_error[0] or 'unknown error'}\n"
                        f"{t('asr_restore_succeeded')}"
                    )
                ),
            )
            log.info(
                f"Previous ASR worker restored: "
                f"{old_state.get('type')} on {old_state.get('device')}"
            )
            return

        error = load_error[0] or "unknown error"
        if restore_error[0]:
            error = (
                f"{error}\n"
                f"{t('asr_restore_failed').format(error=restore_error[0])}"
            )
        QMessageBox.warning(
            parent,
            t("error_title"),
            t("error_load_asr").format(error=error),
        )

        if self._overlay:
            self._overlay.update_asr_device("ASR unavailable")

    def _mem_snapshot(self) -> dict:
        rss_mb = self._mem_proc.memory_info().rss / 1024 / 1024
        # The ASR model (and its native-side leak) lives in the worker process now,
        # so sample its RSS too; the main process holds only VAD + Qt.
        worker_rss_mb = 0.0
        client = self._asr
        if client is not None and client.pid is not None:
            try:
                import psutil

                worker_rss_mb = (
                    psutil.Process(client.pid).memory_info().rss / 1024 / 1024
                )
            except Exception:
                worker_rss_mb = 0.0
        gpu_alloc_mb = 0.0
        gpu_reserved_mb = 0.0
        memory = accelerator_memory(self._asr_device)
        if memory:
            gpu_alloc_mb, gpu_reserved_mb, _ = memory
        msgs = len(self._overlay._messages) if self._overlay else 0
        vad_buf = len(self._vad._speech_buffer)
        return {
            "rss": rss_mb,
            "worker_rss": worker_rss_mb,
            "total_rss": rss_mb + worker_rss_mb,
            "gpu_alloc": gpu_alloc_mb,
            "gpu_reserved": gpu_reserved_mb,
            "msgs": msgs,
            "vad_buf": vad_buf,
        }

    def _log_mem_after_asr(self, kind: str, audio_seconds: float, asr_ms: float):
        self._mem_asr_call_count += 1
        snap = self._mem_snapshot()
        delta = snap["rss"] - self._mem_last_mb
        total_delta = snap["rss"] - self._mem_baseline_mb
        self._mem_last_mb = snap["rss"]
        log.info(
            f"MEM[asr#{self._mem_asr_call_count}:{kind}] RSS={snap['rss']:.1f}MB "
            f"(Δ{delta:+.2f} since last, {total_delta:+.1f} since start) "
            f"worker_rss={snap['worker_rss']:.0f}MB "
            f"GPU(main alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"audio={audio_seconds:.1f}s asr={asr_ms:.0f}ms "
            f"outputs={self._asr_count} msgs={snap['msgs']} vad_buf={snap['vad_buf']}"
        )
        self._check_memory_threshold(snap["total_rss"])

    def _release_memory_caches(self):
        gc.collect()
        empty_cache(self._asr_device)

    def _run_asr(self, audio: np.ndarray, kind: str, **kwargs):
        audio_seconds = len(audio) / 16000
        asr_start = time.perf_counter()
        # Snapshot the active client under the lock, then release it: the blocking
        # cross-process transcribe must not hold _asr_lock, or a slow/hung worker
        # would freeze the Qt thread on every settings change. ASRClient serializes
        # its own pipe access, and only this (single) ASR thread calls transcribe.
        with self._asr_lock:
            if not self._asr_ready or self._asr is None:
                return None, 0.0
            client = self._asr
            asr_type = self._asr_type
            funasr_key = self._funasr_model_key
        try:
            self._apply_pending_asr_settings(client, asr_type, funasr_key)
            result = client.transcribe(audio, **kwargs)
        except (ASRWorkerExited, ASRWorkerTimeout, RemoteASRError) as exc:
            # RemoteASRError joins the worker-death path on purpose: from the
            # pipeline's point of view an unreachable ASR server and a dead
            # local worker need the same bounded recovery and the same visible
            # "ASR unavailable" state.
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._log_mem_after_asr(f"{kind}:error", audio_seconds, asr_ms)
            self._recover_asr_worker(client, str(exc))
            raise
        except ASRWorkerError as exc:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._log_mem_after_asr(f"{kind}:error", audio_seconds, asr_ms)
            fatal = False
            with self._asr_lock:
                if self._asr is client:
                    self._asr_error_count += 1
                    fatal = not exc.recoverable or self._asr_error_count >= 3
            if fatal:
                self._mark_asr_unavailable(str(exc), client)
            raise
        except Exception:
            asr_ms = (time.perf_counter() - asr_start) * 1000
            self._log_mem_after_asr(f"{kind}:error", audio_seconds, asr_ms)
            raise
        with self._asr_lock:
            if self._asr is client:
                self._asr_error_count = 0
                self._asr_restart_count = 0
        asr_ms = (time.perf_counter() - asr_start) * 1000
        if self._asr_worker_baseline_mb is None and client.pid is not None:
            try:
                import psutil

                self._asr_worker_baseline_mb = (
                    psutil.Process(client.pid).memory_info().rss / 1024 / 1024
                )
                log.info(
                    "ASR worker post-first-call baseline: %.0fMB",
                    self._asr_worker_baseline_mb,
                )
            except Exception:
                pass
        self._record_latency("asr", asr_ms)
        self._log_mem_after_asr(kind, audio_seconds, asr_ms)
        return result, asr_ms

    def _start_worker_from_state(self, state: dict, expected_gen: int) -> bool:
        """Load a worker from a saved state and activate it only if no newer engine
        switch happened in the meantime (generation guard). Runs on the ASR thread;
        the load is intentionally done outside _asr_lock. Returns True on activation."""
        try:
            client = self._load_engine_client(state["config"])
        except Exception as e:
            log.error(f"ASR worker (re)start failed: {e}", exc_info=True)
            return False
        stale = None
        with self._asr_lock:
            # _stop_event, not `not self._running`: the latter is also false
            # before the first start(), so a worker loaded ahead of the pipeline
            # was discarded as "superseded". The event is set only by stop().
            if self._asr_generation != expected_gen or self._stop_event.is_set():
                stale = client
            else:
                self._asr = client
                self._asr_type = state["type"]
                self._asr_signature = state["signature"]
                self._asr_device = state["device"]
                self._asr_config = dict(state["config"]) if state["config"] else None
                self._funasr_model_key = state["funasr_model_key"]
                self._gigaam_model_key = state["gigaam_model_key"]
                self._whisper_model_size = state["whisper_model_size"]
                self._asr_ready = True
                self._asr_error_count = 0
                self._asr_restart_state = dict(state)
                self._asr_worker_baseline_mb = None
                self._asr_generation += 1
        if stale is not None:
            log.info("Discarding superseded ASR worker (newer switch won the race)")
            try:
                stale.shutdown()
            except Exception:
                pass
            return False
        name = state.get("display_name") or state.get("type")
        if self._running:
            self._record_session_info()
        if self._overlay:
            self._overlay.update_asr_device(
                f"{name} [{state.get('device_label', state['device'])}]"
            )
        return True

    def _recover_asr_worker(self, dead_client, reason: str):
        """Auto-restart a worker that died mid-session. Without this, a single crash
        or transcribe timeout would leave ASR permanently silent for the session."""
        with self._asr_lock:
            if self._asr is not dead_client:
                return  # an engine switch already replaced/cleared it
            state = dict(self._asr_restart_state) if self._asr_restart_state else None
            attempt = self._asr_restart_count + 1
            give_up = (
                state is None
                or not state.get("config")
                or attempt > self._asr_restart_max
            )
            self._asr_restart_count = attempt
            self._asr = None
            self._asr_ready = False
            self._asr_type = None
            self._asr_signature = None
            self._asr_config = None
            self._asr_error_count = 0
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
            gen = self._asr_generation
        try:
            dead_client.shutdown()
        except Exception:
            try:
                dead_client.terminate()
            except Exception:
                pass
        if not self._running:
            return  # shutting down; do not spawn a replacement worker
        if give_up:
            log.error(
                f"ASR worker died and auto-restart gave up after "
                f"{self._asr_restart_max} attempts: {reason}"
            )
            if self._overlay:
                self._overlay.update_asr_device("ASR unavailable")
            return
        log.warning(
            f"ASR worker died ({reason}); auto-restart attempt "
            f"{attempt}/{self._asr_restart_max}"
        )
        self._release_memory_caches()
        if self._start_worker_from_state(state, gen):
            log.info(
                f"ASR worker auto-restarted: {state.get('type')} on "
                f"{state.get('device')}"
            )
        elif self._asr is None and self._overlay:
            self._overlay.update_asr_device("ASR unavailable")

    def _maybe_recycle_asr_worker(self):
        """Recycle the worker once its RSS grows well past the post-load baseline, to
        bound native-side leaks that accumulate in the long-lived worker process.
        Called from the ASR thread between segments so the reload gap costs no audio
        beyond what arrives during it."""
        if not self._running:
            return
        if not self._asr_queue.empty():
            return
        with self._vad_lock:
            if self._vad._is_speaking:
                return
        if time.monotonic() - self._last_speech_activity < 15.0:
            return
        with self._asr_lock:
            client = self._asr
            if not self._asr_ready or client is None or self._asr_recycling:
                return
            state = dict(self._asr_restart_state) if self._asr_restart_state else None
        if state is None or not state.get("config") or client.pid is None:
            return
        try:
            import psutil

            rss = psutil.Process(client.pid).memory_info().rss / 1024 / 1024
        except Exception:
            return
        if self._asr_worker_baseline_mb is None:
            return
        if rss < self._asr_worker_baseline_mb + self._asr_recycle_delta_mb:
            return
        log.warning(
            f"ASR worker RSS={rss:.0f}MB grew "
            f"{rss - self._asr_worker_baseline_mb:.0f}MB over baseline; recycling"
        )
        self._recycle_asr_worker(client, state)

    def _recycle_asr_worker(self, old_client, state: dict):
        # Graceful stop-then-start (no VRAM doubling). The generation guard makes a
        # concurrent engine switch win over this recycle.
        with self._asr_lock:
            if self._asr is not old_client:
                return
            self._asr = None
            self._asr_ready = False
            self._asr_recycling = True
            self._asr_worker_baseline_mb = None
            self._asr_generation += 1
            gen = self._asr_generation
        try:
            old_client.shutdown()
        except Exception:
            try:
                old_client.terminate()
            except Exception:
                pass
        self._release_memory_caches()
        if not self._running:
            with self._asr_lock:
                self._asr_recycling = False
            return
        try:
            started = self._start_worker_from_state(state, gen)
        finally:
            with self._asr_lock:
                self._asr_recycling = False
        if started:
            log.info(f"ASR worker recycled: {state.get('type')} on {state.get('device')}")
        else:
            log.error("ASR worker recycle failed to restart")
            if self._asr is None and self._overlay:
                self._overlay.update_asr_device("ASR unavailable")

    def _check_memory_threshold(self, rss_mb: float):
        if self._mem_warned or rss_mb < self._mem_threshold_mb:
            return
        self._mem_warned = True
        log.warning(
            f"Memory ceiling reached: combined RSS (main+worker)={rss_mb:.0f}MB "
            f"(threshold {self._mem_threshold_mb}MB). "
            f"Recommend restarting LiveTranslate to free C-side allocator caches."
        )
        if self._mem_warning_callback is not None:
            try:
                self._mem_warning_callback(rss_mb)
            except Exception as e:
                log.warning(f"Memory warning callback failed: {e}")

    def set_memory_warning_callback(self, callback):
        self._mem_warning_callback = callback

    def set_notification_callback(self, callback):
        self._notify_callback = callback

    def _notify_user(self, message: str):
        if self._notify_callback is None:
            log.warning("Notification with no sink: %s", message)
            return
        try:
            self._notify_callback(message)
        except Exception:
            log.error("Notification callback failed", exc_info=True)

    def _log_mem_periodic(self):
        snap = self._mem_snapshot()
        total_delta = snap["rss"] - self._mem_baseline_mb
        log.info(
            f"MEM[tick] RSS={snap['rss']:.1f}MB ({total_delta:+.1f} since start) "
            f"worker_rss={snap['worker_rss']:.0f}MB "
            f"GPU(main alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"msgs={snap['msgs']} asr_calls={self._mem_asr_call_count} "
            f"asr_count={self._asr_count} tl_count={self._translate_count}"
        )
        self._check_memory_threshold(snap["total_rss"])

    def _compute_cost(self):
        if self._input_price > 0 or self._output_price > 0:
            return (self._total_prompt_tokens * self._input_price +
                    self._total_completion_tokens * self._output_price) / 1_000_000
        return 0.0

    def _record_latency(self, kind: str, elapsed_ms: float):
        with self._translation_stats_lock:
            values = self._asr_latencies if kind == "asr" else self._translation_latencies
            values.append(float(elapsed_ms))
            self._latency_counts[kind] += 1
            total_count = self._latency_counts[kind]
            if total_count % 20:
                return
            count = len(values)
            p50, p95 = np.percentile(np.asarray(values), [50, 95])
            audio_metrics = self._audio.metrics() if hasattr(self._audio, "metrics") else {}
            worker_rss = self._mem_snapshot().get("worker_rss", 0.0)
            log.info(
                "PERF[%s] samples=%d p50=%.0fms p95=%.0fms asr_queue=%d "
                "audio_dropped=%d worker_rss=%.0fMB translation_pending=%d",
                kind,
                count,
                p50,
                p95,
                self._asr_queue.qsize(),
                audio_metrics.get("dropped_blocks", 0),
                worker_rss,
                self._translation_pending,
            )

    def _translate_async(
        self,
        msg_id,
        text,
        source_lang,
        extra_langs=None,
        request_translator=None,
        generation=None,
        session_generation=None,
        expected_session=None,
    ):
        """Translate text and update UI with streaming display.

        ``session_generation`` is the recording-session generation the msg's
        work count was registered under (the segment's, not the current
        one). It keys the release in the finally: a session begin/end racing
        the translation must not make the release land on the *new* session
        (a no-op there, leaving the old one's wait hanging on a count that
        will never be released). ``expected_session`` is the entry's session
        stamp, threaded through to the writer so the completion is
        identity-checked like the original write.
        """
        try:
            self._translate_async_inner(
                msg_id, text, source_lang, extra_langs,
                request_translator, generation,
                expected_session,
            )
        finally:
            gen = (
                session_generation
                if session_generation is not None
                else self._session_generation
            )
            self._session_work.release_msg(gen, msg_id)

    def _translate_async_inner(
        self,
        msg_id,
        text,
        source_lang,
        extra_langs=None,
        request_translator=None,
        generation=None,
        expected_session=None,
    ):
        if request_translator is None:
            request_translator, generation = self._snapshot_translation_request(
                msg_id, text
            )
        translated = None
        try:
            tl_start = time.perf_counter()
            for partial in request_translator.translate_iter(text, source_lang):
                translated = partial
                if self._overlay and generation == self._translator_generation:
                    self._overlay.update_streaming(msg_id, partial)
            tl_ms = (time.perf_counter() - tl_start) * 1000
            if not self._commit_translation_result(
                msg_id, text, translated, generation
            ):
                # The model changed while this was in flight, so it must not
                # enter the new generation's history — but it is still a valid
                # translation of this line, and the entry has to be closed out.
                # Returning here left the overlay stuck on "translating" and,
                # because the transcript releases entries in order, stalled the
                # whole meeting record behind it for the rest of the session.
                log.info("Translation superseded by a model switch: msg=%s", msg_id)
                if translated:
                    self._transcript.write_translation(
                        msg_id, translated, session=expected_session
                    )
                else:
                    self._transcript.finalize_no_translation(
                        msg_id, session=expected_session
                    )
                if self._overlay:
                    self._overlay.update_translation(msg_id, translated or "", 0)
                return
            pt, ct = request_translator.last_usage
            with self._translation_stats_lock:
                self._translate_count += 1
                self._total_prompt_tokens += pt
                self._total_completion_tokens += ct
                cost = self._compute_cost()
            self._record_latency("translation", tl_ms)
            log.info(f"Translate ({tl_ms:.0f}ms): {translated}")
            if translated:
                self._transcript.write_translation(
                    msg_id, translated, session=expected_session
                )
            else:
                self._transcript.finalize_no_translation(
                    msg_id, session=expected_session
                )
            if self._overlay:
                self._overlay.update_translation(msg_id, translated, tl_ms)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    cost,
                )
            if self._subwin and self._subwin.isVisible() and translated:
                tl_dict = {self._target_language: translated}
                if extra_langs:
                    self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
                self._subwin.update_text(text, tl_dict)
        except RepetitionError:
            self._commit_translation_result(msg_id, text, None, generation)
            log.warning("Repetition loop detected, model may not support structured output well")
            self._transcript.finalize_no_translation(
                msg_id, session=expected_session
            )
            if self._overlay:
                self._overlay.update_translation(
                    msg_id, f"[{t('error_repetition')}]", 0
                )
        except Exception as e:
            current = self._commit_translation_result(msg_id, text, None, generation)
            import openai
            if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                              openai.AuthenticationError, openai.APIStatusError,
                              TimeoutError, ConnectionError)):
                log.warning(f"Translate error: {e}")
            else:
                log.error(f"Translate error: {e}", exc_info=True)
            if not current:
                # Failed *and* superseded. Same reasoning as above: it still has
                # to be closed out or it blocks every later entry.
                self._finalize_untranslated(
                    msg_id, f"superseded during a failed translation: {e}",
                    user_visible=True,
                )
                return
            self._transcript.finalize_no_translation(
                msg_id, session=expected_session
            )
            if self._overlay:
                self._overlay.update_translation(msg_id, f"[error: {e}]", 0)

    def _translate_extra_langs(self, text, source_lang, extra_langs, tl_dict):
        """Translate into additional languages for subtitle window (parallel)."""
        from concurrent.futures import as_completed

        def _do_translate(lang):
            with self._translation_lock:
                base = self._translator
                if base is None:
                    # Same condition as the primary path: an unavailable
                    # translation service is a named failure, not a None
                    # dereference inside a worker thread.
                    raise TranslationUnavailable(
                        "translation service is unavailable"
                    )
                translator = base.fork_for_request(
                    target_language=lang,
                    history_snapshot=list(self._translation_history),
                )
            return lang, translator.translate(text, source_lang)

        executor = self._extra_tl_executor
        if executor is None:
            log.debug("Extra-language executor is gone; skipping %s", extra_langs)
            return
        futures = []
        for lang in extra_langs:
            try:
                futures.append(executor.submit(_do_translate, lang))
            except RuntimeError:
                log.debug("Extra-language executor shut down mid-submit")
                break

        for future in as_completed(futures):
            try:
                lang, result = future.result()
                tl_dict[lang] = result
                log.info(f"Extra translate [{lang}]: {result}")
            except TranslationUnavailable as e:
                log.warning(f"Extra translate skipped: {e}")
            except Exception as e:
                import openai
                if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                                  openai.AuthenticationError, openai.APIStatusError,
                                  TimeoutError, ConnectionError)):
                    log.warning(f"Extra translate error: {e}")
                else:
                    log.error(f"Extra translate error: {e}", exc_info=True)

    def _translate_subwin_only(self, text, source_lang, extra_langs):
        """Translate only for subtitle window when primary target == source language."""
        tl_dict = {self._target_language: text}  # same language, use original
        self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
        if self._subwin and self._subwin.isVisible():
            self._subwin.update_text(text, tl_dict)

    def start(self):
        if self._running:
            return
        self._tl_executor = ThreadPoolExecutor(
            max_workers=self._translation_workers, thread_name_prefix="translate"
        )
        self._extra_tl_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="translate-extra"
        )
        self._asr_queue = queue.Queue(maxsize=16)
        self._stop_event.clear()
        self._stopped = False
        self._running = True
        self._paused = False
        try:
            self._audio.start()
        except Exception:
            self._running = False
            self._stopped = True
            self._tl_executor.shutdown(wait=False, cancel_futures=True)
            self._extra_tl_executor.shutdown(wait=False, cancel_futures=True)
            self._tl_executor = None
            self._extra_tl_executor = None
            raise
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._asr_thread = threading.Thread(
            target=self._asr_loop, daemon=True
        )
        self._capture_thread.start()
        self._asr_thread.start()
        # Periodic memory snapshot every 30s
        if self._mem_periodic_timer is None:
            self._mem_periodic_timer = QTimer()
            self._mem_periodic_timer.timeout.connect(self._log_mem_periodic)
            self._mem_periodic_timer.start(30000)
        snap = self._mem_snapshot()
        log.info(
            f"MEM[start] RSS={snap['rss']:.1f}MB "
            f"GPU(alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
            f"(baseline for delta tracking)"
        )
        self._publish_transcript_paths()
        self._record_session_info()
        # The pipeline runs; the writer may auto-open a session when entries
        # arrive (the legacy no-button path). Explicit begin/end drives the
        # button-driven lifecycle on top of this.
        self._transcript.set_recording(True)
        log.info("Pipeline started (capture + ASR threads)")

    def stop(self):
        """Tear the pipeline down. Idempotent, bounded and complete.

        Every step is wrapped: a failure in one of them must not skip worker,
        file or service reclamation further down (CALL_CHAIN_FIX_TODO 2.3/2.5).
        A repeat call only mops up whatever the first one left behind.
        """
        first_call = not self._stopped
        self._stopped = True
        self._running = False
        self._stop_event.set()
        # App quit: any open meeting ends here as part of the shutdown. The
        # session state machine goes to IDLE through the same notifier so the
        # UI is not left claiming a recording exists. The generation bump
        # runs under the boundary lock like every other generation write.
        with self._session_boundary_lock:
            if self._session_state != SessionState.IDLE:
                self._session_generation += 1  # supersedes any in-flight ENDING
                self._notify_session_state(SessionState.IDLE)
            self._session_end_gating = False
        self._transcript.set_recording(False)
        # Nothing is waited on anymore; late releases for the old
        # generations are no-ops by design.
        self._session_work.discard_all()

        self._stop_step("audio capture", self._audio.stop)
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            if self._capture_thread.is_alive():
                log.warning("Capture thread still running after timeout")
            self._capture_thread = None

        # Best-effort wake-up only. _asr_loop also polls _stop_event, so a full
        # queue can no longer strand this call the way a blocking put() did.
        try:
            self._asr_queue.put_nowait(None)
        except queue.Full:
            log.debug("ASR queue full; relying on the stop event to end the loop")
        if self._asr_thread:
            self._asr_thread.join(timeout=10)
            if self._asr_thread.is_alive():
                log.warning("ASR thread still running after timeout, proceeding with cleanup")
            self._asr_thread = None

        # Flush the remaining VAD buffer once the pipeline threads are done, and
        # only while ASR can still serve it — a flush against a dead worker would
        # queue translation work the executors below are about to refuse.
        if first_call:
            self._stop_step("VAD flush", self._flush_on_stop)
        self._reset_interim_state()

        # After the flush, so translations it produced are awaited rather than
        # cancelled.
        if self._tl_executor is not None:
            self._stop_step("translation executor", self._tl_executor.shutdown)
            self._tl_executor = None
        if self._extra_tl_executor is not None:
            self._stop_step(
                "extra-language executor", self._extra_tl_executor.shutdown
            )
            self._extra_tl_executor = None

        self._stop_step("transcript", self._transcript.close)
        if self._mem_periodic_timer is not None:
            self._stop_step("memory timer", self._mem_periodic_timer.stop)
            self._mem_periodic_timer = None
        if getattr(self, "_mlx_monitor_timer", None) is not None:
            self._stop_step("MLX monitor timer", self._mlx_monitor_timer.stop)
            self._mlx_monitor_timer = None

        if first_call:
            try:
                snap = self._mem_snapshot()
                total_delta = snap["rss"] - self._mem_baseline_mb
                log.info(
                    f"MEM[stop] RSS={snap['rss']:.1f}MB ({total_delta:+.1f} since start) "
                    f"GPU(alloc/reserved)={snap['gpu_alloc']:.0f}/{snap['gpu_reserved']:.0f}MB "
                    f"asr_calls={self._mem_asr_call_count} outputs={self._asr_count}"
                )
            except Exception:
                log.debug("Final memory snapshot failed", exc_info=True)

        self._stop_step("ASR worker", self._shutdown_asr_worker)
        self._stop_step("MLX service", self._mlx_service.stop)
        if first_call:
            log.info("Pipeline stopped")
        else:
            log.debug("Pipeline stop: residual cleanup only")

    def _stop_step(self, what: str, action):
        """Run one cleanup step; log and continue so later steps still run."""
        try:
            action()
        except Exception:
            log.error("Cleanup step failed: %s", what, exc_info=True)

    def _flush_on_stop(self):
        if not self._asr_ready:
            with self._vad_lock:
                self._vad._reset()
            return
        if self._interim_active:
            with self._vad_lock:
                remaining = self._vad.force_flush()
            if remaining is not None:
                self._process_interim_final(remaining)
        else:
            with self._vad_lock:
                remaining = self._vad.flush()
            if remaining is not None:
                self._process_segment(remaining)

    def wait_until_stopped(self, timeout: float = 15.0) -> bool:
        """True once no pipeline thread and no ASR worker process is left.

        Used by the delete-cache-and-exit path, which must not unlink model
        directories a worker still holds open. Checks the actual child
        processes: _shutdown_asr_worker clears self._asr before the worker has
        finished dying, so that field proves nothing here.
        """
        deadline = time.monotonic() + timeout
        for thread in (self._capture_thread, self._asr_thread):
            if thread is None:
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        while time.monotonic() < deadline:
            if not self._live_worker_pids():
                return True
            time.sleep(0.1)
        remaining = self._live_worker_pids()
        if remaining:
            log.warning("ASR worker processes still alive: %s", remaining)
        return not remaining

    @staticmethod
    def _live_worker_pids():
        import multiprocessing as mp_mod

        return [
            proc.pid
            for proc in mp_mod.active_children()
            if (proc.name or "").startswith("ASRWorker")
        ]

    def pause(self):
        self._paused = True
        # Hand off whatever was mid-utterance. Leaving it in the buffer meant
        # audio from after the resume was appended to it, so a pause taken in
        # the middle of a sentence produced one line spliced across the gap —
        # however long the gap was. flush()/force_flush() clear the buffer
        # either way, so the next utterance starts clean.
        with self._vad_lock:
            remaining = (
                self._vad.force_flush() if self._interim_active else self._vad.flush()
            )
        if remaining is not None and self._asr_ready:
            # Queued for the ASR thread rather than transcribed here: this runs
            # on the Qt thread. Its vad_flush handler resets the interim state.
            # The boundary lock is the producer fence _enqueue_asr expects.
            with self._session_boundary_lock:
                self._enqueue_asr("vad_flush", remaining)
        else:
            self._reset_interim_state()
        if self._overlay:
            self._overlay.update_monitor(0.0, 0.0)
        log.info("Pipeline paused")

    def resume(self):
        self._paused = False
        log.info("Pipeline resumed")

    def _process_segment(self, speech_segment, work_id=None, generation=None,
                         expected_session=None):
        """Run ASR + translation on a speech segment. Called from ASR thread
        and stop(). ``work_id``/``generation``/``expected_session`` identify
        this queue item's session work and meeting; ``generation`` guards
        the *stale segment* case — an end+begin racing the queue means the
        item's session is closed while the writer already has a *new*
        session open: without this guard the old audio's original would be
        written into the new meeting's files (ghost cross-session writing).
        The guard runs twice: at entry (before ASR) and again inside the
        session boundary fence right before the write — recognition can
        take seconds and a begin/end may land inside that window, and the
        writer's ``session=None`` wildcard (the legacy auto-open path)
        cannot refuse a write on its own.
        A segment whose session ended but whose generation is still current
        (end completed, no new begin) is safe to process: the writer itself
        refuses the write and the tracker refuses the msg count.
        ``expected_session`` is re-checked by the writer inside its own
        lock, the final authority for what lands in a session's files."""
        if generation is not None and generation != self._session_generation:
            log.info(
                "Dropping segment from superseded session generation %s "
                "(current %s)", generation, self._session_generation,
            )
            return
        seg_len = len(speech_segment) / 16000
        log.info(f"Speech segment: {seg_len:.1f}s")

        try:
            result, asr_ms = self._run_asr(speech_segment, "segment")
        except Exception as e:
            log.error(f"ASR error: {e}", exc_info=True)
            return
        if asr_ms == 0:
            return
        if asr_ms > 10000:
            log.warning(f"ASR took {asr_ms:.0f}ms, possible hang")
        validated = validate_asr_result(result, "segment")
        if validated is None:
            return
        original_text, source_lang = validated

        # Skip punctuation-only ASR results
        if not any(c.isalnum() for c in original_text):
            log.debug(f"ASR returned punctuation-only, skipping: '{original_text}'")
            return

        # Skip suspiciously short text from long segments (likely noise)
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(
                f"Noise filter: {seg_len:.1f}s segment produced only '{original_text}', skipping"
            )
            return

        asr_lang_setting = self._get_asr_language_setting()
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(
                f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', "
                f"discarding: {original_text[:60]}"
            )
            return

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms): {original_text}")

        # Final identity check + write + count registration + adoption, one
        # linearized section under the session boundary fence. The stale
        # guard at the top ran *before* ASR; recognition can take seconds
        # and a begin/end may land inside that window. Inside this fence:
        #   * the re-check refuses audio whose generation went stale — an
        #     explicit begin completed first, and its new meeting must not
        #     receive pre-begin audio (the writer's session=None wildcard
        #     would have accepted it);
        #   * the write — and the auto-open it may perform — cannot
        #     interleave with begin_recording_session's session-open +
        #     generation bump, and the adoption below decides under the
        #     exact boundary state the write saw.
        # Only short local work runs here (a queued signal emit, buffered
        # file appends, tracker set-adds): no ASR, translation or network —
        # recognition has already finished and the translation submit below
        # runs after the fence. The queue item's work_id is released by the
        # ASR loop's finally only after this whole processing returns, so
        # the ENDING wait cannot pass between the write and the
        # registration.
        with self._session_boundary_lock:
            if generation is not None and generation != self._session_generation:
                log.info(
                    "Dropping segment from superseded session generation %s "
                    "(current %s; the boundary moved during recognition)",
                    generation, self._session_generation,
                )
                return
            # Registered under the *segment's* generation, not the current
            # one — a begin racing the queue must not hang the old session's
            # work on the new session's wait.
            msg_generation = (
                generation if generation is not None
                else self._session_generation
            )
            if self._overlay:
                self._overlay.add_message(
                    msg_id, timestamp, original_text, source_lang, asr_ms
                )
            # expected_session (from the queue item) is the write-time identity
            # check: the writer refuses the entry inside its own lock when the
            # session that is open now is not the one this audio belongs to.
            # The result decides the entry's fate:
            #   recorded  — the translation is counted and submitted;
            #   mismatch  — the audio belongs to a closed/replaced session: no
            #               translation is submitted (its late result could not
            #               be written anywhere) and the overlay entry is closed
            #               out as untranslated;
            #   skipped/failed — subtitle-only (nothing recorded) or a file
            #               error: recognition is displayed, the translation may
            #               still run for display, and the writer's _complete
            #               drops the file write because no original is pending.
            result = self._transcript.write_original(
                msg_id, timestamp, original_text,
                language=source_lang, duration=seg_len,
                session=expected_session,
            )
            if result == TranscriptWriter.WRITE_SESSION_MISMATCH:
                self._finalize_untranslated(
                    msg_id, "segment belongs to a closed session",
                    user_visible=False,
                )
                return
            recorded = result == TranscriptWriter.WRITE_RECORDED
            if recorded:
                # The translation job gets its own count, registered only now
                # that the original is actually in the session's files (a count
                # for a refused entry would hold an ENDING wait for nothing).
                self._session_work.register_msg(msg_generation, msg_id)
                # Legacy auto-open adoption — the single authoritative helper,
                # shared with _process_segment_text. Same fence (RLock
                # re-entry): the adoption decides under the same boundary
                # state the write saw. No generation bump: items enqueued
                # before the auto-open carry this generation and are this
                # meeting's opening speech (see the helper).
                msg_generation = self._adopt_auto_opened_session(
                    msg_id, msg_generation
                )

        # Store for subtitle window (translation will be added later)
        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language

        # Collect extra languages needed by subtitle window (beyond the primary target)
        extra_langs = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            # Remove primary target and source (no need to translate those)
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            # No translation job will exist, so the msg count registered
            # above is released here (the _translate_async finally that
            # normally owns it never runs on this path), under the same
            # generation it was registered in. The session argument keeps
            # the completion identity-checked like every other path.
            self._session_work.release_msg(msg_generation, msg_id)
            self._transcript.finalize_no_translation(
                msg_id, session=expected_session
            )
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    self._compute_cost(),
                )
            if self._subwin and self._subwin.isVisible():
                # Primary is same language; still need to translate extra langs
                if extra_langs:
                    try:
                        self._extra_tl_executor.submit(
                            self._translate_subwin_only, original_text, source_lang, extra_langs
                        )
                    except RuntimeError:
                        pass
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            try:
                self._submit_translation(
                    msg_id, original_text, source_lang, extra_langs or None,
                    session_generation=msg_generation,
                    expected_session=expected_session,
                )
            except TranslationUnavailable as exc:
                # No job was submitted: release the msg count here.
                self._session_work.release_msg(msg_generation, msg_id)
                self._finalize_untranslated(msg_id, str(exc), user_visible=True)
            except RuntimeError:
                # Executor already shut down (we are exiting) — not worth a
                # user-facing error, but still has to be closed out.
                self._session_work.release_msg(msg_generation, msg_id)
                self._finalize_untranslated(
                    msg_id, "translation executor shut down", user_visible=False
                )

    # ── Incremental ASR ──

    _segmenter_cache = {}  # lang -> yasbd pysbd_adapter.Segmenter

    @staticmethod
    def _get_segmenter(lang: str):
        # This import sits on the interim-ASR path, which has no try/except of
        # its own — an ImportError here used to kill the ASR thread outright.
        # yasbd-lib is declared in requirements*.txt and installed by every
        # entrypoint, so a failure means a broken environment: degrade to
        # no-splitting instead of taking the pipeline down.
        try:
            from yasbd import get_supported_langs, pysbd_adapter
        except ImportError:
            log.error(
                "yasbd-lib is unavailable; incremental ASR sentence splitting is "
                "disabled. Reinstall dependencies to restore it.", exc_info=True,
            )
            LiveTranslateApp._segmenter_cache[lang] = None
            return None
        if lang not in LiveTranslateApp._segmenter_cache:
            seg_lang = lang if lang in get_supported_langs() else "en"
            LiveTranslateApp._segmenter_cache[lang] = pysbd_adapter.Segmenter(
                language=seg_lang, clean=False
            )
        return LiveTranslateApp._segmenter_cache[lang]

    def _split_sentences(self, text: str, lang: str = "en") -> list[str]:
        """Split text into sentences using yasbd, with comma fallback for long text."""
        seg = self._get_segmenter(lang)
        if seg is None:
            return [text]
        parts = [p for p in seg.segment(text) if p.strip()]
        if len(parts) > 1:
            return parts

        # Comma fallback for long unsplit text — split at last balanced comma
        # CJK 「、」at 25 chars; all commas at 60 chars (long sentence, reduce latency)
        min_len = 25 if any(c == '、' for c in text) else 60
        if len(text) > min_len:
            for i in range(len(text) - 8, 5, -1):
                if text[i] in ',，;；、':
                    before = text[:i + 1].strip()
                    after = text[i + 1:].strip()
                    if before and after and len(before) > 15 and len(after) > 3:
                        return [before, after]

        return parts

    @staticmethod
    def _is_short_utterance(text: str) -> bool:
        """Check if text has ≤8 alphanumeric chars (likely noise/filler/fragment)."""
        alnum = sum(1 for c in text if c.isalnum())
        return alnum <= 8

    # Punctuation and spacing that ends a committed sentence but never begins
    # the next recognition.
    _ECHO_BOUNDARY = " \t\n。．.!！?？,，、;；:：\"'）)》」』"
    # Minimum overlap for a script that does not separate words with spaces.
    _ECHO_MIN_UNSPACED = 6

    def _strip_committed_overlap(self, text: str) -> str:
        """Drop a re-recognized repeat of text already committed this utterance.

        Two things shape this, and they pull in opposite directions:

        * Committed text is always a complete sentence, so it always ends in
          punctuation. Comparing it verbatim meant this never fired at all —
          a new recognition never begins with a full stop. The tail is
          therefore matched with its trailing punctuation removed.
        * A lecturer opening the next sentence with the previous one's keyword
          ("...производную функции. Функции бывают...") is common, and deleting
          that word costs the sentence its subject. A genuine echo comes from
          an under-trimmed buffer replaying audio already consumed, which
          reproduces a phrase rather than a single word — so an overlap only
          counts when it spans a word boundary, or is several characters long
          in a script that has none.

        Erring toward keeping text is deliberate: a duplicated word is easy to
        read past, a deleted one is unrecoverable and mistranslates the line.
        """
        tail = self._interim_committed_tail
        if not tail:
            return text
        tail = tail.lower().rstrip(self._ECHO_BOUNDARY)
        if not tail:
            return text
        lowered = text.lower()
        max_check = min(len(tail), len(lowered))
        for overlap_len in range(max_check, 2, -1):
            if lowered[:overlap_len] != tail[-overlap_len:]:
                continue
            # The longest match is found first; every shorter one is a prefix
            # of it, so if this is not substantial none of them are either.
            if not self._is_substantial_echo(lowered[:overlap_len]):
                return text
            stripped = text[overlap_len:].lstrip(self._ECHO_BOUNDARY)
            log.debug(
                f"Stripped echo overlap ({overlap_len} chars): "
                f"'{text[:overlap_len]}'"
            )
            return stripped
        return text

    @classmethod
    def _is_substantial_echo(cls, overlap: str) -> bool:
        """Whether an overlap is a replay rather than one repeated word.

        A single-word overlap is genuinely ambiguous in a script that separates
        words — "функции" could be a replay or the next sentence's subject — so
        it is kept. The cost is a duplicated word when a replay happens to be
        one word long; the alternative cost is deleting a real one.
        """
        if " " in overlap.strip():
            return True
        # No word boundary at all: only meaningful where the script has none.
        return (
            len(overlap) >= cls._ECHO_MIN_UNSPACED
            and cls._is_unspaced_script(overlap)
        )

    @staticmethod
    def _is_unspaced_script(text: str) -> bool:
        """Han/Hiragana/Katakana, where a run of characters is several words.

        Not `not text.isascii()`: Cyrillic and Greek are also non-ASCII but do
        separate words with spaces, so that test wrongly treated a single
        Russian word as a multi-word phrase.
        """
        letters = [ch for ch in text if ch.isalnum()]
        if not letters:
            return False
        unspaced = sum(
            1
            for ch in letters
            if "\u4e00" <= ch <= "\u9fff"      # CJK unified ideographs
            or "\u3040" <= ch <= "\u309f"      # hiragana
            or "\u30a0" <= ch <= "\u30ff"      # katakana
        )
        return unspaced >= len(letters) * 0.6

    def _do_interim_asr(self, generation=None, expected_session=None) -> bool:
        """Run ASR on current VAD buffer, output complete sentences, trim consumed audio.
        Returns True if any sentences were committed. ``generation``/
        ``expected_session`` are the interim queue item's session identity,
        threaded into each committed sentence so a session superseded
        mid-pass cannot receive them. A sentence the identity guard refuses
        is not consumed: no trim, no echo-tail update — its audio stays in
        the buffer for a later pass, and any buffered fragments spliced
        into it stay pending (their audio was already trimmed by the pass
        that buffered them, so this text is the only copy — see
        _process_segment_text's return)."""
        with self._vad_lock:
            peek = self._vad.peek_buffer()
        if peek is None:
            return False
        audio, duration = peek

        # Don't bother with very short buffers
        if duration < 1.5:
            return False

        # Word timestamp alignment is expensive for repeated interim passes.
        # The proportional trim path below is less exact but keeps long runs stable.
        use_word_ts = False

        try:
            result, asr_ms = self._run_asr(
                audio, "interim", word_timestamps=use_word_ts
            ) if use_word_ts else self._run_asr(audio, "interim")
        except Exception as e:
            log.error(f"Interim ASR error: {e}", exc_info=True)
            return False

        if asr_ms == 0:
            return False

        validated = validate_asr_result(result, "interim")
        if validated is None:
            return False
        full_text, result_lang = validated
        if not any(c.isalnum() for c in full_text):
            return False

        # Strip echo from previous commit's overlap
        full_text = self._strip_committed_overlap(full_text)
        if not full_text:
            return False

        split_start = time.perf_counter()
        sentences = self._split_sentences(full_text, result_lang)
        split_ms = (time.perf_counter() - split_start) * 1000
        if len(sentences) <= 1:
            return False
        log.debug(f"Interim split [{result_lang}] ({split_ms:.1f}ms): {len(sentences)} parts -> {sentences}")

        # All but last are complete; last is still being spoken
        complete = sentences[:-1]

        # Output committed sentences first: the trim point and the echo
        # tail below are derived from what *actually* committed. A sentence
        # the identity guard refuses (stale session generation, or a
        # session-mismatch write — see _process_segment_text) landed in no
        # session and no overlay entry, so it is NOT consumed: it stays out
        # of the committed prefix and its audio stays in the VAD buffer for
        # a later pass. The pass's remaining sentences carry the same
        # identity, so the loop stops at the first refusal.
        committed_parts: list[str] = []
        actually_committed = False
        consumed_any = False
        for sent in complete:
            text = sent.strip()
            if not text:
                continue
            if self._is_short_utterance(text):
                self._buffer_interim_fragment(text)
                # Buffering still consumes the audio: the trim and the echo tail
                # below must run, or the next pass re-recognizes the same words
                # and appends them to _interim_pending all over again.
                committed_parts.append(text)
                consumed_any = True
                continue

            # Splice the buffered fragments in, but snapshot them first and
            # keep them pending until the commit is confirmed: their audio
            # was already trimmed away by the pass that buffered them, so a
            # refusal here is the last chance to keep their text — without
            # the restore, a later pass could never re-recognize it and the
            # fragments were lost forever.
            pending_prefix = self._interim_pending
            if pending_prefix:
                text = pending_prefix + text

            if not self._process_segment_text(
                text, result_lang, asr_ms,
                generation=generation, expected_session=expected_session,
            ):
                # Identity refusal: neither the sentence nor the spliced
                # pending prefix was consumed. Restore the pending from the
                # snapshot (its audio is gone; a later pass can only commit
                # it from here); the refused sentence's own audio stays in
                # the VAD buffer for that same later pass.
                self._interim_pending = pending_prefix
                break
            # Confirmed consumed: only now is the pending cleared, and only
            # the committed text below participates in the trim, the echo
            # tail and _interim_active.
            self._interim_pending = ""
            committed_parts.append(text)
            actually_committed = True
            consumed_any = True

        if not consumed_any:
            # Nothing consumed — every sentence was refused (or there was
            # nothing to commit): the buffer, the echo tail and the interim
            # state stay untouched so a later pass can retry them.
            return False
        committed_text = "".join(committed_parts)

        # Determine trim point — from the committed prefix only, so audio
        # for refused sentences is never trimmed away.
        total_samples = len(audio)
        if use_word_ts and result.get("words"):
            words = result["words"]
            committed_lower = committed_text.lower().rstrip()
            char_pos = 0
            last_word_end = 0.0
            for w in words:
                word_text = w["word"].strip()
                idx = committed_lower.find(word_text.lower(), char_pos)
                if idx >= 0:
                    char_pos = idx + len(word_text)
                    last_word_end = w["end"]
                if char_pos >= len(committed_lower):
                    break
            trim_samples = int(last_word_end * 16000)
        else:
            # Proportional trim with safety margin to reduce echo
            ratio = len(committed_text) / max(len(full_text), 1)
            margin = int(0.3 * 16000)  # 0.3s extra trim to avoid re-recognition
            trim_samples = int(ratio * total_samples) + margin
            # Don't over-trim: keep at least 0.5s for the remaining sentence
            max_trim = total_samples - int(0.5 * 16000)
            trim_samples = min(trim_samples, max(max_trim, 0))
            # Minimum trim to prevent re-recognition loops
            min_trim = int(0.3 * 16000)
            if trim_samples < min_trim and trim_samples > 0:
                trim_samples = min(min_trim, total_samples // 2)

        if trim_samples > 0:
            with self._vad_lock:
                self._vad.trim_front(trim_samples)

        # Track committed text tail for echo dedup
        self._interim_committed_tail = committed_text[-50:] if len(committed_text) > 50 else committed_text

        self._interim_active = True
        log.info(
            f"Interim ASR: consumed {len(committed_parts)} sentence(s) "
            f"({'committed' if actually_committed else 'buffered only'}), "
            f"trimmed {trim_samples / 16000:.2f}s"
        )
        return actually_committed

    _INTERIM_PENDING_MAX = 200

    def _buffer_interim_fragment(self, text: str):
        """Hold a short fragment until the next real sentence absorbs it.

        Bounded and tail-deduplicated: a re-recognized fragment must not be
        appended twice, and a buffer that somehow keeps growing must not grow
        without limit.
        """
        pending = self._interim_pending
        if pending.endswith(text):
            log.debug(f"Interim fragment already buffered, skipping: '{text}'")
            return
        pending += text
        if len(pending) > self._INTERIM_PENDING_MAX:
            dropped = len(pending) - self._INTERIM_PENDING_MAX
            pending = pending[-self._INTERIM_PENDING_MAX:]
            log.debug(f"Interim pending buffer capped; dropped {dropped} oldest chars")
        self._interim_pending = pending
        log.debug(f"Interim short utterance buffered: '{text}', pending='{pending}'")

    def _reset_interim_state(self):
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""

    def _process_segment_text(self, text: str, source_lang: str, asr_ms: float = 0,
                              work_id=None, generation=None, expected_session=None):
        """Output a text result (from interim or final) — similar to
        _process_segment but skips ASR. ``generation`` guards the stale-
        segment case exactly as in _process_segment, checked at entry and
        again inside the session boundary fence right before the write (an
        interim sentence committed by a queue item whose session ended while
        a newer one began must not land in the newer meeting — the interim
        pass that recognized it may itself have taken seconds);
        ``expected_session`` is the writer-side identity check for the same
        purpose.

        Returns True when the sentence was accounted for — recorded,
        displayed, filtered as noise, or failed-but-closed-out — and False
        only when an identity guard refused it (stale session generation at
        entry or at the fence re-check, or the writer's session mismatch):
        the caller (the interim path) must treat a refused sentence as
        *not consumed*, so its audio stays in the VAD buffer for a later
        pass instead of being trimmed away.
        """
        if generation is not None and generation != self._session_generation:
            log.info(
                "Dropping interim text from superseded session generation %s",
                generation,
            )
            return False
        original_text = text.strip()
        if not original_text or not any(c.isalnum() for c in original_text):
            return True

        asr_lang_setting = self._get_asr_language_setting()
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', discarding: {original_text[:60]}")
            return True

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms, interim): {original_text}")

        # (Session-work bookkeeping mirrors _process_segment, including its
        # boundary fence: identity re-check → write → registration →
        # adoption are one linearized section against an explicit begin,
        # because the interim pass that produced this text may have run for
        # seconds after the entry guard above. See _process_segment's fence
        # comment; only short local work runs under the lock.)
        with self._session_boundary_lock:
            if generation is not None and generation != self._session_generation:
                log.info(
                    "Dropping interim text from superseded session generation "
                    "%s (current %s; the boundary moved during the pass)",
                    generation, self._session_generation,
                )
                return False
            msg_generation = (
                generation if generation is not None
                else self._session_generation
            )
            if self._overlay:
                self._overlay.add_message(
                    msg_id, timestamp, original_text, source_lang, asr_ms
                )
            result = self._transcript.write_original(
                msg_id, timestamp, original_text, language=source_lang,
                session=expected_session,
            )
            if result == TranscriptWriter.WRITE_SESSION_MISMATCH:
                self._finalize_untranslated(
                    msg_id, "interim text belongs to a closed session",
                    user_visible=False,
                )
                return False
            recorded = result == TranscriptWriter.WRITE_RECORDED
            if recorded:
                self._session_work.register_msg(msg_generation, msg_id)
                # Legacy auto-open adoption — the single authoritative helper,
                # shared with _process_segment. Same fence (RLock re-entry):
                # the adoption decides under the same boundary state the
                # write saw.
                msg_generation = self._adopt_auto_opened_session(
                    msg_id, msg_generation
                )

        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language
        extra_langs = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            # No translation job will exist: release the msg count here
            # (see _process_segment's same-language branch), under the
            # generation it was registered in.
            self._session_work.release_msg(msg_generation, msg_id)
            self._transcript.finalize_no_translation(
                msg_id, session=expected_session
            )
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(self._asr_count, self._translate_count, self._total_prompt_tokens, self._total_completion_tokens, self._compute_cost())
            if self._subwin and self._subwin.isVisible():
                if extra_langs:
                    try:
                        self._extra_tl_executor.submit(self._translate_subwin_only, original_text, source_lang, extra_langs)
                    except RuntimeError:
                        pass
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            try:
                self._submit_translation(
                    msg_id, original_text, source_lang, extra_langs or None,
                    session_generation=msg_generation,
                    expected_session=expected_session,
                )
            except TranslationUnavailable as exc:
                self._session_work.release_msg(msg_generation, msg_id)
                self._finalize_untranslated(msg_id, str(exc), user_visible=True)
            except RuntimeError:
                # Executor already shut down (we are exiting) — not worth a
                # user-facing error, but still has to be closed out.
                self._session_work.release_msg(msg_generation, msg_id)
                self._finalize_untranslated(
                    msg_id, "translation executor shut down", user_visible=False
                )
        return True

    def _process_interim_final(self, speech_segment, work_id=None, generation=None,
                               expected_session=None):
        """Handle VAD flush after interim outputs were already made."""
        seg_len = len(speech_segment) / 16000
        log.info(f"Interim final segment: {seg_len:.1f}s")

        try:
            result, asr_ms = self._run_asr(speech_segment, "interim_final")
        except Exception as e:
            log.error(f"Interim final ASR error: {e}", exc_info=True)
            return
        if asr_ms == 0:
            return

        if result is None:
            # Flush any remaining pending
            if self._interim_pending:
                text = self._interim_pending
                self._interim_pending = ""
                lang = self._get_asr_language_setting()
                if lang == "auto":
                    lang = "unknown"
                self._process_segment_text(text, lang, work_id=work_id,
                                            generation=generation,
                                            expected_session=expected_session)
            return

        validated = validate_asr_result(result, "interim_final")
        if validated is None:
            original_text, result_lang = "", ""
        else:
            original_text, result_lang = validated

        # Strip echo from previous commit's overlap
        original_text = self._strip_committed_overlap(original_text)

        # Take the buffered fragments now, but keep them separate: the noise
        # filter below judges *this* segment's own recognition, and those
        # fragments came from earlier audio that already passed it. Folding them
        # in first meant a short reply ("да") followed by a quiet tail was
        # discarded along with the noise — the user said something and nothing
        # ever appeared.
        pending = self._interim_pending
        self._interim_pending = ""

        if original_text:
            alnum_chars = sum(1 for c in original_text if c.isalnum())
            if seg_len >= 2.0 and alnum_chars <= 3:
                log.debug(
                    f"Noise filter: {seg_len:.1f}s segment produced only "
                    f"'{original_text}', dropping it"
                )
                original_text = ""

        original_text = pending + original_text
        if not original_text or not any(c.isalnum() for c in original_text):
            return

        if not result_lang:
            result_lang = self._get_asr_language_setting()
            if result_lang == "auto":
                result_lang = "unknown"
        self._process_segment_text(original_text, result_lang, asr_ms,
                                   work_id=work_id, generation=generation,
                                   expected_session=expected_session)

    def _capture_loop(self):
        silence_chunk = np.zeros(
            int(
                self._config["audio"]["sample_rate"]
                * self._config["audio"]["chunk_duration"]
            ),
            dtype=np.float32,
        )
        last_discarded = self._vad.discarded_segments
        while self._running:
            try:
                item = self._audio.get_audio(timeout=1.0)
            except CaptureRuntimeError as exc:
                log.error("Audio capture stopped unexpectedly: %s", exc)
                self._running = False
                try:
                    self._audio.stop()
                except Exception as stop_exc:
                    log.warning("Failed to clean up audio capture: %s", stop_exc)
                break
            if item is None:
                # Silence-feed branch (mid-utterance): runs under the session
                # boundary lock like the main chunk path, so the fence covers
                # every producer of the queue.
                with self._session_boundary_lock:
                    if (
                        self._vad._is_speaking
                        and not self._paused
                        and not self._session_end_gating
                    ):
                        n = self._vad._get_effective_silence_limit() + 1
                        for _ in range(n):
                            with self._vad_lock:
                                seg = self._vad.process_chunk(silence_chunk)
                            if seg is not None and self._asr_ready:
                                self._enqueue_asr("vad_flush", seg)
                                break
                continue

            chunk, mic_rms = item

            if self._paused:
                continue

            # np.dot avoids allocating a temporary squared array for every
            # 32 ms audio block on the real-time capture thread.
            rms = float(np.sqrt(np.dot(chunk, chunk) / max(chunk.size, 1)))

            if self._overlay:
                self._overlay.update_monitor(rms, self._vad.last_confidence, mic_rms)

            # Producer fence: the gate check, the VAD step and the enqueue
            # run under the session boundary lock, the same lock the end
            # thread holds when it raises the gate and flips the generation
            # to CLOSING. Speech the VAD accepted before the end is therefore
            # enqueued (and counted) before the close can start — the last
            # utterance is kept, never silently dropped; audio after the end
            # never reaches the VAD at all.
            with self._session_boundary_lock:
                if self._session_end_gating:
                    # Session ENDING: no new audio into VAD for the closing
                    # meeting. The monitor above still updates (the user sees
                    # the level); the buffer is left to the ENDING thread's
                    # flush.
                    continue

                with self._vad_lock:
                    speech_segment = self._vad.process_chunk(chunk)
                    speaking = self._vad._is_speaking
                    discarded = self._vad.discarded_segments

                if discarded != last_discarded:
                    # The density filter dropped a segment. It emits nothing, so
                    # without this the utterance's interim state (pending fragments
                    # and the echo tail) would leak into the next one.
                    last_discarded = discarded
                    if self._interim_active or self._interim_pending:
                        log.debug("Low-density segment discarded; resetting interim state")
                    self._reset_interim_state()

                if speaking or speech_segment is not None:
                    self._last_speech_activity = time.monotonic()

                if speech_segment is None:
                    # Still accumulating — check for interim ASR
                    # Unlocked reads by design: these only throttle *whether* to
                    # consider an interim pass. _do_interim_asr re-reads the buffer
                    # under _vad_lock, so a torn read here costs at most one extra
                    # or skipped poll (see VADProcessor's class docstring).
                    if (self._incremental_enabled and self._asr_ready
                            and self._vad._is_speaking):
                        buf_samples = self._vad._speech_samples
                        total_dur = buf_samples / 16000
                        elapsed = (buf_samples - self._last_interim_samples) / 16000
                        now = time.perf_counter()
                        cooldown = now - self._last_interim_check_time
                        if total_dur >= self._interim_interval and elapsed >= self._interim_interval and cooldown >= 1.0:
                            self._last_interim_check_time = now
                            self._enqueue_asr("interim", None)
                    continue

                if not self._asr_ready:
                    log.debug("ASR not ready, dropping segment")
                    continue

                self._enqueue_asr("vad_flush", speech_segment)

    def _enqueue_asr(self, seg_type: str, segment):
        """Queue a segment for the ASR thread, dropping the oldest on overflow.

        Never raises: this runs on the capture thread, which has no outer
        handler and must not die because the consumer fell behind.

        Called with ``_session_boundary_lock`` held (the producer fence — the
        capture loop holds it from the gate check through VAD to this
        enqueue), so the "segment produced but not yet enqueued when the user
        clicked end" interleaving cannot occur: either the whole
        check→VAD→enqueue ran before the end's gate+CLOSING (the segment is
        registered under the still-OPEN generation and kept), or the gate is
        already up and the audio never reached the VAD. The ENDING flush
        goes through _enqueue_final_segment instead.

        The item carries both the registration generation and the writer's
        session stamp, snapshotted together under the same fence, so the
        ASR result can be identity-checked at write time: a session that
        ended (and possibly re-opened as a new one) between enqueue and
        recognition can never receive the old audio's entries.
        """
        if self._stop_event.is_set():
            return
        generation, expected_session = self._session_snapshot()
        decision = self._session_work.admit(generation)
        if decision == "drop":
            # Unreachable under the producer fence (the gate precedes any
            # CLOSING enqueue); kept as a defensive guard for stray callers.
            log.debug("Dropping %s segment for a closing/closed session", seg_type)
            return
        # Both the session-tracked item ("register") and the subtitle-only
        # one ("pass" — no session exists yet) carry a work count: a legacy
        # auto-open session adopted moments later reuses this same
        # generation (adoption never bumps it — see
        # _adopt_auto_opened_session), so items already in flight are that
        # meeting's opening speech and the ENDING wait must cover them.
        # register() auto-creates an unknown generation as OPEN; a
        # CLOSING/SUPERSEDED one refuses, and only then does a "register"
        # item stay dropped while a "pass" item falls back to uncounted.
        work_id = self._next_session_work_id()
        if not self._session_work.register(generation, work_id, seg_type):
            if decision == "register":
                return
            work_id = None
        item = (seg_type, segment, work_id, generation, expected_session)
        try:
            self._asr_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            dropped = self._asr_queue.get_nowait()
        except queue.Empty:
            dropped = None
        if dropped is None:
            # Either the queue drained underneath us, or we just pulled out the
            # stop sentinel. Put the sentinel back rather than swallowing it.
            self._requeue_stop_sentinel()
        else:
            log.warning(f"ASR queue full, dropped {dropped[0]} segment")
            self._release_queued_work(dropped)
        try:
            self._asr_queue.put_nowait(item)
        except queue.Full:
            log.warning("ASR queue still full after drop, skipping segment")
            self._release_queued_work(item)

    def _release_queued_work(self, item) -> None:
        """Release the work count of a queue item that will never be
        processed (dropped on overflow, or requeued mid-handoff). Keyed to
        the generation the item carries, so a begin/end racing the drop
        releases the count in the session it was counted under."""
        if not isinstance(item, tuple) or len(item) < 4:
            return
        work_id, generation = item[2], item[3]
        if work_id is not None:
            self._session_work.release(generation, work_id)

    def _requeue_stop_sentinel(self):
        if not self._stop_event.is_set():
            return
        try:
            self._asr_queue.put_nowait(None)
        except queue.Full:
            pass

    def _asr_loop(self):
        while self._running and not self._stop_event.is_set():
            try:
                # 1s, unchanged: the idle branch below paces the worker-recycle
                # check, and stop() no longer depends on this timeout — the loop
                # condition sees _stop_event, and the sentinel usually wakes it
                # immediately anyway.
                item = self._asr_queue.get(timeout=1.0)
            except queue.Empty:
                # Idle moment: recycle a bloated worker while no audio is waiting.
                # Guarded so an unexpected error can never kill this thread (which
                # would itself silence ASR permanently).
                try:
                    self._maybe_recycle_asr_worker()
                except Exception:
                    log.error("ASR worker recycle check failed", exc_info=True)
                continue

            if item is None:
                break

            seg_type, segment, work_id, work_generation, expected_session = item

            # One bad result must not take the ASR thread down with it: without
            # this the pipeline goes permanently silent while capture keeps
            # filling a queue nobody drains, and no worker-restart path fires.
            # The finally also guarantees the session-work count of this queue
            # item is released no matter which terminal path the processing
            # took (success, filters, exceptions, superseded session) — and
            # only once, because release() is idempotent on absent ids.
            # The release is keyed to the generation the item was registered
            # under (work_generation), so an end+begin racing this item can
            # neither steal its release from the old session nor leave the
            # new one waiting on it.
            try:
                if seg_type == "vad_flush":
                    try:
                        if self._interim_active:
                            self._process_interim_final(
                                segment, work_id, work_generation, expected_session
                            )
                        else:
                            self._process_segment(
                                segment, work_id, work_generation, expected_session
                            )
                    finally:
                        self._reset_interim_state()
                elif seg_type == "interim":
                    self._drain_interim_duplicates()
                    # Each committed sentence registers its own msg count in
                    # _process_segment_text (keyed to this item's
                    # generation); this queue item's work_id is released by
                    # the finally below, so no hand-off here.
                    self._do_interim_asr(work_generation, expected_session)
                    with self._vad_lock:
                        self._last_interim_samples = self._vad._speech_samples
            except ASRProtocolError as exc:
                log.error("ASR contract violation (%s): %s", seg_type, exc)
            except Exception:
                seg_len = len(segment) / 16000 if segment is not None else 0.0
                log.error(
                    "Unhandled error processing %s segment (%.1fs); dropping it "
                    "and continuing", seg_type, seg_len, exc_info=True,
                )
            finally:
                if work_id is not None:
                    self._session_work.release(work_generation, work_id)

    def _drain_interim_duplicates(self):
        while True:
            try:
                item = self._asr_queue.get_nowait()
            except queue.Empty:
                break
            if item is None or item[0] != "interim":
                # put_nowait, not put: this runs on the ASR thread, which is the
                # only consumer — a blocking put on a full queue would deadlock.
                try:
                    self._asr_queue.put_nowait(item)
                except queue.Full:
                    if item is None:
                        self._stop_event.set()
                    else:
                        # The requeue failed: the item is out of the queue and
                        # will never be processed, so its work count must be
                        # released here or an ENDING wait would block on it
                        # until the timeout.
                        log.warning(
                            "ASR queue full while requeueing %s; dropping it "
                            "and releasing its session work", item[0],
                        )
                        self._release_queued_work(item)
                break
            # A duplicate interim request dropped before processing: its
            # work count must go or ENDING would wait for it forever.
            self._release_queued_work(item)


def main():
    setup_logging()
    log.info("LiveTranslate starting...")
    config = load_config()
    config.setdefault("asr", {})
    config["asr"].setdefault("asr_engine", "funasr")
    config["asr"].setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    config["asr"].setdefault("gigaam_model", DEFAULT_GIGAAM_MODEL)
    saved = _load_saved_settings()
    migrate_funasr_settings(saved)

    # Log actual effective config
    _asr_eng = (saved or {}).get("asr_engine", config["asr"].get("asr_engine", "funasr"))
    _funasr_model = (saved or {}).get(
        "funasr_model", config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
    )
    _gigaam_model = (saved or {}).get(
        "gigaam_model", config["asr"].get("gigaam_model", DEFAULT_GIGAAM_MODEL)
    )
    _active_idx = (saved or {}).get("active_model", 0)
    _models = (saved or {}).get("models", [])
    if 0 <= _active_idx < len(_models):
        _m = _models[_active_idx]
        _model_info = f"{_m.get('name', '?')} ({_m.get('model', '?')})"
    else:
        _model_info = f"{config['translation']['model']} (default)"
    if _asr_eng == "funasr":
        log.info(
            f"Config loaded: ASR={_asr_eng}/{_funasr_model}, "
            f"Translator={_model_info}"
        )
    elif _asr_eng == "gigaam":
        log.info(
            f"Config loaded: ASR={_asr_eng}/{_gigaam_model}, "
            f"Translator={_model_info}"
        )
    else:
        log.info(f"Config loaded: ASR={_asr_eng}, Translator={_model_info}")

    # Apply UI language before creating any widgets
    if saved and saved.get("ui_lang"):
        set_lang(saved["ui_lang"])

    os.environ["QT_LOGGING_RULES"] = (
        "qt.text.font.db=false;qt.qpa.fonts.warning=false"
    )
    app = QApplication(sys.argv)
    apply_theme(app)
    dock_visible = bool((saved or {}).get("dock_visible", True))
    configure_application(app, dock_visible=dock_visible)
    # Pin the platform UI family when Qt has it available; this also avoids
    # bitmap fallback fonts on older desktop environments.
    ui_font = default_ui_font_family()
    if ui_font in QFontDatabase.families():
        app.setFont(QFont(ui_font, 9))
    _app_icon = create_app_icon()
    app.setWindowIcon(_app_icon)

    # First launch → setup wizard (hub + download) → configure translation API
    if not SETTINGS_FILE.exists():
        wizard = SetupWizardDialog()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        saved = _load_saved_settings()
        log.info("Setup wizard completed")

        # Prompt user to configure translation API
        from dialogs import ModelEditDialog

        info = QMessageBox(
            QMessageBox.Icon.Information,
            t("window_setup"),
            t("setup_api_hint"),
        )
        info.exec()

        from mlx_service import hy_mt_model_config

        default_model_data = (
            hy_mt_model_config()
            if sys.platform == "darwin" and MLXServiceManager().is_model_ready()
            else {
                "name": "hunyuan-mt-chimera-7b",
                "api_base": DEFAULT_TRANSLATION_API_BASE,
                "api_key": translation_api_key(""),
                "model": "hunyuan-mt-chimera-7b",
            }
        )
        dlg = ModelEditDialog(None, default_model_data)
        dlg.setWindowTitle(t("setup_api_title"))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if data.get("api_key"):
                saved["models"] = [data]
                saved["active_model"] = 0
                _save_settings(saved)
                log.info(f"Translation API configured: {data['name']}")
        # If user skips, ControlPanel will create default placeholder from config.yaml

    # Non-first launch but models missing → download dialog
    else:
        saved = saved or {}
        current_engine = saved.get("asr_engine", config["asr"].get("asr_engine", "funasr"))
        if current_engine == "funasr":
            cache_key = saved.get("funasr_model", config["asr"].get("funasr_model"))
        elif current_engine == "gigaam":
            cache_key = saved.get("gigaam_model", config["asr"].get("gigaam_model"))
        else:
            cache_key = saved.get(
                "whisper_model_size", config["asr"]["model_size"]
            )
        missing = get_missing_models(
            current_engine,
            cache_key,
            saved.get("hub", "ms"),
        )
        if missing:
            log.info(f"Missing models: {[m['name'] for m in missing]}")
            dlg = ModelDownloadDialog(
                missing,
                hub=saved.get("hub", "ms"),
                proxy=saved.get("download_proxy", "system"),
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                sys.exit(0)

    log_window = LogWindow()
    log_handler = log_window.get_handler()
    logging.getLogger().addHandler(log_handler)

    panel = ControlPanel(config, saved_settings=saved)

    overlay = SubtitleOverlay(config["subtitle"])
    if saved:
        ox = saved.get("overlay_x")
        oy = saved.get("overlay_y")
        ow = saved.get("overlay_w")
        oh = saved.get("overlay_h")
        if ox is not None and oy is not None:
            if SubtitleWindow._is_pos_visible(ox, oy):
                overlay.move(ox, oy)
            else:
                screen = QApplication.primaryScreen()
                geo = screen.availableGeometry()
                overlay.move(geo.right() - overlay.width() - 20, geo.bottom() - overlay.height() - 60)
        if ow and oh:
            overlay.resize(ow, oh)
    overlay.show()

    # Subtitle window
    subwin_cfg = (saved or {}).get("subtitle_mode")
    subwin = SubtitleWindow(subwin_cfg)
    subwin_was_enabled = (subwin_cfg or {}).get("enabled", False)

    live_trans = LiveTranslateApp(config)
    # Application-level registry for AI-summary workers: strong references
    # past page destruction, one cancel point at app exit. Held on the app
    # object as well as this frame so it cannot be garbage-collected while
    # a worker is still running (main()'s frame dies at process exit, but
    # between quit() and interpreter teardown the registry must stay alive).
    from summary_task_registry import SummaryTaskRegistry

    summary_registry = SummaryTaskRegistry()
    live_trans._summary_task_registry = summary_registry
    live_trans.set_overlay(overlay)
    live_trans.set_subtitle_window(subwin)
    live_trans.set_panel(panel)
    # The records page marks the session currently being recorded; it reads
    # that from the writer the app owns (the panel is built before the app).
    # Injected through the panel's public API — no reaching into fields.
    panel.set_transcript_writer(live_trans._transcript)
    panel.set_summary_registry(summary_registry)

    # --- meeting-session state bridge -----------------------------------
    # Background-thread transitions land on the Qt loop through a queued
    # signal, never by touching widgets from the ENDING thread.
    class _SessionBridge(QObject):
        state_changed = pyqtSignal(str, object, dict)

    session_bridge = _SessionBridge()

    def _on_session_state_signal(state: str, session_id, summary):
        live_trans._on_session_state_ui(state, session_id)
        if state == SessionState.IDLE:
            # Meeting over, pipeline stays paused: the tray and the overlay's
            # run/pause button reflect that, so a later "resume" is a
            # deliberate act rather than a leftover "Running".
            if live_trans._running and live_trans._paused:
                _is_running[0] = False
                overlay.set_running(False)
                pause_action.setText(t("tray_resume"))

    session_bridge.state_changed.connect(_on_session_state_signal)
    live_trans.set_session_ui_callback(
        lambda state, session_id, summary=None:
        session_bridge.state_changed.emit(state, session_id, summary or {})
    )
    # Fallback only: every deliberate exit goes through on_quit() below, which
    # has already run stop(). stop() is idempotent, so a window-manager quit or
    # an unhandled exit still reclaims threads, files, worker and MLX service.
    app.aboutToQuit.connect(live_trans.stop)

    def _deferred_init():
        panel._apply_settings()
        models = panel.get_settings().get("models", [])
        active_idx = panel.get_settings().get("active_model", 0)
        overlay.set_models(models, active_idx)
        target = panel.get_settings().get("target_language", "zh")
        overlay.set_target_language(target)
        asr_lang = panel.get_settings().get("asr_language", "auto")
        overlay.set_source_language(asr_lang)
        style = panel.get_settings().get("style")
        if style:
            overlay.apply_style(style)
        active_model = panel.get_active_model()
        if active_model:
            if is_hy_mt_model(active_model) and not live_trans._mlx_service.is_running():
                # Do not let the constructor's generic translator handle audio
                # while the managed local service is still booting.
                live_trans._disable_translator()
                if panel.auto_start_mlx_service():
                    log.info("HY-MT is prepared; starting local service automatically")
                else:
                    log.info("HY-MT is selected but its local service is unavailable")
            else:
                live_trans._on_model_changed(active_model)

    QTimer.singleShot(100, _deferred_init)

    tray = QSystemTrayIcon()
    tray.setToolTip(t("tray_tooltip"))
    tray.setIcon(_app_icon)

    menu = QMenu()

    # --- Pause / Resume toggle ---
    pause_action = QAction(t("tray_resume"))
    # The pipeline only starts on the deferred callback below, so it is not
    # running yet — claiming otherwise made the tray and overlay lie for the
    # first 500ms and let that callback overwrite a pause the user got in first.
    _is_running = [False]  # mutable for closure
    _start_cancelled = [False]
    _quitting = [False]
    _cache_delete_thread = []
    overlay.set_running(False)

    def on_start():
        if _start_cancelled[0]:
            log.info("Deferred start skipped: paused or quit before it ran")
            return
        try:
            live_trans.start()
        except Exception as e:
            log.error(f"Start error: {e}", exc_info=True)
            overlay.set_running(False)
            _is_running[0] = False
            pause_action.setText(t("tray_resume"))
            QMessageBox.warning(
                panel,
                t("error_title"),
                t("error_audio_start").format(error=_audio_start_error(e)),
            )
            return
        # Only after a successful start, and only if nothing cancelled us while
        # start() was blocking on the audio device.
        if _start_cancelled[0]:
            log.info("Pipeline start superseded by a pause/quit; stopping again")
            live_trans.stop()
            return
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))

    def on_pause():
        # During ENDING the close owns the pipeline's quiet window: pausing
        # now would flush VAD a second time and race the end thread's own
        # flush. The end is bounded; the button works again right after.
        if live_trans.session_state() == SessionState.ENDING:
            log.debug("Pause ignored while a recording session is ending")
            return
        _start_cancelled[0] = True
        live_trans.pause()
        overlay.set_running(False)
        _is_running[0] = False
        pause_action.setText(t("tray_resume"))
        # Pipeline pause also pauses the meeting record (same session; the
        # session-state machine is the single authority, so the state change
        # goes through it, not by poking the overlay directly).
        if live_trans.session_state() == SessionState.ACTIVE:
            live_trans._notify_session_state(SessionState.PAUSED)

    def on_resume():
        # During ENDING the close owns the pipeline state; resuming now would
        # feed new audio into the session being finalized.
        if live_trans.session_state() == SessionState.ENDING:
            log.debug("Resume ignored while a recording session is ending")
            return
        _start_cancelled[0] = False
        if not live_trans._running:
            on_start()
            return
        # Resuming the pipeline with no recording session: the pipeline would
        # recognise and call the translation API while nothing is recorded.
        # Offer to start a new recording in the same click; declining resumes
        # plain (subtitles only, nothing recorded).
        if live_trans.session_state() == SessionState.IDLE:
            answer = QMessageBox.question(
                overlay,
                t("session_resume_no_session_title"),
                t("session_resume_no_session_msg"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                if live_trans.begin_recording_session():
                    overlay.set_running(True)
                    _is_running[0] = True
                    pause_action.setText(t("tray_pause"))
                    return
                # Session could not open; stay paused rather than run
                # unrecorded.
                QMessageBox.warning(
                    overlay, t("error_title"), t("session_start_unavailable")
                )
                return
        live_trans.resume()
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))
        if live_trans.session_state() == SessionState.PAUSED:
            live_trans._notify_session_state(SessionState.ACTIVE)

    def on_toggle_pause():
        if _is_running[0]:
            on_pause()
        else:
            on_resume()

    pause_action.triggered.connect(on_toggle_pause)
    menu.addAction(pause_action)
    menu.addSeparator()

    # --- Show/hide overlay ---
    overlay_toggle_action = QAction(t("tray_hide_overlay"))

    _hide_notified = [False]

    def on_toggle_overlay():
        if overlay.isVisible():
            overlay.hide()
            overlay_toggle_action.setText(t("tray_show_overlay"))
            if not _hide_notified[0]:
                _hide_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("hide_tray_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            overlay.show()
            overlay.raise_()
            overlay_toggle_action.setText(t("tray_hide_overlay"))

    overlay_toggle_action.triggered.connect(on_toggle_overlay)
    menu.addAction(overlay_toggle_action)

    # --- Subtitle window toggle ---
    def _save_overlay_pos():
        settings = panel.get_settings()
        pos = overlay.pos()
        size = overlay.size()
        settings["overlay_x"] = pos.x()
        settings["overlay_y"] = pos.y()
        settings["overlay_w"] = size.width()
        settings["overlay_h"] = size.height()
        panel._current_settings.update({
            "overlay_x": pos.x(), "overlay_y": pos.y(),
            "overlay_w": size.width(), "overlay_h": size.height(),
        })
        _save_settings(settings)

    overlay.position_changed.connect(_save_overlay_pos)

    subwin_toggle_action = QAction(t("subwin_show"), checkable=True)

    def _save_subwin_state():
        settings = panel.get_settings()
        sm = settings.get("subtitle_mode") or {}
        sm["enabled"] = subwin.isVisible()
        pos = subwin.pos()
        sm["window_x"] = pos.x()
        sm["window_y"] = pos.y()
        settings["subtitle_mode"] = sm
        panel._current_settings["subtitle_mode"] = sm
        _save_settings(settings)

    _subwin_notified = [False]

    def on_toggle_subwin(checked):
        if checked:
            subwin.show()
            subwin.raise_()
            if not _subwin_notified[0]:
                _subwin_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("subwin_drag_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            subwin.hide()
        overlay.set_subtitle_checked(checked)
        _save_subwin_state()

    subwin_toggle_action.toggled.connect(on_toggle_subwin)
    subwin.position_changed.connect(_save_subwin_state)

    # Sync when subtitle window is manually closed (e.g. Alt+F4)
    def _on_subwin_closed():
        subwin_toggle_action.blockSignals(True)
        subwin_toggle_action.setChecked(False)
        subwin_toggle_action.blockSignals(False)
        overlay.set_subtitle_checked(False)
        _save_subwin_state()

    subwin.window_closed.connect(_on_subwin_closed)

    # Restore subtitle window visibility from saved state
    if subwin_was_enabled:
        subwin_toggle_action.setChecked(True)

    menu.addAction(subwin_toggle_action)

    # Quick toggle for subtitle-window click-through (mirrors the settings checkbox).
    subwin_ct_action = QAction(t("subwin_click_through_tray"), checkable=True)
    _subwin_init = panel.get_settings().get("subtitle_mode") or {}
    subwin_ct_action.setChecked(bool(_subwin_init.get("click_through", False)))

    def on_toggle_subwin_ct(checked):
        subwin.set_click_through(checked)
        settings = panel.get_settings()
        sm = settings.get("subtitle_mode") or {}
        sm["click_through"] = checked
        settings["subtitle_mode"] = sm
        panel._current_settings["subtitle_mode"] = sm
        _save_settings(settings)
        w = panel._subtitle_widget
        w._click_through_check.blockSignals(True)
        w._click_through_check.setChecked(checked)
        w._click_through_check.blockSignals(False)
        w._settings["click_through"] = checked

    subwin_ct_action.toggled.connect(on_toggle_subwin_ct)
    menu.addAction(subwin_ct_action)

    # Connect overlay subtitle button
    def _on_overlay_subtitle_toggle():
        subwin_toggle_action.setChecked(not subwin_toggle_action.isChecked())

    overlay.subtitle_toggled.connect(_on_overlay_subtitle_toggle)

    # Connect panel subtitle settings changes
    def _on_panel_subtitle_changed(s):
        subwin.apply_settings(s)
        subwin_ct_action.blockSignals(True)
        subwin_ct_action.setChecked(bool(s.get("click_through", False)))
        subwin_ct_action.blockSignals(False)

    panel.subtitle_settings_changed.connect(_on_panel_subtitle_changed)

    def _on_reset_positions():
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        subwin.move(100, 100)
        _save_subwin_state()
        ow, oh = overlay.width(), overlay.height()
        overlay.move(geo.right() - ow - 50, geo.bottom() - oh - 100)
        _save_overlay_pos()

    panel.reset_positions.connect(_on_reset_positions)

    menu.addSeparator()

    # --- Show log / panel ---
    log_action = QAction(t("tray_show_log"))
    panel_action = QAction(t("tray_show_panel"))

    def on_toggle_log():
        if log_window.isVisible():
            log_window.hide()
        else:
            log_window.show()
            log_window.raise_()

    def on_toggle_panel():
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()
            panel.raise_()

    log_action.triggered.connect(on_toggle_log)
    panel_action.triggered.connect(on_toggle_panel)
    menu.addAction(panel_action)
    menu.addAction(log_action)
    menu.addSeparator()

    # --- Overlay submenu (click-through, topmost, auto-scroll, taskbar) ---
    overlay_menu = QMenu(t("tray_menu_overlay"))

    ct_action = QAction(t("click_through"), checkable=True)
    topmost_action = QAction(t("top_most"), checkable=True)
    topmost_action.setChecked(True)
    autoscroll_action = QAction(t("auto_scroll"), checkable=True)
    autoscroll_action.setChecked(True)
    taskbar_action = QAction(t("taskbar"), checkable=True)
    dock_action = QAction(t("dock_icon"), checkable=True)
    dock_action.setChecked(dock_visible)
    dock_action.setVisible(sys.platform == "darwin")

    # Tray → overlay sync
    ct_action.toggled.connect(lambda v: overlay._handle._ct_check.setChecked(v))
    topmost_action.toggled.connect(
        lambda v: overlay._handle._topmost_check.setChecked(v)
    )
    autoscroll_action.toggled.connect(
        lambda v: overlay._handle._auto_scroll.setChecked(v)
    )
    taskbar_action.toggled.connect(
        lambda v: overlay._handle._taskbar_check.setChecked(v)
    )
    def _set_dock_policy(visible):
        if set_dock_visible(visible) or sys.platform != "darwin":
            settings = panel.get_settings()
            settings["dock_visible"] = bool(visible)
            panel._current_settings["dock_visible"] = bool(visible)
            _save_settings(settings)
    dock_action.toggled.connect(_set_dock_policy)

    # Overlay → tray sync
    overlay._handle.click_through_toggled.connect(lambda v: ct_action.setChecked(v))
    overlay._handle.topmost_toggled.connect(lambda v: topmost_action.setChecked(v))
    overlay._handle.auto_scroll_toggled.connect(
        lambda v: autoscroll_action.setChecked(v)
    )
    overlay._handle.taskbar_toggled.connect(lambda v: taskbar_action.setChecked(v))

    overlay_menu.addAction(ct_action)
    overlay_menu.addAction(topmost_action)
    overlay_menu.addAction(autoscroll_action)
    overlay_menu.addAction(taskbar_action)
    overlay_menu.addAction(dock_action)
    menu.addMenu(overlay_menu)

    # --- Model submenu ---
    model_menu = QMenu(t("tray_menu_model"))
    model_action_group = QActionGroup(model_menu)
    model_action_group.setExclusive(True)

    def _rebuild_model_menu():
        for a in model_action_group.actions():
            model_action_group.removeAction(a)
        model_menu.clear()
        settings = panel.get_settings()
        models = settings.get("models", [])
        active = settings.get("active_model", 0)
        for i, m in enumerate(models):
            name = m.get("name", m.get("model", "?"))
            action = QAction(name, checkable=True)
            if i == active:
                action.setChecked(True)
            model_action_group.addAction(action)
            action.triggered.connect(lambda checked, idx=i: _on_tray_model_switch(idx))
            model_menu.addAction(action)

    def _on_tray_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
            overlay.set_models(models, index)

    def on_overlay_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
        _rebuild_model_menu()

    model_menu.aboutToShow.connect(_rebuild_model_menu)
    menu.addMenu(model_menu)

    # --- Target language submenu ---
    lang_menu = QMenu(t("tray_menu_target_lang"))
    lang_action_group = QActionGroup(lang_menu)
    lang_action_group.setExclusive(True)
    _lang_actions = {}
    lang_more_menu = QMenu(t("tray_more_langs"))

    for code, native in LANGUAGES:
        if code == "auto":
            continue
        action = QAction(f"{code} - {native}", checkable=True)
        lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, lc=code: _on_tray_lang_switch(lc))
        if code in COMMON_LANG_CODES:
            lang_menu.addAction(action)
        else:
            lang_more_menu.addAction(action)
        _lang_actions[code] = action

    lang_menu.addMenu(lang_more_menu)

    current_target = panel.get_settings().get("target_language", "zh")
    if current_target in _lang_actions:
        _lang_actions[current_target].setChecked(True)

    def _on_tray_lang_switch(lang_code):
        overlay.set_target_language(lang_code)
        live_trans._on_target_language_changed(lang_code)
        from control_panel import _save_settings

        settings = panel.get_settings()
        settings["target_language"] = lang_code
        panel._current_settings["target_language"] = lang_code
        _save_settings(settings)

    # Overlay → tray lang sync
    def _on_overlay_lang_changed(lang_code):
        if lang_code in _lang_actions:
            _lang_actions[lang_code].setChecked(True)

    overlay.target_language_changed.connect(_on_overlay_lang_changed)

    menu.addMenu(lang_menu)

    # --- ASR language hint submenu ---
    asr_lang_menu = QMenu(t("tray_menu_asr_lang"))
    asr_lang_action_group = QActionGroup(asr_lang_menu)
    asr_lang_action_group.setExclusive(True)
    _asr_lang_actions = {}
    asr_more_menu = QMenu(t("tray_more_langs"))

    for code, native in LANGUAGES:
        label = t("asr_lang_auto") if code == "auto" else native
        action = QAction(f"{code} - {label}", checkable=True)
        action.setData(code)
        asr_lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, c=code: _on_tray_asr_lang(c))
        if code in COMMON_LANG_CODES:
            asr_lang_menu.addAction(action)
        else:
            asr_more_menu.addAction(action)
        _asr_lang_actions[code] = action

    asr_lang_menu.addMenu(asr_more_menu)

    def _gigaam_locks_ru(index=None):
        """True when the GigaAM engine + selected variant is Russian-only."""
        is_gigaam = (panel._asr_engine.currentIndex() if index is None else index) == 3
        if not is_gigaam:
            return False
        combo = getattr(panel, "_gigaam_model_combo", None)
        return gigaam_is_russian_only(combo.currentData() if combo else None)

    def _sync_asr_language_controls(index=None):
        """Keep every source-language entry point consistent with GigaAM."""
        lock_ru = _gigaam_locks_ru(index)
        if lock_ru:
            ru_idx = panel._asr_lang.findData("ru")
            if ru_idx >= 0 and panel._asr_lang.currentData() != "ru":
                panel._asr_lang.blockSignals(True)
                panel._asr_lang.setCurrentIndex(ru_idx)
                panel._asr_lang.blockSignals(False)
            overlay.set_source_language("ru")
        panel._asr_lang.setEnabled(not lock_ru)
        overlay.set_source_language_enabled(not lock_ru)
        selected = "ru" if lock_ru else (panel._asr_lang.currentData() or "auto")
        for action in _asr_lang_actions.values():
            action.setEnabled(not lock_ru)
            action.setChecked(action.data() == selected)

    panel._asr_engine.currentIndexChanged.connect(_sync_asr_language_controls)
    if hasattr(panel, "_gigaam_model_combo"):
        panel._gigaam_model_combo.currentIndexChanged.connect(
            lambda _idx: _sync_asr_language_controls()
        )
    _sync_asr_language_controls()

    current_asr_lang = (
        "ru"
        if _gigaam_locks_ru()
        else panel.get_settings().get("asr_language", "auto")
    )
    if current_asr_lang in _asr_lang_actions:
        _asr_lang_actions[current_asr_lang].setChecked(True)

    def _on_tray_asr_lang(code):
        from control_panel import _save_settings

        if _gigaam_locks_ru():
            code = "ru"
        live_trans._set_asr_language(code)
        settings = panel.get_settings()
        settings["asr_language"] = code
        panel._current_settings["asr_language"] = code
        _save_settings(settings)
        # Sync control panel combo
        idx = panel._asr_lang.findData(code)
        if idx >= 0:
            panel._asr_lang.blockSignals(True)
            panel._asr_lang.setCurrentIndex(idx)
            panel._asr_lang.blockSignals(False)

    menu.addMenu(asr_lang_menu)
    menu.addSeparator()

    # --- Export submenu ---
    export_menu = QMenu(t("export_menu"))
    export_orig_action = QAction(t("export_original"))
    export_trans_action = QAction(t("export_translation"))
    export_all_action = QAction(t("export_all"))
    export_orig_action.triggered.connect(lambda: overlay.export_messages("original", parent=panel))
    export_trans_action.triggered.connect(lambda: overlay.export_messages("translation", parent=panel))
    export_all_action.triggered.connect(lambda: overlay.export_messages("both", parent=panel))
    export_menu.addAction(export_orig_action)
    export_menu.addAction(export_trans_action)
    export_menu.addAction(export_all_action)
    menu.addMenu(export_menu)
    menu.addSeparator()

    # --- Quit ---
    quit_action = QAction(t("quit"))

    def on_quit():
        """The single exit path. Tray, overlay, panel and SIGINT all land here."""
        if _quitting[0]:
            log.debug("Quit already in progress; ignoring repeat request")
            return
        _quitting[0] = True
        _start_cancelled[0] = True
        try:
            live_trans.stop()
        except Exception:
            log.error("Pipeline stop failed during quit", exc_info=True)
        try:
            # Cooperative cancel of any AI-summary worker still draining:
            # flags set, transports closed, no terminate() and no GUI-thread
            # wait. A worker mid-request surfaces the closed-transport error
            # promptly; a worker on a hung socket is bounded by its 120s
            # request timeout (see SummaryTaskRegistry.cancel_all).
            summary_registry.cancel_all()
        except Exception:
            log.error("Summary worker cancel failed during quit", exc_info=True)
        try:
            panel.stop_background_tasks()
        except Exception:
            log.error("Panel task cleanup failed during quit", exc_info=True)
        app.quit()

    def on_delete_cache_and_quit(entries):
        """Stop everything, then delete the model cache, then exit.

        Order matters: the ASR worker holds model files open, and on Windows a
        directory with an open handle simply refuses to go away.
        """
        if _quitting[0]:
            return
        _quitting[0] = True
        _start_cancelled[0] = True

        busy = QProgressDialog(t("cache_stopping"), "", 0, 0, panel)
        busy.setWindowTitle(t("dialog_delete_title"))
        busy.setCancelButton(None)
        busy.setWindowModality(Qt.WindowModality.ApplicationModal)
        busy.show()
        QApplication.processEvents()

        try:
            live_trans.stop()
        except Exception:
            log.error("Pipeline stop failed before cache delete", exc_info=True)
        if not live_trans.wait_until_stopped():
            log.warning(
                "ASR worker still present after stop; cache delete may fail"
            )

        busy.setLabelText(t("cache_deleting"))
        QApplication.processEvents()

        thread = _CacheDeleteThread([path for _, path, _ in entries])
        _cache_delete_thread.append(thread)  # keep a reference alive

        def _delete_done(failures):
            busy.close()
            if failures:
                QMessageBox.critical(
                    panel,
                    t("error_title"),
                    t("cache_delete_failed").format(
                        paths="\n".join(f"{p}: {e}" for p, e in failures)
                    ),
                )
            app.quit()

        thread.done.connect(_delete_done)
        thread.start()

    panel.delete_cache_and_quit_requested.connect(on_delete_cache_and_quit)

    quit_action.triggered.connect(on_quit)
    menu.addAction(quit_action)

    # --- Connect overlay signals ---
    overlay.settings_requested.connect(on_toggle_panel)
    overlay.target_language_changed.connect(live_trans._on_target_language_changed)

    # --- Meeting-session lifecycle buttons (overlay + records page) -----
    # Both entry points call the same app-level methods; neither keeps its
    # own session state.

    def _confirm_end_session() -> bool:
        """The one confirmation dialog for ending a recording session."""
        return QMessageBox.question(
            overlay,
            t("session_end_confirm_title"),
            t("session_end_confirm_msg"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def _on_session_toggle():
        state = live_trans.session_state()
        if state == SessionState.ENDING:
            return  # the close is running; the button is disabled anyway
        if SessionState.is_recording(state):
            # Fix the target *before* the dialog: the confirmation pumps a
            # nested event loop, and an end plus a new begin can land under
            # it — without the identity, the toggle after the dialog would
            # end whichever meeting happens to be active then.
            target = live_trans._transcript.active_session()
            if not _confirm_end_session():
                return
            if not live_trans.end_recording_session(expected_session=target):
                QMessageBox.information(
                    overlay, t("error_title"), t("session_end_target_gone")
                )
            return
        # IDLE: start a new recording. The session is created first and the
        # pipeline resumed second (inside begin_recording_session), so a
        # failed start cannot leave a running pipeline with no session.
        if not live_trans._running:
            QMessageBox.information(
                overlay, t("error_title"), t("session_start_pipeline_needed")
            )
            return
        if not live_trans.begin_recording_session():
            # Auto-save off or files unwritable; tell the user why nothing
            # opened rather than failing silently behind the button.
            QMessageBox.warning(
                overlay, t("error_title"), t("session_start_unavailable")
            )
            return
        # The session opened and the pipeline resumed inside the call; sync
        # the run/pause visuals to what the pipeline now actually is.
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))

    overlay.session_toggle_requested.connect(_on_session_toggle)

    def _on_end_recording_requested(session_id):
        # Identity closure: the request names the meeting it was issued on
        # (the records page's end button only shows for that record). If
        # the writer's live session is no longer that stamp — the meeting
        # already ended, or a new one began while the click was in flight —
        # the request must not end whichever meeting happens to be active
        # now. active_session() is the writer's authority; PAUSED keeps the
        # session open, so a paused meeting still ends correctly.
        if session_id:
            live = live_trans._transcript.active_session()
            if live != session_id:
                QMessageBox.information(
                    overlay, t("error_title"), t("session_end_target_gone")
                )
                return
        if not _confirm_end_session():
            return
        # Re-validate inside the authoritative entry, atomically with the
        # ENDING flip: the dialog pumped the event loop, so the meeting
        # open now may differ from the one the user clicked on. expected
        # is None only for a legacy caller with no identity to assert.
        if not live_trans.end_recording_session(
            expected_session=session_id or None
        ):
            QMessageBox.information(
                overlay, t("error_title"), t("session_end_target_gone")
            )

    panel.end_recording_requested.connect(_on_end_recording_requested)

    def _on_overlay_source_lang(code):
        """Overlay source language combo → sync to panel + ASR engine + tray."""
        if panel._asr_engine.currentIndex() == 3:
            code = "ru"
        _on_tray_asr_lang(code)
        overlay.set_source_language(code)

    def _on_panel_asr_lang_changed(_index):
        """Panel ASR language combo → sync to overlay."""
        code = "ru" if panel._asr_engine.currentIndex() == 3 else (
            panel._asr_lang.currentData() or "auto"
        )
        overlay.set_source_language(code)

    overlay.source_language_changed.connect(_on_overlay_source_lang)
    panel._asr_lang.currentIndexChanged.connect(_on_panel_asr_lang_changed)
    overlay.model_switch_requested.connect(on_overlay_model_switch)
    overlay.start_requested.connect(on_resume)
    overlay.stop_requested.connect(on_pause)
    overlay.hide_requested.connect(on_toggle_overlay)
    overlay.quit_requested.connect(on_quit)

    tray.setContextMenu(menu)
    tray.show()

    def _on_memory_warning(rss_mb: float):
        tray.showMessage(
            "LiveTranslate",
            t("mem_warning_msg").format(rss=int(rss_mb)),
            QSystemTrayIcon.MessageIcon.Warning,
            10000,
        )

    live_trans.set_memory_warning_callback(_on_memory_warning)

    def _on_notification(message: str):
        tray.showMessage(
            "LiveTranslate", message, QSystemTrayIcon.MessageIcon.Warning, 10000
        )

    live_trans.set_notification_callback(_on_notification)

    QTimer.singleShot(500, on_start)

    def _on_sigint(*_):
        # Post the request; do NOT join threads, close files or reap child
        # processes from inside a signal handler. stop()'s idempotence covers a
        # second Ctrl-C arriving while the first one is still being serviced.
        log.info("SIGINT received, requesting shutdown")
        QTimer.singleShot(0, on_quit)

    signal.signal(signal.SIGINT, _on_sigint)
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing as _multiprocessing

    _multiprocessing.freeze_support()
    main()
