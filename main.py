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
    apply_cache_env,
    funasr_display_name,
    funasr_supports_padding,
    get_missing_models,
    is_asr_cached,
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
    translation_api_base,
    translation_api_key,
    translation_model,
)
from vad_processor import VADProcessor
from asr_client import ASRClient, ASRWorkerError, ASRWorkerExited, ASRWorkerTimeout
from asr_remote import RemoteASRError
from translator import Translator, RepetitionError
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
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal

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
        import shutil

        failures = []
        for path in self._paths:
            try:
                shutil.rmtree(path)
                log.info(f"Deleted: {path}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                log.error(f"Failed to delete {path}: {exc}")
                failures.append((str(path), str(exc)))
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
        self._translator = Translator(
            api_base=translation_api_base(config["translation"].get("api_base")),
            api_key=translation_api_key(config["translation"].get("api_key")),
            model=translation_model(config["translation"].get("model")),
            target_language=self._target_language,
            max_tokens=config["translation"]["max_tokens"],
            temperature=config["translation"]["temperature"],
            streaming=config["translation"]["streaming"],
            system_prompt=config["translation"].get("system_prompt"),
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

    def set_overlay(self, overlay: SubtitleOverlay):
        self._overlay = overlay
        self._publish_transcript_paths()

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

    def set_panel(self, panel: ControlPanel):
        self._panel = panel
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
                        "Audio device switch to %r failed; keeping the previous device",
                        new_device,
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
        if self._asr_type == "gigaam":
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
        return "ru" if self._asr_type == "gigaam" else configured

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
            settings = self._panel.get_settings()
            settings["target_language"] = lang
            from control_panel import _save_settings

            _save_settings(settings)

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

        new_translator = Translator(
            api_base=translation_api_base(model_config.get("api_base")),
            api_key=translation_api_key(model_config.get("api_key")),
            model=translation_model(model_config.get("model")),
            target_language=self._target_language,
            max_tokens=self._config["translation"]["max_tokens"],
            temperature=self._config["translation"]["temperature"],
            streaming=model_config.get("streaming", True),
            system_prompt=prompt,
            proxy=model_config.get("proxy", "none"),
            no_system_role=model_config.get("no_system_role", False),
            no_think=model_config.get("no_think", True),
            thinking_style=model_config.get("thinking_style"),
            json_response=model_config.get("json_response", False),
            timeout=timeout,
            overrides=model_config.get("overrides"),
            extra_body=model_config.get("extra_body"),
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

    def _submit_translation(self, msg_id, text, source_lang, extra_langs=None):
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
            )
        except RuntimeError:
            self._commit_translation_result(msg_id, text, None, generation)
            raise

    def _switch_asr_engine(self, engine_type: str):
        settings = self._panel.get_settings() if self._panel else {}
        engine_type, funasr_model = normalize_asr_engine_selection(
            engine_type, settings.get("funasr_model", self._funasr_model_key)
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
            signature_model = "ai-sage/GigaAM-v3@e2e_rnnt"
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
            display_name = ASR_DISPLAY_NAMES["gigaam"]

        parent = (
            self._panel if self._panel and self._panel.isVisible() else self._overlay
        )

        worker_config = {
            "engine_type": engine_type,
            "funasr_model": funasr_model,
            "model_size": cache_model_key,
            "device": device,
            "compute_type": compute,
            "hub": hub,
            "language": (
                "ru"
                if engine_type == "gigaam"
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
                self._whisper_model_size = state["whisper_model_size"]
                self._asr_ready = True
                self._asr_error_count = 0
                self._asr_restart_state = dict(state)
                self._asr_restart_count = 0
                self._asr_worker_baseline_mb = None
                self._asr_generation += 1

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
            if self._asr_generation != expected_gen or not self._running:
                stale = client
            else:
                self._asr = client
                self._asr_type = state["type"]
                self._asr_signature = state["signature"]
                self._asr_device = state["device"]
                self._asr_config = dict(state["config"]) if state["config"] else None
                self._funasr_model_key = state["funasr_model_key"]
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
    ):
        """Translate text and update UI with streaming display."""
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
                log.info("Discarding translation from superseded model: msg=%s", msg_id)
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
                self._transcript.write_translation(msg_id, translated)
            else:
                self._transcript.finalize_no_translation(msg_id)
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
            self._transcript.finalize_no_translation(msg_id)
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
                return
            self._transcript.finalize_no_translation(msg_id)
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
        log.info("Pipeline stopped")

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
        self._reset_interim_state()
        if self._overlay:
            self._overlay.update_monitor(0.0, 0.0)
        log.info("Pipeline paused")

    def resume(self):
        self._paused = False
        log.info("Pipeline resumed")

    def _process_segment(self, speech_segment):
        """Run ASR + translation on a speech segment. Called from ASR thread and stop()."""
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

        if self._overlay:
            self._overlay.add_message(
                msg_id, timestamp, original_text, source_lang, asr_ms
            )
        self._transcript.write_original(msg_id, timestamp, original_text)

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
            self._transcript.finalize_no_translation(msg_id)
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
                    msg_id, original_text, source_lang, extra_langs or None
                )
            except TranslationUnavailable as exc:
                self._finalize_untranslated(msg_id, str(exc), user_visible=True)
            except RuntimeError:
                # Executor already shut down (we are exiting) — not worth a
                # user-facing error, but still has to be closed out.
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

    def _strip_committed_overlap(self, text: str) -> str:
        """Remove text that overlaps with previously committed content."""
        if not self._interim_committed_tail:
            return text
        tail = self._interim_committed_tail.lower().rstrip()
        text_lower = text.lower()
        # Check if text starts with a suffix of the committed tail
        max_check = min(len(tail), len(text_lower))
        for overlap_len in range(max_check, 2, -1):
            if text_lower[:overlap_len] == tail[-overlap_len:]:
                stripped = text[overlap_len:].strip()
                if stripped:
                    log.debug(f"Stripped echo overlap ({overlap_len} chars): '{text[:overlap_len]}...'")
                    return stripped
                return ""
        return text

    def _do_interim_asr(self) -> bool:
        """Run ASR on current VAD buffer, output complete sentences, trim consumed audio.
        Returns True if any sentences were committed."""
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

        committed_text = ""
        for sent in complete:
            committed_text += sent

        if not committed_text.strip():
            return False

        # Determine trim point
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

        # Output committed sentences
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
                consumed_any = True
                continue

            if self._interim_pending:
                text = self._interim_pending + text
                self._interim_pending = ""

            self._process_segment_text(text, result_lang, asr_ms)
            actually_committed = True
            consumed_any = True

        if not consumed_any:
            return False

        if trim_samples > 0:
            with self._vad_lock:
                self._vad.trim_front(trim_samples)

        # Track committed text tail for echo dedup
        self._interim_committed_tail = committed_text[-50:] if len(committed_text) > 50 else committed_text

        self._interim_active = True
        log.info(
            f"Interim ASR: consumed {len(complete)} sentence(s) "
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

    def _process_segment_text(self, text: str, source_lang: str, asr_ms: float = 0):
        """Output a text result (from interim or final) — similar to _process_segment but skips ASR."""
        original_text = text.strip()
        if not original_text or not any(c.isalnum() for c in original_text):
            return

        asr_lang_setting = self._get_asr_language_setting()
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', discarding: {original_text[:60]}")
            return

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms, interim): {original_text}")

        if self._overlay:
            self._overlay.add_message(msg_id, timestamp, original_text, source_lang, asr_ms)
        self._transcript.write_original(msg_id, timestamp, original_text)

        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language
        extra_langs = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            self._transcript.finalize_no_translation(msg_id)
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
                    msg_id, original_text, source_lang, extra_langs or None
                )
            except TranslationUnavailable as exc:
                self._finalize_untranslated(msg_id, str(exc), user_visible=True)
            except RuntimeError:
                # Executor already shut down (we are exiting) — not worth a
                # user-facing error, but still has to be closed out.
                self._finalize_untranslated(
                    msg_id, "translation executor shut down", user_visible=False
                )
    def _process_interim_final(self, speech_segment):
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
                self._process_segment_text(text, lang)
            return

        validated = validate_asr_result(result, "interim_final")
        if validated is None:
            original_text, result_lang = "", ""
        else:
            original_text, result_lang = validated

        # Strip echo from previous commit's overlap
        original_text = self._strip_committed_overlap(original_text)

        # Prepend any remaining pending short utterances
        if self._interim_pending:
            original_text = self._interim_pending + original_text
            self._interim_pending = ""

        if not original_text or not any(c.isalnum() for c in original_text):
            return

        # Apply noise filter like _process_segment
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(f"Noise filter: {seg_len:.1f}s segment produced only '{original_text}', skipping")
            return

        if not result_lang:
            result_lang = self._get_asr_language_setting()
            if result_lang == "auto":
                result_lang = "unknown"
        self._process_segment_text(original_text, result_lang, asr_ms)

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
                if self._vad._is_speaking and not self._paused:
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
        """
        if self._stop_event.is_set():
            return
        try:
            self._asr_queue.put_nowait((seg_type, segment))
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
        try:
            self._asr_queue.put_nowait((seg_type, segment))
        except queue.Full:
            log.warning("ASR queue still full after drop, skipping segment")

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

            seg_type, segment = item

            # One bad result must not take the ASR thread down with it: without
            # this the pipeline goes permanently silent while capture keeps
            # filling a queue nobody drains, and no worker-restart path fires.
            try:
                if seg_type == "vad_flush":
                    try:
                        if self._interim_active:
                            self._process_interim_final(segment)
                        else:
                            self._process_segment(segment)
                    finally:
                        self._reset_interim_state()
                elif seg_type == "interim":
                    self._drain_interim_duplicates()
                    self._do_interim_asr()
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
                        log.warning("ASR queue full while requeueing %s", item[0])
                break


def main():
    setup_logging()
    log.info("LiveTranslate starting...")
    config = load_config()
    config.setdefault("asr", {})
    config["asr"].setdefault("asr_engine", "funasr")
    config["asr"].setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    saved = _load_saved_settings()
    migrate_funasr_settings(saved)

    # Log actual effective config
    _asr_eng = (saved or {}).get("asr_engine", config["asr"].get("asr_engine", "funasr"))
    _funasr_model = (saved or {}).get(
        "funasr_model", config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL)
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
        missing = get_missing_models(
            current_engine,
            (
                saved.get("funasr_model", config["asr"].get("funasr_model"))
                if current_engine == "funasr"
                else saved.get("whisper_model_size", config["asr"]["model_size"])
            ),
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
    live_trans.set_overlay(overlay)
    live_trans.set_subtitle_window(subwin)
    live_trans.set_panel(panel)
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
        _start_cancelled[0] = True
        live_trans.pause()
        overlay.set_running(False)
        _is_running[0] = False
        pause_action.setText(t("tray_resume"))

    def on_resume():
        _start_cancelled[0] = False
        if not live_trans._running:
            on_start()
            return
        live_trans.resume()
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))

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

    def _sync_asr_language_controls(index=None):
        """Keep every source-language entry point consistent with GigaAM."""
        is_gigaam = (panel._asr_engine.currentIndex() if index is None else index) == 3
        if is_gigaam:
            ru_idx = panel._asr_lang.findData("ru")
            if ru_idx >= 0 and panel._asr_lang.currentData() != "ru":
                panel._asr_lang.blockSignals(True)
                panel._asr_lang.setCurrentIndex(ru_idx)
                panel._asr_lang.blockSignals(False)
            overlay.set_source_language("ru")
        panel._asr_lang.setEnabled(not is_gigaam)
        overlay.set_source_language_enabled(not is_gigaam)
        selected = "ru" if is_gigaam else (panel._asr_lang.currentData() or "auto")
        for action in _asr_lang_actions.values():
            action.setEnabled(not is_gigaam)
            action.setChecked(action.data() == selected)

    panel._asr_engine.currentIndexChanged.connect(_sync_asr_language_controls)
    _sync_asr_language_controls()

    current_asr_lang = (
        "ru"
        if panel._asr_engine.currentIndex() == 3
        else panel.get_settings().get("asr_language", "auto")
    )
    if current_asr_lang in _asr_lang_actions:
        _asr_lang_actions[current_asr_lang].setChecked(True)

    def _on_tray_asr_lang(code):
        from control_panel import _save_settings

        if panel._asr_engine.currentIndex() == 3:
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
