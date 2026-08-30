"""End-to-end pipeline diagnostic: replay an audio file through the real chain.

This exists because neither chain can otherwise be exercised without a person
speaking into a microphone. It wires a file into the *actual* pipeline —
``LiveTranslateApp`` with its real VAD, its real ASR worker subprocess, its real
``Translator`` and its real ``TranscriptWriter`` — so a run here proves the same
code paths the GUI uses.

    # both chains, real settings, real API
    .venv/bin/python debug_pipeline.py --audio sample.mp3

    # transcription only, no network
    .venv/bin/python debug_pipeline.py --audio sample.mp3 --no-translate

    # a specific engine/model instead of what user_settings.json selects
    .venv/bin/python debug_pipeline.py --audio sample.mp3 \\
        --engine funasr --asr-model sensevoice-small --lang zh --target en

Everything at DEBUG level is captured, and the run ends with a report of what
each stage produced plus every warning and error, so a failure says which link
broke rather than just going quiet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def _prepare_environment():
    """Apply the cache env before torch is imported, exactly as main.py does."""
    sys.path.insert(0, str(APP_DIR))
    from model_manager import apply_cache_env

    apply_cache_env()


_prepare_environment()

from debug_recorder import DiagnosticRecorder  # noqa: E402

log = logging.getLogger("LiveTranslate.Debug")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay an audio file through the real LiveTranslate pipeline"
    )
    parser.add_argument("--audio", required=True, help="WAV/MP3/M4A/FLAC to replay")
    parser.add_argument(
        "--engine",
        help="ASR engine (funasr, whisper, gigaam, anime-whisper, remote-whisper). "
             "Defaults to user_settings.json.",
    )
    parser.add_argument(
        "--asr-model",
        help="FunASR model key or Whisper size. Defaults to user_settings.json.",
    )
    parser.add_argument("--lang", help="Source language hint, or 'auto'")
    parser.add_argument("--target", help="Translation target language")
    parser.add_argument(
        "--model-index",
        type=int,
        help="Index into the models list in user_settings.json (default: active_model)",
    )
    parser.add_argument("--device", help="ASR device: mps, cuda, cpu")
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Exercise capture -> VAD -> ASR -> transcript only; no API calls",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Feed audio as fast as the pipeline drains instead of at wall-clock speed",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Force incremental (interim) ASR on, regardless of settings",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Give up if the run has not settled after this many seconds",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Also write the machine-readable report here",
    )
    parser.add_argument(
        "--transcripts-dir",
        default=str(APP_DIR / "transcripts" / "diagnostic"),
        help="Where this run writes its transcript. Defaults to a diagnostic "
             "subdirectory so test runs do not clutter real meeting records; "
             "pass the real transcripts/ path to exercise that location.",
    )
    return parser


def load_settings() -> dict:
    """Load settings through the app's own path, migrations included.

    Reading the JSON directly meant the diagnostic ran against a config the app
    would never actually use — it missed the managed-model migrations that
    _load_saved_settings applies (and persists) at every launch.
    """
    try:
        from control_panel import _load_saved_settings

        return _load_saved_settings() or {}
    except Exception as exc:
        log.warning("Could not load settings through the app path: %s", exc)

    path = APP_DIR / "user_settings.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read user_settings.json: %s", exc)
        return {}


def resolve_worker_config(args, settings: dict, config: dict) -> tuple[dict, dict]:
    """Build the ASR worker config and its restart state.

    Mirrors what _switch_asr_engine assembles, without the Qt download dialog:
    a diagnostic must not pop a window, and a missing model should be reported
    rather than silently downloaded mid-run.
    """
    from model_manager import (
        ASR_DISPLAY_NAMES,
        MODELS_DIR,
        funasr_display_name,
        normalize_asr_engine_selection,
        resolve_custom_whisper_model,
    )

    engine = args.engine or settings.get("asr_engine") or config["asr"]["asr_engine"]
    funasr_model = args.asr_model or settings.get("funasr_model") or config["asr"].get(
        "funasr_model"
    )
    engine, funasr_model = normalize_asr_engine_selection(engine, funasr_model)

    model_size = args.asr_model or settings.get(
        "whisper_model_size", config["asr"]["model_size"]
    )
    cache_key = model_size
    if engine == "whisper":
        custom = resolve_custom_whisper_model(model_size)
        if custom:
            cache_key = custom
    elif engine == "funasr":
        cache_key = funasr_model

    language = args.lang or settings.get("asr_language") or config["asr"].get(
        "language", "auto"
    )
    if engine == "gigaam":
        language = "ru"

    device = args.device or settings.get("asr_device") or "cpu"
    hub = settings.get("hub", "ms")

    display = ASR_DISPLAY_NAMES.get(engine, engine)
    if engine == "funasr":
        display = funasr_display_name(funasr_model)

    worker_config = {
        "engine_type": engine,
        "funasr_model": funasr_model,
        "model_size": cache_key,
        "device": device,
        "compute_type": config["asr"]["compute_type"],
        "hub": hub,
        "language": language,
        "pad_seconds": (
            settings.get("sensevoice_pad_seconds", 0.5)
            if engine == "funasr"
            else settings.get("whisper_pad_seconds", 0.5)
            if engine == "whisper"
            else None
        ),
        "download_root": str((MODELS_DIR / "huggingface" / "hub").resolve()),
        "display_name": display,
        "remote_asr_url": settings.get("remote_asr_url"),
    }
    state = {
        "type": engine,
        "signature": (engine, cache_key, device, hub, config["asr"]["compute_type"]),
        "device": device,
        "funasr_model_key": funasr_model,
        "whisper_model_size": model_size,
        "config": worker_config,
        "display_name": display,
        "device_label": device,
    }
    return worker_config, state


def check_model_cached(worker_config: dict, recorder: DiagnosticRecorder) -> bool:
    from model_manager import is_asr_cached

    engine = worker_config["engine_type"]
    if engine == "remote-whisper":
        return True
    cached = is_asr_cached(engine, worker_config["model_size"], worker_config["hub"])
    if not cached:
        recorder.problem(
            f"ASR model is not cached: {engine}/{worker_config['model_size']} "
            f"(hub={worker_config['hub']}). Launch the app once to download it, "
            f"or pass --engine/--asr-model for a model you already have."
        )
    return bool(cached)


def configure_translator(app, args, settings: dict, recorder: DiagnosticRecorder):
    """Point the app's Translator at the selected model, as the panel would."""
    if args.no_translate:
        app._translator = None
        recorder.note("translation disabled by --no-translate")
        return None

    models = settings.get("models") or []
    if not models:
        recorder.problem("no translation models configured in user_settings.json")
        return None
    index = args.model_index
    if index is None:
        index = settings.get("active_model", 0)
    if not (isinstance(index, int) and 0 <= index < len(models)):
        recorder.problem(f"model index {index} is out of range (have {len(models)})")
        return None

    model = models[index]
    from mlx_service import is_hy_mt_model

    if is_hy_mt_model(model) and not app._mlx_service.is_running():
        # The app auto-starts this service when the managed model is active, and
        # stop() shuts it down again — so a diagnostic that just ran will have
        # left it down. Start it the same way rather than reporting a failure
        # the user cannot act on.
        recorder.note(f"starting the managed MLX service for '{model['name']}'")
        started = time.perf_counter()
        try:
            app._mlx_service.ensure_running(timeout=args.timeout)
        except Exception as exc:
            recorder.problem(f"could not start the local MLX service: {exc}")
            return None
        recorder.note(f"MLX service ready in {time.perf_counter() - started:.1f}s")

    app._on_model_changed(model)
    if app._translator is None:
        recorder.problem(f"translator was not created for model '{model['name']}'")
        return None
    if args.target:
        app._on_target_language_changed(args.target)
    recorder.note(
        f"translator: {model['name']} ({model['model']}) -> {app._target_language}"
    )
    return model


def run(args) -> int:
    recorder = DiagnosticRecorder(APP_DIR / "logs")
    recorder.install()
    log.info("Pipeline diagnostic starting")

    import main as app_module
    from audio_capture_file import FileAudioCapture

    settings = load_settings()
    config = app_module.load_config()

    try:
        source = FileAudioCapture(
            args.audio, realtime=not args.fast, trailing_silence=2.0
        )
    except Exception as exc:
        recorder.problem(f"could not load audio: {exc}")
        recorder.report(sys.stdout)
        return 2
    recorder.note(f"audio: {source.path.name}, {source.duration:.1f}s @16kHz mono")

    app = app_module.LiveTranslateApp(config)
    app._audio = source
    # Same TranscriptWriter, different directory: a diagnostic must not file
    # itself among the user's real meeting records.
    from transcript_writer import TranscriptWriter

    app._transcript = TranscriptWriter(Path(args.transcripts_dir))
    recorder.note(f"transcripts -> {args.transcripts_dir}")
    observer = PipelineObserver(app, recorder)
    observer.install()

    worker_config, state = resolve_worker_config(args, settings, config)
    recorder.note(
        f"ASR: {worker_config['engine_type']} / {worker_config['model_size']} "
        f"on {worker_config['device']} (lang={worker_config['language']})"
    )
    if not check_model_cached(worker_config, recorder):
        recorder.report(sys.stdout)
        return 2

    # VAD and incremental settings come from the real settings path.
    app._on_settings_changed(_pipeline_settings(settings, args))

    load_started = time.perf_counter()
    try:
        if not app._start_worker_from_state(state, app._asr_generation):
            recorder.problem("ASR worker failed to load; see the log for the cause")
            recorder.report(sys.stdout)
            return 2
    except Exception as exc:
        recorder.problem(f"ASR worker load raised: {exc}")
        log.error("ASR worker load failed", exc_info=True)
        recorder.report(sys.stdout)
        return 2
    recorder.note(f"ASR worker ready in {time.perf_counter() - load_started:.1f}s")

    translating = configure_translator(app, args, settings, recorder) is not None

    exit_code = 0
    try:
        app.start()
        _wait_for_completion(app, source, recorder, args.timeout)
    except Exception as exc:
        recorder.problem(f"pipeline raised: {exc}")
        log.error("Pipeline error", exc_info=True)
        exit_code = 1
    finally:
        try:
            app.stop()
        except Exception as exc:
            recorder.problem(f"stop() raised: {exc}")
            log.error("stop() failed", exc_info=True)
            exit_code = 1

    recorder.finish(observer, app, expect_translation=translating)
    recorder.report(sys.stdout)
    if args.json:
        Path(args.json).write_text(
            json.dumps(recorder.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.json}")
    print(f"Full log:    {recorder.log_path}")

    return exit_code or (1 if recorder.problems or recorder.errors else 0)


def _pipeline_settings(settings: dict, args) -> dict:
    """The subset of settings the pipeline consumes, with CLI overrides."""
    picked = {
        key: settings[key]
        for key in (
            "vad_mode", "vad_threshold", "energy_threshold",
            "min_speech_duration", "max_speech_duration",
            "silence_mode", "silence_duration",
            "incremental_asr", "interim_interval",
            "auto_save_transcript", "translation_workers",
        )
        if key in settings
    }
    picked.setdefault("auto_save_transcript", True)
    if args.incremental:
        picked["incremental_asr"] = True
    if args.lang:
        picked["asr_language"] = args.lang
    return picked


def _wait_for_completion(app, source, recorder, timeout: float):
    """Wait for the replay to drain and the in-flight work to settle."""
    deadline = time.monotonic() + timeout
    while not source.finished.is_set() and time.monotonic() < deadline:
        time.sleep(0.2)
    if not source.finished.is_set():
        recorder.problem(f"replay did not finish within {timeout:g}s")
        return

    # Let the tail segment reach ASR, then let its translation land.
    quiet_since = None
    while time.monotonic() < deadline:
        busy = (
            not app._asr_queue.empty()
            or app._translation_pending > 0
        )
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > 3.0:
            return
        time.sleep(0.2)
    recorder.problem(f"pipeline had not settled after {timeout:g}s")


class PipelineObserver:
    """Record what each stage actually produced, by wrapping the real methods."""

    def __init__(self, app, recorder):
        self.app = app
        self.recorder = recorder
        self.segments = []          # (seconds,)
        self.asr_results = []       # (text, language, ms)
        self._by_id = {}            # msg_id -> (source, translated)
        self.interim = []

    @property
    def translations(self):
        """Utterance order, matching what the transcript records."""
        return [self._by_id[key] for key in sorted(self._by_id)]

    def install(self):
        app = self.app
        original_run_asr = app._run_asr
        original_commit = app._commit_translation_result

        def run_asr(audio, kind, **kwargs):
            if kind == "segment":
                self.segments.append(len(audio) / 16000)
            result, ms = original_run_asr(audio, kind, **kwargs)
            if isinstance(result, dict) and result.get("text"):
                entry = (result["text"], result.get("language", "?"), ms)
                (self.interim if kind == "interim" else self.asr_results).append(entry)
            return result, ms

        def commit(msg_id, text, translated, generation):
            accepted = original_commit(msg_id, text, translated, generation)
            if accepted and translated:
                # Keyed by msg_id, not appended: commits arrive in whatever
                # order the worker pool finishes, and a report that listed them
                # that way would misrepresent the record's actual order.
                self._by_id[msg_id] = (text, translated)
            return accepted

        app._run_asr = run_asr
        app._commit_translation_result = commit


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return run(args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
