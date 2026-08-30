import gc
import inspect
import logging
import sys
import traceback
from typing import Any

import numpy as np

log = logging.getLogger("LiveTranslate.ASRWorker")


def _setup_logging():
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)


def _error_response(msg_id: str | None, exc: BaseException, recoverable: bool) -> dict:
    return {
        "id": msg_id,
        "ok": False,
        "type": "error",
        "error": {
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "recoverable": recoverable,
        },
    }


def _ok_response(msg_id: str | None, response_type: str, payload: Any = None) -> dict:
    return {
        "id": msg_id,
        "ok": True,
        "type": response_type,
        "payload": payload,
    }


def _parse_device(device: str) -> tuple[str, int]:
    from torch_backend import normalize_device

    device = str(device or "cpu").split(" (", 1)[0].strip()
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        return "cuda", index
    # CTranslate2/faster-whisper has no MPS backend; keep its public device
    # selection while mapping the engine to Apple Silicon CPU.
    return normalize_device(device, for_ct2=True), 0


def _load_engine(config: dict):
    from model_manager import MODELS_DIR, apply_cache_env

    apply_cache_env()

    engine_type = config["engine_type"]
    device = config.get("device", "cpu")
    hub = config.get("hub", "ms")
    language = config.get("language", "auto")
    pad_seconds = config.get("pad_seconds")

    parsed_device, device_index = _parse_device(device)
    torch_device = normalize_torch_device(device)

    if engine_type == "funasr":
        from asr_funasr import FunASREngine

        engine = FunASREngine(
            model_key=config.get("funasr_model"),
            device=torch_device,
            hub=hub,
            pad_seconds=pad_seconds,
        )
    elif engine_type == "anime-whisper":
        from asr_anime_whisper import AnimeWhisperEngine

        if torch_device == "cpu":
            worker_device = "cpu"
        elif torch_device.startswith("cuda"):
            worker_device = f"cuda:{device_index}"
        else:
            worker_device = torch_device
        engine = AnimeWhisperEngine(device=worker_device, hub=hub)
    elif engine_type == "gigaam":
        from asr_gigaam import GigaAMEngine

        engine = GigaAMEngine(device=torch_device, hub="hf")
    else:
        from asr_engine import ASREngine

        compute_type = config.get("compute_type", "float16")
        if parsed_device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        download_root = config.get("download_root")
        if not download_root:
            download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
        engine = ASREngine(
            model_size=config["model_size"],
            device=parsed_device,
            device_index=device_index,
            compute_type=compute_type,
            language=language,
            download_root=download_root,
            pad_seconds=pad_seconds,
        )

    if hasattr(engine, "set_language"):
        engine.set_language("ru" if engine_type == "gigaam" else language)
    return engine


def normalize_torch_device(device: str | None) -> str:
    from torch_backend import normalize_device

    return normalize_device(device)


# Payload keys that are part of the request envelope rather than transcribe
# kwargs. Everything else is forwarded when the backend accepts it.
_ENVELOPE_KEYS = frozenset({"audio"})

_SIGNATURE_CACHE: dict[int, tuple] = {}


def _accepted_kwargs(engine) -> tuple[set, bool]:
    """(parameter names, accepts **kwargs) for this engine's transcribe.

    Cached per engine instance: a worker handles one backend for its whole life,
    and reflecting on every real-time call is pure overhead.
    """
    key = id(engine)
    cached = _SIGNATURE_CACHE.get(key)
    if cached is not None:
        return cached
    signature = inspect.signature(engine.transcribe)
    names = set(signature.parameters)
    var_kw = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    _SIGNATURE_CACHE.clear()  # only ever one live engine per worker process
    _SIGNATURE_CACHE[key] = (names, var_kw)
    return names, var_kw


def _transcribe(engine, payload: dict):
    """Forward a transcribe request, saying out loud what it could not forward.

    Anything the backend does not accept used to vanish without a trace, so a
    client-side option could be ignored with no way to tell from either side.
    """
    audio = payload.get("audio")
    if not isinstance(audio, np.ndarray):
        raise TypeError("transcribe payload audio must be a numpy.ndarray")

    names, accepts_var_kwargs = _accepted_kwargs(engine)
    kwargs = {}
    ignored = []
    for key, value in payload.items():
        if key in _ENVELOPE_KEYS:
            continue
        if key in names or accepts_var_kwargs:
            kwargs[key] = value
        else:
            ignored.append(key)
    if "word_timestamps" in kwargs:
        kwargs["word_timestamps"] = bool(kwargs["word_timestamps"])
    if ignored:
        log.debug(
            "Backend %s does not accept transcribe kwargs %s; ignoring",
            type(engine).__name__, sorted(ignored),
        )
    return engine.transcribe(audio, **kwargs)


def _cleanup_engine(engine):
    if engine is not None and hasattr(engine, "unload"):
        try:
            engine.unload()
        except Exception:
            log.warning("ASR engine unload failed", exc_info=True)
    gc.collect()
    try:
        from torch_backend import empty_cache

        empty_cache()
    except Exception:
        pass


def worker_main(conn, config: dict):
    _setup_logging()
    engine = None
    try:
        log.info(
            "ASR worker loading: "
            f"{config.get('engine_type')} on {config.get('device')} "
            f"(pid config={config.get('display_name', '')})"
        )
        engine = _load_engine(config)
        conn.send(
            _ok_response(
                None,
                "ready",
                {
                    "engine_type": config.get("engine_type"),
                    "display_name": config.get("display_name"),
                    "device": config.get("device"),
                },
            )
        )
    except BaseException as exc:
        log.error(f"ASR worker load failed: {exc}", exc_info=True)
        try:
            conn.send(_error_response(None, exc, recoverable=False))
        finally:
            _cleanup_engine(engine)
            conn.close()
        return

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            msg_id = msg.get("id")
            msg_type = msg.get("type")
            payload = msg.get("payload") or {}

            try:
                if msg_type == "shutdown":
                    conn.send(_ok_response(msg_id, "shutdown"))
                    break
                if msg_type == "transcribe":
                    result = _transcribe(engine, payload)
                    conn.send(_ok_response(msg_id, "result", result))
                    continue
                if msg_type == "set_language":
                    if hasattr(engine, "set_language"):
                        engine.set_language(payload.get("language", "auto"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                if msg_type == "set_input_padding":
                    if hasattr(engine, "set_input_padding"):
                        engine.set_input_padding(payload.get("pad_seconds"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                raise ValueError(f"Unknown ASR worker command: {msg_type}")
            except Exception as exc:
                log.error(f"ASR worker command failed: {msg_type}: {exc}", exc_info=True)
                conn.send(_error_response(msg_id, exc, recoverable=True))
    finally:
        _cleanup_engine(engine)
        try:
            conn.close()
        except Exception:
            pass
        log.info("ASR worker stopped")
