"""
Remote ASR server for LiveTranslate, using faster-whisper.

Run this on a machine with a GPU, then point LiveTranslate's "Remote Whisper"
engine at it (Settings -> VAD/ASR -> Remote ASR Server URL). The client
(asr_remote.py) POSTs raw float32 PCM (16 kHz mono) to /transcribe and gets
back the transcription as JSON.

    pip install faster-whisper fastapi uvicorn numpy
    python asr_server.py --model large-v3 --device cuda --compute-type float16

Two ways to start it, both reading the same configuration:

    python asr_server.py [--host ...] [--port ...] [--model ...]
        Command-line flags override the defaults below.

    uvicorn asr_server:app
        No argument parsing happens, so the defaults below apply as-is. Override
        them with the LIVETRANSLATE_ASR_* environment variables (see DEFAULTS).

Security: this server has no authentication. It binds to 127.0.0.1 by default;
pass `--host 0.0.0.0` (or set LIVETRANSLATE_ASR_HOST) only on a trusted network,
and consider setting LIVETRANSLATE_ASR_TOKEN so callers must present a matching
`X-ASR-Token` header.

Notes:
- For CUDA, faster-whisper/CTranslate2 needs the CUDA 12 cuBLAS and cuDNN 9
  libraries on the library path (e.g. `pip install nvidia-cublas-cu12
  nvidia-cudnn-cu12`, or a system CUDA install).
- The model is downloaded from Hugging Face on first run; set the HF_ENDPOINT
  env var to a mirror if direct access is slow.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import struct
import time
from types import SimpleNamespace

import numpy as np

try:
    from fastapi import FastAPI, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse
    from faster_whisper import WhisperModel
    import uvicorn
except ImportError as exc:  # pragma: no cover - server stack is optional
    # This module is also the configuration surface documented in
    # REMOTE_ASR.md. Keeping default_config()/parse_args()/_parse_request()
    # importable lets the offline test job verify them without installing the
    # GPU serving stack; create_app() below fails loudly if it is really needed.
    _IMPORT_ERROR = exc
    FastAPI = Request = JSONResponse = WhisperModel = uvicorn = None
    run_in_threadpool = None
else:
    _IMPORT_ERROR = None

log = logging.getLogger("ASR-Server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# 16 kHz float32 mono: 5 minutes is ~19 MB. Anything larger is rejected before
# it is read into memory rather than after.
MAX_AUDIO_SECONDS = 300
MAX_BODY_BYTES = MAX_AUDIO_SECONDS * 16000 * 4 + 1024


def default_config() -> SimpleNamespace:
    """The single source of truth for server configuration.

    Both entry points read this, so `uvicorn asr_server:app` no longer starts
    with no app.state.args at all and dies in the startup hook.
    """
    return SimpleNamespace(
        # 127.0.0.1, not 0.0.0.0: an unauthenticated GPU inference endpoint
        # should not be reachable from the network unless someone says so.
        host=os.getenv("LIVETRANSLATE_ASR_HOST", "127.0.0.1"),
        port=int(os.getenv("LIVETRANSLATE_ASR_PORT", "8765")),
        model=os.getenv("LIVETRANSLATE_ASR_MODEL", "medium"),
        device=os.getenv("LIVETRANSLATE_ASR_DEVICE", "cuda"),
        compute_type=os.getenv("LIVETRANSLATE_ASR_COMPUTE_TYPE", "float16"),
        token=os.getenv("LIVETRANSLATE_ASR_TOKEN", ""),
    )


def parse_args(argv=None) -> SimpleNamespace:
    defaults = default_config()
    parser = argparse.ArgumentParser(description="Remote ASR Server")
    parser.add_argument("--host", default=defaults.host, help="Bind address")
    parser.add_argument("--port", type=int, default=defaults.port, help="Bind port")
    parser.add_argument("--model", default=defaults.model, help="Whisper model size")
    parser.add_argument("--device", default=defaults.device, help="Device: cuda or cpu")
    parser.add_argument(
        "--compute-type", default=defaults.compute_type, help="Compute type"
    )
    parser.add_argument(
        "--token",
        default=defaults.token,
        help="Shared secret required in the X-ASR-Token header (empty = disabled)",
    )
    return parser.parse_args(argv)


@contextlib.asynccontextmanager
async def lifespan(application):
    """Load the model on startup. Replaces the deprecated on_event hook."""
    global _model
    args = application.state.args
    log.info(f"Loading model: {args.model} on {args.device} ({args.compute_type})")
    _model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    log.info(f"Model ready: {args.model}")
    if getattr(args, "token", ""):
        log.info("Token authentication is enabled")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "Binding to %s exposes an unauthenticated GPU endpoint; "
            "deploy only on a trusted network", args.host,
        )
    yield
    _model = None


def create_app(args=None):
    """Build the ASGI app with its configuration attached."""
    if FastAPI is None:
        raise RuntimeError(
            "The remote ASR server needs fastapi, uvicorn and faster-whisper: "
            "pip install faster-whisper fastapi uvicorn numpy"
        ) from _IMPORT_ERROR
    application = FastAPI(title="Remote ASR Server", lifespan=lifespan)
    application.state.args = args or default_config()
    application.post("/transcribe")(transcribe)
    application.get("/health")(health)
    return application


_model = None
# Serialize GPU access: the model can only run one transcription at a time.
_gpu_lock = asyncio.Lock()


def _parse_request(request_body: bytes):
    """Decode the wire format: [uint32 lang_len][lang utf-8][float32 PCM]. Raises
    ValueError on any malformed/attacker-supplied body so the caller returns 400."""
    if len(request_body) < 4:
        raise ValueError("request too short")
    lang_len = struct.unpack("<I", request_body[:4])[0]
    if 4 + lang_len > len(request_body):
        raise ValueError("language length exceeds body")
    language = (
        request_body[4 : 4 + lang_len].decode("utf-8", errors="replace")
        if lang_len > 0
        else None
    )
    if language in ("auto", ""):
        language = None
    audio_bytes = request_body[4 + lang_len :]
    if len(audio_bytes) % 4 != 0:
        raise ValueError("audio byte length is not a multiple of 4")
    return language, np.frombuffer(audio_bytes, dtype=np.float32)


def _run_transcription(audio: np.ndarray, language):
    segments, info = _model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text_parts = [seg.text.strip() for seg in segments]
    return " ".join(text_parts).strip(), info.language


def check_token(request):
    """None when the request may proceed, else a JSONResponse to return."""
    expected = getattr(request.app.state.args, "token", "")
    if not expected:
        return None
    if request.headers.get("X-ASR-Token") != expected:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def oversized(request):
    """Reject an over-long body from the Content-Length header, before reading it."""
    declared = request.headers.get("content-length")
    if declared is None:
        return False
    try:
        return int(declared) > MAX_BODY_BYTES
    except ValueError:
        return False


async def transcribe(request: "Request"):
    """Accept raw float32 PCM audio at 16kHz mono. Return transcription."""
    denied = check_token(request)
    if denied is not None:
        return denied
    if oversized(request):
        return JSONResponse(
            {"error": f"body exceeds {MAX_BODY_BYTES} bytes "
                      f"(~{MAX_AUDIO_SECONDS}s of 16 kHz float32 audio)"},
            status_code=413,
        )

    request_body = await request.body()
    if len(request_body) > MAX_BODY_BYTES:
        # A chunked upload has no Content-Length to check up front.
        return JSONResponse({"error": "body too large"}, status_code=413)
    try:
        language, audio = _parse_request(request_body)
    except (ValueError, struct.error) as e:
        return JSONResponse({"error": f"bad request: {e}"}, status_code=400)

    duration = len(audio) / 16000
    t0 = time.time()
    # Run the blocking GPU call off the event loop, one at a time.
    async with _gpu_lock:
        full_text, detected_lang = await run_in_threadpool(
            _run_transcription, audio, language
        )
    elapsed = time.time() - t0

    log.info(
        f"Transcribed {duration:.1f}s audio in {elapsed:.2f}s: "
        f"[{detected_lang}] {full_text[:80]}"
    )

    if not full_text:
        return {"text": None, "language": detected_lang, "elapsed": elapsed}

    return {
        "text": full_text,
        "language": detected_lang,
        "language_name": detected_lang,
        "elapsed": elapsed,
    }


async def health(request: "Request"):
    return {"status": "ok", "model": request.app.state.args.model}


# Module-level app for `uvicorn asr_server:app`. It carries the defaults, so the
# ASGI entry point no longer depends on __main__ having run first.
app = create_app() if FastAPI is not None else None


if __name__ == "__main__":
    args = parse_args()
    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
