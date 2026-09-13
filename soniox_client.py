"""Soniox cloud realtime STT service manager (ru -> zh).

Owns the WebSocket session lifecycle for the "Soniox Cloud Realtime" ASR
engine: a supervisor thread (connect / receive events / reconnect) and a send
thread (bounded PCM16 buffer -> send_byte_chunk). The capture thread only
ever calls feed(), which never blocks and never raises — network slowness
must never stall ScreenCaptureKit/WASAPI callbacks.

Key invariants:
  * a chunk is popped from the send deque only AFTER send_byte_chunk returns
    — on reconnect the unsent remainder is replayed to the new session once,
    and already-sent audio is never re-sent (no double billing, no repeated
    transcription);
  * the audio buffer is bounded (drop-oldest + counted metric — the server
    simply sees a silence gap);
  * every sink callback is dispatched under a generation check so late
    results from a torn-down session no-op;
  * the API key never appears in logs: mask_key() output is the only form
    ever logged, and the RealtimeSTTConfig (which embeds the key) is never
    logged at all;
  * no i18n / Qt imports — the sink adapter localizes; this module emits
    status/metric facts only.

The SDK import is lazy (via _import_soniox) so a missing dependency degrades
into SonioxNotInstalledError at engine load, and tests can monkeypatch the
seam with a fake client.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np

from soniox_accumulator import SonioxAccumulator, SonioxLiveState, SonioxSegment

log = logging.getLogger("LiveTranslate.soniox")

SONIOX_MODEL = "stt-rt-v5"
SONIOX_SAMPLE_RATE = 16000
SONIOX_LANGUAGE_HINTS = ["ru"]
# Code-switching profile: measured on mixed ru+en lecture audio, a strict
# ["ru"]-only hint transliterates English terms into Cyrillic ("algorithm
# complexity" -> "алгоритм комплексити"), while ["ru", "en"] keeps them in
# Latin script and the Chinese translation preserves the terms verbatim
# ("学习 algorithm complexity") — the classroom reality of ru lecturers
# reading English terminology.
SONIOX_MIXED_LANGUAGE_HINTS = ["ru", "en"]
SONIOX_TARGET_LANGUAGE = "zh"

# Reconnect backoff schedule (seconds), capped at the last value.
RECONNECT_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0)
# How long to wait for finals to quiesce after a finalize() request before
# flushing what we have (the server answers <end>/finals within ~max_endpoint_delay).
FINALIZE_QUIESCE_S = 3.0
# Silence gap fed around finalize boundaries (Soniox docs recommend ~200ms).
FINALIZE_SILENCE_S = 0.25


class SonioxStatus(str, Enum):
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    PAUSED = "paused"
    FAILED = "failed"
    FINISHED = "finished"


class SonioxError(ConnectionError):
    """Base Soniox engine error (ConnectionError so engine-switch treats it
    as an expected load failure, not a crash)."""


class SonioxMissingKeyError(SonioxError):
    pass


class SonioxNotInstalledError(SonioxError):
    pass


SEGMENTATION_PRESETS: dict[str, dict[str, Any]] = {
    # Endpoint-detection presets. "accuracy" is the Soniox default profile;
    # "low_latency" follows the official recommended low-latency starting
    # point; "balanced" is this project's proposed middle ground.
    "accuracy": dict(
        endpoint_latency_adjustment_level=0,
        endpoint_sensitivity=0.0,
        max_endpoint_delay_ms=2000,
    ),
    "balanced": dict(
        endpoint_latency_adjustment_level=1,
        endpoint_sensitivity=0.15,
        max_endpoint_delay_ms=2000,
    ),
    "low_latency": dict(
        endpoint_latency_adjustment_level=2,
        endpoint_sensitivity=0.3,
        max_endpoint_delay_ms=1500,
    ),
}


@dataclass
class SonioxRuntimeConfig:
    api_key: str
    context_text: str = ""
    segmentation: str = "accuracy"
    mixed_language: bool = False
    enable_language_identification: bool = False


class SonioxSink(Protocol):
    def on_live(self, state: SonioxLiveState) -> None: ...
    def on_segments(self, segments: list[SonioxSegment]) -> None: ...
    def on_status(self, status: SonioxStatus) -> None: ...
    def on_error(self, message: str) -> None: ...
    def on_metrics(self, metrics: dict) -> None: ...


def _import_soniox():
    """Lazy SDK import seam (tests monkeypatch this)."""
    try:
        from soniox import SonioxClient
        from soniox.types import RealtimeSTTConfig, TranslationConfig
    except ImportError as exc:
        raise SonioxNotInstalledError(
            "The 'soniox' package is not installed. "
            "Run: pip install \"soniox>=2.9,<3\""
        ) from exc
    return SonioxClient, RealtimeSTTConfig, TranslationConfig


def float32_to_pcm_s16le(chunk: np.ndarray) -> bytes:
    """Convert a float32 [-1, 1] mono block to little-endian PCM16 bytes,
    clipping out-of-range values (no wraparound)."""
    clipped = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def mask_key(key: str) -> str:
    """The only form of the API key that may ever reach a log."""
    if not key:
        return "(empty)"
    if len(key) <= 8:
        return f"{key[:2]}… ({len(key)} chars)"
    return f"{key[:4]}…{key[-2:]} ({len(key)} chars)"


def redact_key(text: str, key: str) -> str:
    """Strip the raw key from any exception/message text before logging.

    SDK and transport exceptions can embed the request URL (which carries
    the key as a query parameter) or the key itself; every log path in this
    module must pass its message through this filter.
    """
    if not key or not text:
        return text or ""
    return text.replace(key, mask_key(key))


def resolve_api_key(settings: dict | None) -> str | None:
    """SONIOX_API_KEY env var wins; else the settings-page value."""
    env_key = (os.environ.get("SONIOX_API_KEY") or "").strip()
    if env_key:
        return env_key
    stored = ((settings or {}).get("soniox_api_key") or "").strip()
    return stored or None


def parse_context(text: str) -> Any | None:
    """Parse the course-terms free text into a soniox StructuredContext.

    Lines of the form "source => target" become forced translation terms;
    every other non-empty line becomes a general "topic" context item; the
    first plain line doubles as the free-form background text. Empty input
    returns None (no context is sent at all — an empty context would be
    meaningless payload).
    """
    if not text or not text.strip():
        return None
    translation_terms: list[tuple[str, str]] = []
    topics: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=>" in line:
            source, _, target = line.partition("=>")
            source, target = source.strip(), target.strip()
            if source and target:
                translation_terms.append((source, target))
            continue
        topics.append(line)
    if not translation_terms and not topics:
        return None

    from soniox.types import (
        StructuredContext,
        StructuredContextGeneralItem,
        StructuredContextTranslationTerm,
    )

    return StructuredContext(
        general=[StructuredContextGeneralItem(key="topic", value=t) for t in topics],
        text=topics[0] if topics else None,
        translation_terms=[
            StructuredContextTranslationTerm(source=s, target=t)
            for s, t in translation_terms
        ],
    )


@dataclass
class _ManagerMetrics:
    sent_bytes: int = 0
    dropped_chunks: int = 0
    reconnects: int = 0
    queued_bytes: int = 0


class SonioxServiceManager:
    def __init__(
        self,
        config: SonioxRuntimeConfig,
        sink: SonioxSink,
        *,
        max_buffered_bytes: int = SONIOX_SAMPLE_RATE * 2 * 10,
        connect_timeout_sec: float = 10.0,
        max_reconnect_attempts: int = 5,
    ) -> None:
        self._config = config
        self._sink = sink
        self._max_buffered_bytes = max_buffered_bytes
        self._connect_timeout_sec = connect_timeout_sec
        self._max_reconnect_attempts = max_reconnect_attempts

        self._lock = threading.Lock()
        self._send_cond = threading.Condition(self._lock)
        # (bytes, sent_flag) pairs — popped from the left only after a
        # successful send, so the remainder is exactly the unsent audio.
        self._queue: deque[tuple[bytes, bool]] = deque()
        self._metrics = _ManagerMetrics()

        self._status = SonioxStatus.CONNECTING
        self._generation = 0
        self._stop_requested = False
        self._finish_requested = False
        self._pause_requested = False
        self._paused = False
        self._reconnect_requested = False
        self._restart_session = threading.Event()
        self._drained = threading.Event()

        self._supervisor_thread: threading.Thread | None = None
        self._send_thread: threading.Thread | None = None
        self._session: Any = None  # current SDK session (send-thread target)
        self._accumulator = SonioxAccumulator()
        self._last_event_key: Any = None  # identical-event drop
        # The error_type of the event that ended the last session (None =
        # clean close). Initialized here because the supervisor reads it
        # after _receive_loop returns, and a session can end without the
        # loop body ever running (e.g. shutdown closing the socket).
        self._last_error_type: str | None = None

    # ------------------------------------------------------------------
    # Public API (called from the Qt thread / capture thread / end thread)

    def start(self) -> None:
        """Spawn the supervisor thread. Non-blocking: connection happens on
        the supervisor; audio fed meanwhile is buffered (bounded)."""
        with self._lock:
            if self._supervisor_thread is not None:
                return
            self._stop_requested = False
            self._finish_requested = False
            self._set_status_locked(SonioxStatus.CONNECTING)
        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop, name="soniox-session", daemon=True
        )
        self._supervisor_thread.start()

    def feed(self, chunk: np.ndarray) -> None:
        """Capture thread: enqueue one audio block. Never blocks, never
        raises; overflow drops the oldest chunks (counted)."""
        try:
            data = float32_to_pcm_s16le(chunk)
        except Exception:  # noqa: BLE001 — capture thread must never raise
            return
        with self._lock:
            if self._stop_requested:
                return
            self._queue.append((data, False))
            self._trim_locked()
            self._metrics.queued_bytes = sum(len(d) for d, _ in self._queue)
            self._send_cond.notify_all()

    def pause(self) -> None:
        """Stop sending new audio, finalize current content, bounded-wait for
        the final segment, commit it. The connection stays open (the SDK
        session's own keepalive need is covered by its server-side timeout
        window; our finalize keeps it warm)."""
        with self._lock:
            if self._stop_requested or self._paused:
                return
            self._pause_requested = True
            self._send_cond.notify_all()
        self._drain_session("pause")
        with self._lock:
            self._pause_requested = False
            if not self._stop_requested:
                self._paused = True
                self._set_status_locked(SonioxStatus.PAUSED)

    def resume(self) -> None:
        with self._lock:
            if self._stop_requested or not self._paused:
                return
            self._paused = False
            self._set_status_locked(SonioxStatus.LIVE)
            self._send_cond.notify_all()

    def apply_config(self, **changes: Any) -> None:
        """Runtime config change (context / segmentation): the SDK sends the
        config per connection, so a graceful reconnect with the new config is
        the only way to apply it. The current session is finalized and its
        tail committed first (no segment is lost mid-flight)."""
        with self._lock:
            if self._stop_requested:
                return
            changed = False
            for key in ("context_text", "segmentation", "mixed_language"):
                if key in changes and changes[key] != getattr(self._config, key):
                    setattr(self._config, key, changes[key])
                    changed = True
            if not changed:
                return
        # Graceful: drain the old session, then reconnect.
        self._drain_session("apply_config")
        self._teardown_current_session()
        with self._lock:
            self._reconnect_requested = True
            self._send_cond.notify_all()

    def _teardown_current_session(self) -> None:
        """Close the current session so the supervisor's receive loop
        unblocks and reconnects with the (new) config."""
        with self._lock:
            self._restart_session.set()
            session = self._session
            self._send_cond.notify_all()
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                log.debug("Soniox session close raised", exc_info=True)

    def finalize_for_end(self, deadline: float) -> bool:
        """Session ENDING: stop accepting new audio (the capture gate is up),
        finalize the current utterance and commit the last segment. Bounded
        by ``deadline`` (a monotonic time). Returns True when clean."""
        budget = max(0.0, deadline - time.monotonic())
        ok = self._drain_session("end", timeout=budget)
        return ok

    def finish_and_drain(self, timeout: float = 5.0) -> bool:
        """Stop/quit path: signal end-of-audio, drain final events (committing
        the trailing segment), bounded by ``timeout``."""
        with self._lock:
            if self._stop_requested:
                return True
            self._finish_requested = True
            self._send_cond.notify_all()
        if self._supervisor_thread is not None:
            self._supervisor_thread.join(timeout=timeout + 2.0)
        with self._lock:
            return self._status == SonioxStatus.FINISHED

    def shutdown(self, timeout: float = 5.0) -> None:
        """Bounded, idempotent teardown. Never raises; never joins forever."""
        with self._lock:
            if self._stop_requested:
                return
            self._stop_requested = True
            self._finish_requested = True
            self._reconnect_requested = False
            self._send_cond.notify_all()
            self._restart_session.set()
        # Close whatever session exists NOW (the supervisor may have rotated
        # to a fresh one between our flag set and this read): close() sends
        # end-of-audio and closes the socket — the only way to unblock a
        # receive_events() parked on recv(). The loop-top stop check makes
        # any reconnect race exit before connecting again; a session created
        # in that window is closed on the retry below.
        for _ in range(3):
            with self._lock:
                session = self._session
            if session is None:
                break
            try:
                session.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                log.debug("Soniox session close during shutdown raised", exc_info=True)
            time.sleep(0.05)
        if self._supervisor_thread is not None:
            self._supervisor_thread.join(timeout=timeout + 2.0)
        send_thread = self._send_thread
        if send_thread is not None and send_thread is not threading.current_thread():
            send_thread.join(timeout=2.0)
        self._safe_call(lambda: self._sink.on_status(SonioxStatus.FINISHED))

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def bump_generation(self) -> None:
        with self._lock:
            self._generation += 1

    def metrics(self) -> dict:
        with self._lock:
            return dict(
                sent_bytes=self._metrics.sent_bytes,
                dropped_chunks=self._metrics.dropped_chunks,
                reconnects=self._metrics.reconnects,
                queued_bytes=sum(len(d) for d, _ in self._queue),
            )

    def status(self) -> SonioxStatus:
        with self._lock:
            return self._status

    # ------------------------------------------------------------------
    # Internals

    def _set_status_locked(self, status: SonioxStatus) -> None:
        # Called with self._lock held. The sink callback runs under the lock,
        # so sink implementations must never call back into this manager
        # (the app adapter only touches the overlay facade / Qt signals).
        if self._status != status:
            self._status = status
            self._safe_call(lambda: self._sink.on_status(status))

    def _safe_call(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001 — callbacks must not kill threads
            log.exception("Soniox sink callback raised")

    def _trim_locked(self) -> None:
        dropped = 0
        queued = sum(len(d) for d, _ in self._queue)
        while queued > self._max_buffered_bytes and len(self._queue) > 1:
            data, _ = self._queue.popleft()
            queued -= len(data)
            dropped += 1
        if dropped:
            self._metrics.dropped_chunks += dropped
            log.warning(
                "Soniox audio buffer overflow: dropped %d oldest chunks "
                "(network slower than realtime)", dropped,
            )

    def _build_config(self) -> Any:
        SonioxClient, RealtimeSTTConfig, TranslationConfig = _import_soniox()
        preset = SEGMENTATION_PRESETS.get(
            self._config.segmentation, SEGMENTATION_PRESETS["accuracy"]
        )
        hints = (
            SONIOX_MIXED_LANGUAGE_HINTS
            if self._config.mixed_language
            else SONIOX_LANGUAGE_HINTS
        )
        cfg = RealtimeSTTConfig(
            model=SONIOX_MODEL,
            audio_format="pcm_s16le",
            sample_rate=SONIOX_SAMPLE_RATE,
            num_channels=1,
            language_hints=hints,
            # strict is documented as "best results with one language hint";
            # with two hints (code-switching) the non-strict bias lets the
            # model follow the actual switches.
            language_hints_strict=not self._config.mixed_language,
            enable_endpoint_detection=True,
            translation=TranslationConfig(
                type="one_way", target_language=SONIOX_TARGET_LANGUAGE,
            ),
            **preset,
        )
        context = parse_context(self._config.context_text)
        if context is not None:
            cfg.context = context
        return cfg

    def _supervisor_loop(self) -> None:
        # The SonioxClient is created per connection attempt: its base client
        # owns an httpx pool we do not otherwise need, and reconnecting with
        # a fresh session is the documented pattern.
        try:
            self._supervisor_loop_inner()
        finally:
            # Whatever path the supervisor exits by, the send thread must be
            # released too (it only observes _stop_requested / the restart
            # event — neither is implied by a supervisor return).
            with self._lock:
                self._stop_requested = True
                self._send_cond.notify_all()
                self._restart_session.set()

    def _supervisor_loop_inner(self) -> None:
        attempts = 0
        while True:
            with self._lock:
                if self._stop_requested:
                    self._set_status_locked(SonioxStatus.FINISHED)
                    return
                self._reconnect_requested = False
                self._restart_session.clear()
                if self._finish_requested:
                    self._set_status_locked(SonioxStatus.FINISHED)
                    return
                current_status = self._status

            session = None
            try:
                SonioxClient, _, _ = _import_soniox()
                client = SonioxClient(api_key=self._config.api_key)
                cfg = self._build_config()
                if current_status != SonioxStatus.PAUSED:
                    with self._lock:
                        self._set_status_locked(
                            SonioxStatus.RECONNECTING
                            if attempts > 0
                            else SonioxStatus.CONNECTING
                        )
                session = client.realtime.stt.connect(
                    config=cfg, connect_timeout_sec=self._connect_timeout_sec
                )
                session.__enter__()
            except (SonioxNotInstalledError, ImportError):
                # Permanent condition (package missing): no amount of
                # retrying installs a dependency. ImportError is caught too
                # because the lazy-import seam raises it before wrapping
                # whenever the SDK is absent.
                log.error("Soniox SDK is not installed")
                with self._lock:
                    self._set_status_locked(SonioxStatus.FAILED)
                self._safe_call(
                    lambda: self._sink.on_error(
                        "The 'soniox' package is not installed; "
                        "see the settings page for install instructions"
                    )
                )
                return
            except Exception as exc:
                attempts += 1
                log.warning(
                    "Soniox connect failed (attempt %d/%d): %s",
                    attempts, self._max_reconnect_attempts,
                    redact_key(str(exc), self._config.api_key),
                )
                if self._should_stop_after_failure(attempts):
                    self._emit_connect_failure()
                    return
                backoff = RECONNECT_BACKOFF_S[
                    min(attempts - 1, len(RECONNECT_BACKOFF_S) - 1)
                ]
                if not self._sleep_interruptible(backoff):
                    return
                continue

            # Connected.
            attempts = 0
            with self._lock:
                if not self._paused and not self._finish_requested:
                    self._set_status_locked(SonioxStatus.LIVE)
                self._session = session
                self._last_event_key = None
                self._accumulator.reset()
                self._metrics.reconnects += 1
                # Snapshot the generation at session start: every result this
                # session later produces is checked against it, so a bump
                # (engine switch / teardown) retroactively invalidates the
                # whole in-flight session — capturing it per-event instead
                # would make the guard a tautology.
                session_generation = self._generation
            self._safe_call(lambda: self._sink.on_metrics(self.metrics()))

            send_thread = threading.Thread(
                target=self._send_loop, name="soniox-send", daemon=True
            )
            with self._lock:
                self._send_thread = send_thread
            send_thread.start()

            reason = self._receive_loop(session, session_generation)
            # Session ended (error, finish, or restart request).
            with self._lock:
                self._session = None
                self._send_thread = None
            send_thread.join(timeout=2.0)
            self._drained.set()

            if reason == "fatal":
                # Auth rejected: permanent — stop, surface, never retry.
                with self._lock:
                    self._set_status_locked(SonioxStatus.FAILED)
                return
            if self._last_error_type in ("max_duration_reached",
                                         "service_unavailable"):
                # Normal operational endings, not failures: the session cap
                # is 300 minutes (a double-header lecture exceeds it) and
                # the docs instruct an immediate new request. Reconnect at
                # once with the attempt budget intact.
                log.info(
                    "Soniox session ended (%s); reconnecting immediately",
                    self._last_error_type,
                )
                continue
            with self._lock:
                restart = self._reconnect_requested
                finish = (
                    self._finish_requested
                    or self._stop_requested
                    or reason == "finished"
                )
            if restart and not (self._stop_requested or self._finish_requested):
                # Config change / send-failure teardown: reconnect with the
                # (possibly updated) config, unsent audio replays.
                with self._lock:
                    self._reconnect_requested = False
                continue
            if finish:
                # Tail already committed by the receive loop (finished event).
                with self._lock:
                    self._set_status_locked(SonioxStatus.FINISHED)
                return
            if not restart:
                # Unexpected disconnect -> reconnect with backoff.
                attempts += 1
                log.warning(
                    "Soniox session dropped (reconnect attempt %d/%d)",
                    attempts, self._max_reconnect_attempts,
                )
                if self._should_stop_after_failure(attempts):
                    self._emit_connect_failure()
                    return
                if not self._sleep_interruptible(
                    RECONNECT_BACKOFF_S[
                        min(attempts - 1, len(RECONNECT_BACKOFF_S) - 1)
                    ]
                ):
                    return

    def _emit_connect_failure(self) -> None:
        """The reconnect budget is exhausted: surface an actionable error."""
        def _report() -> None:
            self._sink.on_error(
                "Soniox connection failed after repeated retries; "
                "check the network and the API key"
            )

        self._safe_call(_report)

    def _should_stop_after_failure(self, attempts: int) -> bool:
        """True when the supervisor must give up (stop/finish requested or
        the reconnect budget is exhausted)."""
        with self._lock:
            if self._stop_requested or self._finish_requested:
                self._set_status_locked(SonioxStatus.FINISHED)
                return True
            if attempts >= self._max_reconnect_attempts:
                self._set_status_locked(SonioxStatus.FAILED)
                return True
            return False

    def _sleep_interruptible(self, seconds: float) -> bool:
        """Sleep unless stop/finish/restart is requested. Returns False when
        the loop must exit."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            with self._lock:
                if self._stop_requested or self._finish_requested:
                    self._set_status_locked(SonioxStatus.FINISHED)
                    return False
                if self._restart_session.is_set():
                    return True
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        return True

    # Server-side error types that no reconnect can fix. The official
    # error-handling docs say to branch on error_type ("stable across
    # releases"), not error_code or the human-readable message — error_code
    # is kept as a pre-extra-fields fallback. The WebSocket handshake does
    # NOT validate the API key (verified live: a bogus key connects fine
    # and the 401/unauthenticated arrives as an in-session error event
    # ~0.2s later), so without this set a bad key loops
    # RECONNECTING->LIVE forever, hiding the real problem from the user.
    FATAL_ERROR_TYPES = frozenset({
        "unauthenticated",      # bad/expired key: retrying cannot fix it
        "api_key_invalid",
        "permission_denied",
    })

    def _receive_loop(self, session: Any, session_generation: int) -> str:
        """Iterate receive_events until the session closes. Returns the
        reason: 'finished' | 'error' | 'restart' | 'fatal'. Every dispatched
        result carries ``session_generation`` (the generation at connect
        time) so a later bump() drops the whole in-flight session's output."""
        try:
            for event in session.receive_events():
                error_code = getattr(event, "error_code", None)
                error_type = getattr(event, "error_type", None)
                self._last_error_type = error_type
                if error_type in self.FATAL_ERROR_TYPES or (
                    error_type is None and error_code in (401, 403)
                ):
                    message = (
                        getattr(event, "error_message", "") or "authentication failed"
                    )
                    log.error(
                        "Soniox auth error %s: %s",
                        error_type or error_code,
                        redact_key(str(message), self._config.api_key),
                    )
                    self._safe_call(
                        lambda m=message: self._sink.on_error(m)
                    )
                    return "fatal"
                if error_code:
                    message = (
                        getattr(event, "error_message", "") or "unknown error"
                    )
                    log.error(
                        "Soniox server error %s: %s",
                        error_code,
                        redact_key(str(message), self._config.api_key),
                    )
                    self._safe_call(lambda m=message: self._sink.on_error(m))
                    return "error"
                tokens = list(getattr(event, "tokens", []) or [])
                finished = bool(getattr(event, "finished", False))

                # Identical-event drop (transport-level duplicate defense).
                event_key = (
                    tuple(
                        (t.text, bool(t.is_final), t.translation_status)
                        for t in tokens
                    ),
                    finished,
                )
                with self._lock:
                    duplicate = self._last_event_key == event_key
                    self._last_event_key = event_key
                if duplicate:
                    continue

                result = self._accumulator.process_event(tokens, finished=finished)
                if result.committed:
                    self._safe_call(
                        lambda segs=result.committed: self._dispatch_segments(
                            session_generation, segs
                        )
                    )
                self._safe_call(
                    lambda live=result.live: self._dispatch_live(
                        session_generation, live
                    )
                )
                if finished:
                    return "finished"
                with self._lock:
                    if self._restart_session.is_set():
                        return "restart"
        except Exception as exc:
            if not (self._stop_requested or self._finish_requested):
                log.warning(
                    "Soniox receive loop ended: %s",
                    redact_key(str(exc), self._config.api_key),
                )
            return "error"
        # Generator ended without a finished event: the connection closed
        # (server- or client-initiated, e.g. our apply_config teardown) —
        # not a completed stream.
        return "closed"

    def _dispatch_live(self, generation: int, live: SonioxLiveState) -> None:
        if generation != self.generation():
            return
        self._sink.on_live(live)

    def _dispatch_segments(self, generation: int, segments: list[SonioxSegment]) -> None:
        if generation != self.generation():
            return
        self._sink.on_segments(segments)

    def _send_loop(self) -> None:
        silence = float32_to_pcm_s16le(
            np.zeros(
                int(SONIOX_SAMPLE_RATE * FINALIZE_SILENCE_S), dtype=np.float32
            )
        )
        while True:
            # Phase 1 (locked): decide what to do. Peek the leftmost chunk;
            # it is popped only after a successful send (the key invariant
            # separating sent from unsent audio). The actual send happens in
            # phase 2 WITHOUT the lock — a slow socket must never block
            # feed() or the drain/pause paths.
            with self._send_cond:
                while True:
                    if self._stop_requested:
                        return
                    if self._paused and not self._finish_requested:
                        # User pause: nothing leaves — queued audio waits for
                        # resume (the app gate should already stop feeding;
                        # this is belt-and-braces).
                        self._send_cond.wait(timeout=0.5)
                        continue
                    if self._queue:
                        # Send pending audio first — including on the finish
                        # path, so end-of-audio never discards the tail.
                        break
                    if self._finish_requested or self._pause_requested:
                        # One-shot boundaries: consume the flag atomically so
                        # the loop cannot spin re-issuing finalize/finish.
                        finish_boundary = self._finish_requested
                        self._finish_requested = False
                        self._pause_requested = False
                        session = self._session
                        if session is None:
                            # No live session: nothing to finalize. A finish
                            # request survives (the supervisor's loop-top
                            # check exits as FINISHED); a pause has nothing
                            # to wait for.
                            if finish_boundary:
                                self._finish_requested = True
                            self._drained.set()
                            self._send_cond.wait(timeout=0.2)
                            continue
                        self._send_cond.release()
                        try:
                            session.send_byte_chunk(silence)
                            if finish_boundary:
                                # End-of-audio: the server finalizes pending
                                # tokens and closes; the receive loop commits.
                                session.finish()
                            else:
                                session.finalize()
                        except Exception as exc:
                            self._handle_send_failure(exc, session)
                            return
                        finally:
                            self._send_cond.acquire()
                        self._drained.set()
                        continue
                    if self._restart_session.is_set():
                        return
                    self._send_cond.wait(timeout=0.5)

                session = self._session
                if session is None:
                    # Connecting: keep the chunk queued for the next session.
                    self._send_cond.wait(timeout=0.2)
                    continue
                chunk = self._queue[0]

            # Phase 2 (unlocked): the network send. Re-check the session and
            # the restart flag — a reconnect may have torn it down while we
            # waited for the lock.
            try:
                session.send_byte_chunk(chunk[0])
            except Exception as exc:
                self._handle_send_failure(exc, session)
                return

            # Phase 3 (locked): pop only after a successful send.
            with self._lock:
                if self._queue:
                    self._queue.popleft()
                    self._metrics.sent_bytes += len(chunk[0])
                    self._metrics.queued_bytes = sum(
                        len(d) for d, _ in self._queue
                    )

    def _handle_send_failure(self, exc: Exception, session: Any = None) -> None:
        """A send failed: close the session (unblocking the receive loop),
        then let the supervisor reconnect immediately and replay the unsent
        remainder."""
        if not self._stop_requested:
            log.warning(
                "Soniox send failed: %s",
                redact_key(str(exc), self._config.api_key),
            )
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                log.debug("Soniox session close after send failure", exc_info=True)
        self._reconnect_requested = True
        self._restart_session.set()

    def _drain_session(self, reason: str, timeout: float | None = None) -> bool:
        """Finalize the current session's tail and wait for the commit.
        Returns True when the drain completed within the budget."""
        with self._lock:
            if self._stop_requested:
                return True
            self._drained.clear()
            # Trigger the send thread's pause-boundary branch (silence +
            # finalize) without losing the LIVE status on success paths.
            self._pause_requested = True
            self._send_cond.notify_all()
        deadline = time.monotonic() + (
            timeout if timeout is not None else FINALIZE_QUIESCE_S
        )
        got = self._drained.wait(timeout=max(0.0, deadline - time.monotonic()))
        if not got:
            log.warning("Soniox %s drain timed out; flushing local finals", reason)
        # Whatever finals arrived so far are committed by the accumulator;
        # flush the remainder so nothing is silently lost.
        segment = self._accumulator.flush()
        if segment is not None:
            self._dispatch_segments(self.generation(), [segment])
        with self._lock:
            self._pause_requested = False
            # pause() re-raises this flag after the drain; apply_config and
            # the ENDING drain must not leave the manager wedged paused.
            self._paused = False
        return got
