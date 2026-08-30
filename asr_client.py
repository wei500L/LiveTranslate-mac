import logging
import multiprocessing as mp
import threading
import time
import uuid
from multiprocessing.connection import Connection

from asr_worker import worker_main

log = logging.getLogger("LiveTranslate.ASRClient")


class ASRClientError(RuntimeError):
    pass


class ASRWorkerError(ASRClientError):
    def __init__(self, error: dict):
        self.error = error
        self.recoverable = bool(error.get("recoverable", True))
        super().__init__(error.get("message", "ASR worker error"))


class ASRWorkerTimeout(ASRClientError):
    pass


class ASRWorkerExited(ASRClientError):
    pass


class ASRClient:
    """Main-process proxy for a single ASR worker process."""

    def __init__(
        self,
        config: dict,
        ready_timeout: float = 180.0,
        request_timeout: float = 120.0,
        shutdown_timeout: float = 5.0,
    ):
        self.config = dict(config)
        self.ready_timeout = ready_timeout
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self._ctx = mp.get_context("spawn")
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        # Two locks, never one. _io_lock serializes pipe round-trips and can be
        # held for up to request_timeout/ready_timeout; _state_lock only guards
        # the _conn/_process/_status fields and is always constant-time.
        # Sharing a single lock made shutdown() wait behind an in-flight
        # transcribe, which put the Qt thread's freeze right back after
        # _run_asr had deliberately released its own lock to avoid it.
        self._io_lock = threading.RLock()
        self._state_lock = threading.RLock()
        # Set by shutdown()/terminate() so a blocked _recv_response gives up
        # instead of polling until its deadline.
        self._cancelled = threading.Event()
        self._status = "created"

    _TERMINAL_STATUSES = ("failed", "stopping", "stopped", "exited")

    @property
    def status(self) -> str:
        with self._state_lock:
            if self._process is not None and self._process.exitcode is not None:
                if self._status not in self._TERMINAL_STATUSES:
                    self._status = "exited"
            elif self._process is None and self._status not in self._TERMINAL_STATUSES:
                # Handles already closed but the status still claims the worker
                # is usable; callers reuse a client on `status == "ready"`.
                self._status = "exited"
            return self._status

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            return self._process.pid if self._process is not None else None

    def start(self):
        with self._state_lock:
            if self._process is not None:
                return
            self._cancelled.clear()
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            name = f"ASRWorker-{self.config.get('engine_type', 'unknown')}"
            process = self._ctx.Process(
                target=worker_main,
                args=(child_conn, self.config),
                name=name,
            )
            process.daemon = True
            process.start()
            child_conn.close()
            self._conn = parent_conn
            self._process = process
            self._status = "starting"
            log.info(f"ASR worker started: pid={process.pid}, name={name}")

    def wait_ready(self, timeout: float | None = None):
        timeout = self.ready_timeout if timeout is None else timeout
        with self._io_lock:
            self._ensure_started()
            self._set_status("loading")
            response = self._recv_response(timeout, expected_id=None)
            if not response.get("ok"):
                self._set_status("failed")
                raise ASRWorkerError(response.get("error") or {})
            if response.get("type") != "ready":
                self._set_status("failed")
                raise ASRClientError(
                    f"Unexpected ASR worker startup response: {response.get('type')}"
                )
            self._set_status("ready")
            log.info(
                f"ASR worker ready: pid={self.pid}, "
                f"{response.get('payload') or {}}"
            )
            return response.get("payload")

    def transcribe(self, audio, word_timestamps: bool = False, **kwargs):
        payload = {"audio": audio, "word_timestamps": word_timestamps}
        payload.update(kwargs)
        response = self._request("transcribe", payload, timeout=self.request_timeout)
        return response.get("payload")

    def set_language(self, language: str):
        self._request(
            "set_language",
            {"language": language},
            timeout=min(10.0, self.request_timeout),
        )

    def set_input_padding(self, pad_seconds):
        self._request(
            "set_input_padding",
            {"pad_seconds": pad_seconds},
            timeout=min(10.0, self.request_timeout),
        )

    def shutdown(self):
        """Stop the worker within a bounded time, gracefully when possible.

        Deliberately does NOT block on _io_lock: an in-flight transcribe can
        hold that for request_timeout (120s by default), and this runs on the
        Qt thread during quit and engine switches.
        """
        with self._state_lock:
            if self._process is None:
                return
            self._status = "stopping"
            process = self._process
            conn = self._conn

        # Tell any in-flight _recv_response to give up.
        self._cancelled.set()

        # Try for a clean handshake, but only if the pipe is free right now.
        graceful = self._io_lock.acquire(timeout=0.5)
        try:
            if graceful and process.is_alive() and conn is not None:
                msg_id = uuid.uuid4().hex
                try:
                    conn.send({"id": msg_id, "type": "shutdown", "payload": {}})
                    if conn.poll(self.shutdown_timeout):
                        try:
                            conn.recv()
                        except EOFError:
                            pass
                except (BrokenPipeError, EOFError, OSError):
                    pass
            elif not graceful:
                log.info(
                    "ASR worker busy; skipping graceful shutdown handshake "
                    "and terminating pid=%s", process.pid,
                )
        finally:
            if graceful:
                self._io_lock.release()

        process.join(timeout=self.shutdown_timeout)
        if process.is_alive():
            log.warning(f"ASR worker did not exit, terminating pid={process.pid}")
            process.terminate()
            process.join(timeout=self.shutdown_timeout)
        if process.is_alive():
            log.error("ASR worker ignored SIGTERM, killing pid=%s", process.pid)
            try:
                process.kill()
            except Exception:
                log.debug("kill() failed", exc_info=True)
            process.join(timeout=self.shutdown_timeout)

        with self._state_lock:
            self._close_handles()
            self._status = "stopped"
        log.info("ASR worker stopped")

    def terminate(self):
        """Kill the worker now. Same lock discipline as shutdown()."""
        self._cancelled.set()
        with self._state_lock:
            process = self._process
            # Unconditional: a process that already exited on its own must not
            # leave the status reading "ready", or _switch_asr_engine will reuse
            # a dead client and the resulting ASRClientError never reaches the
            # worker-recovery path.
            self._status = "failed"
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=self.shutdown_timeout)
        with self._state_lock:
            self._close_handles()

    def _set_status(self, status: str):
        with self._state_lock:
            self._status = status

    def _request(self, request_type: str, payload: dict, timeout: float):
        with self._io_lock:
            self._ensure_ready()
            msg_id = uuid.uuid4().hex
            with self._state_lock:
                conn = self._conn
            if conn is None:
                raise ASRWorkerExited("ASR worker pipe is already closed")
            try:
                conn.send({"id": msg_id, "type": request_type, "payload": payload})
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._set_status("exited")
                raise ASRWorkerExited(f"ASR worker pipe closed: {exc}") from exc

            with self._state_lock:
                previous_status = self._status
                if request_type == "transcribe":
                    self._status = "busy"
            try:
                response = self._recv_response(timeout, expected_id=msg_id)
            finally:
                with self._state_lock:
                    if self._status == "busy":
                        self._status = previous_status

            if not response.get("ok"):
                raise ASRWorkerError(response.get("error") or {})
            return response

    def _recv_response(self, timeout: float, expected_id: str | None):
        deadline = time.monotonic() + timeout
        while True:
            if self._cancelled.is_set():
                # shutdown()/terminate() is tearing the worker down; stop
                # polling instead of holding the caller until the deadline.
                self._set_status("exited")
                raise ASRWorkerExited("ASR worker shutdown was requested")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.terminate()
                raise ASRWorkerTimeout(
                    f"ASR worker response timed out after {timeout:g}s"
                )

            with self._state_lock:
                conn = self._conn
                process = self._process
            try:
                ready = conn is not None and conn.poll(min(0.2, remaining))
            except OSError as exc:
                # shutdown()/terminate() closed the handle out from under this
                # poll. That is a cancellation, not a protocol failure.
                self._set_status("exited")
                raise ASRWorkerExited("ASR worker pipe was closed") from exc
            if ready:
                try:
                    response = conn.recv()
                except (EOFError, OSError) as exc:
                    self._set_status("exited")
                    raise ASRWorkerExited("ASR worker pipe closed") from exc
                if expected_id is None or response.get("id") == expected_id:
                    return response
                # A stale/mismatched frame leaves the pipe desynchronized, so
                # every later request would mismatch too. Raise the worker-death
                # type instead of a bare ASRClientError: only the former reaches
                # _run_asr's recovery path, and the client is unusable either way.
                self._set_status("exited")
                raise ASRWorkerExited(
                    "ASR worker response id mismatch (pipe desynchronized): "
                    f"expected={expected_id}, got={response.get('id')}"
                )

            if process is not None and process.exitcode is not None:
                self._set_status("exited")
                raise ASRWorkerExited(
                    f"ASR worker exited with code {process.exitcode}"
                )
            if process is None:
                self._set_status("exited")
                raise ASRWorkerExited("ASR worker handles were closed")

    def _ensure_started(self):
        with self._state_lock:
            if self._process is None or self._conn is None:
                raise ASRClientError("ASR worker has not been started")

    def _ensure_ready(self):
        self._ensure_started()
        current = self.status
        if current != "ready":
            raise ASRClientError(f"ASR worker is not ready: {current}")

    def _close_handles(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._process = None
