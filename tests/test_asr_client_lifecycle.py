"""ASRClient lock discipline and status fidelity (CALL_CHAIN_FIX_TODO 2.6 / A7).

The client is exercised with fake process/pipe objects: ASRClient.__init__ does
not spawn anything, so injecting the two handles is enough to drive every
lifecycle path without a real worker.
"""

import threading
import time

import pytest

from asr_client import ASRClient, ASRWorkerExited


class FakeProcess:
    def __init__(self, alive=True, exitcode=None, pid=4242):
        self._alive = alive
        self.exitcode = exitcode
        self.pid = pid
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        return None


class FakeConn:
    """A pipe that never answers, so recv paths must end on cancellation."""

    def __init__(self, responses=None):
        self.sent = []
        self.closed = False
        self._responses = list(responses or [])

    def send(self, msg):
        self.sent.append(msg)

    def poll(self, timeout=None):
        if self.closed:
            raise OSError("handle is closed")
        if self._responses:
            return True
        if timeout:
            time.sleep(min(timeout, 0.01))
        return False

    def recv(self):
        return self._responses.pop(0)

    def close(self):
        self.closed = True


def _client(process=None, conn=None, status="ready"):
    client = ASRClient({"engine_type": "test"}, shutdown_timeout=0.1)
    client._process = process if process is not None else FakeProcess()
    client._conn = conn if conn is not None else FakeConn()
    client._status = status
    return client


def test_shutdown_does_not_wait_for_an_in_flight_request():
    """The whole point of splitting the lock: quit must not queue behind a
    120s transcribe."""
    client = _client()
    held = threading.Event()
    release = threading.Event()

    def hog():
        with client._io_lock:
            held.set()
            release.wait(5)

    worker = threading.Thread(target=hog, daemon=True)
    worker.start()
    assert held.wait(2)

    start = time.monotonic()
    client.shutdown()
    elapsed = time.monotonic() - start

    release.set()
    worker.join(2)
    # 0.5s handshake attempt plus bounded joins, nowhere near request_timeout.
    assert elapsed < 3.0
    assert client._process is None
    assert client.status == "stopped"


def test_shutdown_terminates_the_worker_when_the_pipe_is_busy():
    process = FakeProcess()
    conn = FakeConn()
    client = _client(process=process, conn=conn)
    held = threading.Event()
    release = threading.Event()

    # Must be a different thread: _io_lock is reentrant, so holding it here
    # would let shutdown() take the graceful path and prove nothing.
    def hog():
        with client._io_lock:
            held.set()
            release.wait(5)

    worker = threading.Thread(target=hog, daemon=True)
    worker.start()
    assert held.wait(2)
    try:
        client.shutdown()
    finally:
        release.set()
        worker.join(2)
    assert process.terminated
    assert conn.sent == []  # no handshake attempted on a busy pipe


def test_recv_response_gives_up_when_shutdown_is_requested():
    conn = FakeConn()
    client = _client(conn=conn)

    result = {}

    def receive():
        try:
            client._recv_response(timeout=60.0, expected_id="abc")
        except BaseException as exc:  # noqa: BLE001 - recorded for the assert
            result["exc"] = exc

    reader = threading.Thread(target=receive, daemon=True)
    reader.start()
    time.sleep(0.05)
    client._cancelled.set()
    reader.join(2)

    assert isinstance(result.get("exc"), ASRWorkerExited)


def test_recv_response_reports_a_closed_handle_as_worker_death():
    conn = FakeConn()
    conn.closed = True
    client = _client(conn=conn)
    with pytest.raises(ASRWorkerExited):
        client._recv_response(timeout=1.0, expected_id="abc")


def test_response_id_mismatch_is_recoverable_worker_death():
    """A desynchronized pipe must reach _run_asr's recovery path, which only
    catches ASRWorkerExited/Timeout — a bare ASRClientError stalls there."""
    conn = FakeConn(responses=[{"id": "other", "ok": True}])
    client = _client(conn=conn)
    with pytest.raises(ASRWorkerExited):
        client._recv_response(timeout=1.0, expected_id="expected")
    assert client.status != "ready"


def test_terminate_marks_failed_even_when_the_process_already_exited():
    process = FakeProcess(alive=False, exitcode=1)
    client = _client(process=process)
    client.terminate()
    assert client.status == "failed"
    assert not process.terminated  # nothing left to signal


def test_status_never_reports_ready_without_a_process():
    client = _client()
    client._close_handles()
    assert client.status != "ready"


def test_status_reports_a_crashed_worker_as_exited():
    client = _client(process=FakeProcess(alive=False, exitcode=3))
    assert client.status == "exited"


def test_shutdown_is_idempotent():
    client = _client()
    client.shutdown()
    client.shutdown()  # no process left; must not raise
    assert client.status == "stopped"


def test_remote_engine_stops_reporting_ready_once_unloaded():
    pytest.importorskip("httpx")
    from asr_remote import RemoteASRError, RemoteASREngine

    engine = RemoteASREngine.__new__(RemoteASREngine)
    engine._closed = False
    engine._client = type("C", (), {"close": lambda self: None})()
    engine._url = "http://example.invalid/transcribe"
    engine.language = None

    assert engine.status == "ready"
    engine.shutdown()
    assert engine.status != "ready"
    engine.terminate()  # unload twice must stay harmless

    import numpy as np

    with pytest.raises(RemoteASRError):
        engine.transcribe(np.zeros(16, dtype=np.float32))
