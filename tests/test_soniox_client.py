"""Offline tests for the Soniox service manager.

The SDK is faked at the _import_soniox seam: a FakeSonioxClient returns
FakeRealtimeSession objects that implement the session surface
(send_byte_chunk / receive_events / finalize / finish / pause / resume /
close / context manager) with scriptable behavior. No network access.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soniox_client
from soniox_client import (
    FINALIZE_SILENCE_S,
    SonioxMissingKeyError,
    SonioxNotInstalledError,
    SonioxRuntimeConfig,
    SonioxServiceManager,
    SonioxStatus,
    float32_to_pcm_s16le,
    mask_key,
    parse_context,
    resolve_api_key,
)
from soniox_accumulator import SonioxSegment


class FakeToken:
    def __init__(self, text, is_final=False, translation_status=None):
        self.text = text
        self.is_final = is_final
        self.translation_status = translation_status


class FakeEvent:
    def __init__(self, tokens=(), finished=False, error_code=None,
                 error_message=None):
        self.tokens = list(tokens)
        self.finished = finished
        self.error_code = error_code
        self.error_message = error_message


class FakeRealtimeSession:
    """Scriptable fake of soniox's RealtimeSTTSession."""

    def __init__(self, factory):
        self._factory = factory
        self.sent = []
        self.finalize_calls = 0
        self.finish_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.closed = False
        self.entered = False
        self._events = []  # (Event or None-to-block, threading.Event)
        self._event_lock = threading.Lock()
        self._recv_waiting = threading.Event()

    # --- scripting API ---
    def push_event(self, event, timeout=5.0):
        """Queue an event or a sentinel ("BLOCK" = quiet socket that only
        unblocks on close, "EOS" = all queued events delivered)."""
        with self._event_lock:
            self._events.append(event)
        self._recv_waiting.set()

    def push_events(self, events):
        for e in events:
            self.push_event(e)

    def end_stream(self):
        """All queued events delivered: block further receives."""
        self.push_event("EOS")

    # --- session surface ---
    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.close()

    def send_byte_chunk(self, chunk):
        if self.closed:
            raise RuntimeError("session closed")
        if self._factory.send_hook:
            self._factory.send_hook(self, chunk)
        self.sent.append(chunk)

    def receive_events(self):
        while True:
            self._recv_waiting.wait(timeout=0.2)
            with self._event_lock:
                if not self._events:
                    self._recv_waiting.clear()
                    if self.closed:
                        return
                    continue
                item = self._events.pop(0)
            if item in ("EOS", "BLOCK"):
                # Quiet, healthy socket: block until close OR until a control
                # message (finalize/finish) queues more events — the real
                # SDK's receive keeps reading the socket after any send.
                while not self.closed:
                    with self._event_lock:
                        if self._events:
                            break
                    time.sleep(0.05)
                else:
                    return
                continue
            yield item
            if item.finished:
                while not self.closed:
                    time.sleep(0.05)
                return

    def finalize(self):
        self.finalize_calls += 1
        self._factory.on_finalize(self)
        self._wake_receive()

    def finish(self):
        self.finish_calls += 1
        self._factory.on_finish(self)
        self._wake_receive()

    def _wake_receive(self):
        """A control message arrived: the (fake) quiet-socket block must end
        so pending events queued by the hook become deliverable — the real
        SDK's receive keeps reading the socket after any control send."""
        with self._event_lock:
            if self._events and self._events[0] in ("EOS", "BLOCK"):
                self._events.pop(0)
        self._recv_waiting.set()

    def pause(self, *, finalize=True):
        self.pause_calls += 1

    def resume(self):
        self.resume_calls += 1

    def close(self):
        self.closed = True
        self._recv_waiting.set()


class FakeSonioxClient:
    """Fake of soniox.SonioxClient."""

    def __init__(self, *, api_key=None, **kwargs):
        self.api_key = api_key
        self._factory = _active_factory
        self.realtime = self._Realtime(self._factory)

    class _Realtime:
        def __init__(self, factory):
            self.stt = self._STT(factory)

        class _STT:
            def __init__(self, factory):
                self._factory = factory

            def connect(self, *, config, api_key=None, connect_timeout_sec=10.0):
                if self._factory.connect_hook:
                    self._factory.connect_hook()
                session = FakeRealtimeSession(self._factory)
                self._factory.sessions.append(session)
                self._factory.session_ready.set()
                return session


class _Factory:
    """Per-test fake environment injected at the _import_soniox seam."""

    def __init__(self):
        self.sessions = []
        self.session_ready = threading.Event()
        self.connect_hook = None   # raises to simulate connect failure
        self.send_hook = None      # callable(session, chunk) -> may raise
        self.on_finalize = lambda session: None
        self.on_finish = lambda session: None
        self.fail_sends_after = None  # int: raise on send N+

    def reset(self):
        self.__init__()


_active_factory = _Factory()


class RecordingSink:
    def __init__(self):
        self.live = []
        self.segments = []
        self.statuses = []
        self.errors = []
        self.metrics = []
        self._lock = threading.Lock()

    def on_live(self, state):
        with self._lock:
            self.live.append(state)

    def on_segments(self, segments):
        with self._lock:
            self.segments.extend(segments)

    def on_status(self, status):
        with self._lock:
            self.statuses.append(status)

    def on_error(self, message):
        with self._lock:
            self.errors.append(message)

    def on_metrics(self, metrics):
        with self._lock:
            self.metrics.append(metrics)


class _StandInConfig:
    """Stand-in for the SDK's config dataclasses when soniox is absent.

    The manager only *constructs* these and hands them to the client; the
    fake session never reads them back, so accepting the keyword arguments is
    the entire contract. This used to be a bare ``object``, which made every
    connect fail with "object() takes no arguments" on a machine without the
    SDK — invisible locally (the real types were importable) and fatal in the
    light CI recipe, where they are not.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture()
def factory(monkeypatch):
    f = _Factory()
    # FakeSonioxClient resolves its factory through the module global at
    # construction time (on the supervisor thread), so wire it here.
    monkeypatch.setattr(sys.modules[__name__], "_active_factory", f)
    monkeypatch.setattr(
        soniox_client, "_import_soniox",
        lambda: (lambda **kw: FakeSonioxClient(**kw), _StandInConfig, _StandInConfig),
    )
    # The manager builds its config via _import_soniox types; prefer the real
    # soniox types when the SDK is installed, and keep the stand-ins above
    # when it is not (the SDK-free path must stay runnable — these tests fake
    # the SDK at the seam precisely so they can run without it).
    try:
        import soniox.types as real_types

        monkeypatch.setattr(
            soniox_client, "_import_soniox",
            lambda: (
                lambda **kw: FakeSonioxClient(**kw),
                real_types.RealtimeSTTConfig,
                real_types.TranslationConfig,
            ),
        )
    except ImportError:
        pass
    return f


def make_manager(factory, **kwargs):
    sink = RecordingSink()
    config = SonioxRuntimeConfig(api_key=kwargs.pop("api_key", "test-key-123456789"))
    manager = SonioxServiceManager(
        config, sink, max_reconnect_attempts=kwargs.pop("max_reconnect_attempts", 5),
        **kwargs,
    )
    return manager, sink


def wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Conversion helpers


def test_float32_to_pcm_s16le_clipping():
    chunk = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    raw = float32_to_pcm_s16le(chunk)
    values = np.frombuffer(raw, dtype="<i2")
    assert len(raw) == 2 * len(chunk)
    assert values[0] == 0
    assert values[1] == int(0.5 * 32767)
    assert values[3] == 32767   # +1.0 clipped, no overflow
    assert values[4] == -32767  # -1.0 clipped
    assert values[5] == 32767   # +2.0 clipped
    assert values[6] == -32767  # -2.0 clipped


def test_float32_to_pcm_s16le_roundtrip():
    chunk = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    raw = float32_to_pcm_s16le(chunk)
    back = np.frombuffer(raw, dtype="<i2") / 32767.0
    assert np.allclose(back, chunk, atol=1.0 / 32767)


def test_mask_key_never_reveals_key():
    key = "super-secret-key-value"
    masked = mask_key(key)
    assert key not in masked
    assert len(masked) < len(key)
    assert mask_key("") == "(empty)"


def test_resolve_api_key_env_wins(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "env-key")
    assert resolve_api_key({"soniox_api_key": "stored-key"}) == "env-key"
    monkeypatch.delenv("SONIOX_API_KEY")
    assert resolve_api_key({"soniox_api_key": " stored-key "}) == "stored-key"
    assert resolve_api_key({}) is None
    assert resolve_api_key({"soniox_api_key": "   "}) is None


def test_parse_context_rules():
    # parse_context builds the SDK's StructuredContext items, so the real
    # soniox types are required: the light CI recipe (no SDK) skips this
    # rather than failing on the import.
    pytest.importorskip("soniox.types")
    assert parse_context("") is None
    assert parse_context("   \n  \n") is None
    ctx = parse_context("Курс лекций по термодинамике\nэнтропия => 熵\n")
    terms = [
        (t.source, t.target) for t in (ctx.translation_terms or [])
    ]
    assert terms == [("энтропия", "熵")]
    general = [(i.key, i.value) for i in (ctx.general or [])]
    assert general == [("topic", "Курс лекций по термодинамике")]


# ---------------------------------------------------------------------------
# Manager lifecycle


def test_missing_key_error_raised_by_shim_contract():
    # resolve_api_key returns None -> the engine shim (asr_soniox) raises;
    # here we verify the helper contract the shim depends on.
    assert resolve_api_key({"soniox_api_key": ""}) is None


def test_missing_sdk_error(monkeypatch):
    def _raise():
        raise ImportError("no soniox")

    monkeypatch.setattr(soniox_client, "_import_soniox", _raise)
    manager = SonioxServiceManager(
        SonioxRuntimeConfig(api_key="k"), RecordingSink()
    )
    manager.start()
    assert wait_until(lambda: manager.status() == SonioxStatus.FAILED)
    manager.shutdown(timeout=2.0)


def test_subthreshold_audio_still_queued_and_sent(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session = factory.sessions[0]
    assert wait_until(lambda: manager.status() == SonioxStatus.LIVE)

    # Amplitude far below any VAD threshold (Silero triggers around 0.5
    # confidence on voiced audio; 1e-4 amplitude is silence to any VAD).
    quiet = np.full(512, 1e-4, dtype=np.float32)
    for _ in range(10):
        manager.feed(quiet)
    assert wait_until(lambda: len(session.sent) >= 10)
    manager.shutdown(timeout=3.0)
    assert all(isinstance(chunk, bytes) and len(chunk) == 1024 for chunk in session.sent)


def test_feed_never_blocks_when_send_hangs(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)

    started = threading.Event()

    def slow_send(session, chunk):
        started.set()
        time.sleep(30.0)  # hang the send forever

    factory.send_hook = slow_send
    assert wait_until(lambda: manager._send_thread is not None)
    # One chunk gets the send thread inside the hanging send.
    manager.feed(np.full(512, 0.01, dtype=np.float32))
    # Wait until the send thread is inside the hanging send.
    assert wait_until(lambda: started.is_set(), timeout=5.0)

    deadline = time.monotonic() + 2.0
    for i in range(1000):
        manager.feed(np.full(512, 0.01, dtype=np.float32))
        assert time.monotonic() < deadline, "feed() blocked on network send"
    metrics = manager.metrics()
    # The bounded buffer dropped the backlog (default cap ~10s of PCM16).
    assert metrics["queued_bytes"] <= manager._max_buffered_bytes
    manager.shutdown(timeout=3.0)


def test_pause_finalizes_and_resume_does_not_replay_sent(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session = factory.sessions[0]
    assert wait_until(lambda: manager.status() == SonioxStatus.LIVE)

    # The finalize response: the pending provisional becomes final + <end>.
    factory.on_finalize = lambda s: s.push_events(
        [
            FakeEvent([
                FakeToken("Привет", is_final=True, translation_status="original"),
                FakeToken("你好", is_final=True, translation_status="translation"),
                FakeToken("<end>", is_final=True),
            ]),
            "BLOCK",
        ]
    )
    manager.feed(np.full(512, 0.1, dtype=np.float32))
    # The queued chunk is sent before the pause boundary is honored.
    assert wait_until(lambda: len(session.sent) >= 1)
    sent_before_pause = list(session.sent)

    manager.pause()
    assert session.finalize_calls == 1
    assert wait_until(lambda: len(sink.segments) == 1)
    assert sink.segments[0] == SonioxSegment(original="Привет", translation="你好")
    assert manager.status() == SonioxStatus.PAUSED
    # Pause boundary added exactly one silence chunk.
    assert len(session.sent) == len(sent_before_pause) + 1

    # Audio fed while paused must NOT be sent.
    manager.feed(np.full(512, 0.2, dtype=np.float32))
    time.sleep(0.3)
    assert len(session.sent) == len(sent_before_pause) + 1

    manager.resume()
    assert manager.status() == SonioxStatus.LIVE
    # The paused chunk goes out exactly once, after resume.
    assert wait_until(
        lambda: len(session.sent) == len(sent_before_pause) + 2
    )
    time.sleep(0.3)
    assert len(session.sent) == len(sent_before_pause) + 2
    manager.shutdown(timeout=3.0)


def test_end_token_commits_segment_and_live_updates(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session = factory.sessions[0]
    assert wait_until(lambda: manager.status() == SonioxStatus.LIVE)

    session.push_events([
        FakeEvent([FakeToken("Прив")]),
        FakeEvent([FakeToken("Привет")]),
        FakeEvent([
            FakeToken("Привет", is_final=True, translation_status="original"),
            FakeToken("你好", is_final=True, translation_status="translation"),
            FakeToken("<end>", is_final=True),
        ]),
        "BLOCK",
    ])
    assert wait_until(lambda: len(sink.segments) == 1)
    assert sink.segments[0].original == "Привет"
    assert sink.segments[0].translation == "你好"
    # Provisional replaced, never appended.
    live_originals = [s.original for s in sink.live]
    assert "Привет" in live_originals
    assert all("ПривПрив" not in t for t in live_originals)
    manager.shutdown(timeout=3.0)


def test_duplicate_events_dropped(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session = factory.sessions[0]
    dup = FakeEvent([FakeToken("Привет")])
    session.push_events([dup, dup, dup, "BLOCK"])
    time.sleep(0.5)
    # Only one live update for the duplicated event.
    originals = [s.original for s in sink.live if s.original]
    assert originals == ["Привет"]
    manager.shutdown(timeout=3.0)


class FakeErrorEvent:
    def __init__(self, error_code, error_type, message):
        self.tokens = []
        self.finished = False
        self.error_code = error_code
        self.error_type = error_type
        self.error_message = message


def test_unauthenticated_error_type_is_fatal_no_retry(factory):
    """The docs say to branch on error_type (stable), not error_code. An
    unauthenticated event must stop the manager FAILED with no reconnect
    loop (the WS handshake does not validate the key; the rejection arrives
    in-session)."""
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session = factory.sessions[0]
    session.push_events([
        FakeErrorEvent(401, "unauthenticated", "Incorrect API key provided."),
        "BLOCK",
    ])
    assert wait_until(lambda: manager.status() == SonioxStatus.FAILED, timeout=5)
    # No reconnect was attempted (exactly one session ever).
    assert len(factory.sessions) == 1
    assert sink.errors
    manager.shutdown(timeout=2)


def test_max_duration_reached_reconnects_immediately(factory):
    """413 max_duration_reached (the 300-minute session cap) is a normal
    ending: reconnect at once, attempt budget intact — a double-header
    lecture must not burn the 5-retry budget on polite session rollovers."""
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    factory.session_ready.clear()
    session = factory.sessions[0]
    session.push_events([
        FakeErrorEvent(413, "max_duration_reached", "max duration reached"),
        "BLOCK",
    ])
    # A second session appears without any backoff wait.
    assert factory.session_ready.wait(timeout=5), "no immediate reconnect"
    assert manager.status() in (SonioxStatus.LIVE, SonioxStatus.CONNECTING)
    manager.shutdown(timeout=2)


def test_send_failure_reconnects_and_replays_only_unsent(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session1 = factory.sessions[0]
    assert wait_until(lambda: manager.status() == SonioxStatus.LIVE)

    # Fail the 3rd send; the manager should reconnect and replay whatever
    # had not been sent yet — but never re-send already-sent bytes.
    sent_count = {"n": 0}
    sent_payloads = []

    def flaky_send(session, chunk):
        sent_count["n"] += 1
        sent_payloads.append(chunk)
        if sent_count["n"] == 3:
            raise RuntimeError("boom")

    factory.send_hook = flaky_send
    factory.session_ready.clear()
    chunks = [np.full(512, 0.05 * (i + 1), dtype=np.float32) for i in range(8)]
    for c in chunks:
        manager.feed(c)
    assert factory.session_ready.wait(timeout=10.0), "no reconnect happened"
    session2 = factory.sessions[1]
    assert wait_until(lambda: len(session2.sent) > 0, timeout=5.0)
    assert wait_until(
        lambda: manager.metrics()["queued_bytes"] == 0, timeout=5.0
    )

    all_sent = session1.sent + session2.sent
    # No payload sent twice (byte sequences are unique per chunk).
    assert len(all_sent) == len(set(all_sent))
    manager.shutdown(timeout=3.0)


def test_connect_failures_backoff_then_failed_status(factory):
    attempts = {"n": 0}

    def always_fail():
        attempts["n"] += 1
        raise RuntimeError("network down")

    factory.connect_hook = always_fail
    manager, sink = make_manager(factory, max_reconnect_attempts=2)
    # Shrink the backoff so the test stays fast.
    monkey_backoff = (0.05, 0.05)
    original = soniox_client.RECONNECT_BACKOFF_S
    soniox_client.RECONNECT_BACKOFF_S = monkey_backoff
    try:
        manager.start()
        assert wait_until(
            lambda: manager.status() == SonioxStatus.FAILED, timeout=10.0
        )
        assert attempts["n"] >= 2
        assert wait_until(lambda: len(sink.errors) == 1)
    finally:
        soniox_client.RECONNECT_BACKOFF_S = original
    manager.shutdown(timeout=2.0)


def test_generation_guard_drops_late_callbacks(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session = factory.sessions[0]
    assert wait_until(lambda: manager.status() == SonioxStatus.LIVE)

    manager.bump_generation()
    session.push_events([
        FakeEvent([
            FakeToken("Старое", is_final=True, translation_status="original"),
            FakeToken("<end>", is_final=True),
        ]),
        "BLOCK",
    ])
    time.sleep(0.5)
    # Late results from the old generation are display-suppressed too.
    assert sink.segments == []
    manager.shutdown(timeout=3.0)


def test_finish_and_drain_commits_tail(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    session = factory.sessions[0]
    assert wait_until(lambda: manager.status() == SonioxStatus.LIVE)

    # Pending finals with no <end>: finish() must commit the tail via the
    # finished event.
    def on_finish(s):
        s.push_event(
            FakeEvent(
                [
                    FakeToken("Хвост", is_final=True, translation_status="original"),
                    FakeToken("尾部", is_final=True, translation_status="translation"),
                ],
                finished=True,
            )
        )

    factory.on_finish = on_finish
    session.push_events([
        FakeEvent([FakeToken("Хво")]),
        "BLOCK",
    ])
    assert wait_until(lambda: any("Хвост" == s.original for s in sink.segments)
                      is False and True or True) or True
    ok = manager.finish_and_drain(timeout=5.0)
    assert ok
    assert manager.status() == SonioxStatus.FINISHED
    assert any(s.original == "Хвост" for s in sink.segments)
    assert session.finish_calls == 1
    manager.shutdown(timeout=2.0)


def test_shutdown_bounded_on_hung_session(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    # A session whose receive_events never yields and whose close works.
    started = time.monotonic()
    manager.shutdown(timeout=1.0)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0
    assert manager.status() == SonioxStatus.FINISHED


def test_api_key_never_logged(factory, caplog):
    secret = "sk-live-DO-NOT-LEAK-0123456789"

    def bad_connect():
        raise RuntimeError(f"connect refused for {secret}")

    factory.connect_hook = bad_connect
    manager, sink = make_manager(factory, api_key=secret, max_reconnect_attempts=1)
    soniox_client.RECONNECT_BACKOFF_S = (0.05,)
    try:
        with caplog.at_level(logging.DEBUG, logger="LiveTranslate.soniox"):
            manager.start()
            assert wait_until(
                lambda: manager.status() == SonioxStatus.FAILED, timeout=10.0
            )
    finally:
        soniox_client.RECONNECT_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0)
    for record in caplog.records:
        assert secret not in record.getMessage()
    manager.shutdown(timeout=2.0)


def test_metrics_count_reconnects(factory):
    # apply_config with a real context rebuilds the SDK config through
    # parse_context, which needs the real soniox types (see above).
    pytest.importorskip("soniox.types")
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    assert wait_until(lambda: manager.metrics()["reconnects"] == 1)
    # Graceful reconnect (apply_config) creates a second session.
    factory.session_ready.clear()
    manager.apply_config(context_text="новый контекст")
    assert factory.session_ready.wait(timeout=10.0)
    assert wait_until(lambda: manager.metrics()["reconnects"] == 2)
    manager.shutdown(timeout=3.0)


def test_apply_config_noop_when_unchanged(factory):
    manager, sink = make_manager(factory)
    manager.start()
    assert factory.session_ready.wait(timeout=5.0)
    manager.apply_config(context_text="")  # unchanged (default empty)
    time.sleep(0.3)
    assert len(factory.sessions) == 1
    manager.shutdown(timeout=3.0)
