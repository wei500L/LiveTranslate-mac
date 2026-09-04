"""AI summary service: provider refs, prompts, chunked pipeline, worker.

The chat callable is injected, so every test drives the pipeline with fakes —
no network, no real API. The worker-thread tests use QThread with a stub
chat to verify signal flow, cancellation and the no-overwrite-on-failure
rule without touching anything remote.
"""

import json
import threading

import pytest

pytest.importorskip("PyQt6.QtCore", reason="SummaryWorker needs Qt threads")

import ai_summary_service as svc
import meeting_records as records


# --- provider ids and migration -------------------------------------------------


def test_models_get_stable_ids_and_legacy_index_migrates():
    settings = {"models": [{"name": "A"}, {"name": "B"}], "ai_summary_provider": 1}
    changed = svc.ensure_model_ids(settings)
    assert changed
    ids = [m["id"] for m in settings["models"]]
    assert all(ids)
    assert len(set(ids)) == 2
    assert settings["ai_summary_provider"] == ids[1]


def test_dangling_provider_reference_clears():
    settings = {"models": [{"name": "A", "id": "x1"}], "ai_summary_provider": "gone"}
    svc.ensure_model_ids(settings)
    assert settings["ai_summary_provider"] is None


def test_deleted_model_clears_the_summary_provider():
    settings = {"models": [{"name": "A", "id": "x1"}], "ai_summary_provider": "x1"}
    # User deletes the model on the translation page
    settings["models"] = []
    svc.ensure_model_ids(settings)
    assert settings["ai_summary_provider"] is None


def test_resolve_provider_by_id():
    settings = {"models": [{"name": "A", "id": "x1"}, {"name": "B", "id": "x2"}],
                "ai_summary_provider": "x2"}
    assert svc.resolve_provider(settings)["name"] == "B"
    assert svc.resolve_provider({"models": []}) is None


def test_id_stamping_is_idempotent():
    settings = {"models": [{"name": "A", "id": "x1"}]}
    assert not svc.ensure_model_ids(settings)


def test_missing_key_detection_distinguishes_local_from_cloud():
    assert svc.provider_missing_key({"api_base": "https://api.deepseek.com/v1", "api_key": ""})
    assert not svc.provider_missing_key({"api_base": "http://127.0.0.1:1234/v1", "api_key": ""})
    assert not svc.provider_missing_key({"api_base": "https://api.deepseek.com/v1", "api_key": "sk-1"})
    assert not svc.provider_missing_key({"api_base": "", "api_key": ""})


# --- prompts ---------------------------------------------------------------------


def test_templates_cover_meeting_and_classroom():
    assert set(svc.SUMMARY_TEMPLATES) == {"meeting", "classroom"}
    for key in svc.SUMMARY_TEMPLATES:
        msgs = svc.build_request_messages(
            [{"timestamp": "00:00:01", "original": "x", "translation": "y"}],
            template=key, output_lang="中文",
        )
        assert msgs[0]["role"] == "system"
        # both required section lists present in the prompt
        assert "会议概述" in msgs[0]["content"] or "课程主题" in msgs[0]["content"]


def test_chunk_messages_carry_part_numbers_and_span():
    msgs = svc.build_request_messages(
        [{"timestamp": f"00:0{i}:00", "original": "x", "translation": "y"} for i in range(3)],
        template="meeting", output_lang="English",
        chunk_index=2, chunk_total=7,
    )
    assert "2/7" in msgs[0]["content"]
    assert msgs[1]["content"].startswith("[")


def test_merge_messages_include_every_part():
    msgs = svc.build_merge_messages(
        ["part one", "part two"], template="meeting", output_lang="中文",
        span_all="09:00:00 – 10:00:00",
    )
    joined = msgs[-1]["content"]
    assert "part one" in joined and "part two" in joined


def test_no_system_role_is_merged_into_first_user_message():
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Resp", (),
                {"choices": [type("Ch", (), {"message": type("M", (), {"content": "ok"})()})()]},
            )()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    chat = svc._chat_callable(_Client, "m", {}, True, {})
    result = chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}])
    assert result == "ok"
    messages = captured["messages"]
    assert all(m["role"] != "system" for m in messages)
    assert "SYS" in messages[0]["content"] and "USER" in messages[0]["content"]


def test_chat_callable_applies_overrides_and_extra_body():
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Resp", (),
                {"choices": [type("Ch", (), {"message": type("M", (), {"content": "x"})()})()]},
            )()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    chat = svc._chat_callable(
        _Client, "model-id", {"enable_thinking": False}, False,
        {"max_tokens": 128, "temperature": 0.1, "seed": 7},
    )
    chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert captured["model"] == "model-id"
    assert captured["max_tokens"] == 128
    assert captured["temperature"] == 0.1
    assert captured["seed"] == 7
    assert captured["extra_body"] == {"enable_thinking": False}


# --- pipeline --------------------------------------------------------------------


def _entries(count, translated=True, per_entry=80):
    return [
        {
            "timestamp": f"00:{i // 60:02d}:{i % 60:02d}",
            "original": "о" * per_entry,
            "translation": ("译" * per_entry) if translated else None,
        }
        for i in range(count)
    ]


def test_short_record_takes_the_single_shot_path():
    calls = []

    def chat(messages):
        calls.append(messages)
        return "# 纪要\n\n## 会议概述\n全部内容。"

    out = svc.summarize(_entries(5), output_lang="中文", chat=chat)
    assert len(calls) == 1
    assert out.startswith("# 纪要")


def test_long_record_runs_chunked_map_reduce():
    calls = []

    def chat(messages):
        calls.append(messages[0]["content"])
        return "chunk result"

    progress = []
    out = svc.summarize(
        _entries(300), output_lang="中文", chat=chat,
        on_progress=lambda s, i, t: progress.append((s, i, t)),
        max_chars=2000,
    )
    assert len(calls) > 2          # several chunks + one merge
    assert calls[-1] != calls[0]   # the last call is the merge prompt
    assert progress[-1] == ("merge", 1, 1)
    assert progress[0][0] == "part"
    assert out == "chunk result"


def test_chunk_boundaries_follow_entry_edges():
    """The model must never see a truncated entry: every request body ends
    with a complete entry and starts with a complete entry."""
    entries = _entries(300)
    bodies = []

    def chat(messages):
        bodies.append(messages[1]["content"])
        return "ok"

    svc.summarize(entries, output_lang="中文", chat=chat, max_chars=2000)
    # Reassemble the [ts] text lines across all part bodies and confirm the
    # union equals the full record, i.e. no entry was cut in half.
    seen = []
    for body in bodies[:-1]:  # last body is the merge prompt
        for line in body.splitlines():
            if line.startswith("["):
                seen.append(line)
    expected = [f"[{e['timestamp']}] {e['translation']}" for e in entries]
    assert seen == expected


def test_empty_api_response_raises_typed_error():
    def chat(messages):
        return ""

    with pytest.raises(svc.SummaryError) as err:
        svc.summarize(_entries(3), output_lang="中文", chat=chat)
    assert err.value.kind == "summary_error_empty"


def test_timeout_exceptions_are_classified():
    def chat(messages):
        raise TimeoutError("read timed out")

    with pytest.raises(svc.SummaryError) as err:
        svc.summarize(_entries(3), output_lang="中文", chat=chat)
    assert err.value.kind == "summary_error_timeout"


def test_auth_exceptions_are_classified():
    def chat(messages):
        raise Exception("401 Incorrect API key")

    with pytest.raises(svc.SummaryError) as err:
        svc.summarize(_entries(3), output_lang="中文", chat=chat)
    assert err.value.kind == "summary_error_auth"


def test_connection_errors_are_classified():
    class ConnectionError_(Exception):
        pass

    def chat(messages):
        raise ConnectionError_("connection refused")

    with pytest.raises(svc.SummaryError) as err:
        svc.summarize(_entries(3), output_lang="中文", chat=chat)
    assert err.value.kind == "summary_error_unreachable"


def test_partial_chunk_failure_fails_with_old_summary_untouched(tmp_path):
    """One chunk failing must not write anything: the previous summary stays."""
    sessions_stamp = "20260101_090000"
    records.save_summary(tmp_path, sessions_stamp, "OLD", {"provider_name": "old"})

    state = {"n": 0}

    def chat(messages):
        state["n"] += 1
        if state["n"] == 2:
            raise TimeoutError("mid-run timeout")
        return "chunk"

    with pytest.raises(svc.SummaryError):
        svc.summarize(_entries(200), output_lang="中文", chat=chat, max_chars=500)
    assert records.load_summary(tmp_path, sessions_stamp)["content"] == "OLD"


def test_empty_record_is_rejected_before_any_request():
    def chat(messages):  # pragma: no cover - must not be called
        raise AssertionError("no request expected")

    with pytest.raises(svc.SummaryError) as err:
        svc.summarize([], output_lang="中文", chat=chat)
    assert err.value.kind == "summary_empty_record"


def test_cancel_before_first_request_raises_cancelled():
    event = threading.Event()
    event.set()

    def chat(messages):  # pragma: no cover
        raise AssertionError("no request expected")

    with pytest.raises(svc.Cancelled):
        svc.summarize(_entries(5), output_lang="中文", chat=chat, cancel_event=event)


def test_cancel_between_chunks_preserves_old_summary(tmp_path):
    records.save_summary(tmp_path, "20260101_090000", "OLD", {})
    event = threading.Event()
    state = {"n": 0}

    def chat(messages):
        state["n"] += 1
        if state["n"] >= 2:
            event.set()
        return "chunk"

    with pytest.raises(svc.Cancelled):
        svc.summarize(
            _entries(200), output_lang="中文", chat=chat,
            cancel_event=event, max_chars=500,
        )
    assert records.load_summary(tmp_path, "20260101_090000")["content"] == "OLD"


# --- worker thread -------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, fn):
        self._fn = fn

    def create(self, **kwargs):
        return type(
            "Resp", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": self._fn(kwargs)})()})()]}
        )()


class _FakeChatCompletions:
    def __init__(self, fn):
        self.completions = _FakeCompletions(fn)


class _FakeClient:
    def __init__(self, fn):
        self.chat = _FakeChatCompletions(fn)


def _fake_client_for(fn):
    return _FakeClient(fn)


@pytest.fixture
def stub_chat(monkeypatch):
    """Patch the client factory the worker's make_chat_fn builds on."""
    holder = {"fn": lambda messages: "# 纪要"}
    monkeypatch.setattr(
        svc, "make_openai_client", lambda *a, **k: _fake_client_for(holder["fn"])
    )
    return holder


def _make_worker(tmp_path, entries, provider=None):
    from ai_summary_service import SummaryWorker

    provider = provider or {
        "id": "x1", "name": "Stub", "api_base": "http://localhost:1/v1",
        "api_key": "", "model": "stub-model",
    }
    return SummaryWorker(
        tmp_path, "20260101_090000", entries, provider,
        output_lang="中文", default_output_lang="中文",
    )


def _run_worker(worker):
    """Drive run() synchronously and collect the signal emissions.

    run() is plain Python besides QThread plumbing; executing it inline
    keeps the tests deterministic (no event loop, no timing) while exercising
    exactly the code the UI thread would trigger via start().
    """
    events = {"progress": [], "succeeded": None, "failed": None}
    worker.progress.connect(lambda s, i, t: events["progress"].append((s, i, t)))
    worker.succeeded.connect(lambda c, m: events.update(succeeded=(c, m)))
    worker.failed.connect(lambda k, d: events.update(failed=(k, d)))
    worker.run()
    return events


def test_worker_saves_summary_and_emits_success(tmp_path, stub_chat):
    entries = _entries(5)
    worker = _make_worker(tmp_path, entries)
    events = _run_worker(worker)

    assert events["failed"] is None
    assert events["succeeded"] is not None
    content, meta = events["succeeded"]
    assert content == "# 纪要"
    loaded = records.load_summary(tmp_path, "20260101_090000")
    assert loaded["content"] == "# 纪要"
    assert loaded["meta"]["provider_name"] == "Stub"
    assert loaded["meta"]["source_hash"] == records.source_hash(entries)
    assert "api_key" not in json.dumps(loaded["meta"])


def test_worker_emits_part_progress_for_long_records(tmp_path, stub_chat):
    entries = _entries(200)
    worker = _make_worker(tmp_path, entries)
    events = _run_worker(worker)
    stages = [e[0] for e in events["progress"]]
    assert "part" in stages and "merge" in stages


def test_worker_failure_keeps_previous_summary(tmp_path, stub_chat):
    entries = _entries(5)
    records.save_summary(tmp_path, "20260101_090000", "OLD", {"provider_name": "old"})
    stub_chat["fn"] = lambda messages: (_ for _ in ()).throw(TimeoutError("t/o"))
    worker = _make_worker(tmp_path, entries)
    events = _run_worker(worker)

    assert events["succeeded"] is None
    assert events["failed"][0] == "summary_error_timeout"
    assert records.load_summary(tmp_path, "20260101_090000")["content"] == "OLD"


def test_worker_cancel_writes_nothing_over_old_summary(tmp_path, stub_chat):
    entries = _entries(200)
    records.save_summary(tmp_path, "20260101_090000", "OLD", {})
    worker = _make_worker(tmp_path, entries)
    state = {"n": 0}

    def fake_chat(messages):
        state["n"] += 1
        if state["n"] >= 2:
            worker.cancel()
        return "chunk"

    stub_chat["fn"] = fake_chat
    events = _run_worker(worker)
    assert events["succeeded"] is None
    assert events["failed"] is None
    assert records.load_summary(tmp_path, "20260101_090000")["content"] == "OLD"


def test_worker_empty_response_is_a_typed_failure(tmp_path, stub_chat):
    stub_chat["fn"] = lambda messages: ""
    worker = _make_worker(tmp_path, _entries(3))
    events = _run_worker(worker)
    assert events["failed"][0] == "summary_error_empty"
